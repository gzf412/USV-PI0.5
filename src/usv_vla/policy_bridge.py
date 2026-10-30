"""Bounded asynchronous RPC to a model-Python worker, independent of Gazebo.

The simulation Python never imports JAX. Pending inference means zero thrust;
responses are consumed once and must still satisfy wall/source-time budgets.
"""
import base64
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image


def validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Policy metadata must be an object")
    if type(metadata.get("state_dim")) is not int or metadata["state_dim"] not in (33, 39):
        raise ValueError("Policy metadata requires state_dim=33 or 39")
    for key, expected in (("action_dim", 2), ("action_units", "N")):
        if metadata.get(key) != expected or isinstance(metadata.get(key), bool):
            raise ValueError(f"Policy metadata requires {key}={expected!r}")
    return metadata


def first_action(actions):
    array = np.asarray(actions, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 2:
        raise ValueError("Policy output must be a nonempty [horizon, 2] action block")
    if not np.isfinite(array).all():
        raise ValueError("Policy returned non-finite thrust")
    if np.any(np.abs(array) > 15):
        raise ValueError("Policy returned thrust outside executed +/-15 N limits")
    return array[0].tolist()


def request_message(request_id, state, rgb, prompt, decision_sim_ns, decision_wall_ns, sources):
    state = np.asarray(state, dtype=np.float32)
    rgb = np.asarray(rgb)
    if state.shape not in ((33,), (39,)) or not np.isfinite(state).all():
        raise ValueError("Policy requires finite raw state[33] or state[39]")
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Policy requires HWC uint8 RGB")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Policy requires a task prompt")
    if not sources:
        raise ValueError("Policy requires causal sensor provenance")
    for source in sources.values():
        if source["source_time_ns"] > decision_sim_ns or source["arrival_monotonic_ns"] > decision_wall_ns:
            raise ValueError("Future sensor input is forbidden")
    # JPEG keeps IPC bounded; model-side transforms apply the training resize.
    encoded = io.BytesIO()
    Image.fromarray(rgb).save(encoded, format="JPEG", quality=95)
    return dict(type="infer", id=request_id, state=state.tolist(), prompt=prompt,
                image_jpeg=base64.b64encode(encoded.getvalue()).decode("ascii"),
                decision_sim_time_ns=int(decision_sim_ns), decision_monotonic_ns=int(decision_wall_ns),
                sources=sources)


class PolicyBridge:
    def __init__(self, *, python=None, host="127.0.0.1", port=8000, timeout_s=.5,
                 max_source_age_s=.75, startup_timeout_s=15., log_path=None, worker_command=None):
        if not all(np.isfinite(value) and value > 0 for value in (timeout_s, max_source_age_s, startup_timeout_s)):
            raise ValueError("Policy time budgets must be positive")
        self.timeout_ns = int(timeout_s*1e9)
        self.max_age_ns = int(max_source_age_s*1e9)
        self.pending = None
        self.worker_error = None
        self.next_id = 0
        self.incoming, self.outgoing = queue.Queue(), queue.Queue(maxsize=1)
        self.log = open(log_path, "w") if log_path is not None else subprocess.DEVNULL
        command = worker_command or [python or os.environ.get("USV_MODEL_PYTHON", sys.executable),
                                      "-m", "usv_vla.policy_worker", "--host", host, "--port", str(port)]
        environment = os.environ.copy()
        root = Path(__file__).resolve().parents[2]
        environment["PYTHONPATH"] = str(root/"src")+os.pathsep+environment.get("PYTHONPATH", "")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1, env=environment)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._write, daemon=True).start()
        try:
            try:
                ready = self.incoming.get(timeout=startup_timeout_s)
            except queue.Empty as error:
                raise TimeoutError("Policy worker startup exceeded deadline; check model Python and server") from error
            if ready.get("type") != "ready":
                raise RuntimeError(f"Policy worker startup failed: {ready}")
            self.metadata = validate_metadata(ready.get("metadata"))
        except Exception:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if message.get("type") == "error":
                    self.worker_error = message.get("error")
                self.incoming.put(message)
            self.incoming.put(dict(type="error", error="Policy worker disconnected"))
        except Exception as error:
            self.incoming.put(dict(type="error", error=f"Invalid worker protocol: {error}"))

    def _write(self):
        try:
            while True:
                message = self.outgoing.get()
                if message is None:
                    return
                self.process.stdin.write(json.dumps(message, allow_nan=False)+"\n")
                self.process.stdin.flush()
        except Exception as error:
            self.incoming.put(dict(type="error", error=f"Policy worker pipe failed: {error}"))

    def check_deadline(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if self.pending and now_ns-self.pending["submitted_ns"] > self.timeout_ns:
            raise TimeoutError("Policy inference exceeded wall-clock budget")
        if self.process.poll() is not None:
            detail = f": {self.worker_error}" if self.worker_error else "; see policy_worker.log"
            raise RuntimeError("Policy worker exited"+detail)

    def submit(self, state, rgb, prompt, decision_sim_ns, decision_wall_ns, sources, row_index):
        self.check_deadline()
        if np.asarray(state).shape != (self.metadata['state_dim'],):
            raise ValueError('Observation state width does not match policy metadata')
        if self.pending is not None:
            raise RuntimeError("Only one policy request may be in flight")
        started_ns = time.monotonic_ns()
        if decision_wall_ns > started_ns:
            raise ValueError("Policy decision wall time is in the future")
        if started_ns-decision_wall_ns > self.timeout_ns:
            raise RuntimeError("Policy decision is already too old")
        message = request_message(self.next_id, state, rgb, prompt, decision_sim_ns, decision_wall_ns, sources)
        oldest_ns = min(source["source_time_ns"] for source in sources.values())
        if decision_sim_ns-oldest_ns > self.max_age_ns:
            raise RuntimeError("Policy observation is already stale")
        self.pending = dict(id=self.next_id, submitted_ns=int(decision_wall_ns),
                            decision_sim_time_ns=int(decision_sim_ns), oldest_source_ns=int(oldest_ns),
                            observation_row_index=row_index)
        self.next_id += 1
        self.outgoing.put_nowait(message)
        return dict(status="submitted", **self.pending)

    def poll(self, current_sim_ns, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        self.check_deadline(now_ns)
        try:
            response = self.incoming.get_nowait()
        except queue.Empty:
            return None
        if response.get("type") != "result":
            raise RuntimeError(f"Policy worker failed: {response.get('error', response)}")
        pending = self.pending
        if pending is None or response.get("id") != pending["id"]:
            raise RuntimeError("Policy response does not match the current observation")
        if current_sim_ns < pending["decision_sim_time_ns"]:
            raise RuntimeError("Policy source time is in the future")
        if current_sim_ns-pending["oldest_source_ns"] > self.max_age_ns:
            raise RuntimeError("Policy result rejected: source observation is stale")
        action = first_action(response.get("actions"))
        self.pending = None
        info = dict(status="executed_first_step", **pending,
                    latency_s=(now_ns-pending["submitted_ns"])/1e9,
                    worker_rpc_s=response.get("rpc_s"),
                    source_age_s=(current_sim_ns-pending["oldest_source_ns"])/1e9)
        return action, info

    def close(self):
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if hasattr(self.log, "close"):
            self.log.close()


class CommandLease:
    """A wall-clock watchdog independent of inference and sensor callbacks."""
    def __init__(self, publisher, timeout_s=.25):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Command lease must be positive")
        self.publisher, self.timeout_s = publisher, timeout_s
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.deadline, self.failure = None, None
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def publish(self, command):
        with self.lock:
            if self.failure and any(command):
                raise RuntimeError(self.failure)
            published_ns = self.publisher(command)
            self.deadline = time.monotonic()+self.timeout_s if any(command) else None
            return published_ns

    def _watch(self):
        while not self.stop.wait(min(.01, self.timeout_s/4)):
            with self.lock:
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.publisher([0., 0.])
                    self.deadline = None
                    self.failure = "Policy command lease expired; thrust set to zero"

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.publish([0., 0.])

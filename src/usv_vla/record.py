"""Record one isolated dynamic Gazebo episode with causal observations.

Initial state source is explicitly privileged simulation pose, not GPS fusion.
Manual input is an intent mailbox; this process alone publishes motor commands.
"""

import argparse
from collections import defaultdict, deque
import contextlib
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from PIL import Image

from usv_vla.core import TaskConfig, TaskEvaluator, ScriptedTeacher, body_xy, limited_action, pack_state
from usv_vla.world import add_arguments, from_arguments
from usv_vla.policy_bridge import CommandLease, PolicyBridge
from usv_vla.paths import OUTPUTS
from usv_vla.video import export_episode_videos, resolve_ffmpeg


def stamp(msg):
    return msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nsec


def pose_stamp(message):
    # Gazebo PosePublisher stamps each Pose, leaving Pose_V.header empty.
    root = next((pose for pose in message.pose if pose.name == "small_cora_float"), None)
    if root is None:
        raise ValueError("Root boat pose missing from Pose_V.")
    return stamp(root)


def scheduled_decision(clock_ns, previous_ns):
    """Keep causal 10 Hz ticks despite small transport/scheduler jitter.

    Never catch up a missed control interval by inventing commands. A tick may
    execute at most 50 ms late; both target time and actual clock are recorded.
    """
    if clock_ns < 2_500_000_000:
        return None
    if previous_ns < 0:
        return clock_ns
    target = previous_ns+100_000_000
    if clock_ns < target:
        return None
    if clock_ns-target > 50_000_000:
        raise RuntimeError("Control tick exceeded the 50ms simulation-time lateness budget")
    return target


def policy_task_instruction(task):
    """Return a deployable task, keeping evaluator-only progress out of inference."""
    route = getattr(task, 'route', None)
    if route is not None:
        instruction = route.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip() or instruction != instruction.strip():
            raise ValueError('Policy inference requires an explicit route instruction.')
    else:
        # Legacy non-route TaskEvaluator builds its user instruction before this
        # fixed evaluator suffix. Require the complete format rather than
        # accepting arbitrary text if the legacy evaluator changes.
        context = task.context()
        instruction, marker, suffix = context.partition(' Current phase: ')
        if (not marker or not instruction or
                'Frozen approach reference relative to episode initial heading:' not in suffix):
            raise ValueError('Cannot safely extract the legacy task instruction.')
    if 'Current phase:' in instruction or 'frozen initial route heading:' in instruction:
        raise ValueError('Policy instruction contains evaluator-only context.')
    return instruction


def rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1., min(1., 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


class SensorStore:
    def __init__(self):
        self.condition = threading.Condition()
        self.messages = defaultdict(lambda: deque(maxlen=120))
        self.collision = False
        self.request = None
        self.request_arrival_ns = None

    def add(self, kind, source_ns, msg, arrival_ns=None):
        arrival_ns = time.monotonic_ns() if arrival_ns is None else arrival_ns
        with self.condition:
            self.messages[kind].append((source_ns, arrival_ns, msg))
            self.condition.notify_all()

    def select(self, kind, source_cutoff_ns, arrival_cutoff_ns, max_age_ns):
        candidates = [entry for entry in self.messages[kind]
                      if entry[0] <= source_cutoff_ns and entry[1] <= arrival_cutoff_ns
                      and source_cutoff_ns-entry[0] <= max_age_ns]
        return max(candidates, key=lambda entry: (entry[0], entry[1])) if candidates else None


class RawWriter:
    def __init__(self, output, *, record_aerial=False, jpeg_quality=95):
        self.output = output
        self.record_aerial = record_aerial
        self.jpeg_quality = jpeg_quality
        (output/"images").mkdir()
        if record_aerial:
            (output/"aerial_images").mkdir()
        self.items = queue.Queue(maxsize=512)
        self.error = None
        self.counts = defaultdict(int)
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def put(self, kind, source_ns, arrival_ns, message):
        if self.error:
            return
        try:
            self.items.put_nowait((kind, source_ns, arrival_ns, message))
        except queue.Full:
            self.error = RuntimeError("Raw recorder overflow; episode is invalid.")

    def work(self):
        from google.protobuf.json_format import MessageToDict
        with contextlib.ExitStack() as stack:
            raw = stack.enter_context((self.output/"raw_sensors.jsonl").open("w"))
            frames = stack.enter_context((self.output/"frames.jsonl").open("w"))
            aerial_frames = stack.enter_context((self.output/"aerial_frames.jsonl").open("w")) if self.record_aerial else None
            while True:
                item = self.items.get()
                try:
                    if item is None:
                        return
                    kind, source_ns, arrival_ns, msg = item
                    self.counts[kind] += 1
                    row = dict(kind=kind, source_time_ns=source_ns, arrival_monotonic_ns=arrival_ns)
                    if kind in ("image", "aerial_image"):
                        if msg.pixel_format_type != 3:  # gz.msgs.PixelFormatType.RGB_INT8
                            raise ValueError(f"Expected RGB_INT8 camera, got {msg.pixel_format_type}")
                        pixels = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
                        rgb = pixels[:, :msg.width*3].reshape(msg.height, msg.width, 3)
                        folder = "images" if kind == "image" else "aerial_images"
                        relative = f"{folder}/{source_ns:019d}.jpg"
                        Image.fromarray(rgb).save(self.output/relative, quality=self.jpeg_quality)
                        row.update(image=relative, width=msg.width, height=msg.height)
                        frame_index = frames if kind == "image" else aerial_frames
                        if frame_index is None:
                            raise RuntimeError("Received aerial image with aerial recording disabled")
                        frame_index.write(json.dumps(row)+"\n")
                    else:
                        row["message"] = MessageToDict(msg, preserving_proto_field_name=True)
                        raw.write(json.dumps(row)+"\n")
                except Exception as error:
                    self.error = error
                finally:
                    self.items.task_done()

    def close(self):
        self.items.put(None)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Raw writer did not finish.")
        if self.error:
            raise self.error


def conservative_clearance(xy, previous_xy, obstacles):
    # Radius encloses the simplified vessel's hull and rear propeller visuals.
    a = np.asarray(previous_xy if previous_xy is not None else xy, dtype=float)
    delta = np.asarray(xy)-a
    length2 = float(delta@delta)
    nearest = math.inf
    for obstacle in obstacles:
        center = np.array([obstacle["x"], obstacle["y"]])
        weight = max(0., min(1., float((center-a)@delta)/length2)) if length2 else 0.
        nearest = min(nearest, float(np.linalg.norm(a+weight*delta-center))-.85-obstacle["radius_m"])
    return nearest


def terminate(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)


def run(args):
    jpeg_quality = getattr(args, "jpeg_quality", 95)
    if not 70 <= jpeg_quality <= 95:
        raise ValueError("JPEG quality must be in 70..95.")
    for name in ("camera_rate_hz", "aerial_rate_hz"):
        rate = getattr(args, name, 30 if name == "camera_rate_hz" else 10)
        if not 5 <= rate <= 30:
            raise ValueError(f"{name} must be in 5..30 Hz.")
    videos_enabled = getattr(args, "videos", True)
    # Resolve the encoder before launching the boat. Encoding runs after safe teardown.
    video_ffmpeg = resolve_ffmpeg(getattr(args, "ffmpeg", None)) if videos_enabled else None
    output = (Path(args.output).expanduser().resolve() if args.output else
              OUTPUTS/"episodes"/datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"episode_directory": str(output)}), flush=True)
    spec = from_arguments(output/"world.sdf", args)
    os.environ["GZ_PARTITION"] = "usv_episode_"+uuid.uuid4().hex[:12]
    # Transport must be imported after selecting this episode's private partition.
    from gz.transport13 import Node
    from gz.msgs10.clock_pb2 import Clock
    from gz.msgs10.image_pb2 import Image as ImageMsg
    from gz.msgs10.camera_info_pb2 import CameraInfo
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.imu_pb2 import IMU
    from gz.msgs10.navsat_pb2 import NavSat
    from gz.msgs10.contacts_pb2 import Contacts
    from gz.msgs10.laserscan_pb2 import LaserScan
    from gz.msgs10.double_pb2 import Double
    from gz.msgs10.double_v_pb2 import Double_V
    from gz.msgs10.stringmsg_pb2 import StringMsg

    store, writer, node = SensorStore(), RawWriter(output, record_aerial=videos_enabled,
                                                 jpeg_quality=jpeg_quality), Node()
    stop_event = threading.Event()
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, lambda *_: stop_event.set())
    def receive(kind, msg):
        arrival = time.monotonic_ns()
        source = ((msg.sim.sec*1_000_000_000+msg.sim.nsec) if kind == "clock"
                  else pose_stamp(msg) if kind == "pose" else stamp(msg))
        # Aerial frames are inspection-only and never enter the policy observation store.
        if kind not in ("aerial_image", "aerial_camera_info"):
            store.add(kind, source, msg, arrival)
        if kind == "contacts" and len(msg.contact):
            with store.condition:
                store.collision = True
        if kind != "clock":
            writer.put(kind, source, arrival, msg)
    subscriptions = [
        (Clock, "/world/usv_training/clock", "clock"),
        (Pose_V, "/model/small_cora_float/pose", "pose"),
        (ImageMsg, "/small_cora/camera/image", "image"),
        (CameraInfo, "/small_cora/camera/camera_info", "camera_info"),
        (IMU, "/small_cora/imu/data", "imu"),
        (NavSat, "/small_cora/gps/fix", "gps"),
        (Contacts, "/usv/contacts", "contacts"),
        (LaserScan, "/small_cora/lidar/scan", "lidar"),
    ]
    if videos_enabled:
        subscriptions += [
            (ImageMsg, spec["aerial_camera"]["topic"], "aerial_image"),
            (CameraInfo, spec["aerial_camera"]["camera_info_topic"], "aerial_camera_info"),
        ]
    for cls, topic, kind in subscriptions:
        if not node.subscribe(cls, topic, lambda msg, kind=kind: receive(kind, msg)):
            raise RuntimeError(f"Cannot subscribe to {topic}")
    def request(msg):
        try:
            command = limited_action(msg.data)
        except ValueError:
            command = None
        with store.condition:
            store.request = command
            store.request_arrival_ns = time.monotonic_ns()
    node.subscribe(Double_V, "/usv/control_request", request)
    publishers = [node.advertise(f"/small_cora/thrusters/{side}/thrust", Double) for side in ("left", "right")]
    status_publisher = node.advertise("/usv/task_status", StringMsg)
    def publish_raw(command):
        command = limited_action(command)
        for pub, value in zip(publishers, command):
            pub.publish(Double(data=value))
        return time.monotonic_ns()

    lease = CommandLease(publish_raw, args.policy_command_lease_s) if args.controller == "policy" else None
    publish = lease.publish if lease is not None else publish_raw
    policy = None
    policy_source = None
    if args.controller == "policy":
        policy_source = dict(kind="external_websocket_policy_service", host=args.policy_host, port=args.policy_port,
                             checkpoint_training_verified=False,
                             inference_timeout_s=args.policy_timeout_s,
                             max_source_age_s=args.policy_max_source_age_s,
                             command_lease_s=args.policy_command_lease_s,
                             pending_action="zero_thrust", execute_steps=1, simulation_paused_for_inference=False)

    spec.update(controller=args.controller, is_trained_vla=None if args.controller == "policy" else False,
                policy_source=policy_source, transport_partition=os.environ["GZ_PARTITION"],
                camera_extrinsic_xyzrpy=[.52, 0, .45, 0, 0, 0],
                camera_extrinsic_frame="Gazebo sensor pose in hull FLU; optical-frame rotation in world.sdf",
                raw_recording="continuous RGB, calibration, GPS, IMU, pose, contacts and lidar",
                source_time_alignment="source <= decision and arrival <= decision wall time",
                action_semantics="published command N; next-decision interval, not measured thrust",
                row_state_previous_command_semantics="last command published before this decision",
                normalization={"position_scale_m": 32, "action_scale_N": 15},
                jpeg_quality=jpeg_quality,
                decision_schedule=("causal_10hz_ticks_max_50ms_lateness" if spec.get("route") else
                                   "latest_clock_at_decision"),
                source_fingerprints={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                     for name in ("record.py", "world.py", "core.py", "route.py", "scene_plan.py", "video.py")
                                     if (Path(__file__).parent/name).is_file()},
                video_recording={"enabled": videos_enabled, "directory": "videa",
                                 "fps": 10, "time_basis": "simulation_time",
                                 "aerial_used_for_training": False},
                collision_monitor="hull contact sensor plus conservative swept map footprints; not full hull geometry clearance to banks")
    (output/"episode.json").write_text(json.dumps(spec, indent=2)+"\n")
    process = manual = None
    error = None
    result = dict(success=False, termination_reason="initialization_failed", done=True)
    rows, previous_command, last_pose, initial_yaw, task = 0, [0., 0.], None, None, None
    gps_origin = None
    first_wall, last_sensor_wall, last_decision, sim_started = time.monotonic(), time.monotonic(), -1, None
    buoys = {row["slot"]: (row["x"], row["y"]) for row in spec["buoys"]}
    try:
        with (output/"gazebo.log").open("w") as log, (output/"steps.jsonl").open("w") as steps:
            if args.controller == "policy":
                policy = PolicyBridge(python=args.policy_python, host=args.policy_host, port=args.policy_port,
                                      timeout_s=args.policy_timeout_s, max_source_age_s=args.policy_max_source_age_s,
                                      startup_timeout_s=args.policy_startup_timeout_s,
                                      log_path=output/"policy_worker.log")
                policy_source["server_metadata"] = policy.metadata
                (output/"episode.json").write_text(json.dumps(spec, indent=2)+"\n")
            command = ["gz", "sim", "-r"]
            if not args.gui:
                command += ["-s", "--headless-rendering"]
            command.append(spec["world"])
            process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
            if args.controller == "manual":
                manual = subprocess.Popen([sys.executable, "-m", "usv_vla.manual"], start_new_session=True)
            while True:
                if lease is not None and lease.failure:
                    raise RuntimeError(lease.failure)
                if policy is not None:
                    policy.check_deadline()
                if stop_event.is_set():
                    raise RuntimeError("interrupted")
                if process.poll() is not None:
                    raise RuntimeError("Gazebo exited; see gazebo.log")
                if writer.error:
                    raise RuntimeError(f"Recorder failed: {writer.error}")
                if rows and time.monotonic()-last_sensor_wall > 2:
                    raise RuntimeError("Observation/clock heartbeat stopped")
                if time.monotonic()-first_wall > args.max_wall_s:
                    raise TimeoutError("Wall-clock episode limit reached")
                if manual is not None and manual.poll() is not None:
                    raise RuntimeError("Manual window closed")
                with store.condition:
                    clocks = store.messages["clock"]
                    if not clocks:
                        store.condition.wait(.05)
                        continue
                    clock_ns = clocks[-1][0]
                    decision_ns = (scheduled_decision(clock_ns, last_decision) if spec.get("route") else clock_ns)
                    decision_wall_ns = time.monotonic_ns()
                    if decision_ns is None or decision_ns < 2_500_000_000 or decision_ns-last_decision < 100_000_000:
                        store.condition.wait(.01)
                        continue
                    selected = {kind: store.select(kind, decision_ns, decision_wall_ns, age)
                                for kind, age in {"pose":250_000_000,"image":250_000_000,
                                                  "imu":150_000_000,"gps":300_000_000}.items()}
                    collision = store.collision
                    intent, intent_arrival = store.request, store.request_arrival_ns
                if not all(selected.values()) or not all(p.has_connections() for p in publishers):
                    if time.monotonic()-last_sensor_wall > (60 if rows == 0 else 2):
                        raise RuntimeError("Required sensors or thrusters are unavailable/stale")
                    time.sleep(.01)
                    continue
                pose_ns, _, poses = selected["pose"]
                if last_pose is not None and pose_ns <= last_pose[0]:
                    if time.monotonic()-last_sensor_wall > 2:
                        raise RuntimeError("Pose source clock stopped advancing")
                    time.sleep(.005)
                    continue
                boat = next((p for p in poses.pose if p.name == "small_cora_float"), None)
                if boat is None:
                    raise RuntimeError("Boat root pose missing")
                roll, pitch, yaw = rpy(boat.orientation)
                pose = [boat.position.x, boat.position.y, yaw, roll, pitch]
                if last_pose is None:
                    last_pose = (pose_ns, pose)
                    last_sensor_wall = time.monotonic()
                    continue
                delta_t = (pose_ns-last_pose[0])/1e9
                u, v = body_xy((pose[0]-last_pose[1][0])/delta_t, (pose[1]-last_pose[1][1])/delta_t, yaw)
                velocity = [u, v, selected["imu"][2].angular_velocity.z]
                if initial_yaw is None:
                    initial_yaw = yaw
                    if spec.get("route"):
                        from usv_vla.route import RouteTaskEvaluator, RouteTeacher
                        task = RouteTaskEvaluator(TaskConfig(**spec["task"]), buoys, pose[:2], initial_yaw,
                                                  route=spec["route"], obstacles=spec["obstacles"])
                        teacher = RouteTeacher(task)
                    else:
                        task = TaskEvaluator(TaskConfig(**spec["task"]), buoys, pose[:2], initial_yaw)
                        teacher = ScriptedTeacher(task)
                    sim_started = decision_ns
                state = pack_state(pose, velocity, initial_yaw, previous_command, buoys)
                from usv_vla.gps_state import extend_state
                fix = selected['gps'][2]
                gps = (fix.latitude_deg, fix.longitude_deg)
                if gps_origin is None:
                    gps_origin = gps
                    spec['gps_state_v2_origin_lat_lon_deg'] = list(gps_origin)
                    (output/'episode.json').write_text(json.dumps(spec, indent=2)+'\n')
                # Existing evaluator handles buoy tasks only; no GPS goal is fabricated.
                state_gps_v2 = extend_state(state, gps=gps, origin=gps_origin)
                policy_state = state_gps_v2 if policy is not None and policy.metadata['state_dim'] == 39 else state
                # Use the same model-side scales to reject unencodable pilot samples.
                scales = np.array([2,2,1,math.pi,math.pi,1,1,1,1]+[32,32,1]*8)
                if np.any(np.abs(state/scales) > 1):
                    raise RuntimeError("Observation exceeds pi05 engineering scales")
                clearance = conservative_clearance(pose[:2], last_pose[1][:2], spec["obstacles"])
                result = task.update(pose_ns/1e9, pose[:2], math.hypot(u,v), collision=collision, clearance_m=clearance)
                prompt = task.context()
                sources = {kind:dict(source_time_ns=entry[0],arrival_monotonic_ns=entry[1])
                           for kind, entry in selected.items()}
                policy_info = None
                submitted_policy_prompt = None
                status_publisher.publish(StringMsg(data=json.dumps(dict(result, prompt=prompt,
                    boat_xy=pose[:2], elapsed_s=(decision_ns-sim_started)/1e9,
                    buoys={str(slot):state[9+3*slot:12+3*slot].tolist() for slot in buoys}))))
                if result["done"] or args.controller == "zero":
                    action = [0., 0.]
                elif args.controller == "scripted":
                    action = teacher.command(pose, velocity)
                elif args.controller == "policy":
                    response = policy.poll(decision_ns)
                    if response is not None:
                        action, policy_info = response
                    else:
                        # Pending inference never extends a preceding nonzero command.
                        action = [0., 0.]
                        if policy.pending is None:
                            publish([0., 0.])
                            camera = selected["image"][2]
                            if camera.pixel_format_type != 3:
                                raise ValueError("Policy camera must be RGB_INT8")
                            pixels = np.frombuffer(camera.data, np.uint8).reshape(camera.height, camera.step)
                            rgb = pixels[:, :camera.width*3].reshape(camera.height, camera.width, 3)
                            submitted_policy_prompt = policy_task_instruction(task)
                            policy_info = policy.submit(policy_state, rgb, submitted_policy_prompt, decision_ns, decision_wall_ns,
                                                        sources, rows)
                        else:
                            policy_info = dict(status="pending_zero_thrust", **policy.pending)
                elif intent is None or intent_arrival is None or decision_wall_ns-intent_arrival > 500_000_000:
                    raise RuntimeError("Manual command heartbeat missing or stale")
                else:
                    action = intent
                action = limited_action(action)
                published_wall_ns = publish(action)
                row = dict(index=rows, decision_sim_time_ns=decision_ns, decision_monotonic_ns=decision_wall_ns,
                           decision_latest_clock_sim_time_ns=clock_ns,
                           decision_tick_lateness_ns=clock_ns-decision_ns,
                           state=state.tolist(), state_source="simulator_pose_debug_not_real_localization",
                           state_gps_v2=state_gps_v2.tolist(),
                           image=f"images/{selected['image'][0]:019d}.jpg", prompt=prompt,
                           pose_xyzrpy=[pose[0],pose[1],boat.position.z,roll,pitch,yaw],
                           action=action, action_publish_monotonic_ns=published_wall_ns,
                           sources=sources, policy_inference=policy_info,
                           policy_input_prompt=submitted_policy_prompt,
                           policy_input_state=(policy_state.tolist() if policy is not None and policy.metadata['state_dim'] == 39 else None),
                           clearance_m=clearance, **result)
                steps.write(json.dumps(row)+"\n")
                steps.flush()
                previous_command, last_pose, last_decision = action, (pose_ns, pose), decision_ns
                last_sensor_wall = time.monotonic()
                rows += 1
                if rows % 50 == 0 or result["done"]:
                    print(json.dumps(dict(step=rows,elapsed_s=(decision_ns-sim_started)/1e9,**result)), flush=True)
                if result["done"]:
                    break
    except Exception as caught:
        error = str(caught)
        result = dict(done=True,success=False,termination_reason=error)
    finally:
        for _ in range(5):
            publish([0., 0.])
            time.sleep(.02)
        if lease is not None:
            lease.close()
        if policy is not None:
            policy.close()
        terminate(manual)
        terminate(process)
        for _, topic, _ in subscriptions:
            node.unsubscribe(topic)
        node.unsubscribe("/usv/control_request")
        try:
            writer.close()
        except Exception as caught:
            error = str(caught)
            result.update(success=False,termination_reason=error)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        missing = []
        if (output/"steps.jsonl").exists():
            for line in (output/"steps.jsonl").read_text().splitlines():
                row = json.loads(line)
                if not (output/row["image"]).exists():
                    missing.append(row["index"])
        if missing:
            error = "Recorded image references are missing"
            result.update(success=False,termination_reason=error)
        summary = dict(result,episode_directory=str(output),steps=rows,controller=args.controller,
                       is_trained_vla=None if args.controller == "policy" else False,policy_source=policy_source,
                       state_source="simulator_pose_debug_not_real_localization",raw_counts=dict(writer.counts),
                       missing_image_steps=missing,recording_valid=error is None and rows>0,
                       published_zero_on_exit=True,processes_stopped=True)
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if videos_enabled and rows:
        try:
            export_episode_videos(output, ffmpeg=video_ffmpeg, fps=10, require_aerial=True)
            # Aerial frames are temporary review material, never policy input.
            # Only discard them after both videos have been published successfully.
            shutil.rmtree(output/"aerial_images")
            summary["videos"] = {"status": "completed", "directory": "videa",
                                 "metadata": "videa/metadata.json",
                                 "files": ["videa/first_person.mp4", "videa/aerial_global.mp4"],
                                 "aerial_raw_frames": "deleted_after_video_export"}
        except Exception as caught:
            summary["videos"] = {"status": "failed", "error": str(caught), "raw_frames_retained": True}
    else:
        summary["videos"] = {"status": "disabled" if not videos_enabled else "no_decisions"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    videos_ok = not videos_enabled or summary["videos"]["status"] == "completed"
    return 0 if summary["success"] and summary["recording_valid"] and videos_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output", help="New episode directory; default: outputs/episodes/<local timestamp>")
    parser.add_argument("--controller", choices=["scripted","manual","zero","policy"], default="manual")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--policy-python", default=os.environ.get("USV_MODEL_PYTHON", sys.executable),
                        help="Python with openpi_client dependencies; separate from system Gazebo Python")
    parser.add_argument("--policy-timeout-s", type=float, default=.5, help="Wall-clock inference deadline")
    parser.add_argument("--policy-max-source-age-s", type=float, default=.75,
                        help="Maximum sim-source age when accepting an action; simulation keeps advancing")
    parser.add_argument("--policy-startup-timeout-s", type=float, default=15.)
    parser.add_argument("--policy-command-lease-s", type=float, default=.25,
                        help="Independent wall-clock limit on keeping a nonzero command")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--videos", action=argparse.BooleanOptionalAction, default=True,
                        help="Record front/aerial views and generate videa/*.mp4 after the episode")
    parser.add_argument("--ffmpeg", help="Optional encoder path; otherwise use USV_FFMPEG/PATH/imageio-ffmpeg")
    parser.add_argument("--camera-rate-hz", type=float, default=30., help="Front RGB raw recording rate")
    parser.add_argument("--aerial-rate-hz", type=float, default=10., help="Review-only aerial raw recording rate")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-wall-s", type=float, default=240.)
    raise SystemExit(run(parser.parse_args()))

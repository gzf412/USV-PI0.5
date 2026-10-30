"""Model-Python JSON-lines worker. Stdout is reserved for the IPC protocol."""
import argparse
import base64
import contextlib
import io
import json
import sys
import time

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    protocol = sys.stdout
    def emit(value):
        protocol.write(json.dumps(value, allow_nan=False)+"\n")
        protocol.flush()
    try:
        # Keep third-party startup messages away from protocol stdout.
        with contextlib.redirect_stdout(sys.stderr):
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            client = WebsocketClientPolicy(host=args.host, port=args.port)
        emit(dict(type="ready", metadata=client.get_server_metadata()))
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("type") != "infer":
                raise ValueError("Unknown policy worker request")
            width = client.get_server_metadata()['state_dim']
            state = np.asarray(request["state"], dtype=np.float64 if width == 35 else np.float32)
            if state.shape != (client.get_server_metadata()['state_dim'],) or not np.isfinite(state).all():
                raise ValueError("Worker state shape differs from policy metadata")
            rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(request["image_jpeg"], validate=True))).convert("RGB"))
            observation = dict(state=state, images={"front":rgb}, prompt=request["prompt"])
            started = time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):
                result = client.infer(observation)
            emit(dict(type="result", id=request["id"], actions=np.asarray(result["actions"]).tolist(),
                      rpc_s=time.monotonic()-started))
    except Exception as error:
        failure = dict(type="error", error=f"{type(error).__name__}: {error}")
        # Persist diagnosis even if the parent sees process exit before IPC EOF.
        print(json.dumps(failure), file=sys.stderr, flush=True)
        emit(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

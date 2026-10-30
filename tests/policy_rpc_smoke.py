"""Explicit loopback integration check; test stub only, no model weights.

This is separate from offline unittest discovery because loopback access and a
model Python with websockets/msgpack are required. Exercises full 1280x720 RGB.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
from usv_vla.policy_bridge import PolicyBridge


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-python',default=os.environ.get('USV_MODEL_PYTHON',sys.executable))
    parser.add_argument('--port',type=int,default=18088)
    args=parser.parse_args()
    environment=os.environ.copy()
    environment['PYTHONPATH']=str(PROJECT_ROOT/'src')+os.pathsep+environment.get('PYTHONPATH','')
    with tempfile.TemporaryDirectory(prefix='usv_policy_rpc_') as directory:
        root=Path(directory)
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen([args.model_python,str(PROJECT_ROOT/'tests/fake_policy_server.py'),
                                     '--port',str(args.port),'--action','3','-6'],stdout=log,stderr=log,env=environment)
            bridge=None
            try:
                # Worker connection startup has its own bounded deadline.
                time.sleep(.3)
                if server.poll() is not None:
                    raise RuntimeError((root/'server.log').read_text())
                bridge=PolicyBridge(python=args.model_python,port=args.port,timeout_s=3.,
                                    max_source_age_s=.75,startup_timeout_s=8.,log_path=root/'worker.log')
                wall=time.monotonic_ns()
                rgb=np.random.default_rng(1).integers(0,256,(720,1280,3),dtype=np.uint8)
                bridge.submit(np.zeros(33),rgb,'Approach red round buoy.',1_000_000_000,wall,
                              {'image':dict(source_time_ns=1_000_000_000,arrival_monotonic_ns=wall)},0)
                deadline=time.monotonic()+3.
                while time.monotonic()<deadline:
                    response=bridge.poll(1_100_000_000)
                    if response is not None:
                        action,info=response
                        assert action==[3.,-6.],action
                        print(json.dumps(dict(test='full_resolution_rpc',is_trained_vla=False,
                                              rgb_shape=list(rgb.shape),rgb_bytes=rgb.nbytes,
                                              action_N=action,latency_s=info['latency_s'],
                                              server_metadata=bridge.metadata)))
                        return 0
                    time.sleep(.005)
                raise TimeoutError('Full-resolution test did not return')
            except Exception:
                if (root/'worker.log').exists():
                    print((root/'worker.log').read_text(),file=sys.stderr)
                raise
            finally:
                if bridge is not None: bridge.close()
                server.terminate()
                try: server.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=3)


if __name__=='__main__': raise SystemExit(main())

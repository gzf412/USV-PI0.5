"""Protocol/timeout tests use a stub process, never model weights."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

from usv_vla.policy_bridge import CommandLease, PolicyBridge, first_action, request_message, validate_metadata


METADATA = dict(state_dim=33, action_dim=2, action_units="N", policy_kind="unit_test_stub_not_vla")


class PolicyBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.rgb = np.zeros((24,32,3), np.uint8)
        self.sources = {"image": dict(source_time_ns=990_000_000, arrival_monotonic_ns=10)}

    def worker(self, response=None, delay=0, metadata=None, timeout=.8):
        path = Path(self.temporary.name)/"worker.py"
        code = ("import json,sys,time,base64,io\n"
                f"print(json.dumps(dict(type='ready', metadata={metadata or METADATA!r})),flush=True)\n"
                "for line in sys.stdin:\n"
                " request=json.loads(line)\n"
                " assert request['type']=='infer' and len(request['state'])==33\n"
                " assert base64.b64decode(request['image_jpeg']).startswith(b'\\xff\\xd8')\n"
                f" time.sleep({delay!r})\n"
                f" actions={response if response is not None else [[3,-6],[9,9]]!r}\n"
                " print(json.dumps(dict(type='result',id=request['id'],actions=actions,rpc_s=0.001)),flush=True)\n")
        path.write_text(code)
        bridge = PolicyBridge(worker_command=[sys.executable,"-u",str(path)], timeout_s=timeout,
                              max_source_age_s=.5, startup_timeout_s=2)
        self.addCleanup(bridge.close)
        return bridge

    def submit(self, bridge):
        return bridge.submit(np.zeros(33),self.rgb,"Approach red round buoy.",
                             1_000_000_000,time.monotonic_ns(),self.sources,7)

    def poll_ready(self, bridge, sim_ns=1_100_000_000):
        for _ in range(100):
            response = bridge.poll(sim_ns)
            if response is not None:
                return response
            time.sleep(.005)
        self.fail("Stub did not respond")

    def test_metadata_requires_state_action_and_newton_units(self):
        self.assertEqual(validate_metadata(METADATA),METADATA)
        for key,value in (("state_dim",32),("action_dim",33),("action_units","normalized")):
            with self.assertRaises(ValueError): validate_metadata({**METADATA,key:value})

    def test_action_block_first_step_and_invalid_outputs(self):
        self.assertEqual(first_action([[3,-6],[15,15]]),[3,-6])
        for value in ([3,-6],[],[[1,2,3]],[[float('nan'),0]],[[float('inf'),0]],[[16,0]]):
            with self.assertRaises(ValueError): first_action(value)

    def test_bad_worker_metadata_rejected_before_commands(self):
        with self.assertRaisesRegex(ValueError,'action_units'):
            self.worker(metadata={**METADATA,'action_units':'normalized'})

    def test_pipe_protocol_consumes_once_and_tracks_source(self):
        bridge=self.worker()
        submitted=self.submit(bridge)
        self.assertEqual(submitted['observation_row_index'],7)
        action,info=self.poll_ready(bridge)
        self.assertEqual(action,[3,-6])
        self.assertEqual(info['status'],'executed_first_step')
        self.assertEqual(info['observation_row_index'],7)
        self.assertAlmostEqual(info['source_age_s'],.11)
        self.assertGreaterEqual(info['latency_s'],0)
        self.assertIsNone(bridge.pending)
        self.assertIsNone(bridge.poll(1_100_000_000))

    def test_timeout_is_bounded_while_worker_waits(self):
        bridge=self.worker(delay=.3,timeout=.03)
        self.submit(bridge)
        time.sleep(.05)
        with self.assertRaisesRegex(TimeoutError,'budget'): bridge.poll(1_100_000_000)

    def test_stale_source_result_rejected(self):
        bridge=self.worker()
        self.submit(bridge)
        with self.assertRaisesRegex(RuntimeError,'stale'):
            self.poll_ready(bridge,1_600_000_000)

    def test_wrong_shape_from_worker_rejected(self):
        bridge=self.worker(response=[3,-6])
        self.submit(bridge)
        with self.assertRaises(ValueError): self.poll_ready(bridge)

    def test_no_future_source_or_arrival(self):
        for sources in ({'image':dict(source_time_ns=101,arrival_monotonic_ns=90)},
                        {'image':dict(source_time_ns=90,arrival_monotonic_ns=101)}):
            with self.assertRaisesRegex(ValueError,'Future'):
                request_message(0,np.zeros(33),self.rgb,'test',100,100,sources)

    def test_disconnect_is_failure(self):
        bridge=self.worker()
        self.submit(bridge)
        bridge.process.kill()
        bridge.process.wait(timeout=2)
        with self.assertRaisesRegex(RuntimeError,'exited'): bridge.poll(1_100_000_000)

    def test_wrong_response_id_rejected(self):
        bridge=self.worker(delay=.1)
        self.submit(bridge)
        bridge.incoming.put(dict(type='result',id=999,actions=[[0,0]]))
        with self.assertRaisesRegex(RuntimeError,'match'): bridge.poll(1_100_000_000)

    def test_independent_command_lease_publishes_zero(self):
        commands=[]
        lease=CommandLease(lambda command: commands.append(list(command)) or time.monotonic_ns(),timeout_s=.03)
        self.addCleanup(lease.close)
        lease.publish([3,-6])
        time.sleep(.07)
        self.assertEqual(commands[-1],[0,0])
        self.assertIn('expired',lease.failure)
        with self.assertRaises(RuntimeError): lease.publish([1,1])

    def test_zero_command_cancels_lease(self):
        lease=CommandLease(lambda command: time.monotonic_ns(),timeout_s=.03)
        self.addCleanup(lease.close)
        lease.publish([3,-6])
        lease.publish([0,0])
        time.sleep(.07)
        self.assertIsNone(lease.failure)


if __name__=='__main__': unittest.main()

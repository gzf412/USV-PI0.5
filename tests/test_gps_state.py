import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from usv_vla.gps_state import extend_state, local_east_north, normalize_extra
from usv_vla.export_lerobot import gps_v2_states


class GpsStateTests(unittest.TestCase):
    def test_wgs84_local_and_body_frame(self):
        origin = (0., 0.)
        goal = (0., 100 / 6378137 * 180 / math.pi)
        east, north = local_east_north(origin, goal)
        self.assertAlmostEqual(east, 100, places=5)
        self.assertAlmostEqual(north, 0, places=5)
        state = extend_state(np.zeros(33), gps=origin, origin=origin, goal=goal, yaw=math.pi/2)
        np.testing.assert_allclose(state[33:], [0, 0, 1, 0, -100, 1], atol=1e-4)
        self.assertEqual(state.shape, (39,))

    def test_validity_and_range(self):
        np.testing.assert_array_equal(extend_state(np.zeros(33))[33:], np.zeros(6))
        with self.assertRaises(ValueError):
            extend_state(np.zeros(33), goal=(0., 0.))
        with self.assertRaises(ValueError):
            extend_state(np.zeros(33), gps=(0., 0.), origin=(0., 0.), goal=(0., .01), yaw=0.)
        with self.assertRaises(ValueError):
            normalize_extra([0, 0, 0, 1, 0, 1])
        with self.assertRaises(ValueError):
            local_east_north((0., 0.), (float('nan'), 0.))

    def test_more_resolution_near_target_and_boundary(self):
        near = normalize_extra([0, 0, 1, .2, 0, 1])[3]
        far = normalize_extra([0, 0, 1, 100.2, 0, 1])[3] - normalize_extra([0, 0, 1, 100, 0, 1])[3]
        self.assertGreater(near, far)
        self.assertAlmostEqual(normalize_extra([0, 0, 1, 150, 0, 1])[3], 1.)

    def test_historical_gps_uses_exact_causal_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [dict(kind='gps', source_time_ns=1, arrival_monotonic_ns=10,
                            message=dict(latitude_deg=0., longitude_deg=0.)),
                       dict(kind='gps', source_time_ns=2, arrival_monotonic_ns=20,
                            message=dict(latitude_deg=0., longitude_deg=.00001))]
            (root/'raw_sensors.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
            rows = [dict(state=[0.]*33, sources={'gps': {'source_time_ns': x['source_time_ns'],
                    'arrival_monotonic_ns': x['arrival_monotonic_ns']}}) for x in records]
            states = gps_v2_states(root, rows)
            self.assertEqual(states[0][35], 1.)
            self.assertGreater(states[1][33], 1.)
            self.assertEqual(states[1][38], 0.)
            rows[1]['sources']['gps']['arrival_monotonic_ns'] = 21
            with self.assertRaises(ValueError):
                gps_v2_states(root, rows)

    def test_policy_rpc_accepts_gps_width(self):
        from usv_vla.policy_bridge import request_message, validate_metadata
        metadata = validate_metadata({'state_dim': 39, 'action_dim': 2, 'action_units': 'N'})
        state = extend_state(np.zeros(33), gps=(0., 0.), origin=(0., 0.))
        message = request_message(1, state, np.zeros((8, 8, 3), dtype=np.uint8), 'Reach goal', 2, 2,
                                  {'gps': {'source_time_ns': 1, 'arrival_monotonic_ns': 1}})
        self.assertEqual(len(message['state']), metadata['state_dim'])


if __name__ == '__main__':
    unittest.main()

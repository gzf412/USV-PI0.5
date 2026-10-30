"""Check yellow-target identity across the existing multi-buoy route families."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_yellow_batch import yellow_scene
from usv_vla.batch_collect import _scene_seed
from usv_vla.green_routes import generate_scene
from usv_vla.green_shore import generate_green_shore
from usv_vla.scene_plan import validate_scene


class YellowBatchTests(unittest.TestCase):
    def test_all_route_families_keep_valid_yellow_target(self):
        for mode in ('direct_visible', 'direct_offscreen', 'buoy_pass', 'buoy_gap', 'mixed_multi', 'shore'):
            with self.subTest(mode=mode):
                seed = _scene_seed(20261030, mode, 0)
                original = generate_green_shore(0, seed) if mode == 'shore' else generate_scene(mode, 0, seed)
                scene = yellow_scene(original)
                self.assertEqual(scene['target_slot'], 2)
                self.assertEqual(scene['dataset_type'], 'yellow_multi_v1')
                self.assertIn('yellow round buoy', scene['route']['instruction'])
                self.assertEqual(next(b for b in scene['buoys'] if b['slot'] == 2)['x'], scene['target_xy'][0])
                validate_scene(scene)


if __name__ == '__main__':
    unittest.main()

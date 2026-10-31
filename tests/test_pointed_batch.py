import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_pointed_batch import SLOTS, make_pointed
from usv_vla.batch_collect import _scene_seed
from usv_vla.scene_plan import validate_scene


class PointedBatchTests(unittest.TestCase):
    def test_each_color_and_route_family(self):
        for color, slot in SLOTS.items():
            for mode, index in [('direct_visible', 0), ('direct_offscreen', 0), ('buoy_pass', 0),
                                ('buoy_gap', 0), ('mixed_multi', 0), ('shore', 0), ('shore', 30)]:
                with self.subTest(color=color, mode=mode, index=index):
                    seed = _scene_seed(20261031+slot, mode, index)
                    scene = make_pointed(color, mode, index, seed)
                    self.assertEqual(scene['seed'], seed)
                    self.assertEqual(scene['target_slot'], slot)
                    self.assertIn(f'{color} pointed buoy', scene['route']['instruction'])
                    self.assertEqual(sum(b['slot'] == slot for b in scene['buoys']), 1)
                    if mode == 'shore' and index >= 30:
                        self.assertEqual(len(scene['cylinders']), 0)
                        self.assertEqual(len(scene['buoys']), 8)
                    validate_scene(scene)


if __name__ == '__main__':
    unittest.main()

"""Geometric and contract checks for generated multi-buoy scenes, no simulator."""
import math
from pathlib import Path
import tempfile
import unittest

from usv_vla.core import TaskConfig
from usv_vla.green_routes import MODES, generate_scene
from usv_vla.green_scene import validate_green_scene
from usv_vla.route import RouteTaskEvaluator
from usv_vla.world import build_world


class GreenRoutesTests(unittest.TestCase):
    def test_modes_are_reproducible_and_build_the_world_and_evaluator(self):
        buoy_counts, cylinder_counts, angles = set(), set(), []
        with tempfile.TemporaryDirectory() as tmp:
            for mode in MODES:
                for i in range(10):
                    with self.subTest(mode=mode,index=i):
                        s = generate_scene(mode, i, 10001+19*i)
                        self.assertEqual(s, generate_scene(mode, i, 10001+19*i))
                        self.assertEqual(s['initial_visibility'], validate_green_scene(s))
                        self.assertEqual(s['target_slot'], 4)
                        self.assertEqual(len(s['buoys']),len({b['slot'] for b in s['buoys']}))
                        self.assertFalse(s['route']['prompt_include_phase'])
                        self.assertIn('green round buoy',s['route']['instruction'].lower())
                        if mode == 'direct_offscreen':
                            self.assertTrue(s['initial_visibility']['target']['entirely_outside_horizontal'])
                        if mode in ('buoy_pass','buoy_gap','mixed_multi') and i % 3 == 0:
                            self.assertNotIn('left',s['route']['instruction'].lower())
                            self.assertNotIn('right',s['route']['instruction'].lower())
                        if mode == 'buoy_gap':
                            self.assertEqual(len(s['route']['required_obstacle_models']),2)
                            self.assertTrue(all(x.startswith('v2_buoy_') for x in s['route']['required_obstacle_models']))
                        if mode == 'mixed_multi':
                            self.assertTrue(any(x.startswith('v2_buoy_') for x in s['route']['required_obstacle_models']))
                            self.assertTrue(any(x.startswith('usv_cylinder_') for x in s['route']['required_obstacle_models']))
                        spec = build_world(Path(tmp)/'world.sdf',scene=s)
                        RouteTaskEvaluator(TaskConfig(**spec['task']),{b['slot']:(b['x'],b['y']) for b in spec['buoys']},
                                           s['initial_pose'][:2],s['initial_pose'][3],route=s['route'],obstacles=spec['obstacles'])
                        buoy_counts.add(len(s['buoys'])); cylinder_counts.add(len(s['cylinders']))
                        angles.append(abs(math.degrees(s['initial_pose'][3]-s['route']['reference_yaw_rad'])))
        self.assertIn(1,buoy_counts); self.assertIn(8,buoy_counts)
        self.assertEqual(cylinder_counts,{1,2,3,4})
        self.assertGreater(max(angles),50.)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown green scene mode'):
            generate_scene('other',0,1)


if __name__ == '__main__':
    unittest.main()

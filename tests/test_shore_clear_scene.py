import math
import unittest

from usv_vla.green_scene import buoy_radius
from usv_vla.scene_plan import _segment_distance, validate_scene, xy_from
from usv_vla.shore_clear_scene import generate
from usv_vla.training_selection import accepted_supplements


class ShoreClearSceneTests(unittest.TestCase):
    def test_three_targets_both_directions_have_seven_clear_buoys(self):
        for target in (0, 4, 6):
            for index in (0, 1):
                with self.subTest(target=target,index=index):
                    scene=generate(target,index,2026102900+target*10+index)
                    validate_scene(scene)
                    self.assertEqual({b['slot'] for b in scene['buoys']},set(range(8)))
                    self.assertEqual(scene['cylinders'],[])
                    self.assertEqual(scene['route']['kind'],'bank_follow')
                    self.assertEqual(scene['route']['gates'],[])
                    self.assertIn(('red','green','blue')[(0,4,6).index(target)]+' round buoy',scene['route']['instruction'])
                    bank=scene['route']['bank_corridor']
                    stop=xy_from(bank['origin'],bank['yaw_rad'],bank['progress_end_m']+.9,0.)
                    for b in scene['buoys']:
                        if b['slot']==target: continue
                        d=_segment_distance([b['x'],b['y']],bank['origin'],stop)
                        self.assertGreaterEqual(d,buoy_radius(b['slot'])+.85+1.)


if __name__ == '__main__':
    unittest.main()

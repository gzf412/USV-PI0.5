"""Multi-buoy geometry, semantic targeting and world generation regressions."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from usv_vla.audit import _selected_target
from usv_vla.core import pack_state
from usv_vla.scene_plan import initial_visibility, validate_scene
from usv_vla.world import ORIGIN, build_world


def scene():
    return dict(dataset_type="green_multi_v1", category="approach_visible", seed=11,
                initial_pose=list(ORIGIN), target_slot=4, target_xy=[-532., 178.],
                buoys=[dict(slot=0,x=-525.,y=170.), dict(slot=4,x=-532.,y=178.),
                       dict(slot=7,x=-539.,y=171.)], cylinders=[],
                route=dict(kind="direct", instruction="Approach the green round buoy and stop.",
                           prompt_include_phase=False, waypoints=[], required_obstacle_models=[], cylinder_models=[]))


class GreenSceneTests(unittest.TestCase):
    def test_world_preserves_semantic_positions_and_state(self):
        data = scene()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scene.sdf"
            result = build_world(path, scene=data)
            model_names = {m.attrib['name'] for m in ET.parse(path).findall('.//world/model')}
        self.assertEqual(result['task']['target'], 4)
        self.assertEqual([x['slot'] for x in result['buoys']], [0,4,7])
        self.assertEqual([(x['x'],x['y']) for x in result['buoys']], [(-525.,170.),(-532.,178.),(-539.,171.)])
        self.assertEqual({o['model'] for o in result['obstacles']}, {'v2_buoy_1','v2_buoy_5','v2_buoy_8'})
        self.assertTrue({'v2_buoy_1','v2_buoy_5','v2_buoy_8'}.issubset(model_names))
        self.assertEqual(result['scene_plan'], data)
        state = pack_state([*ORIGIN[:2],ORIGIN[3],0.,0.],[0.,0.,0.],ORIGIN[3],[0.,0.],
                           {x['slot']:(x['x'],x['y']) for x in result['buoys']})
        self.assertEqual([int(state[11+3*i]) for i in range(8)], [1,0,0,0,1,0,0,1])
        self.assertEqual(_selected_target(result)['model'], 'v2_buoy_5')

    def test_duplicate_slots_mismatched_target_and_stale_phase_rejected(self):
        for mutation in ('duplicate','wrong_target','phase','instruction'):
            plan = scene()
            if mutation == 'duplicate': plan['buoys'][0]['slot'] = 4
            if mutation == 'wrong_target': plan['target_xy'][0] += 1.
            if mutation == 'phase': plan['route']['prompt_include_phase'] = True
            if mutation == 'instruction': plan['route']['instruction'] = 'Approach the red round buoy.'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_scene(plan)
        meta = {'task':{'target':4}, 'buoys':[{'slot':0},{'slot':4},{'slot':4}]}
        with self.assertRaises(ValueError): _selected_target(meta)

    def test_collisions_and_32m_policy_scale_are_rejected(self):
        collision = scene(); collision['buoys'][0]['x'] = -532.; collision['buoys'][0]['y'] = 170.
        with self.assertRaisesRegex(ValueError, 'clear'): validate_scene(collision)
        far = scene(); far['buoys'][0]['x'] = -500.; far['buoys'][0]['y'] = 165.
        with self.assertRaisesRegex(ValueError, '31 m'): validate_scene(far)

    def test_actual_initial_visibility_checks_the_required_object_and_occlusion(self):
        data = scene(); data['route']['required_obstacle_models'] = ['v2_buoy_1']
        with self.assertRaisesRegex(ValueError, 'visible'): validate_scene(data)
        data['route']['required_obstacle_models'] = []
        self.assertTrue(validate_scene(data)['target']['unobstructed_visible'])
        pose = data['initial_pose'].copy(); pose[3] += math.pi/2
        self.assertFalse(initial_visibility(data, pose)['target']['fully_visible'])

if __name__ == '__main__': unittest.main()

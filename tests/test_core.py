import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from usv_vla.core import TaskConfig, TaskEvaluator, limited_action, pack_state
from usv_vla.record import SensorStore, conservative_clearance, pose_stamp
from usv_vla.world import build_world


class ObservationTests(unittest.TestCase):
    def test_missing_and_off_camera_are_distinct(self):
        state = pack_state([0,0,0,0,0], [1,.1,.2], 0, [3,-6], {0:(-10,5),7:(8,-3)})
        self.assertEqual(state.shape,(33,))
        np.testing.assert_allclose(state[7:9],[.2,-.4])
        np.testing.assert_allclose(state[9:12],[-10,5,1])
        np.testing.assert_array_equal(state[12:30],0)
        np.testing.assert_allclose(state[30:],[8,-3,1])

    def test_rotation(self):
        state=pack_state([0,0,math.pi/2,0,0],[0,0,0],0,[0,0],{0:(0,10)})
        np.testing.assert_allclose(state[9:12],[10,0,1],atol=1e-6)

    def test_motor_rejection_and_limits(self):
        self.assertEqual(limited_action([20,-20]),[15,-15])
        for action in ([1,2,3],[float('nan'),0]):
            with self.assertRaises(ValueError): limited_action(action)

    def test_causal_selection_checks_source_and_arrival(self):
        store=SensorStore()
        store.add('image',90,'old',90)
        store.add('image',110,'future',95)
        store.add('image',99,'late',120)
        self.assertEqual(store.select('image',100,100,20)[2],'old')
        self.assertIsNone(store.select('image',200,200,20))

    def test_pose_publisher_uses_inner_source_stamp(self):
        root=SimpleNamespace(name='small_cora_float',header=SimpleNamespace(stamp=SimpleNamespace(sec=3,nsec=4)))
        message=SimpleNamespace(pose=[root],header=SimpleNamespace(stamp=SimpleNamespace(sec=0,nsec=0)))
        self.assertEqual(pose_stamp(message),3_000_000_004)

    def test_swept_clearance_catches_between_frames(self):
        clearance=conservative_clearance([2,0],[-2,0],[dict(x=0,y=0,radius_m=.75)])
        self.assertLess(clearance,0)


class TaskTests(unittest.TestCase):
    def make(self,**kwargs):
        return TaskEvaluator(TaskConfig(**kwargs),{0:(20.,0.),7:(30.,0.)},[0.,0.])

    def test_stop_requires_continuous_low_speed(self):
        task=self.make()
        for i in range(20): task.update(i*.1,[17.9,0],.1)
        self.assertFalse(task.done)
        task.update(2.,[17.9,0],.3)
        for i in range(21,41): task.update(i*.1,[17.9,0],.1)
        self.assertFalse(task.done)
        task.update(4.1,[17.9,0],.1)
        self.assertTrue(task.success)

    def test_moving_past_is_not_stopping(self):
        task=self.make()
        for i in range(31): task.update(i*.1,[17.9,0],.5)
        self.assertFalse(task.done)

    def test_wrong_target_is_not_success(self):
        task=self.make()
        for i in range(31): task.update(i*.1,[27.9,0],0)
        self.assertFalse(task.done)

    def test_old_far_stop_and_too_close_stop_do_not_complete(self):
        for distance in (2.7,1.85):
            with self.subTest(distance=distance):
                task=self.make()
                xy=[20-distance,0]
                clearance=conservative_clearance(xy,None,[dict(x=20,y=0,radius_m=.75)])
                self.assertGreater(clearance,task.config.minimum_clearance_m)
                for i in range(31): task.update(i*.1,xy,0,clearance_m=clearance)
                self.assertFalse(task.done)
                self.assertEqual(task.result(3.,xy,0)['hold_s'],0)

    def test_closer_stop_band_keeps_clearance_guard(self):
        obstacles=[dict(x=20,y=0,radius_m=.75)]
        for safe_xy in ([18.1,0],[17.7,0]):
            with self.subTest(safe_xy=safe_xy):
                safe_clearance=conservative_clearance(safe_xy,None,obstacles)
                task=self.make()
                self.assertGreater(safe_clearance,task.config.minimum_clearance_m)
                for i in range(21): task.update(i*.1,safe_xy,0,clearance_m=safe_clearance)
                self.assertTrue(task.success)

        task=self.make()
        unsafe_xy=[18.25,0]
        unsafe_clearance=conservative_clearance(unsafe_xy,None,obstacles)
        task.update(0,unsafe_xy,0,clearance_m=unsafe_clearance)
        self.assertFalse(task.success)
        self.assertEqual(task.reason,'clearance_violation')

        task=self.make()
        task.update(0,[17.9,0],0,clearance_m=.1)
        self.assertFalse(task.success)
        self.assertEqual(task.reason,'clearance_violation')

    def test_context_uses_configured_stop_band(self):
        for kind in ('approach','pass_then_approach'):
            for thresholds,expected in (({},'1.9 to 2.3 meters'),
                                        ({'stop_min_m':2.,'stop_max_m':3.},'2 to 3 meters')):
                with self.subTest(kind=kind,thresholds=thresholds):
                    task=self.make(kind=kind,destination=7 if kind=='pass_then_approach' else None,
                                   **thresholds)
                    self.assertIn(expected,task.context())

    def test_left_gate_crossing_and_exit_hold(self):
        task=self.make(kind='pass',side='left')
        task.update(0,[19,3],.5)
        task.update(.1,[21,3],.5)
        self.assertTrue(task.gate_passed)
        self.assertFalse(task.done)
        for i in range(2,23): task.update(i*.1,[23,3],0)
        self.assertTrue(task.success)

    def test_right_gate_and_wrong_side(self):
        for side,y,expected in [('right',-3,True),('left',-3,False),('left',5,False)]:
            task=self.make(kind='pass',side=side)
            task.update(0,[19,y],.5)
            task.update(.1,[21,y],.5)
            self.assertEqual(task.gate_passed,expected)
            if not expected: self.assertEqual(task.reason,'wrong_side_or_gate')

    def test_combination_requires_A_before_B(self):
        task=self.make(kind='pass_then_approach',destination=7)
        task.update(0,[19,3],.4)
        task.update(.1,[21,3],.4)
        task.update(.2,[23,3],.4)
        self.assertEqual(task.phase,'approach_destination')
        for i in range(3,24): task.update(i*.1,[27.9,0],0)
        self.assertTrue(task.success)
        skipped=self.make(kind='pass_then_approach',destination=7)
        skipped.update(0,[19,0],.5)
        skipped.update(.1,[27.5,0],0)
        self.assertFalse(skipped.success)
        self.assertEqual(skipped.reason,'wrong_side_or_gate')

    def test_collision_timeout_and_stale_are_failures(self):
        for kwargs,expected in [({'collision':True},'collision'),({'clearance_m':.1},'clearance_violation')]:
            task=self.make()
            task.update(0,[0,0],0,**kwargs)
            self.assertEqual(task.reason,expected)
        task=self.make(timeout_s=.2)
        task.update(0,[0,0],0)
        task.update(.2,[0,0],0)
        self.assertEqual(task.reason,'timeout')
        task=self.make()
        task.update(0,[0,0],0)
        task.update(.4,[0,0],0)
        self.assertEqual(task.reason,'stale_state')


class WorldTests(unittest.TestCase):
    def test_distractors_do_not_overlap_cylinders(self):
        with tempfile.TemporaryDirectory() as directory:
            for seed in range(20):
                spec=build_world(Path(directory)/f'{seed}.sdf',count=8,seed=seed)
                for i,a in enumerate(spec['obstacles']):
                    for b in spec['obstacles'][i+1:]:
                        gap=math.hypot(a['x']-b['x'],a['y']-b['y'])-a['radius_m']-b['radius_m']
                        self.assertGreater(gap,.3)

    def test_dynamic_boat_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            for count in [1,2,8]:
                path=Path(directory)/f'{count}.sdf'
                spec=build_world(path,count=count)
                tree=ET.parse(path)
                boat=tree.find('.//model[@name="small_cora_float"]')
                self.assertEqual(boat.findtext('static'),'false')
                self.assertEqual(len(boat.findall('plugin[@name="gz::sim::systems::Thruster"]')),2)
                self.assertIsNotNone(boat.find('plugin[@name="vrx::PolyhedraBuoyancyDrag"]'))
                self.assertEqual(len(spec['buoys']),count)
                self.assertEqual(len(spec['obstacles']),count+3)
                self.assertIsNone(tree.find('.//model[@name="vlm_boat"]'))


if __name__=='__main__': unittest.main()

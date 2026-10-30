"""Guidance geometry and measured, rather than planned, crossing acceptance."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from usv_vla.collection_standard import PROFILE, apply_standard, inspect_standard
from usv_vla import recollect_standard


def world_point(forward, left, yaw, center=(10., -4.)):
    return [center[0] + math.cos(yaw)*forward - math.sin(yaw)*left,
            center[1] + math.sin(yaw)*forward + math.cos(yaw)*left]


def gate_point(point, gate):
    dx, dy = point[0]-gate['center'][0], point[1]-gate['center'][1]
    c, s = math.cos(gate['yaw_rad']), math.sin(gate['yaw_rad'])
    return c*dx+s*dy, -s*dx+c*dy


def scene_for(side='right', yaw=.6, cruise=.4):
    sign = 1 if side == 'left' else -1
    gate = {'center': [10., -4.], 'yaw_rad': yaw,
            'left_min_m': 2. if sign > 0 else -3.7,
            'left_max_m': 3.7 if sign > 0 else -2., 'exit_forward_m': 2.}
    return {'initial_pose': [*world_point(-6., sign*.2, yaw), .14, yaw],
            'target': {'position': world_point(12., 0., yaw), 'color': 'red'},
            'obstacles': [{'position': gate['center'][:], 'radius_m': .405}],
            'route': {'kind': 'cylinder_pass', 'side': side, 'gates': [gate],
                      'cruise_speed_mps': cruise,
                      'waypoints': [world_point(-3., sign*2.6, yaw),
                                    world_point(0., sign*2.6, yaw),
                                    world_point(2.5, sign*2.6, yaw)]}}


class TestApplyCollectionStandard(unittest.TestCase):
    def test_mirrored_curves_align_at_gate_and_preserve_physical_scene(self):
        paths = {}
        for side in ('left', 'right'):
            with self.subTest(side=side):
                source = scene_for(side)
                original = deepcopy(source)
                revised = apply_standard(source)
                self.assertEqual(source, original, 'Planning must not mutate an existing dataset scene')
                for key in ('initial_pose', 'target', 'obstacles'):
                    self.assertEqual(revised[key], original[key])
                self.assertEqual(revised['route']['gates'], original['route']['gates'])
                self.assertEqual(revised['route']['side'], side)
                self.assertAlmostEqual(revised['route']['cruise_speed_mps'], .6)
                gate = revised['route']['gates'][0]
                local = [gate_point(p, gate) for p in revised['route']['waypoints']]
                paths[side] = local
                sign = 1 if side == 'left' else -1
                self.assertAlmostEqual(local[-2][0], 0.)
                self.assertAlmostEqual(local[-2][1], sign*2.1)
                self.assertAlmostEqual(local[-1][0], gate['exit_forward_m']+.5)
                self.assertAlmostEqual(local[-1][1], sign*2.1)
                # The approach has already become almost parallel before abeam;
                # the following exit segment keeps that tangent through the gate.
                df, dl = (local[-2][i]-local[-3][i] for i in (0, 1))
                self.assertLess(abs(math.degrees(math.atan2(dl, df))), 2.)
                self.assertTrue(all(a[0] < b[0] for a, b in zip(local, local[1:])))
                revised['obstacles'][0]['radius_m'] = 99.
                self.assertEqual(source, original, 'The result must own its nested physical metadata')
        for left, right in zip(paths['left'], paths['right']):
            self.assertAlmostEqual(left[0], right[0])
            self.assertAlmostEqual(left[1], -right[1])

    def test_speed_cap_and_direct_route_keep_original_guidance(self):
        source = scene_for(cruise=.7)
        revised = apply_standard(source)
        self.assertEqual(revised['route']['cruise_speed_mps'], .75)
        self.assertLessEqual(revised['route']['maneuver_speed_mps'], .75)
        direct = deepcopy(source)
        direct['route']['kind'] = 'direct'
        revised_direct = apply_standard(direct)
        self.assertEqual(revised_direct['route']['waypoints'], direct['route']['waypoints'])
        self.assertEqual(revised_direct['obstacles'], direct['obstacles'])
        self.assertEqual(revised_direct['route']['cruise_speed_mps'], .75)

    def test_rejects_curve_that_cannot_honor_gate(self):
        for kwargs in ({'control_forward_m': 0.}, {'control_forward_m': float('nan')},
                       {'lateral_m': 1.9}, {'lateral_m': 3.8}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                apply_standard(scene_for(), **kwargs)


class TestInspectCollectionStandard(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.episode = Path(temporary.name)

    def write_recording(self, *, side='right', yaw=.6, peak=30., before=8., after=0., cross=True):
        scene = apply_standard(scene_for(side, yaw))
        sign = 1 if side == 'left' else -1
        # Crossing is one quarter of the interval: f=-1 -> +3.  Planned
        # waypoints are aligned even when the raw boat headings are not.
        samples = [(-6., sign*.2, 0.), (-3., sign*1.8, sign*peak),
                   (-1., sign*2.2, sign*before),
                   (3. if cross else -.5, sign*2.6, sign*after)]
        rows = []
        for forward, left, angle in samples:
            heading = math.atan2(math.sin(yaw+math.radians(angle)), math.cos(yaw+math.radians(angle)))
            rows.append({'pose_xyzrpy': [*world_point(forward, left, yaw), .14, 0., 0., heading]})
        (self.episode/'episode.json').write_text(json.dumps({'scene_plan': scene}))
        (self.episode/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return scene

    def test_crossing_interpolates_raw_position_and_heading_for_both_sides(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                self.write_recording(side=side)
                result = inspect_standard(self.episode)
                self.assertTrue(result['passed'], result['issues'])
                self.assertEqual(result['profile'], PROFILE)
                self.assertEqual(result['crossing']['before_index'], 2)
                self.assertAlmostEqual(result['crossing']['fraction'], .25)
                sign = 1 if side == 'left' else -1
                self.assertAlmostEqual(result['crossing']['heading_relative_route_deg'], sign*6.)
                self.assertAlmostEqual(result['crossing']['left_m'], sign*2.3)
                self.assertAlmostEqual(result['initial_outward_peak_deg'], 30.)

    def test_heading_interpolation_wraps_across_pi(self):
        self.write_recording(side='left', yaw=math.radians(179.))
        result = inspect_standard(self.episode)
        self.assertTrue(result['passed'], result['issues'])
        self.assertAlmostEqual(result['crossing']['heading_relative_route_deg'], 6.)

    def test_rejects_excess_abeam_angle_despite_aligned_waypoints(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                self.write_recording(side=side, before=16., after=0.)
                result = inspect_standard(self.episode)
                self.assertFalse(result['passed'])
                self.assertAlmostEqual(abs(result['crossing']['heading_relative_route_deg']), 12.)
                self.assertEqual(result['issues'], ['Cylinder abeam heading exceeds 10 degrees'])

    def test_rejects_excess_initial_turn_even_with_good_abeam_angle(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                self.write_recording(side=side, peak=36.)
                result = inspect_standard(self.episode)
                self.assertFalse(result['passed'])
                self.assertAlmostEqual(result['initial_outward_peak_deg'], 36.)
                self.assertEqual(result['issues'], ['Initial outward turn exceeds 35 degrees'])

    def test_missing_crossing_or_current_profile_cannot_be_accepted(self):
        scene = self.write_recording(cross=False)
        with self.assertRaisesRegex(ValueError, 'No physical cylinder crossing'):
            inspect_standard(self.episode)
        scene.pop('demonstration_standard')
        (self.episode/'episode.json').write_text(json.dumps({'scene_plan': scene}))
        with self.assertRaisesRegex(ValueError, 'does not declare'):
            inspect_standard(self.episode)

    def test_nonfinite_crossing_yaw_is_rejected(self):
        for bad_yaw in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(yaw=bad_yaw):
                self.write_recording()
                path = self.episode/'steps.jsonl'
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[2]['pose_xyzrpy'][5] = bad_yaw
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.assertRaisesRegex(ValueError, 'Nonfinite.*observation'):
                    inspect_standard(self.episode)


class TestRecollectionSourceProtection(unittest.TestCase):
    def test_source_alias_and_wrong_profile_resume_are_refused_before_collect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'old_batch'
            source.mkdir()
            output = root/'new_batch'
            output.mkdir()
            manifest = output/'manifest.json'
            manifest.write_text(json.dumps({'entries': [{'scene': 'plan.json'}]}))
            (output/'plan.json').write_text(json.dumps({'demonstration_standard': {'profile': 'old'}}))
            before = manifest.read_bytes()
            for destination, expected in ((source/'.', 'must differ'),
                                          (output, 'Cannot resume')):
                with self.subTest(destination=destination):
                    args = SimpleNamespace(source_root=source, output_root=destination, resume=True)
                    parser = Mock()
                    parser.parse_args.return_value = args
                    with patch.object(recollect_standard.bc, 'parser', return_value=parser), \
                         patch.object(recollect_standard.bc, 'collect') as collector:
                        with self.assertRaisesRegex(ValueError, expected):
                            recollect_standard.main()
                        collector.assert_not_called()
                    self.assertEqual(manifest.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()

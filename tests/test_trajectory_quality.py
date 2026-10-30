"""Behavioral checks for recorded demonstration quality, separate from success."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from usv_vla.trajectory_quality import inspect_trajectory_quality, require_trajectory_quality


class TrajectoryQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.episode = Path(self.temp.name)

    def write(self, *, category="approach_visible", heading=None, speed=None,
              duration=20., target=None):
        count = round(duration/.1)+1
        times = [i*.1 for i in range(count)]
        angles = [heading(t) if heading else 0. for t in times]
        speeds = [speed(t) if speed else .5 for t in times]
        rates = [(angles[min(i+1, count-1)]-angles[max(i-1, 0)])/
                 ((min(i+1, count-1)-max(i-1, 0))*.1) for i in range(count)]
        route = {"kind": "direct", "cruise_speed_mps": .5, "waypoints": []}
        if category.startswith("around"):
            route.update(kind="cylinder_pass", waypoints=[[8., 4.], [12., 4.], [16., 4.]])
        elif category == "between":
            route.update(kind="cylinder_gap", waypoints=[[8., 0.], [12., 0.]])
        elif category == "shore":
            route.update(kind="bank_follow", waypoints=[[8., 0.], [12., 0.]])
        metadata = {"initial_pose": [0., 0., .14, angles[0]],
                    "scene_plan": {"category": category, "route": route,
                                   "target_xy": target or [50., 0.]},
                    "task": {"stop_min_m": 1.9, "stop_max_m": 2.3}}
        rows, x, y = [], 0., 0.
        for i, (time, angle, velocity, rate) in enumerate(zip(times, angles, speeds, rates)):
            if i:
                x += .1*velocity*math.cos(angle)
                y += .1*velocity*math.sin(angle)
            rows.append({"index": i, "decision_sim_time_ns": round(time*1e9),
                         "pose_xyzrpy": [x, y, .14, 0., 0., angle],
                         "state": [velocity, 0., rate]+[0.]*30,
                         "action": [3., 3.], "done": i == count-1})
        rows[-1]["action"] = [0., 0.]
        self.save(metadata, rows)
        return metadata, rows

    def save(self, metadata, rows):
        (self.episode/"episode.json").write_text(json.dumps(metadata))
        (self.episode/"steps.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))

    def test_large_single_turn_is_not_mistaken_for_weaving(self):
        angle = math.radians(80.)
        self.write(category="approach_offscreen",
                   heading=lambda t: angle*(1-math.cos(math.pi*t/20))/2,
                   target=[50*math.cos(angle), 50*math.sin(angle)])
        result = require_trajectory_quality(self.episode)
        self.assertGreater(result["metrics"]["yaw_total_variation_deg"], 75.)
        self.assertEqual(result["metrics"]["yaw_reversals_3deg"]["count"], 0)

    def test_slow_snake_is_rejected_even_without_large_yaw_rate(self):
        self.write(heading=lambda t: math.radians(8)*math.sin(4*math.pi*t/20))
        source = self.episode/"steps.jsonl"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        result = inspect_trajectory_quality(self.episode)
        self.assertFalse(result["passed"])
        self.assertLess(result["metrics"]["yaw_rate_abs_peak_rad_s"], .1)
        self.assertTrue(any("heading" in issue for issue in result["issues"]))
        with self.assertRaisesRegex(ValueError, "Trajectory smoothness"):
            require_trajectory_quality(self.episode)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_one_outward_and_return_turn_is_required_for_a_normal_bypass(self):
        self.write(category="around_left",
                   heading=lambda t: math.radians(40)*math.sin(math.pi*t/20))
        result = require_trajectory_quality(self.episode)
        self.assertEqual(result["metrics"]["yaw_reversals_10deg"]["count"], 1)
        self.assertEqual(result["reference"]["planned_sign_reversals"], 1)

    def test_repeated_outward_and_return_turns_are_rejected_for_bypass(self):
        self.write(category="around_right", duration=30.,
                   heading=lambda t: math.radians(40)*math.sin(3*math.pi*t/30))
        result = inspect_trajectory_quality(self.episode)
        self.assertFalse(result["passed"])
        self.assertGreater(result["metrics"]["yaw_reversals_10deg"]["count"], 1)
        self.assertTrue(any("around:" in issue for issue in result["issues"]))

    def test_gap_allows_one_small_correction_but_rejects_repeated_corrections(self):
        self.write(category="between", heading=lambda t: math.radians(4)*math.sin(math.pi*t/20))
        self.assertTrue(inspect_trajectory_quality(self.episode)["passed"])
        self.write(category="between", heading=lambda t: math.radians(4)*math.sin(3*math.pi*t/20))
        result = inspect_trajectory_quality(self.episode)
        self.assertFalse(result["passed"])
        self.assertTrue(any("near_collinear:" in issue for issue in result["issues"]))

    def test_broken_decision_grid_cannot_receive_quality_proof(self):
        metadata, rows = self.write()
        rows[40]["decision_sim_time_ns"] += 10_000_000
        self.save(metadata, rows)
        result = inspect_trajectory_quality(self.episode)
        self.assertFalse(result["passed"])
        self.assertTrue(any("10 Hz" in issue for issue in result["issues"]))

    def test_thrust_slew_checks_motion_but_allows_emergency_zero_on_exit(self):
        metadata, rows = self.write()
        result = require_trajectory_quality(self.episode)
        self.assertEqual(result["metrics"]["excluded_terminal_action_transitions"], 1)
        rows[80]["action"] = [6., 3.]
        self.save(metadata, rows)
        self.assertTrue(any("thrust slew" in issue for issue in
                            inspect_trajectory_quality(self.episode)["issues"]))
        rows[80]["action"] = [3., 3.]
        rows[-1]["done"] = False
        self.save(metadata, rows)
        self.assertFalse(inspect_trajectory_quality(self.episode)["passed"])

    def test_abrupt_speed_jump_fails_even_with_a_straight_heading(self):
        self.write(speed=lambda t: .5 if t < 10 else .65)
        result = inspect_trajectory_quality(self.episode)
        self.assertFalse(result["passed"])
        self.assertTrue(any("speed acceleration" in x or "speed jerk" in x for x in result["issues"]))

    def test_low_speed_limit_has_a_duration_boundary_after_motion_started(self):
        def background(t):
            if t < 2:
                return .3
            if t < 6:
                return .121+(.3-.121)*(1+math.cos(math.pi*(t-2)/4))/2
            return .121
        for samples, expected in ((11, True), (12, False)):
            with self.subTest(samples=samples):
                self.write(category="around_left",
                           speed=lambda t: .119 if 8-1e-8 <= t <= 8+(samples-1)*.1+1e-8 else background(t))
                result = inspect_trajectory_quality(self.episode)
                self.assertEqual(result["passed"], expected, result["issues"])
                self.assertAlmostEqual(result["metrics"]["navigation_low_speed_longest_s"], (samples-1)*.1)

    def test_initial_stationary_period_and_target_braking_are_not_mid_route_stalls(self):
        self.write(category="around_left", speed=lambda t: 0. if t < 4 else
                   .3*(1-math.cos(math.pi*min(t-4, 4)/4))/2)
        self.assertTrue(inspect_trajectory_quality(self.episode)["passed"])
        self.write(category="around_left", target=[4.2, 0.], speed=lambda t:
                   .3 if t < 2 else .05+.25*(1+math.cos(math.pi*min(t-2, 4)/4))/2)
        self.assertTrue(inspect_trajectory_quality(self.episode)["passed"])

    def test_speed_valley_catches_brake_then_accelerate_before_stopping_zone(self):
        valley = lambda t: .5-.25*math.exp(-((t-10)/1.5)**2/2)
        self.write(category="shore", speed=valley)
        result = inspect_trajectory_quality(self.episode)
        self.assertFalse(result["passed"])
        self.assertGreater(result["metrics"]["navigation_speed_valley_prominence_mps"], .12)
        self.assertTrue(any("speed valley" in x for x in result["issues"]))
        self.write(category="around_left", speed=valley)
        result = require_trajectory_quality(self.episode)
        self.assertTrue(result["metrics"]["navigation_speed_valley_is_diagnostic_only"])

    def test_monotonic_deceleration_has_no_speed_valley(self):
        self.write(speed=lambda t: .5-.25*(1-math.cos(math.pi*t/20))/2)
        result = require_trajectory_quality(self.episode)
        self.assertAlmostEqual(result["metrics"]["navigation_speed_valley_prominence_mps"], 0.)

    def test_nan_input_fails_and_proof_is_deterministic_and_serializable(self):
        metadata, rows = self.write()
        first = inspect_trajectory_quality(self.episode)
        self.assertEqual(first, inspect_trajectory_quality(self.episode))
        json.dumps(first, allow_nan=False)
        rows[7]["state"][0] = float("nan")
        self.save(metadata, rows)
        result = inspect_trajectory_quality(self.episode)
        self.assertFalse(result["passed"])
        self.assertTrue(any("non-finite" in x for x in result["issues"]))

    def test_tiny_initial_settling_is_allowed_but_real_reverse_is_rejected(self):
        for initial, expected in ((-.02, True), (-.04, False)):
            with self.subTest(initial=initial):
                self.write(speed=lambda t: initial+(.3-initial)*
                           (1-math.cos(math.pi*min(t, 4)/4))/2)
                result = inspect_trajectory_quality(self.episode)
                self.assertEqual(result["passed"], expected, result["issues"])
                self.assertAlmostEqual(result["metrics"]["minimum_forward_velocity_m_s"], initial)
                if not expected:
                    self.assertTrue(any("reverse motion" in x for x in result["issues"]))

    def test_approach_distance_overshoot_is_checked_over_entire_recording(self):
        # These straight, constant-speed runs only differ by their target
        # location.  The profile must catch a too-close run independently of
        # task success and of heading/rate/acceleration checks.
        self.write(target=[11.87, 0.])
        self.assertTrue(inspect_trajectory_quality(self.episode)["passed"])
        self.write(target=[11.80, 0.])
        result = inspect_trajectory_quality(self.episode)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["metrics"]["minimum_target_distance_m"], 1.80)
        self.assertTrue(any("target-distance overshoot" in x for x in result["issues"]))

    def test_forward_approach_rules_are_not_applied_to_explicit_other_tasks(self):
        metadata, rows = self.write(speed=lambda t: -.04+.34*
                                    (1-math.cos(math.pi*min(t, 4)/4))/2)
        metadata["task"]["kind"] = "pass"
        self.save(metadata, rows)
        result = require_trajectory_quality(self.episode)
        self.assertFalse(result["metrics"]["forward_approach_checks_applied"])
        metadata, rows = self.write(target=[11.8, 0.])
        metadata["task"]["kind"] = "pass"
        self.save(metadata, rows)
        self.assertTrue(inspect_trajectory_quality(self.episode)["passed"])


if __name__ == "__main__":
    unittest.main()

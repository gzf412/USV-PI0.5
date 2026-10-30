"""Causal route scoring and closed-loop teacher regressions."""
import math
import unittest

import numpy as np

from usv_vla.core import TaskConfig, wrap
from usv_vla.route import RouteTaskEvaluator, RouteTeacher


def gate(left_min=2., left_max=3.5):
    return dict(center=[7., 0.], yaw_rad=0., left_min_m=left_min, left_max_m=left_max, exit_forward_m=1.5)


def pass_route(side="left"):
    sign = 1 if side == "left" else -1
    return dict(kind="cylinder_pass", side=side, reference_yaw_rad=0., cruise_speed_mps=.5,
                waypoints=[[4., sign*2.7], [7., sign*2.7], [9., sign*2.7]],
                gates=[gate(2., 3.5) if sign == 1 else gate(-3.5, -2.)])


class RouteTests(unittest.TestCase):
    def task(self, route=None, obstacles=None):
        return RouteTaskEvaluator(TaskConfig(), {0: [15., 0.]}, [0., 0.], route=route, obstacles=obstacles)

    def test_faster_trial_waits_until_bank_recovery_and_keeps_normal_limit(self):
        route = pass_route()
        route.update(cruise_speed_mps=.84, maneuver_speed_mps=.65)
        with self.assertRaisesRegex(ValueError, "cruise"):
            self.task(route)
        route.update(speed_profile="simulation_speed_trial_v1",
                     speed_trial_bank_recovery=dict(origin=[0., 0.], yaw_rad=0., end_forward_m=9.5))
        task = self.task(route)
        task.route_complete = True
        teacher = RouteTeacher(task)
        teacher.command([10., .6, 0.], [.6, 0., 0.])
        self.assertFalse(teacher.speed_trial_recovered)
        teacher.command([10., .2, math.radians(20.)], [.6, 0., 0.])
        self.assertFalse(teacher.speed_trial_recovered)
        teacher.command([10., .2, math.radians(5.)], [.6, 0., 0.])
        self.assertTrue(teacher.speed_trial_recovered)

    def test_task_instruction_remains_fixed_when_phase_is_hidden(self):
        for instruction in (None, "Pass the first cylinder, then reach the buoy."):
            with self.subTest(instruction=instruction):
                route = pass_route()
                route["prompt_include_phase"] = False
                if instruction is not None:
                    route["instruction"] = instruction
                task = self.task(route)
                initial = task.context()
                self.assertNotIn("Current phase", initial)
                self.assertNotIn("frozen initial route heading", initial)
                if instruction is not None:
                    self.assertEqual(initial, instruction)
                task.update(0., [6.9, 2.7], .4)
                task.update(.1, [7.1, 2.7], .4)
                self.assertEqual(task.phase, "clear_cylinders")
                self.assertEqual(task.context(), initial)
                task.update(.2, [8.6, 2.7], .4)
                self.assertEqual(task.phase, "approach")
                self.assertEqual(task.context(), initial)

    def test_legacy_instruction_still_includes_current_phase(self):
        task = self.task(pass_route())
        initial = task.context()
        self.assertIn("Current phase: cylinder_pass.", initial)
        self.assertIn("frozen initial route heading", initial)
        task.update(0., [6.9, 2.7], .4)
        task.update(.1, [7.1, 2.7], .4)
        self.assertIn("Current phase: clear_cylinders.", task.context())
        task.update(.2, [8.6, 2.7], .4)
        self.assertIn("Current phase: approach.", task.context())

    def test_wrong_side_fails_and_teacher_waypoints_do_not_score(self):
        task = self.task(pass_route())
        teacher = RouteTeacher(task)
        teacher.waypoint = 3
        task.update(0., [6., -2.7], .4)
        result = task.update(.1, [8., -2.7], .4)
        self.assertFalse(result["success"])
        self.assertEqual(result["termination_reason"], "wrong_cylinder_side_or_gap")

    def test_pass_requires_gate_then_exit_before_goal(self):
        task = self.task(pass_route())
        task.update(0., [6.9, 2.7], .4)
        task.update(.1, [7.1, 2.7], .4)
        self.assertFalse(task.route_complete)
        task.update(.2, [8.6, 2.7], .4)
        self.assertTrue(task.route_complete)
        for i in range(3, 24):
            result = task.update(i*.1, [12.9, 0.], 0.)
        self.assertTrue(result["success"])
        self.assertEqual(result["route_gates_completed"], 1)

    def test_direct_goal_cannot_shortcut_gate(self):
        task = self.task(pass_route())
        for i in range(25):
            result = task.update(i*.1, [12.9, 0.], 0.)
        self.assertFalse(result["success"])
        self.assertFalse(result["route_complete"])
        self.assertEqual(result["hold_s"], 0.)

    def test_gap_scores_interpolated_crossing(self):
        route = dict(kind="cylinder_gap", waypoints=[[5., 0.], [9., 0.]], gates=[gate(-1., 1.)])
        task = self.task(route)
        task.update(0., [6.9, -.2], .5)
        task.update(.1, [7.1, .2], .5)
        task.update(.2, [8.6, .2], .5)
        self.assertTrue(task.route_complete)
        task = self.task(route)
        task.update(0., [6.9, 1.2], .5)
        result = task.update(.1, [7.1, 1.2], .5)
        self.assertEqual(result["termination_reason"], "wrong_cylinder_side_or_gap")

    def test_leaving_exit_strip_fails(self):
        task = self.task(pass_route())
        task.update(0., [6.9, 2.7], .4)
        task.update(.1, [7.1, 2.7], .4)
        result = task.update(.2, [7.4, .4], .4)
        self.assertEqual(result["termination_reason"], "left_cylinder_exit_corridor")

    def test_bank_requires_corridor_and_forward_progress(self):
        route = dict(kind="bank_follow", waypoints=[[5., 0.], [8.5, 0.]],
                     bank_corridor=dict(origin=[0., 0.], yaw_rad=0., left_min_m=-1.5,
                                        left_max_m=1.5, progress_start_m=0., progress_end_m=8.))
        task = self.task(route)
        for i in range(81):
            task.update(i*.1, [i*.1, 0.], .5)
        self.assertTrue(task.route_complete)
        shortcut = self.task(route)
        result = shortcut.update(0., [8., 0.], 0.)
        self.assertEqual(result["termination_reason"], "bank_route_not_started_at_corridor_entry")
        departed = self.task(route)
        departed.update(0., [0., 0.], 0.)
        result = departed.update(.1, [1., 2.], .5)
        self.assertEqual(result["termination_reason"], "left_bank_follow_corridor")

    def test_path_preflight_and_actual_swept_clearance(self):
        obstacle = dict(x=7., y=0., radius_m=.405, model="usv_cylinder_1")
        with self.assertRaisesRegex(ValueError, "clearance"):
            self.task(dict(kind="direct"), [obstacle])
        task = self.task(pass_route(), [obstacle])
        task.update(0., [5., 0.], .5)
        result = task.update(.1, [9., 0.], .5)
        self.assertFalse(result["success"])
        self.assertIn(result["termination_reason"], ("clearance_violation", "wrong_cylinder_side_or_gap"))

    def test_thrust_is_feedback_continuous_limited_and_slew_limited(self):
        task = self.task()
        teacher = RouteTeacher(task)
        commands = []
        for i in range(30):
            pose = [i*.02, 0., .25-i*.005, 0., 0.]
            task.update(i*.1, pose[:2], i*.01)
            commands.append(teacher.command(pose, [i*.01, 0., .01*i]))
        commands = np.asarray(commands)
        self.assertLessEqual(np.abs(commands).max(), 14.+1e-9)
        self.assertLessEqual(np.abs(np.diff(commands, axis=0)).max(), 1.+1e-9)
        self.assertGreater(len(np.unique(np.round(commands, 4), axis=0)), 20)
        self.assertFalse(np.allclose(commands[:, 0], commands[:, 1]))

    def test_guidance_rounds_corner_without_waypoint_jump(self):
        task = self.task(pass_route())
        teacher = RouteTeacher(task)
        corner = np.asarray(task.waypoints[0])
        direction = corner/np.linalg.norm(corner)
        before, safe_before = teacher._path_target(corner-.51*direction, .5)
        after, safe_after = teacher._path_target(corner-.49*direction, .5)
        self.assertTrue(safe_before and safe_after)
        # Crossing the old 0.5 m switch radius must not jump the target three
        # metres to the next vertex or command another stop at this corner.
        self.assertLess(np.linalg.norm(after-before), .05)
        self.assertGreater(before[0], corner[0]+.5)

    def test_safe_cylinder_lead_in_reduces_unnecessary_initial_turn(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                task = self.task(pass_route(side), [dict(x=7., y=0., radius_m=.405)])
                teacher = RouteTeacher(task)
                target, safe = teacher._path_target(task.start, 0.)
                self.assertTrue(safe)
                self.assertLess(abs(math.degrees(math.atan2(target[1], target[0]))), 25.)
                np.testing.assert_allclose(teacher.path[1], task.waypoints[1])
                np.testing.assert_allclose(teacher.path[2], task.waypoints[2])
                # Shorter guidance does not perform the physical crossing.
                self.assertFalse(task.route_complete)
                self.assertEqual(task.completed_gates, 0)

    def test_cylinder_lead_in_kept_when_shortcut_hits_obstacle(self):
        task = self.task(pass_route(), [dict(x=5., y=1.2, radius_m=.405)])
        teacher = RouteTeacher(task)
        np.testing.assert_allclose(teacher.path[1], task.waypoints[0])
        self.assertFalse(teacher._safe_preview(task.start, task.waypoints[1]))

    def test_cylinder_lead_in_cannot_skip_its_scored_gate(self):
        route = pass_route()
        route["waypoints"][1] = [7., 4.]
        task = self.task(route)
        teacher = RouteTeacher(task)
        np.testing.assert_allclose(teacher.path[1], task.waypoints[0])
        self.assertFalse(teacher._safe_preview(task.start, task.waypoints[1]))

    def test_displacement_does_not_jump_to_a_future_route_leg(self):
        task = self.task(pass_route())
        teacher = RouteTeacher(task)
        target, safe = teacher._path_target(np.array([10., -5.]), .5)
        self.assertFalse(safe)
        self.assertLessEqual(teacher.progress_m, teacher.lengths[0]+1e-9)
        self.assertLessEqual(target[0], task.gates[0]["center"][0])
        self.assertFalse(task.route_complete)
        self.assertEqual(task.completed_gates, 0)

    def test_preview_cannot_cut_the_cylinder_exit_strip(self):
        task = self.task(pass_route())
        teacher = RouteTeacher(task)
        self.assertFalse(teacher._safe_preview(np.array([8., 2.7]), np.array([9., 0.])))
        self.assertTrue(teacher._safe_preview(np.array([8., 2.7]), np.array([9., 2.7])))
        # The task must still score the physical crossing independently.
        self.assertFalse(task.route_complete)
        self.assertFalse(task.gate_passed)

    def test_preview_tolerates_world_rotation_roundoff_at_gate_boundary(self):
        task = self.task(pass_route())
        teacher = RouteTeacher(task)
        # A boundary waypoint transformed to/from a large world origin may
        # differ by a few ulps. It must not strand a safe boat before its exit.
        self.assertTrue(teacher._safe_preview(np.array([7.2, 2.]), np.array([9., 2.-5e-14])))
        self.assertFalse(teacher._safe_preview(np.array([7.2, 2.]), np.array([9., 2.-1e-4])))

    def test_preview_shortens_before_cutting_an_obstacle_corner(self):
        obstacle = dict(x=5., y=1.2, radius_m=.405, model="corner_obstacle")
        task = self.task(pass_route(), [obstacle])
        teacher = RouteTeacher(task)
        position = np.array([3.6, 2.43])
        target, safe = teacher._path_target(position, .5)
        full_preview = teacher._point_at(teacher.progress_m+1.95)
        self.assertFalse(teacher._safe_preview(position, full_preview))
        self.assertTrue(safe)
        self.assertTrue(teacher._safe_preview(position, target))
        self.assertLess(target[0], full_preview[0])

    def test_preview_stays_in_bank_corridor_until_physical_exit(self):
        route = dict(kind="bank_follow", waypoints=[[4., 0.], [9., 0.]],
                     bank_corridor=dict(origin=[0., 0.], yaw_rad=0., left_min_m=-.5,
                                        left_max_m=.5, progress_start_m=0., progress_end_m=8.))
        task = self.task(route)
        teacher = RouteTeacher(task)
        self.assertFalse(teacher._safe_preview(np.array([7.5, .4]), np.array([9., 2.])))
        self.assertTrue(teacher._safe_preview(np.array([7.5, .4]), np.array([9., .4])))
        task.update(0., [0., 0.], .5)
        task.update(.1, [8., .4], .5)
        self.assertTrue(task.route_complete)
        self.assertTrue(teacher._safe_preview(np.array([8., .4]), np.array([9., 2.])))

    def test_safety_stop_immediately_clears_previous_thrust(self):
        task = self.task()
        teacher = RouteTeacher(task)
        task.update(0., [0., 0.], 0.)
        self.assertGreater(max(teacher.command([0., 0., 0.], [0., 0., 0.])), 0.)
        task.update(.1, [.01, 0.], .1, collision=True)
        self.assertEqual(teacher.command([.01, 0., 0.], [.1, 0., 0.]), [0., 0.])
        self.assertTrue(np.array_equal(teacher.previous_command, [0., 0.]))

    def test_numerical_corner_path_avoids_repeated_large_heading_reversals(self):
        # Same independent damped model as the reach/stop regression below.
        # Task success alone previously admitted several large S-shaped turns.
        for route in (pass_route(), pass_route("right")):
            with self.subTest(side=route["side"]):
                task = self.task(route)
                teacher = RouteTeacher(task)
                xy, yaw, u, r, total_turn = np.zeros(2), 0., 0., 0., 0.
                for i in range(1200):
                    result = task.update(i*.1, xy, abs(u))
                    if result["done"]:
                        break
                    left, right = teacher.command([*xy, yaw, 0., 0.], [u, 0., r])
                    u += .1*((left+right)-28.*u)/20.
                    r += .1*((right-left)*.16-3.*r)/4.1
                    yaw = wrap(yaw+.1*r)
                    total_turn += abs(.1*r)
                    xy += .1*u*np.array([math.cos(yaw), math.sin(yaw)])
                self.assertTrue(result["success"], result)
                self.assertLess(math.degrees(total_turn), 150.)

    def test_numerical_damped_point_boat_reaches_and_stops(self):
        # Independent low-order 20 kg / 4.1 kg m^2 damped vessel model, not a
        # substitute for the actual Gazebo checks in batch collection.
        scenarios = [None, pass_route(), pass_route("right"),
                     dict(kind="cylinder_gap", waypoints=[[4., 0.], [9., 0.]], gates=[gate(-1., 1.)])]
        for route in scenarios:
            for initial_yaw in (0., math.radians(65)):
                with self.subTest(kind=route and route["kind"], yaw=initial_yaw):
                    task = RouteTaskEvaluator(TaskConfig(), {0: [15., 0.]}, [0., 0.], initial_yaw, route=route)
                    teacher = RouteTeacher(task)
                    xy = np.zeros(2)
                    yaw, u, r = initial_yaw, 0., 0.
                    for i in range(1200):
                        result = task.update(i*.1, xy, abs(u))
                        if result["done"]:
                            break
                        left, right = teacher.command([*xy, yaw, 0., 0.], [u, 0., r])
                        u += .1*((left+right)-28.*u)/20.
                        r += .1*((right-left)*.16-3.*r)/4.1
                        yaw = wrap(yaw+.1*r)
                        xy += .1*u*np.array([math.cos(yaw), math.sin(yaw)])
                    self.assertTrue(result["success"], result)
                    self.assertTrue(result["route_complete"])
                    self.assertGreaterEqual(result["distance_m"], 1.9)
                    self.assertLessEqual(result["distance_m"], 2.3)

    def test_short_final_leg_brakes_before_bank_or_gate_exit_without_reversing(self):
        # The short bank scene has a final vertex 2.4 m from the buoy, only
        # 0.3 m before the nominal 2.1 m stop.  Waiting until that vertex to
        # brake at 0.574 m/s previously caused overshoot, reverse recovery and
        # a large speed jerk despite the motor-command slew limit holding.
        for kind in ("bank_follow", "cylinder_gap"):
            for initial_yaw in (0., math.radians(-2.1)):
                with self.subTest(kind=kind, initial_yaw=initial_yaw):
                    route = dict(kind=kind, reference_yaw_rad=0., cruise_speed_mps=.574,
                                 waypoints=[[2.954, 0.], [5.44, 0.], [6.04, 0.]], gates=[])
                    if kind == "bank_follow":
                        route["bank_corridor"] = dict(origin=[0., 0.], yaw_rad=0., left_min_m=-1.8,
                                                      left_max_m=1.8, progress_start_m=0., progress_end_m=5.44)
                    else:
                        route["gates"] = [dict(center=[3., 0.], yaw_rad=0., left_min_m=-.7,
                                               left_max_m=.7, exit_forward_m=2.)]
                    obstacles = [dict(x=8.44, y=0., radius_m=.75, model="target_buoy")]
                    if kind == "cylinder_gap":
                        obstacles.extend(dict(x=3., y=side*2.3, radius_m=.405, model=f"cylinder_{side}")
                                         for side in (-1, 1))
                    task = RouteTaskEvaluator(TaskConfig(), {0: [8.44, 0.]}, [0., 0.], initial_yaw,
                                              route=route, obstacles=obstacles)
                    teacher = RouteTeacher(task)
                    xy, yaw, u, r = np.zeros(2), initial_yaw, 0., 0.
                    surge, distances, positions, commands = [], [], [], []
                    for i in range(1200):
                        result = task.update(i*.1, xy, abs(u))
                        surge.append(u)
                        distances.append(result["distance_m"])
                        positions.append(xy.copy())
                        if result["done"]:
                            break
                        left, right = teacher.command([*xy, yaw, 0., 0.], [u, 0., r])
                        commands.append([left, right])
                        u += .1*((left+right)-28.*u)/20.
                        r += .1*((right-left)*.16-3.*r)/4.1
                        yaw = wrap(yaw+.1*r)
                        xy += .1*u*np.array([math.cos(yaw), math.sin(yaw)])
                    self.assertTrue(result["success"], result)
                    self.assertTrue(result["route_complete"])
                    self.assertEqual(result["route_gates_completed"], len(route["gates"]))
                    self.assertGreaterEqual(result["bank_progress_m"], 5.44 if kind == "bank_follow" else 0.)
                    self.assertGreaterEqual(min(distances), 1.9)
                    self.assertGreaterEqual(min(surge), -1e-9)
                    self.assertGreaterEqual(np.diff(np.asarray(positions)[:, 0]).min(), -1e-9)
                    speed_acceleration = np.diff(np.abs(surge))/.1
                    self.assertLess(np.abs(np.diff(speed_acceleration)/.1).max(), 2.5)
                    self.assertLessEqual(np.abs(np.diff(commands, axis=0)).max(), 1.+1e-9)


if __name__ == "__main__":
    unittest.main()

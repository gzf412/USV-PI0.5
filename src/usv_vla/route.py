"""Geometry-only demonstration controller and independently scored USV routes.

The route and obstacle map are privileged demonstration/evaluation data.  Neither
is added to the policy state or images.  Only the human task and its phase are
included in the language context.
"""

import math

import numpy as np

from usv_vla.core import TaskEvaluator, body_xy, limited_action, wrap

ROUTE_KINDS = ("direct", "cylinder_pass", "cylinder_gap", "bank_follow")
VESSEL_RADIUS_M = .85


def _point(value, name):
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be finite XY coordinates")
    return point


def _segment_distance(a, b, point):
    delta = b-a
    length2 = float(delta@delta)
    weight = np.clip(float((point-a)@delta)/length2, 0., 1.) if length2 else 0.
    return float(np.linalg.norm(a+weight*delta-point))


def _gate_coordinates(gate, xy):
    return body_xy(*(_point(xy, "sample")-_point(gate["center"], "gate center")), gate["yaw_rad"])


class RouteTaskEvaluator(TaskEvaluator):
    """Score physical gate crossings, exit progress, and bank corridor occupancy.

    Route success never depends on the teacher's waypoint index.  The usual
    collision, timestamp, stop-distance, low-speed, and dwell checks still apply.
    """

    def __init__(self, config, buoys, start_xy, initial_yaw=0., *, route=None, obstacles=None):
        super().__init__(config, buoys, start_xy, initial_yaw)
        if config.kind != "approach":
            raise ValueError("Route tasks finish by approaching the unique red-round target")
        self.route = dict(route or {"kind": "direct"})
        self.kind = self.route.get("kind", "direct")
        if self.kind not in ROUTE_KINDS:
            raise ValueError("Unknown route task kind")
        self.reference_yaw = float(self.route.get("reference_yaw_rad", self.reference_yaw))
        self.cruise_speed_mps = float(self.route.get("cruise_speed_mps", .5))
        # Higher speeds are opt-in for separate simulator trials. Existing
        # collection scenes retain the original 0.75 m/s planning limit.
        speed_limit = 1.05 if self.route.get("speed_profile") == "simulation_speed_trial_v1" else .75
        if not math.isfinite(self.reference_yaw) or not .15 <= self.cruise_speed_mps <= speed_limit:
            raise ValueError("Invalid route heading or cruise speed")
        preview = self.route.get("speed_trial_lookahead_m")
        if preview is not None and (self.route.get("speed_profile") != "simulation_speed_trial_v1"
                                    or not math.isfinite(float(preview)) or not 1.5 <= float(preview) <= 4.):
            raise ValueError("Extended preview is only valid for simulation speed trials (1.5..4m)")
        recovery = self.route.get("speed_trial_bank_recovery")
        if recovery is not None:
            if self.route.get("speed_profile") != "simulation_speed_trial_v1" or not isinstance(recovery, dict):
                raise ValueError("Bank recovery is restricted to simulation speed trials")
            _point(recovery.get("origin"), "recovery origin")
            if not np.isfinite([recovery.get("yaw_rad"), recovery.get("end_forward_m")]).all():
                raise ValueError("Invalid bank recovery reference")
        self.maneuver_speed_mps = float(self.route.get("maneuver_speed_mps", self.cruise_speed_mps))
        if not math.isfinite(self.maneuver_speed_mps) or not .15 <= self.maneuver_speed_mps <= self.cruise_speed_mps:
            raise ValueError("Invalid maneuver speed")
        self.waypoints = [_point(p, "waypoint") for p in self.route.get("waypoints", [])]
        self.obstacles = list(obstacles or [])
        self.gates = [dict(gate) for gate in self.route.get("gates", [])]
        self.gate_index, self.gate_crossed = 0, False
        self.completed_gates = 0
        self.bank = self.route.get("bank_corridor")
        self.bank_started = False
        self.bank_progress_m = 0.
        self.route_complete = self.kind == "direct"
        self.phase = "approach" if self.route_complete else self.kind
        self._validate_geometry()

    def _validate_geometry(self):
        for gate in self.gates:
            _point(gate["center"], "gate center")
            values = [gate[key] for key in ("yaw_rad", "left_min_m", "left_max_m", "exit_forward_m")]
            if not np.isfinite(values).all() or gate["left_min_m"] >= gate["left_max_m"] or gate["exit_forward_m"] <= 0:
                raise ValueError("Invalid route gate")
        if self.kind in ("cylinder_pass", "cylinder_gap") and not self.gates:
            raise ValueError("Cylinder routes need an independently scored gate")
        if self.kind == "bank_follow":
            if self.bank is None:
                raise ValueError("Bank-follow route requires a real bank corridor")
            _point(self.bank["origin"], "bank corridor origin")
            values = [self.bank[key] for key in ("yaw_rad", "left_min_m", "left_max_m", "progress_start_m", "progress_end_m")]
            if (not np.isfinite(values).all() or self.bank["left_min_m"] >= self.bank["left_max_m"]
                    or self.bank["progress_end_m"]-self.bank["progress_start_m"] < 3.):
                raise ValueError("Bank corridor needs at least 3m of progress")
        if self.kind != "direct" and not self.waypoints:
            raise ValueError("Non-direct route needs a collision-checked demonstration path")
        if self.waypoints:
            for gate in self.gates:
                if _gate_coordinates(gate, self.waypoints[-1])[0] < gate["exit_forward_m"]+.15:
                    raise ValueError("Final route waypoint must extend beyond the scored gate exit")
            if self.bank is not None:
                end_forward = body_xy(*(self.waypoints[-1]-_point(self.bank["origin"], "bank origin")), self.bank["yaw_rad"])[0]
                if end_forward < self.bank["progress_end_m"]+.15:
                    raise ValueError("Final route waypoint must extend beyond the bank corridor end")
        # This preflight rejects impossible routes before any motor command.
        points = [self.start, *self.waypoints]
        last = points[-1]
        delta = last-self.a
        stop = self.a+delta/max(float(np.linalg.norm(delta)), 1e-9)*((self.config.stop_min_m+self.config.stop_max_m)/2)
        points.append(stop)
        for a, b in zip(points, points[1:]):
            for obstacle in self.obstacles:
                center = _point([obstacle["x"], obstacle["y"]], "obstacle")
                clearance = _segment_distance(a, b, center)-VESSEL_RADIUS_M-float(obstacle["radius_m"])
                if clearance < self.config.minimum_clearance_m-1e-9:
                    raise ValueError(f"Demonstration path violates obstacle clearance: {obstacle.get('model', 'obstacle')}")
        for gate in self.gates:
            if _gate_coordinates(gate, self.start)[0] >= -.5:
                raise ValueError("Route must start before its scored gate")

    def context(self):
        stop = f"stop {self.config.stop_min_m:g} to {self.config.stop_max_m:g} meters from its center"
        instruction = self.route.get("instruction")
        if instruction is None:
            if self.kind == "cylinder_pass":
                side = self.route.get("side", "left")
                instruction = (f"Pass the initially nearest orange cylinder on its {side} side, "
                               f"then approach the red round buoy and {stop}.")
            elif self.kind == "cylinder_gap":
                instruction = ("Pass between the two initially nearest orange cylinders, "
                               f"then approach the red round buoy and {stop}.")
            elif self.kind == "bank_follow":
                instruction = f"Follow the river bank forward, then approach the red round buoy and {stop}."
            else:
                instruction = f"Approach the red round buoy and {stop}."
        # New complex demonstrations can use only the user-given instruction;
        # the teacher's ground-truth progress is unavailable to a real policy.
        if self.route.get("prompt_include_phase", True) is False:
            return instruction
        reference = wrap(self.reference_yaw-self.initial_yaw)
        return (f"{instruction} Current phase: {self.phase}. "
                f"Left and right use the frozen initial route heading: "
                f"sin {math.sin(reference):.3f}, cos {math.cos(reference):.3f} relative to the initial boat heading.")

    def _update_gates(self, xy):
        if self.gate_index >= len(self.gates):
            return
        gate = self.gates[self.gate_index]
        forward, left = _gate_coordinates(gate, xy)
        if not self.gate_crossed and self.previous is not None:
            old_forward, old_left = _gate_coordinates(gate, self.previous)
            if old_forward < 0 <= forward:
                fraction = -old_forward/(forward-old_forward)
                crossing_left = old_left+fraction*(left-old_left)
                if not gate["left_min_m"] <= crossing_left <= gate["left_max_m"]:
                    self.finish("wrong_cylinder_side_or_gap")
                    return
                self.gate_crossed, self.gate_passed = True, True
                self.phase = "clear_cylinders"
        if self.gate_crossed:
            if forward < -.05:
                self.finish("returned_before_cylinder_gate")
            elif not gate["left_min_m"]-.1 <= left <= gate["left_max_m"]+.1:
                # Stay in the protected strip until the hull is safely past it.
                self.finish("left_cylinder_exit_corridor")
            elif forward >= gate["exit_forward_m"]:
                self.completed_gates += 1
                self.gate_index += 1
                self.gate_crossed = False
                if self.gate_index == len(self.gates):
                    self.route_complete = self.pass_complete = True
                    self.phase = "approach"
                else:
                    self.phase = self.kind

    def _update_bank(self, xy):
        forward, left = body_xy(*(_point(xy, "sample")-_point(self.bank["origin"], "bank origin")), self.bank["yaw_rad"])
        if not self.bank_started:
            if abs(forward-self.bank["progress_start_m"]) > .75:
                self.finish("bank_route_not_started_at_corridor_entry")
                return
            self.bank_started = True
        if not self.bank["left_min_m"] <= left <= self.bank["left_max_m"]:
            self.finish("left_bank_follow_corridor")
            return
        if forward < self.bank["progress_start_m"]-.75:
            self.finish("reversed_out_of_bank_corridor")
            return
        self.bank_progress_m = max(self.bank_progress_m, forward-self.bank["progress_start_m"])
        if forward >= self.bank["progress_end_m"]:
            self.route_complete = self.pass_complete = True
            self.phase = "approach"

    def update(self, timestamp_s, xy, speed_mps, *, collision=False, clearance_m=math.inf):
        xy = _point(xy, "sample").copy()
        if not np.isfinite([timestamp_s, speed_mps]).all() or speed_mps < 0 or math.isnan(clearance_m):
            raise ValueError("Invalid evaluation sample")
        if not self.done and self.previous_time is not None and timestamp_s <= self.previous_time:
            raise ValueError("Task samples must have strictly increasing source time")
        if not self.done:
            if self.kind == "bank_follow" and not self.route_complete:
                self._update_bank(xy)
            elif self.gates and not self.route_complete:
                self._update_gates(xy)
            if self.obstacles:
                point, old = _point(xy, "sample"), self.previous if self.previous is not None else _point(xy, "sample")
                swept = min(_segment_distance(old, point, _point([o["x"], o["y"]], "obstacle"))
                            -VESSEL_RADIUS_M-o["radius_m"] for o in self.obstacles)
                clearance_m = min(clearance_m, swept)
        # Parent independently checks low-speed dwell and all common faults.  If
        # the buoy is reached without doing the route, dwell cannot accumulate.
        super().update(timestamp_s, xy, speed_mps, collision=collision, clearance_m=clearance_m)
        if not self.route_complete and not self.done:
            self.hold_start = None
        elif not self.route_complete and self.success:
            self.done = self.success = False
            self.reason = "running"
            self.hold_start = None
        return self.result(timestamp_s, xy, speed_mps)

    def result(self, timestamp_s, xy, speed_mps):
        result = super().result(timestamp_s, xy, speed_mps)
        result.update(route_kind=self.kind, route_complete=self.route_complete,
                      route_gates_completed=self.completed_gates, route_gate_count=len(self.gates),
                      bank_progress_m=self.bank_progress_m)
        return result


class RouteTeacher:
    """Continuous feedback, damped steering, speed limiting, and thrust slew.

    No forward/left/right action lookup table or random thrust perturbation is
    used.  Diverse actions come from positions, headings, speeds, and maneuvers.
    Commands are Newtons and are limited to 14 N per motor, below the 15 N limit.
    """

    def __init__(self, evaluator):
        self.task = evaluator
        self.speed_trial_recovered = False
        self.waypoint = 0
        self.previous_command = np.zeros(2)
        self.previous_time = None
        # Follow a continuous arc-length parameterization of the checked route.
        # The final leg uses the same stopping point as the route preflight.
        last = evaluator.waypoints[-1] if evaluator.waypoints else evaluator.start
        delta = last-evaluator.a
        stop = evaluator.a+delta/max(float(np.linalg.norm(delta)), 1e-9)*(
            (evaluator.config.stop_min_m+evaluator.config.stop_max_m)/2)
        waypoints = list(evaluator.waypoints)
        # A generated cylinder route has a lateral setup point before its gate.
        # If the gate can already be approached safely from the start, seeking
        # that setup point adds an unnecessary sharp outward turn.  Keep the
        # scored gate and exit vertices, and omit only this optional lead-in.
        # The scene, task corridor, and independent crossing score stay intact.
        if evaluator.kind == "cylinder_pass" and len(waypoints) >= 3:
            gate = evaluator.gates[0]
            before = _gate_coordinates(gate, waypoints[0])[0]
            at_gate = _gate_coordinates(gate, waypoints[1])[0]
            if (before < 0. and abs(at_gate) < 1e-6
                    and self._safe_preview(evaluator.start, waypoints[1])):
                waypoints.pop(0)
        vertices = [evaluator.start, *waypoints, stop]
        self.path = [np.asarray(vertices[0], dtype=float).copy()]
        for point in vertices[1:]:
            if np.linalg.norm(point-self.path[-1]) > 1e-6:
                self.path.append(np.asarray(point, dtype=float).copy())
        self.path = np.asarray(self.path)
        self.lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.arcs = np.r_[0., np.cumsum(self.lengths)]
        self.segment = 0
        self.progress_m = 0.

    def _point_at(self, progress):
        segment = min(len(self.lengths)-1, max(0, int(np.searchsorted(self.arcs, progress, side="right")-1)))
        fraction = np.clip((progress-self.arcs[segment])/self.lengths[segment], 0., 1.)
        return self.path[segment]+fraction*(self.path[segment+1]-self.path[segment])

    def _safe_preview(self, position, target):
        task = self.task
        for obstacle in task.obstacles:
            center = _point([obstacle["x"], obstacle["y"]], "obstacle")
            if (_segment_distance(position, target, center)-VESSEL_RADIUS_M-obstacle["radius_m"]
                    < task.config.minimum_clearance_m-1e-9):
                return False
        # A preview may round a corner, but cannot cut across the protected
        # gate/exit strip.  Test the entire straight preview inside that strip.
        strips = [(gate["center"], gate["yaw_rad"], 0., gate["exit_forward_m"],
                   gate["left_min_m"], gate["left_max_m"])
                  for gate in task.gates[task.gate_index:]]
        if task.bank is not None and not task.route_complete:
            bank = task.bank
            strips.append((bank["origin"], bank["yaw_rad"], bank["progress_start_m"],
                           bank["progress_end_m"], bank["left_min_m"], bank["left_max_m"]))
        for origin, yaw, forward_min, forward_max, left_min, left_max in strips:
            a = np.asarray(body_xy(*(position-_point(origin, "strip origin")), yaw))
            b = np.asarray(body_xy(*(target-_point(origin, "strip origin")), yaw))
            delta = b-a
            if abs(delta[0]) < 1e-9:
                if not forward_min <= a[0] <= forward_max:
                    continue
                bounds = (a[1], b[1])
            else:
                ends = sorted(((forward_min-a[0])/delta[0], (forward_max-a[0])/delta[0]))
                lo, hi = max(0., ends[0]), min(1., ends[1])
                if lo > hi:
                    continue
                bounds = (a[1]+lo*delta[1], a[1]+hi*delta[1])
            # Rotating a waypoint back from world coordinates can put an exact
            # boundary 1e-14 m outside its interval. Match the numerical
            # tolerance already used for obstacle clearance, without widening
            # the physically scored gate or exit corridor.
            if not all(left_min-1e-9 <= left <= left_max+1e-9 for left in bounds):
                return False
        return True

    def _path_target(self, position, speed):
        lookahead = float(np.clip(1.5+.9*max(0., speed), 1.5, 2.1))
        if self.task.route.get("speed_trial_lookahead_m") is not None:
            lookahead = max(lookahead, float(self.task.route["speed_trial_lookahead_m"]))
        # Search only the current leg and its immediate successor.  A global
        # nearest-point search could jump over an unperformed gate when the
        # vessel is displaced or when two path legs happen to be close.
        candidates = []
        for segment in range(self.segment, min(self.segment+2, len(self.lengths))):
            if segment > self.segment and np.linalg.norm(position-self.path[segment]) > 2.*lookahead:
                continue
            a, b = self.path[segment:segment+2]
            fraction = float(np.clip((position-a)@(b-a)/self.lengths[segment]**2, 0., 1.))
            projected = a+fraction*(b-a)
            candidates.append((float(np.linalg.norm(position-projected)), segment,
                               float(self.arcs[segment]+fraction*self.lengths[segment])))
        distance, self.segment, progress = min(candidates)
        self.progress_m = max(self.progress_m, progress)
        # Keep the compatibility counter observational; task scoring never
        # consults it, and guidance no longer seeks each waypoint to 0.5 m.
        self.waypoint = min(len(self.task.waypoints), self.segment)
        # A distant localization jump must not rejoin beyond an unperformed
        # gate merely because the shorter lead-in ends at that gate.
        if distance > 2.*lookahead:
            return self._point_at(self.progress_m), False
        for fraction in (1., .75, .5, .25, 0.):
            target = self._point_at(min(self.arcs[-1], self.progress_m+fraction*lookahead))
            if self._safe_preview(position, target):
                return target, True
        return self._point_at(self.progress_m), False

    def command(self, pose, velocity):
        task = self.task
        if task.done:
            self.previous_command[:] = 0.
            return [0., 0.]
        position = _point(pose[:2], "boat position")
        yaw, u, r = float(pose[2]), float(velocity[0]), float(velocity[2])
        now = task.previous_time
        dt = .1 if self.previous_time is None or now is None else max(.02, min(.2, now-self.previous_time))
        self.previous_time = now
        cruise = task.cruise_speed_mps
        if task.kind == "cylinder_pass" and not task.route_complete:
            cruise = min(cruise, task.maneuver_speed_mps)
        recovery = task.route.get("speed_trial_bank_recovery")
        if recovery is not None and not self.speed_trial_recovered:
            forward, lateral = body_xy(*(position-_point(recovery["origin"], "recovery origin")), recovery["yaw_rad"])
            aligned = abs(wrap(yaw-recovery["yaw_rad"])) <= math.radians(10.)
            if forward >= recovery["end_forward_m"] and abs(lateral) <= .4 and aligned:
                self.speed_trial_recovered = True
            else:
                cruise = min(cruise, task.maneuver_speed_mps)
        target, safe_preview = self._path_target(position, u)
        # Braking belongs to the entire route, not only its final segment.
        # A bank exit or cylinder waypoint may lie just centimetres before the
        # stopping point, leaving too little distance to brake after switching
        # segments.  Remaining arc length gives the same continuous slowdown
        # on either side of such a vertex without stopping at the vertex itself.
        remaining_path = max(0., float(self.arcs[-1]-self.progress_m))
        final_leg = self.segment == len(self.lengths)-1 and task.route_complete
        if final_leg:
            delta = task.a-position
            remaining = float(np.linalg.norm(delta))-(task.config.stop_min_m+task.config.stop_max_m)/2
            bearing = wrap(math.atan2(delta[1], delta[0])-yaw)
            # Signed radial error allows a small, controlled reverse correction
            # while maintaining bow-to-target heading if there is overshoot.
            desired = max(-.12, min(cruise, .55*remaining))
        else:
            bearing = wrap(math.atan2(*(target-position)[::-1])-yaw)
            # Intermediate vertices do not ask the vessel to brake to a point.
            # Limit speed against the route's eventual stop, even when its
            # final leg is too short for the whole deceleration maneuver.
            desired = min(cruise, .55*remaining_path)
        if not safe_preview:
            desired = 0.
        desired *= max(0., math.cos(bearing))**2
        if abs(bearing) > 1.15:
            desired = 0.
        throttle = np.clip(14.*desired+45.*(desired-u), -12., 12.)
        steering = np.clip(7.*bearing-8.*r, -7., 7.)
        requested = np.asarray(limited_action([throttle-steering, throttle+steering]))
        peak = float(np.max(np.abs(requested)))
        if peak > 14.:
            requested *= 14./peak
        maximum_change = 10.*dt
        command = np.clip(requested, self.previous_command-maximum_change, self.previous_command+maximum_change)
        self.previous_command = command
        return limited_action(command)

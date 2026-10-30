"""Causal observation packing and deterministic buoy-task evaluation."""

from dataclasses import dataclass
import math

import numpy as np

BUOY_SLOTS = tuple((color, shape) for color in ("red", "yellow", "green", "blue")
                   for shape in ("round", "pointed"))
STATE_DIM, ACTION_DIM, MAX_FORCE = 33, 2, 15.0


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def body_xy(dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def limited_action(action):
    values = np.asarray(action, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("Motor command must be finite [left_N, right_N].")
    return np.clip(values, -MAX_FORCE, MAX_FORCE).tolist()


def pack_state(pose, velocity, initial_yaw, previous_command, buoys):
    """buoys maps stable zero-based semantic slots to valid map XY positions.

    Invalid/stale poses must be gated before this call. Off-camera positions
    retain their sign; absence is represented by the missing map key only.
    """
    x, y, yaw, roll, pitch = pose
    u, v, r = velocity
    previous = limited_action(previous_command)
    state = [u, v, r, roll, pitch, math.sin(yaw-initial_yaw), math.cos(yaw-initial_yaw),
             previous[0]/MAX_FORCE, previous[1]/MAX_FORCE]
    if any(k not in range(8) for k in buoys):
        raise ValueError("Unknown buoy slot.")
    for index in range(8):
        if index not in buoys:
            state.extend([0., 0., 0.])
        else:
            bx, by = buoys[index]
            state.extend([*body_xy(bx-x, by-y, yaw), 1.])
    result = np.asarray(state, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite state or target position.")
    return result


@dataclass(frozen=True)
class TaskConfig:
    kind: str = "approach"
    target: int = 0
    destination: int | None = None
    side: str = "left"
    stop_min_m: float = 1.9
    stop_max_m: float = 2.3
    stop_speed_mps: float = .2
    hold_s: float = 2.
    timeout_s: float = 120.
    gate_min_m: float = 2.
    gate_max_m: float = 4.
    exit_min_m: float = 2.
    exit_max_m: float = 4.
    minimum_clearance_m: float = .2
    maximum_gap_s: float = .3
    maximum_from_start_m: float = 45.

    def __post_init__(self):
        if self.kind not in ("approach", "pass", "pass_then_approach") or self.side not in ("left", "right"):
            raise ValueError("Unsupported task kind or side.")
        if self.target not in range(8) or (self.destination is not None and self.destination not in range(8)):
            raise ValueError("Invalid semantic target slot.")
        if self.kind == "pass_then_approach" and (self.destination is None or self.destination == self.target):
            raise ValueError("Combination task needs a different destination.")
        if not (0 < self.stop_min_m < self.stop_max_m and 0 < self.gate_min_m < self.gate_max_m
                and 0 < self.exit_min_m < self.exit_max_m):
            raise ValueError("Invalid distance bands.")
        if any(not math.isfinite(x) or x <= 0 for x in (
            self.stop_speed_mps, self.hold_s, self.timeout_s, self.minimum_clearance_m,
            self.maximum_gap_s, self.maximum_from_start_m,
        )):
            raise ValueError("Thresholds must be finite and positive.")


class TaskEvaluator:
    def __init__(self, config, buoys, start_xy, initial_yaw=0.):
        self.config, self.buoys, self.start = config, dict(buoys), np.asarray(start_xy, dtype=float)
        if config.target not in buoys or (config.destination is not None and config.destination not in buoys):
            raise ValueError("Task refers to absent buoy.")
        self.a = np.asarray(buoys[config.target], dtype=float)
        direction = self.a-self.start
        if np.linalg.norm(direction) < config.stop_max_m:
            raise ValueError("Start is too close to target for the task.")
        self.reference_yaw = math.atan2(direction[1], direction[0])
        self.initial_yaw = initial_yaw
        self.phase = "approach" if config.kind == "approach" else "pass"
        self.previous = self.previous_time = self.started = self.hold_start = None
        self.done = self.success = self.gate_passed = self.pass_complete = False
        self.reason = "running"

    def coordinates(self, xy):
        return body_xy(*(np.asarray(xy)-self.a), self.reference_yaw)

    def world_point(self, forward, left):
        c, s = math.cos(self.reference_yaw), math.sin(self.reference_yaw)
        return self.a + np.array([c*forward-s*left, s*forward+c*left])

    def finish(self, reason, success=False):
        self.done, self.success, self.reason = True, success, reason

    def context(self):
        cfg = self.config
        name = lambda i: " ".join(BUOY_SLOTS[i]) + " buoy"
        stop = f"stop {cfg.stop_min_m:g} to {cfg.stop_max_m:g} meters away"
        if cfg.kind == "approach":
            original = f"Approach the {name(cfg.target)} and {stop}."
        else:
            original = f"Pass the {name(cfg.target)} on its {cfg.side} side"
            original += (f", then approach the {name(cfg.destination)} and {stop}."
                         if cfg.kind == "pass_then_approach" else ", then stop in the exit region.")
        reference = wrap(self.reference_yaw-self.initial_yaw)
        return (f"{original} Current phase: {self.phase}. "
                f"Frozen approach reference relative to episode initial heading: "
                f"sin {math.sin(reference):.3f}, cos {math.cos(reference):.3f}. "
                f"Left is positive in that frozen reference.")

    def update(self, timestamp_s, xy, speed_mps, *, collision=False, clearance_m=math.inf):
        if self.done:
            return self.result(timestamp_s, xy, speed_mps)
        if not np.isfinite([timestamp_s, *xy, speed_mps]).all() or speed_mps < 0 or math.isnan(clearance_m):
            raise ValueError("Invalid evaluation sample.")
        if self.previous_time is not None and timestamp_s <= self.previous_time:
            raise ValueError("Task samples must have strictly increasing source time.")
        if self.started is None:
            self.started = timestamp_s
        if collision:
            self.finish("collision")
        elif clearance_m < self.config.minimum_clearance_m:
            self.finish("clearance_violation")
        elif timestamp_s-self.started >= self.config.timeout_s:
            self.finish("timeout")
        elif np.linalg.norm(np.asarray(xy)-self.start) > self.config.maximum_from_start_m:
            self.finish("out_of_bounds")
        elif self.previous_time is not None and timestamp_s-self.previous_time > self.config.maximum_gap_s:
            self.finish("stale_state")
        if self.done:
            return self.result(timestamp_s, xy, speed_mps)
        cfg = self.config
        local = self.coordinates(xy)
        side_y = local[1] * (1 if cfg.side == "left" else -1)
        if self.phase == "pass" and self.previous is not None:
            old = self.coordinates(self.previous)
            if old[0] < 0 <= local[0]:
                fraction = -old[0]/(local[0]-old[0])
                crossing = (old[1]+fraction*(local[1]-old[1])) * (1 if cfg.side == "left" else -1)
                if cfg.gate_min_m <= crossing <= cfg.gate_max_m:
                    self.gate_passed, self.phase = True, "exit"
                else:
                    self.finish("wrong_side_or_gate")
        in_exit = cfg.exit_min_m <= local[0] <= cfg.exit_max_m and cfg.gate_min_m <= side_y <= cfg.gate_max_m
        if self.phase == "exit" and local[0] < 0:
            self.finish("returned_before_buoy")
        if self.phase == "exit" and in_exit:
            self.pass_complete = True
            if cfg.kind == "pass_then_approach":
                self.phase = "approach_destination"
        if cfg.kind == "approach" or self.phase == "approach_destination":
            target = self.buoys[cfg.destination if self.phase == "approach_destination" else cfg.target]
            distance = float(np.linalg.norm(np.asarray(xy)-target))
            # Coordinates near a decimal boundary can differ by a few ulps.
            inside = cfg.stop_min_m-1e-9 <= distance <= cfg.stop_max_m+1e-9
        else:
            inside = self.phase == "exit" and in_exit
        if not self.done and inside and speed_mps <= cfg.stop_speed_mps:
            if self.hold_start is None:
                self.hold_start = timestamp_s
            if timestamp_s-self.hold_start >= cfg.hold_s-1e-9:
                self.finish("approach_stop_complete" if cfg.kind == "approach" else "pass_task_complete", True)
        else:
            self.hold_start = None
        self.previous, self.previous_time = np.asarray(xy, dtype=float), timestamp_s
        return self.result(timestamp_s, xy, speed_mps)

    def result(self, timestamp_s, xy, speed_mps):
        target = self.config.destination if self.phase == "approach_destination" else self.config.target
        return dict(done=self.done, success=self.success, termination_reason=self.reason, phase=self.phase,
                    distance_m=float(np.linalg.norm(np.asarray(xy)-self.buoys[target])), speed_mps=speed_mps,
                    hold_s=0. if self.hold_start is None else max(0., timestamp_s-self.hold_start),
                    gate_passed=self.gate_passed, pass_complete=self.pass_complete)


class ScriptedTeacher:
    """Geometry controller for pipeline validation; this is NOT a trained VLA."""
    def __init__(self, evaluator):
        self.task = evaluator
        self.waypoint = 0

    def command(self, pose, velocity):
        task, cfg = self.task, self.task.config
        position = np.array(pose[:2])
        sign = 1 if cfg.side == "left" else -1
        if task.phase in ("pass", "exit"):
            points = [task.world_point(-3, 3*sign), task.world_point(0.5, 3*sign), task.world_point(3, 3*sign)]
            while self.waypoint < 2 and np.linalg.norm(position-points[self.waypoint]) < .5:
                self.waypoint += 1
            target = points[self.waypoint]
            remaining = float(np.linalg.norm(target-position))
        else:
            goal_id = cfg.destination if task.phase == "approach_destination" else cfg.target
            buoy = np.array(task.buoys[goal_id])
            delta = position-buoy
            target = buoy + delta/max(np.linalg.norm(delta), 1e-9) * ((cfg.stop_min_m+cfg.stop_max_m)/2)
            remaining = float(np.linalg.norm(target-position))
        bearing = wrap(math.atan2(*(target-position)[::-1])-pose[2])
        desired = min(.35, remaining*.5) * max(0., math.cos(bearing))
        throttle = 60*(desired-velocity[0]) if abs(bearing) < 1 else -15*velocity[0]
        steering = max(-6., min(6., 5*bearing-2*velocity[2]))
        return limited_action([throttle-steering, throttle+steering])

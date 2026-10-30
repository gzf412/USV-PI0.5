"""Independent smoothness checks on recorded, unmodified simulator trajectories.

This profile is for the current calm-water simulation.  Its numerical limits
are not calibrated limits for a physical boat.  Task success, clearance,
causal alignment and image validity remain separate acceptance requirements.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


PROFILE = "smooth_trajectory_v1"
THRESHOLDS = {
    "nominal_step_s": .1,
    "step_tolerance_s": 1e-6,
    "reversal_persistence_s": .5,
    "direct_heading_excess_deg": 3.,
    "direct_reversal_hysteresis_deg": 3.,
    "direct_max_reversals": 0,
    "collinear_heading_budget_extra_deg": 15.,
    "collinear_reversal_hysteresis_deg": 3.,
    "collinear_max_reversals": 1,
    "around_heading_budget_extra_deg": 45.,
    "around_reversal_hysteresis_deg": 10.,
    "planned_turn_ignore_below_deg": 1.,
    "yaw_rate_abs_max_rad_s": .45,
    "yaw_acceleration_abs_max_rad_s2": .5,
    "speed_acceleration_abs_max_m_s2": .7,
    "speed_jerk_abs_max_m_s3": 2.5,
    "action_slew_abs_max_N_s": 10.1,
    "navigation_started_speed_m_s": .25,
    "navigation_distance_min_m": 3.5,
    "navigation_low_speed_m_s": .12,
    "navigation_low_speed_duration_max_s": 1.,
    "navigation_speed_valley_mean_window_s": .5,
    "navigation_speed_valley_peak_window_s": 3.5,
    "navigation_speed_valley_prominence_max_m_s": .12,
    "navigation_speed_valley_prominence_max_cruise_fraction": .30,
    "approach_minimum_forward_velocity_m_s": -.03,
    "approach_stop_min_distance_tolerance_m": .03,
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _finite_numbers(values, length, name):
    if not isinstance(values, (list, tuple)) or len(values) < length:
        raise ValueError(f"{name} needs at least {length} values")
    result = [float(x) for x in values[:length]]
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _description(metadata):
    """Read both recorder metadata and compact standalone fixture descriptions."""
    description = metadata.get("description") or {}
    scene = metadata.get("scene_plan") or metadata.get("scene") or {}
    route = metadata.get("route") or description.get("route") or scene.get("route") or {"kind": "direct"}
    category = metadata.get("category") or description.get("category") or scene.get("category")
    kind = route.get("kind", "direct")
    if category in ("around_left", "around_right") or kind == "cylinder_pass":
        group = "around"
    elif category in ("between", "shore") or kind in ("cylinder_gap", "bank_follow"):
        group = "near_collinear"
    elif category in (None, "approach_visible", "approach_offscreen") and kind == "direct":
        group = "direct"
    else:
        raise ValueError(f"unsupported smoothness category/kind: {category!r}/{kind!r}")
    return description, scene, route, category, kind, group


def _reference(metadata, description, scene, route, first_pose):
    initial = metadata.get("initial_pose") or description.get("initial_pose") or scene.get("initial_pose")
    if initial is None:
        start, initial_yaw = first_pose[:2], first_pose[5]
    else:
        initial = _finite_numbers(initial, 4, "initial_pose")
        start, initial_yaw = initial[:2], initial[3]
    target = scene.get("target_xy") or description.get("target_xy") or metadata.get("target_xy")
    task = metadata.get("task") or description.get("task") or scene.get("task") or {}
    if target is None:
        buoys = metadata.get("buoys") or description.get("buoys") or []
        selected = task.get("target", 0)
        buoy = next((b for b in buoys if b.get("slot") == selected and b.get("present", True)), None)
        if buoy is not None:
            target = [buoy["x"], buoy["y"]]
    if target is None:
        raise ValueError("target_xy or selected buoy is required for independent smoothness scoring")
    target = _finite_numbers(target, 2, "target_xy")
    waypoints = [_finite_numbers(p, 2, "waypoint") for p in route.get("waypoints", [])]
    last = waypoints[-1] if waypoints else start
    dx, dy = last[0]-target[0], last[1]-target[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("final route anchor coincides with target")
    radius = (float(task.get("stop_min_m", 1.9))+float(task.get("stop_max_m", 2.3)))/2
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("invalid stopping distance")
    stop = [target[0]+dx/length*radius, target[1]+dy/length*radius]
    vertices = [start, *waypoints, stop]
    headings = [initial_yaw]
    for a, b in zip(vertices, vertices[1:]):
        if math.dist(a, b) > 1e-8:
            headings.append(math.atan2(b[1]-a[1], b[0]-a[0]))
    signed_turns = [math.degrees(_wrap(b-a)) for a, b in zip(headings, headings[1:])]
    signs = [1 if x > 0 else -1 for x in signed_turns
             if abs(x) >= THRESHOLDS["planned_turn_ignore_below_deg"]]
    return {
        "target_xy": target,
        "stop_xy": stop,
        "planned_heading_variation_deg": sum(abs(x) for x in signed_turns),
        "planned_signed_turns_deg": signed_turns,
        "planned_sign_reversals": sum(a != b for a, b in zip(signs, signs[1:])),
        "planned_path_length_m": sum(math.dist(a, b) for a, b in zip(vertices, vertices[1:])),
        "cruise_speed_m_s": float(route.get("cruise_speed_mps", .5)),
        "task_kind": task.get("kind", "approach"),
        "stop_min_m": float(task.get("stop_min_m", 1.9)),
    }


def _heading_reversals(headings, times, threshold):
    """Count substantial alternating turns with hysteresis and time persistence.

    A reversal must remain beyond the previous extremum by ``threshold`` for
    0.5 s.  These checks do not filter, resample or change the recorded heading.
    """
    direction = 0
    low = high = headings[0]
    low_i = high_i = 0
    extrema = []
    candidate_time = None
    candidate_value = None
    candidate_i = None
    extreme = headings[0]
    extreme_i = 0
    for i, value in enumerate(headings):
        if not direction:
            if value < low:
                low, low_i = value, i
            if value > high:
                high, high_i = value, i
            if high-low >= threshold:
                direction = 1 if high_i > low_i else -1
                extrema = [(low_i, low)] if direction == 1 else [(high_i, high)]
                extreme, extreme_i = (high, high_i) if direction == 1 else (low, low_i)
            continue
        if direction*(value-extreme) >= 0:
            extreme, extreme_i = value, i
            candidate_time = None
        elif direction*(extreme-value) >= threshold:
            if candidate_time is None:
                candidate_time, candidate_value, candidate_i = times[i], value, i
            elif direction*(candidate_value-value) > 0:
                candidate_value, candidate_i = value, i
            if times[i]-candidate_time >= THRESHOLDS["reversal_persistence_s"]-1e-9:
                extrema.append((extreme_i, extreme))
                direction *= -1
                extreme, extreme_i = candidate_value, candidate_i
                candidate_time = None
        else:
            candidate_time = None
    if direction:
        extrema.append((extreme_i, extreme))
    return {
        "count": max(0, len(extrema)-2),
        "turn_leg_amplitudes_deg": [abs(b[1]-a[1]) for a, b in zip(extrema, extrema[1:])],
        "extrema": [{"index": i, "elapsed_s": times[i], "heading_unwrapped_deg": value}
                    for i, value in extrema],
    }


def _peak(values):
    return max((abs(x) for x in values), default=0.)


def _speed_valleys(speeds, times, eligible):
    # Five raw 10 Hz samples form a 0.5 s mean for this diagnostic only.  The
    # underlying dataset and acceleration/jerk measurements stay unmodified.
    half_width = 2
    means = []
    for i in range(half_width, len(speeds)-half_width):
        if all(eligible[i-half_width:i+half_width+1]):
            means.append((i, sum(speeds[i-half_width:i+half_width+1])/5))
    best = {"prominence_m_s": 0., "index": None, "elapsed_s": None}
    for at, (i, speed) in enumerate(means):
        before = [v for j, v in means[:at]
                  if times[i]-times[j] <= THRESHOLDS["navigation_speed_valley_peak_window_s"]+1e-9]
        after = [v for j, v in means[at+1:]
                 if times[j]-times[i] <= THRESHOLDS["navigation_speed_valley_peak_window_s"]+1e-9]
        if before and after:
            prominence = min(max(before), max(after))-speed
            if prominence > best["prominence_m_s"]:
                best = {"prominence_m_s": prominence, "index": i, "elapsed_s": times[i],
                        "valley_mean_speed_m_s": speed,
                        "before_peak_mean_speed_m_s": max(before),
                        "after_peak_mean_speed_m_s": max(after)}
    return best


def _inspect_valid_rows(metadata, rows, result):
    if not isinstance(metadata, dict):
        raise ValueError("episode.json must contain an object")
    if len(rows) < 3:
        raise ValueError("at least three recorded decisions are required")
    poses, velocities, actions, timestamps = [], [], [], []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {i} must be an object")
        poses.append(_finite_numbers(row.get("pose_xyzrpy"), 6, f"row {i} pose"))
        velocities.append(_finite_numbers(row.get("state"), 3, f"row {i} state u/v/r"))
        actions.append(_finite_numbers(row.get("action"), 2, f"row {i} action"))
        timestamp = row.get("decision_sim_time_ns")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            raise ValueError(f"row {i} has invalid decision_sim_time_ns")
        timestamps.append(timestamp)
    deltas = [(b-a)/1e9 for a, b in zip(timestamps, timestamps[1:])]
    if any(abs(dt-THRESHOLDS["nominal_step_s"]) > THRESHOLDS["step_tolerance_s"] for dt in deltas):
        raise ValueError("decision times must form a strictly continuous 10 Hz grid")
    times = [(t-timestamps[0])/1e9 for t in timestamps]
    description, scene, route, category, kind, group = _description(metadata)
    reference = _reference(metadata, description, scene, route, poses[0])
    result.update(category=category, route_kind=kind, trajectory_group=group, reference=reference)
    speeds = [math.hypot(v[0], v[1]) for v in velocities]
    rates = [v[2] for v in velocities]
    speed_accelerations = [(b-a)/dt for a, b, dt in zip(speeds, speeds[1:], deltas)]
    yaw_accelerations = [(b-a)/dt for a, b, dt in zip(rates, rates[1:], deltas)]
    # Acceleration samples live at interval centers, whose separation is the
    # average adjacent dt.  The grid check above makes that 0.1 s here.
    jerks = [(b-a)/((deltas[i]+deltas[i+1])/2)
             for i, (a, b) in enumerate(zip(speed_accelerations, speed_accelerations[1:]))]
    headings = [0.]
    for a, b in zip(poses, poses[1:]):
        headings.append(headings[-1]+math.degrees(_wrap(b[5]-a[5])))
    total_yaw = sum(abs(b-a) for a, b in zip(headings, headings[1:]))
    net_yaw = headings[-1]-headings[0]
    reversals3 = _heading_reversals(headings, times, 3.)
    reversals10 = _heading_reversals(headings, times, 10.)
    action_slews = []
    excluded_terminal_actions = 0
    for i, (a, b, dt) in enumerate(zip(actions, actions[1:], deltas), start=1):
        if i == len(rows)-1 and rows[i].get("done") is True:
            excluded_terminal_actions += 1
            continue
        action_slews.extend((b[j]-a[j])/dt for j in (0, 1))
    distances = [math.dist(p[:2], reference["target_xy"]) for p in poses]
    started, low_start, longest_low = False, None, 0.
    navigation_eligible = []
    for time, speed, distance in zip(times, speeds, distances):
        if speed >= THRESHOLDS["navigation_started_speed_m_s"]:
            started = True
        navigation_eligible.append(started and distance > THRESHOLDS["navigation_distance_min_m"])
        if (started and distance > THRESHOLDS["navigation_distance_min_m"]
                and speed < THRESHOLDS["navigation_low_speed_m_s"]):
            if low_start is None:
                low_start = time
            longest_low = max(longest_low, time-low_start)
        else:
            low_start = None
    valley = _speed_valleys(speeds, times, navigation_eligible)
    cruise = reference["cruise_speed_m_s"]
    if not math.isfinite(cruise) or cruise <= 0:
        raise ValueError("route cruise speed must be finite and positive")
    metrics = {
        "decision_count": len(rows),
        "duration_s": times[-1],
        "yaw_total_variation_deg": total_yaw,
        "net_yaw_change_deg": net_yaw,
        "yaw_variation_minus_abs_net_deg": max(0., total_yaw-abs(net_yaw)),
        "yaw_variation_minus_planned_deg": total_yaw-reference["planned_heading_variation_deg"],
        "yaw_reversals_3deg": reversals3,
        "yaw_reversals_10deg": reversals10,
        "yaw_rate_abs_peak_rad_s": _peak(rates),
        "yaw_acceleration_abs_peak_rad_s2": _peak(yaw_accelerations),
        "speed_acceleration_abs_peak_m_s2": _peak(speed_accelerations),
        "speed_jerk_abs_peak_m_s3": _peak(jerks),
        "nonterminal_action_slew_abs_peak_N_s": _peak(action_slews),
        "excluded_terminal_action_transitions": excluded_terminal_actions,
        "navigation_started": started,
        "navigation_low_speed_longest_s": longest_low,
        "navigation_speed_valley_prominence_mps": valley["prominence_m_s"],
        "navigation_speed_valley_prominence_cruise_fraction": valley["prominence_m_s"]/cruise,
        "navigation_speed_valley": valley,
        "navigation_speed_valley_is_diagnostic_only": group == "around",
        "path_length_m": sum(math.dist(a[:2], b[:2]) for a, b in zip(poses, poses[1:])),
        "forward_approach_checks_applied": reference["task_kind"] == "approach",
        "minimum_forward_velocity_m_s": min(v[0] for v in velocities),
        "minimum_target_distance_m": min(distances),
    }
    result["metrics"] = metrics
    issues = result["issues"]
    if group == "direct":
        if metrics["yaw_variation_minus_abs_net_deg"] > THRESHOLDS["direct_heading_excess_deg"]+1e-9:
            issues.append("direct: heading reverses by more than the allowed 3 degree excess budget")
        if reversals3["count"] > THRESHOLDS["direct_max_reversals"]:
            issues.append("direct: significant repeated heading reversal")
    elif group == "near_collinear":
        if metrics["yaw_variation_minus_planned_deg"] > THRESHOLDS["collinear_heading_budget_extra_deg"]+1e-9:
            issues.append("near_collinear: heading variation exceeds planned turns plus 15 degrees")
        if reversals3["count"] > THRESHOLDS["collinear_max_reversals"]:
            issues.append("near_collinear: more than one significant heading reversal")
    else:
        if metrics["yaw_variation_minus_planned_deg"] > THRESHOLDS["around_heading_budget_extra_deg"]+1e-9:
            issues.append("around: heading variation exceeds planned turns plus 45 degrees")
        if reversals10["count"] > max(1, reference["planned_sign_reversals"]):
            issues.append("around: unnecessary alternating large heading turns")
    for metric, limit, label in (
        ("yaw_rate_abs_peak_rad_s", "yaw_rate_abs_max_rad_s", "yaw rate"),
        ("yaw_acceleration_abs_peak_rad_s2", "yaw_acceleration_abs_max_rad_s2", "yaw acceleration"),
        ("speed_acceleration_abs_peak_m_s2", "speed_acceleration_abs_max_m_s2", "speed acceleration"),
        ("speed_jerk_abs_peak_m_s3", "speed_jerk_abs_max_m_s3", "speed jerk"),
        ("nonterminal_action_slew_abs_peak_N_s", "action_slew_abs_max_N_s", "nonterminal thrust slew"),
    ):
        if metrics[metric] > THRESHOLDS[limit]+1e-7:
            issues.append(f"{label} exceeds simulation profile limit ({metrics[metric]:.6g} > {THRESHOLDS[limit]:g})")
    if longest_low > THRESHOLDS["navigation_low_speed_duration_max_s"]+1e-9:
        issues.append("mid-route low speed persists for more than 1 second after movement started")
    if group != "around" and (
        valley["prominence_m_s"] > THRESHOLDS["navigation_speed_valley_prominence_max_m_s"]+1e-9
        or valley["prominence_m_s"]/cruise > THRESHOLDS["navigation_speed_valley_prominence_max_cruise_fraction"]+1e-9
    ):
        issues.append("mid-route speed valley followed by reacceleration exceeds simulation profile limit")
    if reference["task_kind"] == "approach":
        if metrics["minimum_forward_velocity_m_s"] < THRESHOLDS["approach_minimum_forward_velocity_m_s"]-1e-9:
            issues.append("forward approach: reverse motion exceeds 0.03 m/s settling tolerance")
        if min(distances) < reference["stop_min_m"]-THRESHOLDS["approach_stop_min_distance_tolerance_m"]-1e-9:
            issues.append("forward approach: target-distance overshoot exceeds 0.03 m below stop_min_m")


def inspect_trajectory_quality(episode):
    """Return deterministic JSON proof; never edit or smooth the source data."""
    episode = Path(episode).resolve()
    result = {
        "schema_version": 1,
        "profile": PROFILE,
        "episode": str(episode),
        "inspector_sha256": _sha256(Path(__file__)),
        "thresholds_status": "current_calm_water_simulation_not_hardware_calibrated",
        "thresholds": dict(THRESHOLDS),
        "passed": False,
        "issues": [],
    }
    try:
        metadata_path, steps_path = episode/"episode.json", episode/"steps.jsonl"
        result["episode_metadata_sha256"] = _sha256(metadata_path)
        result["steps_sha256"] = _sha256(steps_path)
        metadata = json.loads(metadata_path.read_text())
        rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
        _inspect_valid_rows(metadata, rows, result)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        result["issues"].append(f"invalid trajectory input: {error}")
    result["passed"] = not result["issues"]
    return result


def require_trajectory_quality(episode):
    result = inspect_trajectory_quality(episode)
    if not result["passed"]:
        raise ValueError("Trajectory smoothness failed: "+"; ".join(result["issues"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args(argv)
    result = inspect_trajectory_quality(args.episode)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

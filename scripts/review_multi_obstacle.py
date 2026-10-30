#!/usr/bin/env python3
"""Independently review the measured crossings of a multi-cylinder pilot.

Uses the recorded boat positions and headings, not teacher waypoint indices or
saved success flags. Clearance is a conservative swept-circle calculation,
not exact hull-to-object geometry. Outputs a machine-readable JSON report.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np

from usv_vla.route import VESSEL_RADIUS_M, _gate_coordinates, _segment_distance

MAX_CROSSING_ANGLE_DEG = 10.
EXIT_CORRIDOR_TOLERANCE_M = .1  # Matches RouteTaskEvaluator's measured exit strip.


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _xyz_yaw(row):
    pose = row.get("pose_xyzrpy")
    if not isinstance(pose, list) or len(pose) != 6:
        raise ValueError("Each recorded pose must contain six numbers")
    values = [_finite(value, "pose") for value in pose]
    return np.asarray(values[:2]), values[5]


def review_episode(episode):
    episode = Path(episode).resolve()
    metadata_path, rows_path = episode / "episode.json", episode / "steps.jsonl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError("At least two observations are required")
    route = metadata["route"]
    gates = route["gates"]
    models = route["cylinder_models"]
    if route.get("kind") != "cylinder_pass" or len(gates) not in (2, 3) or len(models) != len(gates) or len(set(models)) != len(models):
        raise ValueError("Expected two or three distinct cylinder gates and models")
    obstacles = metadata["obstacles"]
    by_model = {obj["model"]: obj for obj in obstacles}
    if len(by_model) != len(obstacles) or any(model not in by_model for model in models):
        raise ValueError("Obstacle models must be unique and contain the route cylinders")
    poses = [_xyz_yaw(row) for row in rows]
    times = []
    for index, row in enumerate(rows):
        stamp = row.get("decision_sim_time_ns")
        if row.get("index") != index or not isinstance(stamp, int) or isinstance(stamp, bool) or stamp < 0:
            raise ValueError("Invalid row index or decision timestamp")
        if times and stamp <= times[-1]:
            raise ValueError("Decision timestamps must increase strictly")
        times.append(stamp)
    prompts = [row.get("prompt") for row in rows]
    instruction = route.get("instruction")
    issues = []
    fixed = (route.get("prompt_include_phase") is False and isinstance(instruction, str)
             and bool(instruction.strip()) and all(prompt == instruction for prompt in prompts))
    if not fixed:
        issues.append("Policy prompt is not the fixed complete instruction")
    minimum = _finite(metadata["task"]["minimum_clearance_m"], "minimum clearance")
    if minimum <= 0:
        raise ValueError("Minimum clearance must be positive")

    # Check the named near/far sequence against the physical start position,
    # so a different gate order cannot silently pass as the intended instruction.
    cylinders = [by_model[name] for name in models]
    cylinder_centers = []
    for obj in cylinders:
        center = np.asarray([_finite(obj["x"], "obstacle x"), _finite(obj["y"], "obstacle y")])
        radius = _finite(obj["radius_m"], "obstacle radius")
        if radius <= 0:
            raise ValueError("Obstacle radius must be positive")
        cylinder_centers.append(center)
    start_distances = [float(np.linalg.norm(center - poses[0][0])) for center in cylinder_centers]
    if any(a >= b for a, b in zip(start_distances, start_distances[1:])):
        issues.append("Named cylinder order does not match increasing initial distance")
    first_side = route.get("side")
    if first_side not in ("left", "right"):
        raise ValueError("Invalid first-cylinder side")

    obstacle_reports = []
    for obj in obstacles:
        center = np.asarray([_finite(obj["x"], "obstacle x"), _finite(obj["y"], "obstacle y")])
        radius = _finite(obj["radius_m"], "obstacle radius")
        if radius <= 0:
            raise ValueError("Obstacle radius must be positive")
        clearances = [_segment_distance(a[0], b[0], center) - VESSEL_RADIUS_M - radius
                      for a, b in zip(poses, poses[1:])]
        index = int(np.argmin(clearances))
        clearance = float(clearances[index])
        obstacle_reports.append(dict(model=obj["model"], radius_m=radius,
                                     minimum_swept_clearance_m=clearance,
                                     minimum_segment_start_index=index,
                                     minimum_segment_end_index=index + 1,
                                     passed=clearance >= minimum - 1e-9))
        if clearance < minimum - 1e-9:
            issues.append(f"Swept clearance below minimum for {obj['model']}")

    crossings, completed = [], []
    for idx, (gate, center, model) in enumerate(zip(gates, cylinder_centers, models)):
        gcenter = np.asarray([_finite(v, "gate center") for v in gate["center"]])
        if gcenter.shape != (2,) or np.linalg.norm(gcenter - center) > 1e-6:
            raise ValueError(f"Gate {idx+1} does not correspond to {model}")
        yaw = _finite(gate["yaw_rad"], "gate yaw")
        lower = _finite(gate["left_min_m"], "gate left minimum")
        upper = _finite(gate["left_max_m"], "gate left maximum")
        exit_forward = _finite(gate["exit_forward_m"], "gate exit")
        if lower >= upper or exit_forward <= 0:
            raise ValueError(f"Invalid gate {idx+1} bounds")
        side = first_side if idx % 2 == 0 else ("right" if first_side == "left" else "left")
        if (side == "left" and lower <= 0) or (side == "right" and upper >= 0):
            issues.append(f"Gate {idx+1} bounds do not describe the instructed {side} side")
        coords = [_gate_coordinates(gate, point) for point, _ in poses]
        if coords[0][0] >= 0:
            issues.append(f"Recording starts at or beyond gate {idx+1}")
        forward_crossings = []
        for i, ((old_x, old_y), (new_x, new_y)) in enumerate(zip(coords, coords[1:])):
            if old_x < 0 <= new_x:
                fraction = -old_x / (new_x-old_x)
                lateral = old_y + fraction * (new_y-old_y)
                old_yaw, new_yaw = poses[i][1], poses[i+1][1]
                dyaw = math.atan2(math.sin(new_yaw-old_yaw), math.cos(new_yaw-old_yaw))
                heading = old_yaw + fraction*dyaw
                heading_error = math.degrees(math.atan2(math.sin(heading-yaw), math.cos(heading-yaw)))
                crossing_ns = times[i] + fraction * (times[i+1]-times[i])
                forward_crossings.append(dict(segment_start_index=i, fraction=fraction,
                                              elapsed_s=(crossing_ns-times[0])/1e9,
                                              lateral_m=lateral, heading_relative_to_gate_deg=heading_error,
                                              time_ns=crossing_ns))
        if len(forward_crossings) != 1:
            issues.append(f"Gate {idx+1} must have exactly one forward crossing; got {len(forward_crossings)}")
        crossing = forward_crossings[0] if forward_crossings else None
        exit_ns = None
        if crossing is not None:
            if not lower <= crossing["lateral_m"] <= upper:
                issues.append(f"Gate {idx+1} crossed outside its instructed side")
            if abs(crossing["heading_relative_to_gate_deg"]) > MAX_CROSSING_ANGLE_DEG + 1e-9:
                issues.append(f"Gate {idx+1} heading exceeds {MAX_CROSSING_ANGLE_DEG:g} degrees")
            start_index = crossing["segment_start_index"]
            for i in range(start_index, len(coords)-1):
                old_x, old_y = coords[i]
                new_x, new_y = coords[i+1]
                begin = crossing["fraction"] if i == start_index else 0.
                if i > start_index and old_x < -.05:
                    issues.append(f"Returned behind gate {idx+1} before its exit")
                    break
                end = 1.
                if old_x < exit_forward <= new_x:
                    end = (exit_forward-old_x)/(new_x-old_x)
                    exit_ns = times[i] + end * (times[i+1]-times[i])
                # Both endpoints of a linear segment clipped to the exit strip
                # are enough to prove its lateral position stays in bounds.
                for fraction in (begin, end):
                    lateral = old_y + fraction*(new_y-old_y)
                    if not lower-EXIT_CORRIDOR_TOLERANCE_M-1e-9 <= lateral <= upper+EXIT_CORRIDOR_TOLERANCE_M+1e-9:
                        issues.append(f"Gate {idx+1} left its protected exit corridor")
                        break
                if exit_ns is not None or issues and issues[-1].startswith(f"Gate {idx+1} left its"):
                    break
            if exit_ns is None:
                issues.append(f"Gate {idx+1} never reached its protected exit")
        crossings.append(dict(gate=idx+1, cylinder_model=model, instructed_side=side,
                              initial_cylinder_distance_m=start_distances[idx],
                              lateral_bounds_m=[lower, upper], crossing=crossing,
                              exit_elapsed_s=None if exit_ns is None else (exit_ns-times[0])/1e9))
        completed.append(exit_ns)
    for idx, (previous, current) in enumerate(zip(crossings, crossings[1:])):
        if previous["crossing"] is not None and current["crossing"] is not None:
            first, second = previous["crossing"]["time_ns"], current["crossing"]["time_ns"]
            if first >= second:
                issues.append(f"Cylinder gates {idx+1} and {idx+2} crossed in the wrong order")
            if completed[idx] is None or completed[idx] >= second:
                issues.append(f"Cylinder {idx+2} crossed before cylinder {idx+1}'s protected exit")

    return dict(schema_version=1, episode=str(episode), passed=not issues, issues=issues,
                inputs_sha256={"episode.json": _hash(metadata_path), "steps.jsonl": _hash(rows_path)},
                method="physical segment crossings with interpolated angle and conservative swept-circle clearance",
                thresholds=dict(max_crossing_angle_deg=MAX_CROSSING_ANGLE_DEG,
                                minimum_swept_clearance_m=minimum, vessel_radius_m=VESSEL_RADIUS_M,
                                exit_corridor_tolerance_m=EXIT_CORRIDOR_TOLERANCE_M),
                sample_count=len(rows), duration_s=(times[-1]-times[0])/1e9,
                prompt_fixed=fixed, gates=crossings, obstacle_clearances=obstacle_reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, help="Write a new JSON report instead of stdout")
    args = parser.parse_args()
    try:
        report = review_episode(args.episode)
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as error:
        print(f"Cannot review episode: {error}", file=sys.stderr)
        return 2
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(result)
        else:
            print(result, end="")
    except OSError as error:
        print(f"Cannot write report: {error}", file=sys.stderr)
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

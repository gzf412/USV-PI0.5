"""Early-aligned cylinder demonstrations and their independent raw-data check."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from usv_vla.core import wrap
from usv_vla.route import _gate_coordinates

PROFILE = "early_alignment_v1"
MAX_ABEAM_HEADING_DEG = 10.
MAX_INITIAL_OUTWARD_DEG = 35.


def apply_standard(scene, *, control_forward_m=-3., lateral_m=2.1,
                   maneuver_speed_mps=.5435685442559686, start_backoff_m=0.):
    """Copy a physical scene, increasing cruise and shaping only its guidance."""
    result = deepcopy(scene)
    route = result["route"]
    old_speed = float(route["cruise_speed_mps"])
    route["cruise_speed_mps"] = min(.75, 1.5*old_speed)
    description = {"profile": PROFILE, "speed_multiplier": 1.5,
                   "original_cruise_mps": old_speed, "maximum_cruise_mps": .75}
    if route["kind"] == "cylinder_pass":
        if route["side"] not in ("left", "right") or len(route["gates"]) != 1:
            raise ValueError("Early alignment requires one left/right cylinder gate")
        gate = route["gates"][0]
        if not math.isfinite(start_backoff_m) or not 0 <= start_backoff_m <= 3.:
            raise ValueError("Invalid initial approach backoff")
        result["initial_pose"][0] -= start_backoff_m*math.cos(gate["yaw_rad"])
        result["initial_pose"][1] -= start_backoff_m*math.sin(gate["yaw_rad"])
        sign = 1 if route["side"] == "left" else -1
        if not math.isfinite(control_forward_m) or control_forward_m >= 0:
            raise ValueError("Curve control must precede the gate")
        lateral = sign*float(lateral_m)
        if not gate["left_min_m"] <= lateral <= gate["left_max_m"]:
            raise ValueError("Curve endpoint is outside the physical gate")
        start = np.asarray(_gate_coordinates(gate, result["initial_pose"][:2]))
        control = np.array([control_forward_m, lateral])
        end = np.array([0., lateral])
        yaw = gate["yaw_rad"]
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                             [math.sin(yaw), math.cos(yaw)]])
        points = [(1-t)**2*start+2*(1-t)*t*control+t*t*end
                  for t in np.linspace(0., 1., 21)[1:]]
        points.append(np.array([gate["exit_forward_m"]+.5, lateral]))
        route["waypoints"] = [(np.asarray(gate["center"])+rotation@p).tolist() for p in points]
        description.update(curve="quadratic_bezier_20_segments", control_forward_m=control_forward_m,
                           lateral_m=float(lateral_m), max_abeam_heading_deg=MAX_ABEAM_HEADING_DEG,
                           max_initial_outward_deg=MAX_INITIAL_OUTWARD_DEG, start_backoff_m=start_backoff_m)
        # Keep the turn at the speed actually demonstrated in Gazebo; the
        # increased straight cruise resumes after independently scoring exit.
        if not math.isfinite(maneuver_speed_mps) or not .15 <= maneuver_speed_mps <= .5435685442559686+1e-8:
            raise ValueError("Maneuver speed exceeds the validated turn speed")
        route["maneuver_speed_mps"] = min(route["cruise_speed_mps"], maneuver_speed_mps, .5435685442559686)
        description["maneuver_speed_mps"] = route["maneuver_speed_mps"]
    result["demonstration_standard"] = description
    return result


def inspect_standard(episode):
    """Interpolate the physical crossing; never infer alignment from waypoints."""
    episode = Path(episode)
    metadata = json.loads((episode/"episode.json").read_text())
    scene = metadata["scene_plan"]
    if scene.get("demonstration_standard", {}).get("profile") != PROFILE:
        raise ValueError("Episode does not declare the current demonstration standard")
    route = scene["route"]
    if not .15 <= route["cruise_speed_mps"] <= .75:
        raise ValueError("Cruise exceeds the current limit")
    result = {"profile": PROFILE, "passed": True, "issues": [],
              "steps_sha256": hashlib.sha256((episode/"steps.jsonl").read_bytes()).hexdigest(),
              "metadata_sha256": hashlib.sha256((episode/"episode.json").read_bytes()).hexdigest(),
              "cruise_speed_mps": route["cruise_speed_mps"], "crossing": None}
    if route["kind"] != "cylinder_pass":
        return result
    rows = [json.loads(line) for line in (episode/"steps.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError("Too few observations for crossing alignment")
    gate = route["gates"][0]
    if not np.isfinite([*gate["center"], gate["yaw_rad"]]).all():
        raise ValueError("Nonfinite crossing gate")
    if any(len(row.get("pose_xyzrpy", [])) != 6 or not np.isfinite(row["pose_xyzrpy"]).all() for row in rows):
        raise ValueError("Nonfinite or invalid crossing observation")
    sign = 1 if route["side"] == "left" else -1
    start_yaw = rows[0]["pose_xyzrpy"][5]
    outward = [0.]
    for index, (a, b) in enumerate(zip(rows, rows[1:])):
        outward.append(sign*math.degrees(wrap(a["pose_xyzrpy"][5]-start_yaw)))
        af, al = _gate_coordinates(gate, a["pose_xyzrpy"][:2])
        bf, bl = _gate_coordinates(gate, b["pose_xyzrpy"][:2])
        if af < 0 <= bf:
            fraction = -af/(bf-af)
            yaw = a["pose_xyzrpy"][5]+fraction*wrap(b["pose_xyzrpy"][5]-a["pose_xyzrpy"][5])
            outward.append(sign*math.degrees(wrap(yaw-start_yaw)))
            heading = math.degrees(wrap(yaw-gate["yaw_rad"]))
            result["crossing"] = {"before_index": index, "fraction": fraction,
                                  "heading_relative_route_deg": heading,
                                  "left_m": al+fraction*(bl-al)}
            break
    if result["crossing"] is None:
        raise ValueError("No physical cylinder crossing")
    result["initial_outward_peak_deg"] = max(outward)
    result["thresholds"] = {"max_abeam_heading_deg": MAX_ABEAM_HEADING_DEG,
                            "max_initial_outward_deg": MAX_INITIAL_OUTWARD_DEG}
    if abs(result["crossing"]["heading_relative_route_deg"]) > MAX_ABEAM_HEADING_DEG:
        result["issues"].append("Cylinder abeam heading exceeds 10 degrees")
    if result["initial_outward_peak_deg"] > MAX_INITIAL_OUTWARD_DEG:
        result["issues"].append("Initial outward turn exceeds 35 degrees")
    result["passed"] = not result["issues"]
    return result

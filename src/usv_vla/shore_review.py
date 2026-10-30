"""Measure the recorded boat trajectory against the actual shore collision mesh.

This supplements, rather than replaces, task-success and trajectory-quality
checks. Distances are horizontal distances from the recorded boat origin to
the projected, above-water collision mesh, not lidar readings.
"""

import hashlib
import json
import math
from pathlib import Path

from usv_vla.scene_plan import shore_clearance

PROFILE = "long_shore_v1"
MIN_INITIAL_TARGET_M = 25.0
MAX_INITIAL_TARGET_M = 30.0
MIN_CENTER_SHORE_M = 2.5
MAX_CENTER_SHORE_M = 4.7
BOAT_BOUNDING_RADIUS_M = 0.85


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _position(value, label):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{label} must have at least two coordinates")
    xy = [float(value[0]), float(value[1])]
    if not all(math.isfinite(v) for v in xy):
        raise ValueError(f"{label} must be finite")
    return xy


def require_long_shore(episode: Path) -> dict:
    """Return measurement proof for a successful long shore run, or raise.

    Every stored decision position is assessed using the real terrain mesh.
    The continuous-segment hull lower bound uses the 1-Lipschitz property of
    distance to a set, and bounds movement between successive observations by
    a straight line: min(sampled distance) - max segment/2 - bounding radius.
    It is a geometric lower bound for the linear interpolation, not a claim
    about unobserved movement between samples or exact hull-to-bank geometry.
    """
    episode = Path(episode).resolve()
    metadata_path, summary_path, steps_path = (episode / name for name in
                                             ("episode.json", "summary.json", "steps.jsonl"))
    try:
        metadata = json.loads(metadata_path.read_text())
        summary = json.loads(summary_path.read_text())
        rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
        scene = metadata.get("scene_plan") or metadata.get("scene") or {}
        route = scene.get("route") or metadata.get("route") or {}
        if scene.get("category") != "shore" or route.get("kind") != "bank_follow":
            raise ValueError("episode is not a bank-follow shore scenario")
        if not all(summary.get(key) is True for key in ("done", "success", "route_complete")):
            raise ValueError("shore task did not complete successfully")
        if len(rows) < 3 or not isinstance(rows[-1], dict) or not all(
                rows[-1].get(key) is True for key in ("done", "success", "route_complete")):
            raise ValueError("final recorded decision does not certify successful completion")
        positions = [_position(row.get("pose_xyzrpy"), f"row {i} pose") for i, row in enumerate(rows)]
        target = _position(scene.get("target_xy"), "target_xy")
        initial_distance = math.dist(positions[0], target)
        if not MIN_INITIAL_TARGET_M <= initial_distance <= MAX_INITIAL_TARGET_M:
            raise ValueError(f"initial target distance {initial_distance:.3f} m is outside 25–30 m")
        timestamps = [row.get("decision_sim_time_ns") for row in rows]
        if any(not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in timestamps):
            raise ValueError("invalid simulation timestamp")
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("simulation timestamps must be strictly increasing")
        segments = [math.dist(a, b) for a, b in zip(positions, positions[1:])]
        distances = [float(shore_clearance(point)) for point in positions]
        if not all(math.isfinite(d) and d >= 0 for d in distances):
            raise ValueError("shore mesh returned invalid clearance")
        minimum, maximum = min(distances), max(distances)
        if minimum < MIN_CENTER_SHORE_M or maximum > MAX_CENTER_SHORE_M:
            raise ValueError(f"boat-center shore distance {minimum:.3f}–{maximum:.3f} m is outside "
                             f"{MIN_CENTER_SHORE_M}–{MAX_CENTER_SHORE_M} m")
        max_step = max(segments)
        hull_bound = minimum - max_step / 2.0 - BOAT_BOUNDING_RADIUS_M
        if hull_bound <= 0:
            raise ValueError("insufficient conservative hull clearance along interpolated path")
        return {
            "schema_version": 1,
            "profile": PROFILE,
            "passed": True,
            "episode": str(episode),
            "inspector_sha256": _digest(Path(__file__)),
            "episode_metadata_sha256": _digest(metadata_path),
            "summary_sha256": _digest(summary_path),
            "steps_sha256": _digest(steps_path),
            "shore_distance_source": "projected above-water sydney_regatta_shore.dae collision mesh",
            "distance_reference": "recorded simulator boat origin, horizontal XY",
            "initial_target_distance_m": initial_distance,
            "final_target_distance_m": math.dist(positions[-1], target),
            "recorded_path_length_m": sum(segments),
            "duration_sim_s": (timestamps[-1] - timestamps[0]) / 1e9,
            "decision_count": len(rows),
            "center_shore_distance_min_m": minimum,
            "center_shore_distance_max_m": maximum,
            "center_shore_distance_mean_m": sum(distances) / len(distances),
            "max_recorded_segment_m": max_step,
            "boat_bounding_radius_m": BOAT_BOUNDING_RADIUS_M,
            "hull_shore_clearance_lower_bound_m": hull_bound,
            "hull_bound_assumptions": "horizontal 0.85 m bounding circle; linear interpolation between poses",
            "accepted_center_shore_distance_m": [MIN_CENTER_SHORE_M, MAX_CENTER_SHORE_M],
        }
    except (OSError, KeyError, TypeError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid shore episode: {exc}") from exc

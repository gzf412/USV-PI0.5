"""Near-bank green-buoy demonstrations with a real obstacle and independent review.

The planned path is privileged teacher data.  The model receives only the fixed
instruction, onboard observations, and buoy map, never the planned side or
the simulator's route progress.  This module does not run a simulator.
"""

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from usv_vla.core import BUOY_SLOTS, TaskConfig, body_xy, wrap
from usv_vla.route import RouteTaskEvaluator, VESSEL_RADIUS_M, _segment_distance
from usv_vla.scene_plan import project_object, shore_clearance, xy_from
from usv_vla.world import ORIGIN

PROFILE = "green_shore_obstacle_v1"
P0 = np.array([-563.768125, 160.538125])
P1 = np.array([-525.6186875, 154.73040625])
MIN_CENTER_SHORE_M = 2.5
MAX_BASELINE_SHORE_M = 4.7
MAX_DETOUR_SHORE_M = 7.2
MAX_CROSSING_HEADING_DEG = 10.


def _smoothstep(t):
    t = max(0., min(1., t))
    return t*t*(3. - 2.*t)


def _model_for_slot(slot):
    return f"v2_buoy_{slot+1}"


def _radius(slot):
    return .75 if BUOY_SLOTS[slot][1] == "round" else .434


def _dense_segments(points, spacing=.25):
    for a, b in zip(points, points[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        count = max(1, math.ceil(float(np.linalg.norm(b-a))/spacing))
        for i in range(count):
            yield a+(b-a)*(i/count)
    yield np.asarray(points[-1], dtype=float)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _obstacles(scene):
    return ([dict(model=_model_for_slot(b["slot"]), x=b["x"], y=b["y"],
                  radius_m=_radius(b["slot"])) for b in scene["buoys"]]
            + [dict(model=f'usv_cylinder_{c["id"]}', x=c["x"], y=c["y"],
                    radius_m=c["radius_m"]) for c in scene["cylinders"]])


def generate_green_shore(index, seed):
    """Generate a reproducible scene and geometrically preflight its teacher.

    Across groups of four, both shore directions alternate between a cylinder
    and a uniquely identified non-target buoy in the direct path. Each plan has
    1–4 cylinders and at least one non-target buoy.  Only the first, relevant
    obstacle is required in the initial front-camera field of view.
    """
    rng = random.Random(int(seed))
    index = int(index)
    unit = (P1-P0)/np.linalg.norm(P1-P0)
    yaw0 = math.atan2(unit[1], unit[0])
    length = float(np.linalg.norm(P1-P0))
    count = 1 + index % 4
    buoy_obstacle = bool((index // 2) % 2)
    errors = []
    for attempt in range(100):
        distance = rng.uniform(25.2, 28.8)
        along = rng.uniform(1., length-distance-1.)
        offset = rng.uniform(3.6, 3.9)
        start = P0 + unit*along + np.array([-unit[1], unit[0]])*offset
        target = start+unit*distance
        yaw = yaw0
        if index % 2:
            start, target = target.copy(), start.copy()
            yaw += math.pi
        # In route coordinates +left is open water for the first direction.
        sign = -1 if index % 2 else 1
        pose = [float(start[0]), float(start[1]), ORIGIN[2], yaw+rng.uniform(-.025, .025)]
        gate_x = rng.uniform(10.7, 11.6)
        move_begin, move_end = gate_x-7.2, gate_x-1.7
        hold_end, return_end = gate_x+2.5, gate_x+9.
        lateral = rng.uniform(2.32, 2.45)
        if return_end >= distance-3.5:
            continue

        def lateral_at(x):
            if x <= move_begin:
                return 0.
            if x < move_end:
                return sign*lateral*_smoothstep((x-move_begin)/(move_end-move_begin))
            if x <= hold_end:
                return sign*lateral
            if x < return_end:
                return sign*lateral*(1.-_smoothstep((x-hold_end)/(return_end-hold_end)))
            return 0.

        # Dense interpolation prevents path-following from corner-cutting the
        # carefully checked smooth detour. Include exact gate and boundaries.
        last = distance-2.4
        xs = sorted({last*i/math.ceil(last/.3) for i in range(1, math.ceil(last/.3)+1)}
                    | {move_begin, move_end, gate_x, hold_end, return_end, last})
        waypoints = [xy_from(start, yaw, x, lateral_at(x)) for x in xs]
        obstacle_xy = xy_from(start, yaw, gate_x, 0.)
        obstacle_slot = rng.choice([0, 1, 2, 3, 5, 6, 7])
        buoys = [dict(slot=4, x=float(target[0]), y=float(target[1]))]
        cylinders = []
        if buoy_obstacle:
            buoys.append(dict(slot=obstacle_slot, x=obstacle_xy[0], y=obstacle_xy[1]))
            obstacle_model, obstacle_radius = _model_for_slot(obstacle_slot), _radius(obstacle_slot)
        else:
            cylinders.append(dict(id=1, x=obstacle_xy[0], y=obstacle_xy[1], radius_m=.405))
            obstacle_model, obstacle_radius = "usv_cylinder_1", .405
        # A distractor is placed on the water side, clear of the whole route.
        path = [pose[:2], *waypoints, xy_from(start, yaw, distance-2.1, 0.)]
        occupied = [dict(x=b["x"], y=b["y"], radius_m=_radius(b["slot"])) for b in buoys]
        occupied += cylinders.copy()

        def find_distractor(radius):
            for _ in range(180):
                x = rng.uniform(5., min(23., distance-4.))
                y = sign*rng.uniform(5.4, 7.8)
                point = xy_from(start, yaw, x, y)
                if math.dist(point, pose[:2]) > 30.5:
                    continue
                if any(math.dist(point, [o["x"], o["y"]]) < radius+o["radius_m"]+1.1 for o in occupied):
                    continue
                if min(_segment_distance(np.asarray(a), np.asarray(b), np.asarray(point))
                       for a, b in zip(path, path[1:])) < radius+VESSEL_RADIUS_M+.45:
                    continue
                if shore_clearance(point) < MIN_CENTER_SHORE_M:
                    continue
                if any(math.dist(point, p) > 31.5 for p in path):
                    continue
                result = dict(x=point[0], y=point[1], radius_m=radius)
                occupied.append(result)
                return result
            raise ValueError("unable to place an offshore distractor")

        try:
            while len(cylinders) < count:
                other = find_distractor(.405)
                cylinders.append(dict(id=len(cylinders)+1, **other))
            remaining = [slot for slot in range(8) if slot != 4 and all(b["slot"] != slot for b in buoys)]
            rng.shuffle(remaining)
            wanted = 1+(index % 3)
            while len(buoys)-1 < wanted:
                slot = remaining.pop()
                other = find_distractor(_radius(slot))
                buoys.append(dict(slot=slot, x=other["x"], y=other["y"]))

            side = "left" if sign > 0 else "right"
            bank_side = "right" if sign > 0 else "left"
            low, high = ((2.0, 3.3) if sign > 0 else (-3.3, -2.0))
            gate = dict(center=obstacle_xy, yaw_rad=yaw, left_min_m=low,
                        left_max_m=high, exit_forward_m=2.)
            obstacle_desc = ("the orange cylinder" if not buoy_obstacle else
                             "the " + " ".join(BUOY_SLOTS[obstacle_slot]) + " buoy")
            instructions = [
                f"Follow the riverbank on your {bank_side} to the green round buoy and stop.",
                f"Follow the riverbank on your {bank_side}; safely avoid obstacles, then approach the green round buoy and stop 1.9 to 2.3 meters away.",
                f"Follow the riverbank on your {bank_side}, safely pass {obstacle_desc}, return to the bank, and stop near the green round buoy.",
            ]
            route = dict(kind="cylinder_pass", side=side, reference_yaw_rad=yaw,
                         cruise_speed_mps=.64, maneuver_speed_mps=.54,
                         waypoint_radius_m=.85, waypoints=waypoints, gates=[gate],
                         cylinder_models=[] if buoy_obstacle else [obstacle_model],
                         required_obstacle_models=[obstacle_model], prompt_include_phase=False,
                         instruction=instructions[index % len(instructions)])
            scene = dict(schema_version=1, dataset_type="green_multi_v1", category="shore",
                         index=index, seed=int(seed), generation_attempt=attempt,
                         initial_pose=pose, target_slot=4, target_xy=target.tolist(),
                         buoys=buoys, cylinders=cylinders, route=route,
                         shore_obstacle=dict(profile=PROFILE, baseline_origin=start.tolist(),
                                             yaw_rad=yaw, bank_side=bank_side,
                                             obstacle_model=obstacle_model,
                                             detour_window_m=[move_begin, return_end],
                                             gate_forward_m=gate_x, target_distance_m=distance,
                                             nominal_chord_offset_m=offset))
            visible = project_object(pose, *obstacle_xy, obstacle_radius,
                                     z_min=0. if buoy_obstacle else -.23,
                                     z_max=(1.15 if buoy_obstacle and BUOY_SLOTS[obstacle_slot][1] == 'pointed'
                                            else 1.1 if buoy_obstacle else .67))
            if not visible["fully_visible"]:
                raise ValueError("required obstacle not fully visible initially")
            if any(shore_clearance(point) < MIN_CENTER_SHORE_M for point in _dense_segments(path, .3)):
                raise ValueError("planned path too close to physical shore")
            all_obstacles = _obstacles(scene)
            RouteTaskEvaluator(TaskConfig(target=4), {b["slot"]: (b["x"], b["y"]) for b in buoys},
                               pose[:2], pose[3], route=route, obstacles=all_obstacles)
            return scene
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(f"Cannot generate green shore plan after 100 attempts: {errors[-1:]}")


def review_green_shore(episode):
    """Independently audit a recorded detour, return to shore, and stop.

    Uses physical pose samples, never teacher waypoint index or success alone.
    The vessel is bounded by an 0.85m horizontal circle; coast distances are
    to the actual above-water collision mesh. Segment bounds assume linear
    interpolation between observed positions.
    """
    episode = Path(episode).resolve()
    metadata_path, summary_path, steps_path = (episode / name for name in
                                               ("episode.json", "summary.json", "steps.jsonl"))
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
    scene = metadata["scene_plan"]
    route, info = scene["route"], scene["shore_obstacle"]
    target_slot = {'green_multi_v1': 4, 'blue_multi_v1': 6}.get(scene.get('dataset_type'))
    if (target_slot is None or scene.get("target_slot") != target_slot
            or info.get("profile") != PROFILE or route.get("kind") != "cylinder_pass"):
        raise ValueError("not a green shore obstacle episode")
    if len(rows) < 3:
        raise ValueError("not enough recorded poses")
    poses = [row.get("pose_xyzrpy") for row in rows]
    if any(not isinstance(p, list) or len(p) != 6 or not np.isfinite(p).all() for p in poses):
        raise ValueError("invalid recorded pose")
    points = [np.asarray(p[:2], dtype=float) for p in poses]
    times = [row.get("decision_sim_time_ns") for row in rows]
    if any(not isinstance(t, int) or isinstance(t, bool) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("invalid decision timestamps")
    local = [body_xy(*(p-np.asarray(info["baseline_origin"])), info["yaw_rad"]) for p in points]
    distances = [shore_clearance(p) for p in points]
    steps = [float(np.linalg.norm(b-a)) for a, b in zip(points, points[1:])]
    window_start, window_end = info["detour_window_m"]
    before = [i for i, (forward, _) in enumerate(local) if .5 <= forward <= window_start-.3]
    after = [i for i, (forward, _) in enumerate(local) if window_end+.3 <= forward <= info["target_distance_m"]-2.2]
    issues = []
    if not summary.get("success") or not summary.get("route_complete") or not rows[-1].get("success"):
        issues.append("recorded task did not complete successfully")
    if route.get("prompt_include_phase") is not False or any(row.get("prompt") != route["instruction"] for row in rows):
        issues.append("prompt is not a fixed instruction")
    if not before or not after:
        issues.append("missing observed near-bank travel before or after the detour")
    for label, indexes in (("before", before), ("after", after)):
        if indexes:
            if max(abs(local[i][1]) for i in indexes) > .8:
                issues.append(f"{label} detour trajectory does not return to baseline")
            if max(distances[i] for i in indexes) > MAX_BASELINE_SHORE_M:
                issues.append(f"{label} detour is too far from the physical bank")
            progress = max(local[i][0] for i in indexes)-min(local[i][0] for i in indexes)
            if progress < 1.0:
                issues.append(f"{label} detour does not follow the bank for at least 1m")
    min_bank, max_bank = min(distances), max(distances)
    # The signed-distance function is 1-Lipschitz, so half a sample interval
    # bounds the minimum for the entire line segment between adjacent poses.
    minimum_bank_bound = min(min(a, b)-step/2 for a, b, step in zip(distances, distances[1:], steps))
    if minimum_bank_bound < MIN_CENTER_SHORE_M or max_bank > MAX_DETOUR_SHORE_M:
        issues.append("physical bank distance outside safety bounds")
    obstacle_reports = []
    obstacles = metadata["obstacles"]
    expected = {o["model"]: o for o in _obstacles(scene)}
    observed = {o["model"]: o for o in obstacles}
    if len(observed) != len(obstacles) or set(observed) != set(expected):
        issues.append("recorded obstacle models do not match the planned world")
    for name in set(expected) & set(observed):
        a, b = expected[name], observed[name]
        if math.dist([a["x"], a["y"]], [b["x"], b["y"]]) > 1e-5 or abs(a["radius_m"]-b["radius_m"]) > 1e-5:
            issues.append(f"recorded obstacle geometry differs for {name}")
    named = observed.get(info["obstacle_model"])
    if (named is None or info["obstacle_model"] not in route["required_obstacle_models"]
            or math.dist([named["x"], named["y"]], route["gates"][0]["center"]) > 1e-5):
        issues.append("scored gate is not the intended physical obstacle")
    for buoy in scene["buoys"]:
        maximum = max(math.dist(p, [buoy["x"], buoy["y"]]) for p in points)
        if maximum > 32.:
            issues.append(f"buoy slot {buoy['slot']} exceeds the 32m state range")
    for obstacle in obstacles:
        center = [obstacle["x"], obstacle["y"]]
        clearance = min(_segment_distance(a, b, center)-VESSEL_RADIUS_M-float(obstacle["radius_m"])
                        for a, b in zip(points, points[1:]))
        obstacle_reports.append(dict(model=obstacle["model"], minimum_swept_clearance_m=clearance))
        if clearance < float(metadata["task"]["minimum_clearance_m"])-1e-9:
            issues.append(f"insufficient swept clearance for {obstacle['model']}")
    gate = route["gates"][0]
    crossings = []
    gate_coords = [body_xy(*(p-np.asarray(gate["center"])), gate["yaw_rad"]) for p in points]
    for i, (a, b) in enumerate(zip(gate_coords, gate_coords[1:])):
        if a[0] < 0 <= b[0]:
            fraction = -a[0]/(b[0]-a[0])
            lateral = a[1]+fraction*(b[1]-a[1])
            yaw = poses[i][5]+fraction*wrap(poses[i+1][5]-poses[i][5])
            heading = math.degrees(wrap(yaw-gate["yaw_rad"]))
            crossings.append(dict(index=i, lateral_m=lateral, heading_relative_route_deg=heading))
    if len(crossings) != 1:
        issues.append("expected exactly one forward obstacle crossing")
    else:
        crossing = crossings[0]
        if not gate["left_min_m"] <= crossing["lateral_m"] <= gate["left_max_m"]:
            issues.append("crossed on an unsafe or unplanned side")
        if abs(crossing["heading_relative_route_deg"]) > MAX_CROSSING_HEADING_DEG:
            issues.append("heading at obstacle exceeds 10 degrees")
    target = np.asarray(scene["target_xy"])
    target_row = next((b for b in scene["buoys"] if b["slot"] == target_slot), None)
    if (target_row is None or math.dist([target_row["x"], target_row["y"]], target) > 1e-6
            or metadata["task"].get("target") != target_slot):
        issues.append("target slot, position, or evaluator is inconsistent")
    initial_distance = float(np.linalg.norm(points[0]-target))
    final_distance = float(np.linalg.norm(points[-1]-target))
    if not 25 <= initial_distance <= 30:
        issues.append("initial target distance outside 25–30m")
    task = metadata["task"]
    if not float(task["stop_min_m"]) <= final_distance <= float(task["stop_max_m"]):
        issues.append("final boat position outside stop band")
    end = len(rows)-1
    while end > 0 and times[-1]-times[end-1] <= float(task["hold_s"])*1e9+1:
        end -= 1
    if times[-1]-times[end] < float(task["hold_s"])*1e9-100_000_000:
        issues.append("insufficient stop dwell")
    for i in range(end, len(rows)):
        speed = rows[i].get("speed_mps")
        if speed is None or not math.isfinite(float(speed)) or float(speed) > float(task["stop_speed_mps"]) or not float(task["stop_min_m"]) <= np.linalg.norm(points[i]-target) <= float(task["stop_max_m"]):
            issues.append("stop dwell contains an out-of-band position or speed")
            break
    return dict(schema_version=1, profile=PROFILE, passed=not issues, issues=issues,
                inputs_sha256={p.name: _digest(p) for p in (metadata_path, summary_path, steps_path)},
                inspector_sha256=_digest(Path(__file__)), initial_target_distance_m=initial_distance,
                final_target_distance_m=final_distance, recorded_path_length_m=sum(steps),
                duration_sim_s=(times[-1]-times[0])/1e9, sample_count=len(rows),
                center_shore_distance_min_m=min_bank, center_shore_distance_max_m=max_bank,
                interpolated_center_shore_lower_bound_m=minimum_bank_bound,
                hull_shore_lower_bound_m=minimum_bank_bound-VESSEL_RADIUS_M,
                pre_detour_baseline_samples=len(before), post_detour_baseline_samples=len(after),
                crossing=crossings[0] if len(crossings) == 1 else None,
                obstacle_clearances=obstacle_reports)

"""Validated, explicit multi-buoy scenes for green-round target demonstrations.

Slots are zero-based semantic IDs; geometry is simulation map metadata, never a
policy prompt.  The policy's 32 m position scale is protected with a 1 m margin.
"""
import math
import re

import numpy as np

from usv_vla.core import BUOY_SLOTS
from usv_vla.world import POINTED_BUOY_RAISE_M

TARGET_SLOT = 4
POSITION_MARGIN_M = 31.0
VESSEL_RADIUS_M = .85
MIN_CLEARANCE_M = .2


def buoy_radius(slot):
    return .75 if BUOY_SLOTS[slot][1] == "round" else .434


def _xy(row):
    return [row["x"], row["y"]]


def _box_overlap(a, b):
    return (min(a[2], b[2]) > max(a[0], b[0]) and
            min(a[3], b[3]) > max(a[1], b[1]))


def initial_visibility_green(scene, pose=None):
    from usv_vla.scene_plan import front_camera_geometry, project_object

    initial = scene["initial_pose"] if pose is None else pose
    if len(initial) != 4 or not np.isfinite(initial).all():
        raise ValueError("Visibility pose must be finite [x,y,z,yaw].")
    objects = []
    for buoy in scene["buoys"]:
        slot = buoy["slot"]
        shape = BUOY_SLOTS[slot][1]
        view = project_object(initial, *(_xy(buoy)), buoy_radius(slot),
                              z_min=0.,
                              z_max=1.1 if shape == "round" else .8+POINTED_BUOY_RAISE_M)
        objects.append(dict(slot=slot, model=f"v2_buoy_{slot+1}", kind="buoy", **view))
    for cylinder in scene["cylinders"]:
        view = project_object(initial, *(_xy(cylinder)), cylinder["radius_m"])
        objects.append(dict(id=cylinder["id"], model=f"usv_cylinder_{cylinder['id']}", kind="cylinder", **view))
    for obj in objects:
        box = obj["bbox_normalized"]
        blockers = [other["model"] for other in objects if other is not obj and box is not None
                    and other["bbox_normalized"] is not None and other["positive_depth"]
                    and other["center_forward_m"] < obj["center_forward_m"]
                    and _box_overlap(box, other["bbox_normalized"])]
        obj["occluded_by_models"] = blockers
        obj["unobstructed_visible"] = bool(obj["fully_visible"] and not blockers)
    buoy_rows = [obj for obj in objects if obj["kind"] == "buoy"]
    cylinders = [obj for obj in objects if obj["kind"] == "cylinder"]
    target = dict(next(obj for obj in buoy_rows if obj["slot"] == scene["target_slot"]))
    target["occluded_by_cylinder"] = any(name.startswith("usv_cylinder_") for name in target["occluded_by_models"])
    return dict(method="static_sdf_frustum_and_projected_silhouette_bounds",
                camera=front_camera_geometry(), target=target, buoys=buoy_rows, cylinders=cylinders,
                required_obstacle_models=scene["route"].get("required_obstacle_models", []),
                required_cylinder_models=scene["route"].get("cylinder_models", []))


def validate_green_scene(scene):
    from usv_vla.scene_plan import CATEGORIES, _segment_distance, shore_clearance

    target_slot = {'red_multi_v1': 0, 'yellow_multi_v1': 2, 'blue_multi_v1': 6}.get(scene.get('dataset_type'), TARGET_SLOT)
    target_color = BUOY_SLOTS[target_slot][0]

    if scene.get("category") not in CATEGORIES:
        raise ValueError("Unknown collection category.")
    pose = scene["initial_pose"]
    if len(pose) != 4 or not np.isfinite(pose).all():
        raise ValueError("initial_pose must be finite [x,y,z,yaw].")
    target = scene["target_xy"]
    if len(target) != 2 or not np.isfinite(target).all():
        raise ValueError("target_xy must contain two finite coordinates.")
    if scene.get("target_slot") != target_slot:
        raise ValueError("Target must match the dataset's round-buoy slot.")
    buoys = scene["buoys"]
    if not isinstance(buoys, list) or not 1 <= len(buoys) <= 8:
        raise ValueError("Need 1..8 uniquely assigned buoys.")
    slots = [row["slot"] for row in buoys]
    if any(type(slot) is not int or slot not in range(8) for slot in slots) or len(set(slots)) != len(slots) or target_slot not in slots:
        raise ValueError("Buoy slots must be unique integers in 0..7 and include target slot 4.")
    if any(not np.isfinite(_xy(row)).all() for row in buoys):
        raise ValueError("Buoy coordinates must be finite.")
    target_buoy = next(row for row in buoys if row["slot"] == target_slot)
    if math.dist(_xy(target_buoy), target) > 1e-8:
        raise ValueError("target_xy must equal the green-round buoy position.")
    forward = math.cos(pose[3])*(target[0]-pose[0])+math.sin(pose[3])*(target[1]-pose[1])
    if forward <= 0 or not 4.-1e-8 <= math.dist(pose[:2], target) <= 30.+1e-8:
        raise ValueError("Target must start ahead at a distance of 4..30 meters.")
    cylinders = scene["cylinders"]
    if not isinstance(cylinders, list) or len(cylinders) > 4:
        raise ValueError("Collection scenes require 0..4 cylinders.")
    if [c["id"] for c in cylinders] != list(range(1, len(cylinders)+1)):
        raise ValueError("Cylinder ids must follow model order, 1..N.")
    for cylinder in cylinders:
        if not np.isfinite([*(_xy(cylinder)), cylinder["radius_m"]]).all() or abs(cylinder["radius_m"]-.405) > 1e-9:
            raise ValueError("Cylinders require finite coordinates and .405 m collision radius.")
    route = scene["route"]
    if route.get("kind") not in ("direct", "cylinder_pass", "cylinder_gap", "bank_follow"):
        raise ValueError("Unknown route kind.")
    instruction = route.get("instruction")
    if not isinstance(instruction, str) or not re.search(rf"\b{target_color}\s+round\s+buoy\b", instruction, re.IGNORECASE):
        raise ValueError("Instruction must uniquely specify the target round buoy.")
    if route.get("prompt_include_phase") is not False:
        raise ValueError("Only the fixed human instruction may be passed to the policy.")
    objects = [dict(model=f"v2_buoy_{b['slot']+1}", center=_xy(b), radius=buoy_radius(b["slot"])) for b in buoys]
    objects += [dict(model=f"usv_cylinder_{c['id']}", center=_xy(c), radius=c["radius_m"]) for c in cylinders]
    names = {o["model"] for o in objects}
    required = route.get("required_obstacle_models", [])
    cylinder_models = route.get("cylinder_models", [])
    if not isinstance(required, list) or not isinstance(cylinder_models, list) or len(set(required)) != len(required):
        raise ValueError("Required obstacle models must be a unique list.")
    if any(name not in names or name == f'v2_buoy_{target_slot+1}' for name in [*required, *cylinder_models]):
        raise ValueError("Required obstacle model is missing or is the target.")
    points = [pose[:2], *route.get("waypoints", [])]
    if any(len(point) != 2 or not np.isfinite(point).all() for point in points):
        raise ValueError("Route points must be finite XY coordinates.")
    previous = np.asarray(points[-1]); delta = np.asarray(target)-previous
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        raise ValueError("Last waypoint must differ from target.")
    stop = (np.asarray(target)-delta/length*2.1).tolist()
    points.append(stop)
    for obj in objects:
        for a, b in zip(points, points[1:]):
            if _segment_distance(obj["center"], a, b) < VESSEL_RADIUS_M+obj["radius"]+MIN_CLEARANCE_M-1e-9:
                raise ValueError(f"Planned teacher route does not clear {obj['model']}.")
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            if math.dist(a["center"], b["center"]) < a["radius"]+b["radius"]+.2:
                raise ValueError("Scene objects overlap.")
    for buoy in buoys:
        if any(math.dist(_xy(buoy), point) > POSITION_MARGIN_M for point in points):
            raise ValueError("Buoy position exceeds the 31 m planned training-scale margin.")
    checked_points = [*points, *[o["center"] for o in objects]]
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.dist(a, b)/.5))
        checked_points.extend((np.asarray(a)*(1-i/n)+np.asarray(b)*(i/n)).tolist() for i in range(1, n))
    if any(shore_clearance(point) < 2.5 for point in checked_points):
        raise ValueError("Scene or route is too near the actual shore collision mesh.")
    visibility = initial_visibility_green(scene)
    by_model = {obj["model"]: obj for obj in [*visibility["buoys"], *visibility["cylinders"]]}
    for model in set(required).union(cylinder_models):
        if not by_model[model]["unobstructed_visible"]:
            raise ValueError(f"Required obstacle {model} must be visible and unobstructed in the initial image.")
    if scene["category"] == "approach_visible" and not visibility["target"]["unobstructed_visible"]:
        raise ValueError("Visible-target scene needs an unobstructed in-frame target.")
    if scene["category"] == "approach_offscreen" and not visibility["target"]["entirely_outside_horizontal"]:
        raise ValueError("Offscreen target must lie completely outside the horizontal frame.")
    return visibility

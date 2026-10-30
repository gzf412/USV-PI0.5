"""Reproducible, geometry-checked one-buoy collection scenes.

Scene geometry is teacher/evaluator metadata. Only the existing buoy map enters
policy state; cylinder positions and observer cameras are never policy inputs.
"""

from functools import lru_cache
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np

from usv_vla.paths import ASSETS
from usv_vla.world import ORIGIN

CATEGORIES = ("approach_visible", "approach_offscreen", "around_left", "around_right", "between", "shore")


def xy_from(origin, yaw, forward, left):
    c, s = math.cos(yaw), math.sin(yaw)
    return [origin[0]+c*forward-s*left, origin[1]+s*forward+c*left]


@lru_cache(maxsize=1)
def front_camera_geometry():
    model = ET.parse(ASSETS/"models/small_cora_float/model.sdf")
    sensor = model.find('.//sensor[@name="front_camera"]')
    pose = [float(v) for v in sensor.findtext("pose").split()]
    hfov = float(sensor.findtext("camera/horizontal_fov"))
    width, height = (int(sensor.findtext("camera/image/"+name)) for name in ("width", "height"))
    return dict(pose=pose, horizontal_fov_rad=hfov,
                vertical_fov_rad=2*math.atan(math.tan(hfov/2)*height/width),
                width=width, height=height)


def project_object(initial_pose, x, y, radius_m, *, z_min=-.23, z_max=.67):
    """Conservative complete-object bounds in the actual front-camera image.

    Camera uses +X forward, +Y left, +Z up. The cylinder's bounding box is
    deliberately larger than its curved silhouette, making edge tests strict.
    """
    camera = front_camera_geometry()
    yaw = initial_pose[3]
    dx, dy = x-initial_pose[0], y-initial_pose[1]
    c, s = math.cos(yaw), math.sin(yaw)
    forward = c*dx+s*dy-camera["pose"][0]
    left = -s*dx+c*dy-camera["pose"][1]
    camera_z = initial_pose[2]+camera["pose"][2]
    outside_horizontal = abs(left)-radius_m > max(forward+radius_m, 0.)*math.tan(camera["horizontal_fov_rad"]/2)
    projected = []
    for depth in (forward-radius_m, forward+radius_m):
        for lateral in (left-radius_m, left+radius_m):
            for z in (z_min, z_max):
                if depth <= 0:
                    return dict(fully_visible=False, positive_depth=False, bbox_normalized=None,
                                entirely_outside_horizontal=bool(outside_horizontal),
                                center_forward_m=forward, center_left_m=left)
                projected.append((-lateral/(depth*math.tan(camera["horizontal_fov_rad"]/2)),
                                  -(z-camera_z)/(depth*math.tan(camera["vertical_fov_rad"]/2))))
    bounds = [min(p[0] for p in projected), min(p[1] for p in projected),
              max(p[0] for p in projected), max(p[1] for p in projected)]
    fully_visible = bounds[0] >= -.95 and bounds[2] <= .95 and bounds[1] >= -.95 and bounds[3] <= .95
    return dict(fully_visible=bool(fully_visible), positive_depth=True,
                bbox_normalized=bounds, center_forward_m=forward, center_left_m=left,
                entirely_outside_horizontal=bool(outside_horizontal))


def _interval_overlap(a, b, margin=.015):
    return min(a[2], b[2])-max(a[0], b[0]) > -margin


def _segment_distance(point, start, end):
    point, start, end = np.asarray(point), np.asarray(start), np.asarray(end)
    delta = end-start
    alpha = float(np.clip(np.dot(point-start, delta)/max(np.dot(delta, delta), 1e-12), 0., 1.))
    return float(np.linalg.norm(point-start-alpha*delta))


@lru_cache(maxsize=1)
def _shore_triangles():
    """Read actual shore collision triangles, including the model's Z offset."""
    path = ASSETS/"models/sydney_regatta/meshes/sydney_regatta_shore.dae"
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    tree = ET.parse(path)
    vertices = np.fromstring(tree.find('.//c:source[@id="SydneyTerrainCollider-POSITION"]/c:float_array', ns).text,
                             sep=" ").reshape(-1, 3)*.001
    # The COLLADA node scales coordinates by .001; its unit is one meter.
    vertices[:, 2] += .2
    indices = np.fromstring(tree.find('.//c:triangles/c:p', ns).text, sep=" ", dtype=int).reshape(-1, 3)[:, 0]
    triangles = vertices[indices.reshape(-1, 3)]
    return triangles[triangles[:, :, 2].max(axis=1) >= .1, :, :2]


def shore_clearance(point):
    """Conservative horizontal distance to the physical shore mesh projection."""
    triangles = _shore_triangles()
    p = np.asarray(point, dtype=float)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v, w, q = b-a, c-a, p-a
    determinant = v[:, 0]*w[:, 1]-v[:, 1]*w[:, 0]
    good = abs(determinant) > 1e-10
    u = np.zeros(len(triangles)); t = np.zeros(len(triangles))
    u[good] = (q[good, 0]*w[good, 1]-q[good, 1]*w[good, 0])/determinant[good]
    t[good] = (v[good, 0]*q[good, 1]-v[good, 1]*q[good, 0])/determinant[good]
    if np.any(good & (u >= 0) & (t >= 0) & (u+t <= 1)):
        return 0.
    shortest = math.inf
    for start, end in ((a, b), (b, c), (c, a)):
        d = end-start
        alpha = np.clip(np.sum((p-start)*d, axis=1)/np.maximum(np.sum(d*d, axis=1), 1e-12), 0., 1.)
        shortest = min(shortest, float(np.linalg.norm(p-start-alpha[:, None]*d, axis=1).min()))
    return shortest


def initial_visibility(scene, pose=None):
    """Recompute projection at [x,y,z,yaw], including a measured start pose."""
    if scene.get("dataset_type") in ("green_multi_v1", "red_multi_v1", "blue_multi_v1"):
        from usv_vla.green_scene import initial_visibility_green
        return initial_visibility_green(scene, pose)
    initial = scene["initial_pose"] if pose is None else pose
    if len(initial) != 4 or not np.isfinite(initial).all():
        raise ValueError("Visibility pose must be finite [x,y,z,yaw].")
    rows = []
    for cylinder in scene["cylinders"]:
        rows.append(dict(id=cylinder["id"], **project_object(initial, cylinder["x"], cylinder["y"], cylinder["radius_m"])))
    target = project_object(initial, *scene["target_xy"], .75, z_min=0., z_max=1.1)
    target["occluded_by_cylinder"] = any(
        row["positive_depth"] and target["positive_depth"] and
        row["center_forward_m"] < target["center_forward_m"] and
        _interval_overlap(row["bbox_normalized"], target["bbox_normalized"], 0.)
        for row in rows)
    return dict(method="static_sdf_frustum_and_projected_silhouette_bounds",
                camera=front_camera_geometry(), target=target, cylinders=rows,
                required_cylinder_models=scene["route"].get("cylinder_models", []))


def validate_scene(scene):
    """Reject malformed or unsafe collection plans before launching Gazebo."""
    if scene.get("dataset_type") in ("green_multi_v1", "red_multi_v1", "blue_multi_v1"):
        from usv_vla.green_scene import validate_green_scene
        return validate_green_scene(scene)
    if scene.get("category") not in CATEGORIES:
        raise ValueError("Unknown collection category.")
    pose = scene["initial_pose"]
    if len(pose) != 4 or not np.isfinite(pose).all():
        raise ValueError("initial_pose must be finite [x,y,z,yaw].")
    target = scene["target_xy"]
    if len(target) != 2 or not np.isfinite(target).all():
        raise ValueError("target_xy must contain two finite coordinates.")
    forward = math.cos(pose[3])*(target[0]-pose[0])+math.sin(pose[3])*(target[1]-pose[1])
    # Keep the target inside the policy's 32 m relative-position scale while
    # allowing longer routes for every task kind, including obstacle courses.
    max_distance = 30.
    if forward <= 0 or not 4.-1e-8 <= math.dist(pose[:2], target) <= max_distance+1e-8:
        raise ValueError(f"Target must start ahead, at a distance of 4..{max_distance:g} meters.")
    cylinders = scene["cylinders"]
    if not 1 <= len(cylinders) <= 4:
        raise ValueError("Collection scenes require 1..4 cylinders.")
    if [c["id"] for c in cylinders] != list(range(1, len(cylinders)+1)):
        raise ValueError("Cylinder ids must follow their model order, 1..N.")
    route = scene["route"]
    if route.get("kind") not in ("direct", "cylinder_pass", "cylinder_gap", "bank_follow"):
        raise ValueError("Unknown route kind.")
    required_models = route.get("cylinder_models", [])
    model_distances = {f'usv_cylinder_{c["id"]}': math.dist(pose[:2], [c["x"], c["y"]])
                       for c in cylinders}
    if any(name not in model_distances for name in required_models):
        raise ValueError("Route names an absent cylinder.")
    other_distances = [d for name, d in model_distances.items() if name not in required_models]
    if required_models and other_distances and max(model_distances[name] for name in required_models)+1. > min(other_distances):
        raise ValueError("The task cylinders must be the initially nearest cylinders.")
    points = [pose[:2], *route.get("waypoints", [])]
    # Teacher finishes at the near-side 2.1 m buoy band.
    previous = np.asarray(points[-1]); delta = np.asarray(target)-previous
    stop = np.asarray(target)-delta/max(float(np.linalg.norm(delta)), 1e-9)*2.1
    points.append(stop.tolist())
    for cylinder in cylinders:
        if not np.isfinite([cylinder["x"], cylinder["y"], cylinder["radius_m"]]).all() or cylinder["radius_m"] != .405:
            raise ValueError("Cylinders must have finite coordinates and the actual .405 m collision radius.")
        center = [cylinder["x"], cylinder["y"]]
        if math.dist(center, target) < 3.2:
            raise ValueError("Cylinder overlaps the buoy stopping region.")
        if any(_segment_distance(center, a, b) < 1.75 for a, b in zip(points, points[1:])):
            raise ValueError("Planned teacher route does not clear a cylinder.")
    checked_points = [*points, target, *([c["x"], c["y"]] for c in cylinders)]
    # Sample path interiors so a pair of water-side endpoints cannot cut land.
    for a, b in zip(points, points[1:]):
        checked_points.extend((np.asarray(a)*(1-t)+np.asarray(b)*t).tolist() for t in (.25, .5, .75))
    if any(shore_clearance(point) < 2.5 for point in checked_points):
        raise ValueError("Scene or route is too near the actual shore collision mesh.")
    visibility = initial_visibility(scene)
    if route["kind"] in ("cylinder_pass", "cylinder_gap"):
        if not all(row["fully_visible"] for row in visibility["cylinders"]):
            raise ValueError("All cylinders in a cylinder task must initially fit in the front camera.")
        for i, row in enumerate(visibility["cylinders"]):
            if any(_interval_overlap(row["bbox_normalized"], other["bbox_normalized"])
                   for other in visibility["cylinders"][i+1:]):
                raise ValueError("Initial cylinder silhouettes overlap.")
    if scene["category"] == "approach_visible" and (not visibility["target"]["fully_visible"] or visibility["target"]["occluded_by_cylinder"]):
        raise ValueError("Visible-target scene needs an unobscured in-frame target.")
    if scene["category"] == "approach_offscreen" and not visibility["target"]["entirely_outside_horizontal"]:
        raise ValueError("Offscreen-target scene must keep the complete target outside the front image.")
    return visibility


def generate_scene(category, index, seed):
    """Return a JSON-compatible, deterministic plan for one full episode."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown scene category: {category}")
    rng = random.Random(int(seed))
    for attempt in range(400):
        count = 1+(int(index) % 4)
        if category == "between":
            count = 2+(int(index) % 3)
        pose = [ORIGIN[0]+rng.uniform(-5., 5.), ORIGIN[1]+rng.uniform(0., 5.), ORIGIN[2],
                ORIGIN[3]+rng.uniform(-.25, .25)]
        distance = rng.uniform(4., 9.) if index % 3 == 0 else rng.uniform(9., 18.)
        bearing = rng.uniform(-28., 28.)*math.pi/180
        if category == "approach_offscreen":
            bearing = rng.choice((-1, 1))*rng.uniform(52., 75.)*math.pi/180
        reference = pose[3]+bearing
        cylinders, waypoints, gates = [], [], []
        route = dict(kind="direct", reference_yaw_rad=reference, waypoints=waypoints,
                     cruise_speed_mps=rng.uniform(.34, .58), gates=gates,
                     cylinder_models=[], waypoint_radius_m=.85,
                     instruction="Approach the red round buoy and stop 1.9 to 2.3 meters away.")
        def add_cylinder(forward, left):
            x, y = xy_from(pose, reference, forward, left)
            cylinders.append(dict(id=len(cylinders)+1, x=x, y=y, radius_m=.405))
        if category in ("around_left", "around_right", "between"):
            distance = rng.uniform(13., 18.)
            reference = pose[3]+rng.uniform(-.10, .10)
            route["reference_yaw_rad"] = reference
            fwd = rng.uniform(5.8, 8.)
            if category == "between":
                gap_half = rng.uniform(2., 2.45)
                add_cylinder(fwd, gap_half); add_cylinder(fwd, -gap_half)
                center = xy_from(pose, reference, fwd, 0.)
                gate_half = gap_half-.405-.85-.25
                gate = dict(center=center, yaw_rad=reference, left_min_m=-gate_half,
                            left_max_m=gate_half, exit_forward_m=2.)
                waypoints.extend(xy_from(pose, reference, fwd+step, 0.) for step in (-2.5, 0., 2.5))
                route.update(kind="cylinder_gap", cylinder_models=["usv_cylinder_1", "usv_cylinder_2"],
                             instruction="Pass between the two initially nearest orange cylinders, then approach the red round buoy and stop 1.9 to 2.3 meters away.")
            else:
                side = "left" if category == "around_left" else "right"
                sign = 1 if side == "left" else -1
                lateral = rng.uniform(-.35, .35)
                add_cylinder(fwd, lateral)
                offset = rng.uniform(2.65, 2.95)
                center = xy_from(pose, reference, fwd, lateral)
                limits = (2., 3.7) if sign == 1 else (-3.7, -2.)
                gate = dict(center=center, yaw_rad=reference, left_min_m=limits[0],
                            left_max_m=limits[1], exit_forward_m=2.)
                waypoints.extend(xy_from(pose, reference, fwd+step, lateral+sign*offset) for step in (-3., 0., 2.5))
                route.update(kind="cylinder_pass", cylinder_models=["usv_cylinder_1"], side=side,
                             instruction=f"Pass on the {side} side of the initially nearest orange cylinder, using the initial direction toward the buoy as the left-right reference; then approach the red round buoy and stop 1.9 to 2.3 meters away.")
            gates.append(gate)
        if category == "shore":
            # Endpoints are actual water-edge vertices in the shore collision mesh.
            p0, p1 = np.array([-563.768125, 160.538125]), np.array([-525.6186875, 154.73040625])
            delta = p1-p0; length = float(np.linalg.norm(delta)); unit = delta/length
            bank_yaw = math.atan2(unit[1], unit[0])
            distance = rng.uniform(25., 29.5)
            along = rng.uniform(1., length-distance-1.)
            offset = rng.uniform(3.6, 3.9)
            start = p0+unit*along+np.array([-unit[1], unit[0]])*offset
            target = start+unit*distance
            if index % 2:
                start, target = target.copy(), start.copy(); bank_yaw += math.pi
            pose = [*start.tolist(), ORIGIN[2], bank_yaw+rng.uniform(-.04, .04)]
            reference = bank_yaw
            waypoints.extend(xy_from(pose, reference, progress, 0.)
                             for progress in (distance*.35, distance*.65, distance-2.4))
            route.update(kind="bank_follow", reference_yaw_rad=reference, waypoint_radius_m=.35,
                         bank_corridor=dict(origin=pose[:2], yaw_rad=reference, left_min_m=-.4,
                                            left_max_m=.4, progress_start_m=0., progress_end_m=distance-3.),
                         shore_reference=dict(source="sydney_regatta_shore.dae collision-mesh water-edge vertices; offset measured from endpoint chord", vertices=[p0.tolist(), p1.tolist()], offset_m=offset),
                         instruction=f"Follow the riverbank on your {'left' if index % 2 else 'right'}, stay close to the bank, continue to the red round buoy ahead, and stop 1.9 to 2.3 meters away.")
            bearing = 0.
        else:
            target = xy_from(pose, reference, distance, 0.)
        # Remaining objects are distractors, placed safely away from every route segment.
        final_previous = np.asarray(waypoints[-1] if waypoints else pose[:2]); to_target = np.asarray(target)-final_previous
        stop = np.asarray(target)-to_target/np.linalg.norm(to_target)*2.1
        route_points = [pose[:2], *waypoints, stop.tolist()]
        required_distance = (max(math.dist(pose[:2], [c["x"], c["y"]]) for c in cylinders)
                             if route["cylinder_models"] else None)
        placement_ok = True
        while len(cylinders) < count:
            found = False
            for _ in range(150):
                # Wide angular spacing is possible within the real 80-degree image.
                heading = pose[3]
                cf = rng.uniform(5., 13.)
                sign = (-1 if index % 2 else 1) if category == "shore" else rng.choice((-1, 1))
                cl = sign*rng.uniform(2.5, min(7.5, cf*.62))
                x, y = xy_from(pose, heading, cf, cl)
                center = [x, y]
                if required_distance is not None and math.dist(pose[:2], center) <= required_distance+1.:
                    continue
                if math.dist(center, target) < 3.4 or any(math.dist(center, [c["x"], c["y"]]) < 2.0 for c in cylinders):
                    continue
                if any(_segment_distance(center, a, b) < 2.3 for a, b in zip(route_points, route_points[1:])):
                    continue
                cylinders.append(dict(id=len(cylinders)+1, x=x, y=y, radius_m=.405)); found = True; break
            if not found:
                placement_ok = False; break
        if not placement_ok:
            continue
        scene = dict(schema_version=1, category=category, index=int(index), seed=int(seed),
                     generation_attempt=attempt, initial_pose=pose, target_xy=list(target),
                     cylinders=cylinders, route=route)
        try:
            scene["initial_visibility"] = validate_scene(scene)
        except ValueError:
            continue
        return scene
    raise ValueError(f"Cannot create a valid {category} scene for seed {seed} after 400 attempts.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", choices=CATEGORIES)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = json.dumps(generate_scene(args.category, args.index, args.seed), indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()

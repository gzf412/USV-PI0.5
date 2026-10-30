"""Generate a separate dynamic world; leave the VLM image scene untouched."""

import argparse
import copy
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from usv_vla.core import BUOY_SLOTS, TaskConfig
from usv_vla.paths import ASSETS, OUTPUTS

ROOT = Path(__file__).resolve().parent
SOURCE = ASSETS / "worlds/simple_waterway.sdf"
ORIGIN = (-532., 162., .14, 1.436332312998582)
AERIAL_IMAGE_TOPIC = "/usv/aerial_camera/image"
AERIAL_INFO_TOPIC = "/usv/aerial_camera/camera_info"
AERIAL_FRAME_ID = "usv_aerial_camera_optical"
POINTED_BUOY_RAISE_M = .35


def text(parent, tag, value):
    element = parent.find(tag)
    if element is None:
        element = ET.SubElement(parent, tag)
    element.text = str(value)
    return element


def _add_aerial_camera(world, config, positions, obstacles, reference, enabled,
                       initial_pose=ORIGIN, route=None, rate_hz=10):
    """Add a fixed, invisible observer; it is never a model observation input.

    SDF cameras look along local +X. Positive pi/2 pitch therefore points
    straight down, with the initial target direction toward the image top.
    Bound all subjects at their highest relevant Z, not only at water level.
    """
    metadata = dict(enabled=bool(enabled), topic=AERIAL_IMAGE_TOPIC,
                    camera_info_topic=AERIAL_INFO_TOPIC, frame_id=AERIAL_FRAME_ID,
                    pose=None, resolution=[1280, 720], rate_hz=rate_hz,
                    frame_convention="optical_x_right_y_down_z_forward",
                    purpose="recording_only_not_policy_input")
    if not enabled:
        return metadata
    cosine, sine = math.cos(reference), math.sin(reference)

    def local(x, y):
        dx, dy = x-initial_pose[0], y-initial_pose[1]
        return cosine*dx+sine*dy, -sine*dx+cosine*dy

    subjects = []

    def square(x, y, radius):
        forward, left = local(x, y)
        subjects.extend((forward+df, left+dl) for df in (-radius, radius)
                        for dl in (-radius, radius))

    # Include the full hull/mast footprint and object geometry, not just centers.
    square(*initial_pose[:2], 2.)
    for point in (route or {}).get("waypoints", []):
        square(*point, 2.)
    for obstacle in obstacles:
        square(obstacle["x"], obstacle["y"], obstacle["radius_m"])
    stop_slot = config.destination if config.kind == "pass_then_approach" else config.target
    if config.kind in ("approach", "pass_then_approach"):
        square(*positions[stop_slot], config.stop_max_m+2.)
    if config.kind in ("pass", "pass_then_approach"):
        target_forward, target_left = local(*positions[config.target])
        sign = 1 if config.side == "left" else -1
        # Full gate and exit rectangles, plus the teacher's entry waypoint and
        # a hull footprint around each; connecting route segments stay inside.
        for forward in (-3., 0., config.exit_min_m, config.exit_max_m):
            for left in (config.gate_min_m, config.gate_max_m):
                subjects.extend((target_forward+forward+df, target_left+sign*left+dl)
                                for df in (-2., 2.) for dl in (-2., 2.))
    raw_bounds = dict(forward_min=min(p[0] for p in subjects),
                      forward_max=max(p[0] for p in subjects),
                      left_min=min(p[1] for p in subjects),
                      left_max=max(p[1] for p in subjects))
    margin = 1.5
    forward_min, forward_max = raw_bounds["forward_min"]-margin, raw_bounds["forward_max"]+margin
    left_min, left_max = raw_bounds["left_min"]-margin, raw_bounds["left_max"]+margin
    center_forward, center_left = (forward_min+forward_max)/2, (left_min+left_max)/2
    hfov = 1.35
    vfov = 2*math.atan(math.tan(hfov/2)*720/1280)
    subject_top_z = 3.
    height_above_subjects = max((forward_max-forward_min)/(2*math.tan(vfov/2)),
                                (left_max-left_min)/(2*math.tan(hfov/2)), 10.)
    altitude = subject_top_z+height_above_subjects
    pose = [initial_pose[0]+cosine*center_forward-sine*center_left,
            initial_pose[1]+sine*center_forward+cosine*center_left,
            altitude, 0., math.pi/2, reference]
    model = ET.SubElement(world, "model", name="usv_aerial_observer")
    text(model, "static", "true")
    text(model, "pose", " ".join(str(value) for value in pose))
    link = ET.SubElement(model, "link", name="camera_link")
    sensor = ET.SubElement(link, "sensor", name="aerial_camera", type="camera")
    text(sensor, "topic", AERIAL_IMAGE_TOPIC)
    text(sensor, "gz_frame_id", AERIAL_FRAME_ID)
    text(sensor, "always_on", "true")
    text(sensor, "update_rate", rate_hz)
    text(sensor, "visualize", "false")
    camera = ET.SubElement(sensor, "camera")
    text(camera, "horizontal_fov", hfov)
    image = ET.SubElement(camera, "image")
    text(image, "width", 1280)
    text(image, "height", 720)
    text(image, "format", "R8G8B8")
    clip = ET.SubElement(camera, "clip")
    text(clip, "near", .1)
    text(clip, "far", max(500., 2*altitude))
    text(camera, "camera_info_topic", AERIAL_INFO_TOPIC)
    optical_frame = ET.SubElement(model, "frame", name=AERIAL_FRAME_ID, attached_to="camera_link")
    text(optical_frame, "pose", f"0 0 0 {-math.pi/2} 0 {-math.pi/2}").set("relative_to", "camera_link")
    water_half_forward, water_half_left = altitude*math.tan(vfov/2), altitude*math.tan(hfov/2)
    metadata.update(pose=pose, horizontal_fov_rad=hfov, vertical_fov_rad=vfov,
                    view=dict(reference_yaw_rad=reference, center_forward_m=center_forward,
                              center_left_m=center_left, subject_bounds_m=raw_bounds,
                              margin_m=margin, highest_subject_z_m=subject_top_z,
                              water_bounds_m=dict(forward_min=center_forward-water_half_forward,
                                                  forward_max=center_forward+water_half_forward,
                                                  left_min=center_left-water_half_left,
                                                  left_max=center_left+water_half_left)))
    return metadata


def build_world(output, *, count=1, target=0, destination=None, kind="approach", side="left",
                bearing_deg=0., distance_m=8., seed=0, cylinders=3, aerial_camera=True,
                scene=None, front_rate_hz=30, aerial_rate_hz=10):
    if not 1 <= count <= 8 or cylinders not in range(5):
        raise ValueError("Require 1..8 buoys and 0..4 cylinders.")
    if not all(math.isfinite(rate) and 1 <= rate <= 60 for rate in (front_rate_hz, aerial_rate_hz)):
        raise ValueError("Camera rates must be finite and within 1..60 Hz.")
    if scene is not None:
        from usv_vla.scene_plan import validate_scene
        scene = copy.deepcopy(scene)
        computed_visibility = validate_scene(scene)
        if scene.get("dataset_type") in ("green_multi_v1", "red_multi_v1", "blue_multi_v1", "yellow_multi_v1"):
            count, target, destination, kind = len(scene["buoys"]), scene["target_slot"], None, "approach"
        else:
            count, target, destination, kind = 1, 0, None, "approach"
        cylinders, seed = len(scene["cylinders"]), scene["seed"]
    config = TaskConfig(kind=kind, target=target, destination=destination, side=side)
    if not 4 <= distance_m <= 16 or not math.isfinite(bearing_deg):
        raise ValueError("Pilot distance must be 4..16m and bearing finite.")
    rng = random.Random(seed)
    required = [target] + ([destination] if destination is not None else [])
    if len(set(required)) != len(required) or len(required) > count:
        raise ValueError("Scene count cannot omit a required target.")
    available = [i for i in range(8) if i not in required]
    rng.shuffle(available)
    selected = ([row["slot"] for row in scene["buoys"]]
                if scene and scene.get("dataset_type") in ("green_multi_v1", "red_multi_v1", "blue_multi_v1", "yellow_multi_v1")
                else required+available[:count-len(required)])
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(SOURCE)
    world = tree.getroot().find("world")
    world.set("name", "usv_training")
    real_time_factor = float(scene.get("real_time_factor", 4.)) if scene else 2.
    if not math.isfinite(real_time_factor) or not 1 <= real_time_factor <= 8:
        raise ValueError("Scene real_time_factor must be within 1..8.")
    text(world.find("physics"), "real_time_factor", real_time_factor)
    source_buoys = {}
    for model in list(world.findall("model")):
        name = model.get("name", "")
        if name.startswith("v2_buoy_"):
            source_buoys[int(name.rsplit("_", 1)[-1])-1] = copy.deepcopy(model)
        if name.startswith(("v2_", "simple_cylinder_")) or name == "vlm_boat":
            world.remove(model)
    initial_pose = list(scene["initial_pose"]) if scene else list(ORIGIN)
    yaw = initial_pose[3]
    reference = yaw+math.radians(bearing_deg)
    if scene:
        reference = math.atan2(scene["target_xy"][1]-initial_pose[1], scene["target_xy"][0]-initial_pose[0])
    def global_xy(forward, left):
        return (initial_pose[0]+math.cos(reference)*forward-math.sin(reference)*left,
                initial_pose[1]+math.sin(reference)*forward+math.cos(reference)*left)
    positions = ({row["slot"]: [row["x"], row["y"]] for row in scene["buoys"]}
                 if scene and scene.get("dataset_type") in ("green_multi_v1", "red_multi_v1", "blue_multi_v1", "yellow_multi_v1")
                 else {target: scene["target_xy"] if scene else global_xy(distance_m, 0)})
    if destination is not None:
        positions[destination] = global_xy(distance_m+8, 0)
    cylinder_centers = ([(row["x"], row["y"]) for row in scene["cylinders"]] if scene
                        else [global_xy(-3+8*i, -13) for i in range(cylinders)])
    for slot in selected:
        if slot in positions:
            continue
        for _ in range(200):
            xy = global_xy(rng.uniform(3, 17), rng.choice([-1, 1])*rng.uniform(7, 12))
            if (all(math.dist(xy, other) > 3.5 for other in positions.values())
                    and all(math.dist(xy, other) > 1.6 for other in cylinder_centers)):
                positions[slot] = xy
                break
        else:
            raise ValueError("Cannot place separated distractor buoys.")
    obstacles, buoy_rows = [], []
    for slot in sorted(selected):
        model = source_buoys[slot]
        bx, by = positions[slot]
        color, shape = BUOY_SLOTS[slot]
        height = POINTED_BUOY_RAISE_M if shape == "pointed" else 0.
        text(model, "pose", f"{bx} {by} {height} 0 0 0")
        world.append(model)
        radius = .75 if shape == "round" else .434
        row = dict(slot=slot, color=color, shape=shape, x=bx, y=by, present=True,
                   model=model.get("name"), radius_m=radius, z_m=height, source="known_static_simulation_map")
        buoy_rows.append(row)
        obstacles.append(dict(x=bx, y=by, radius_m=radius, model=model.get("name")))
    # Reuse only the simple orange cylinder visual; add real collision geometry.
    template = ET.parse(SOURCE).getroot().find('.//model[@name="simple_cylinder_1"]')
    for i in range(cylinders):
        cylinder = copy.deepcopy(template)
        cylinder.set("name", f"usv_cylinder_{i+1}")
        bx, by = cylinder_centers[i]
        text(cylinder, "pose", f"{bx} {by} 0 0 0 0")
        link = cylinder.find("link")
        collision = ET.SubElement(link, "collision", name="cylinder_collision")
        text(collision, "pose", "0 0 .22 0 0 0")
        geometry = ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder")
        text(geometry, "radius", ".405")
        text(geometry, "length", ".9")
        world.append(cylinder)
        obstacles.append(dict(x=bx, y=by, radius_m=.405, model=cylinder.get("name")))
    boat = ET.parse(ASSETS/"models/small_cora_float/model.sdf").getroot().find("model")
    text(boat, "pose", f"{initial_pose[0]} {initial_pose[1]} {initial_pose[2]} 0 0 {yaw}")
    text(boat.find('.//sensor[@name="front_camera"]'), "update_rate", front_rate_hz)
    hull = boat.find('link[@name="hull"]')
    contact = ET.SubElement(hull, "sensor", name="hull_contacts", type="contact")
    text(contact, "topic", "/usv/contacts")
    text(contact, "always_on", "true")
    text(contact, "update_rate", 50)
    text(ET.SubElement(contact, "contact"), "collision", "hull_collision")
    world.append(boat)
    aerial = _add_aerial_camera(world, config, positions, obstacles, reference, aerial_camera,
                               initial_pose, scene.get("route") if scene else None, aerial_rate_hz)
    original = ET.parse(ASSETS/"worlds/sensor_systems.sdf").getroot().find("world")
    for system in ("Imu", "NavSat", "Contact"):
        name = f"gz::sim::systems::{system}"
        if world.find(f'plugin[@name="{name}"]') is None:
            world.append(copy.deepcopy(original.find(f'plugin[@name="{name}"]')))
    gui = world.find("gui")
    if gui is not None:
        for camera in gui.findall("plugin/camera_pose"):
            camera.text = f"{initial_pose[0]-3} {initial_pose[1]-3} 8 0 .5 {yaw}"
    tree.write(output, encoding="unicode", xml_declaration=True)
    spec = dict(world=str(output), world_name="usv_training", buoys=buoy_rows, obstacles=obstacles,
                initial_pose=initial_pose, task=vars(config), seed=seed, state_dim=33, action_dim=2,
                boat_model="small_cora_float", state_source="simulator_pose_debug_not_real_localization",
                thresholds_status="initial_simulation_values_not_hardware_calibrated",
                policy_hz=10, camera_raw_hz=front_rate_hz, lighting="daylight_no_fog", aerial_camera=aerial,
                real_time_factor=real_time_factor)
    if scene:
        spec.update(scene=scene, scene_plan=scene, category=scene["category"], route=scene["route"],
                    initial_visibility=computed_visibility)
    output.with_suffix(".json").write_text(json.dumps(spec, indent=2)+"\n")
    return spec


def add_arguments(parser):
    parser.add_argument("--buoys", type=int, default=1)
    parser.add_argument("--target", choices=["-".join(x) for x in BUOY_SLOTS], default="red-round")
    parser.add_argument("--destination", choices=["-".join(x) for x in BUOY_SLOTS])
    parser.add_argument("--task", choices=["approach", "pass", "pass_then_approach"], default="approach")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--bearing-deg", type=float, default=0.)
    parser.add_argument("--distance-m", type=float, default=8.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cylinders", type=int, default=3)
    parser.add_argument("--scene-json", type=Path, help="Reproducible one-buoy collection scene plan")


def from_arguments(output, args):
    lookup = {"-".join(x): i for i, x in enumerate(BUOY_SLOTS)}
    scene_path = getattr(args, "scene_json", None)
    scene = json.loads(Path(scene_path).read_text()) if scene_path else None
    return build_world(output, count=args.buoys, target=lookup[args.target],
                       destination=None if args.destination is None else lookup[args.destination],
                       kind=args.task, side=args.side, bearing_deg=args.bearing_deg,
                       distance_m=args.distance_m, seed=args.seed, cylinders=args.cylinders,
                       aerial_camera=getattr(args, "videos", True), scene=scene,
                       front_rate_hz=getattr(args, "camera_rate_hz", 30),
                       aerial_rate_hz=getattr(args, "aerial_rate_hz", 10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output", type=Path, default=OUTPUTS/"worlds/usv_training.sdf")
    args = parser.parse_args()
    spec = from_arguments(args.output, args)
    print(json.dumps({"world": spec["world"], "buoys": len(spec["buoys"]), "task": spec["task"]}, indent=2))

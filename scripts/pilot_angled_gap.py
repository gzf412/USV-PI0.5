#!/usr/bin/env python3
"""Generate a separate, geometry-checked angled cylinder-gap pilot scene.

Example: PYTHONPATH=src /usr/bin/python3 scripts/pilot_angled_gap.py \
  --seed 12 --index 0 --output /tmp/angled_gap.json

No simulator is launched and no dataset is changed. Cylinder layout and planned
waypoints are privileged teacher/evaluation metadata, not policy observations.
"""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np

from usv_vla.collection_standard import apply_standard
from usv_vla.scene_plan import _segment_distance, validate_scene, xy_from
from usv_vla.world import ORIGIN


def generate(seed, index=0, count=2):
    if count not in (2, 3, 4):
        raise ValueError('A gap pilot requires 2 to 4 cylinders')
    rng = random.Random(seed)
    # Balance turns left/right for adjacent indexes. Offset and yaw have
    # opposite signs so the bow starts angled toward the gate, not away from it.
    side = 1 if index % 2 == 0 else -1
    for attempt in range(1200):
        route_yaw = ORIGIN[3] + rng.uniform(-.3, .3)
        angle_deg = side * rng.uniform(12., 25.)
        yaw = route_yaw + math.radians(angle_deg)
        pose = [ORIGIN[0]+rng.uniform(-4., 4.), ORIGIN[1]+rng.uniform(.5, 5.), ORIGIN[2], yaw]
        fwd = rng.uniform(10., 13.)
        # Match the incoming curve tangent to the actual initial yaw. A short
        # offset makes the controller turn sharply, then undo that correction
        # before the gate; the tangent match supports a single smooth turn.
        lateral = math.tan(math.radians(angle_deg)) * (fwd-4.)
        gap_half = rng.uniform(2.12, 2.42)
        distance = rng.uniform(22., 28.)
        center = xy_from(pose, route_yaw, fwd, lateral)
        target = xy_from(pose, route_yaw, math.sqrt(distance*distance-lateral*lateral), lateral)
        cylinders = []
        for offset in (gap_half, -gap_half):
            x, y = xy_from(pose, route_yaw, fwd, lateral+offset)
            cylinders.append(dict(id=len(cylinders)+1, x=x, y=y, radius_m=.405))
        # Quadratic Bezier meets the gate axis tangentially at the gate,
        # followed by a straight, independently scored exit strip.
        waypoints = []
        for t in (.2, .4, .6, .8, 1.):
            forward = 2*(1-t)*t*(fwd-4.) + t*t*fwd
            left = 2*(1-t)*t*lateral+t*t*lateral
            waypoints.append(xy_from(pose, route_yaw, forward, left))
        waypoints.append(xy_from(pose, route_yaw, fwd+2.6, lateral))
        gate_half = gap_half-.405-.85-.25
        gate = dict(center=center, yaw_rad=route_yaw, left_min_m=-gate_half,
                    left_max_m=gate_half, exit_forward_m=2.)
        route = dict(kind='cylinder_gap', reference_yaw_rad=route_yaw,
                     waypoints=waypoints, cruise_speed_mps=rng.uniform(.37, .50),
                     gates=[gate], cylinder_models=['usv_cylinder_1', 'usv_cylinder_2'],
                     waypoint_radius_m=.85, prompt_include_phase=False,
                     instruction=('Pass between the two initially nearest orange cylinders, '
                                  'then approach the red round buoy and stop 1.9 to 2.3 meters away.'))
        last=np.asarray(waypoints[-1]); target_v=np.asarray(target); delta=target_v-last
        stop=target_v-delta/np.linalg.norm(delta)*2.1
        points=[pose[:2], *waypoints, stop.tolist()]
        required_distance=max(math.dist(pose[:2], [c['x'], c['y']]) for c in cylinders)
        placed = True
        while len(cylinders) < count:
            found = False
            for _ in range(300):
                forward = rng.uniform(12., 21.)
                left = rng.uniform(-.63*forward, .63*forward)
                x,y = xy_from(pose, yaw, forward, left)
                center_d=[x,y]
                if (math.dist(pose[:2], center_d) <= required_distance+1. or
                    math.dist(center_d, target) < 3.4 or
                    any(math.dist(center_d, [c['x'], c['y']]) < 2.0 for c in cylinders) or
                    any(_segment_distance(center_d, a, b) < 2.3 for a,b in zip(points,points[1:]))):
                    continue
                cylinders.append(dict(id=len(cylinders)+1,x=x,y=y,radius_m=.405))
                found = True
                break
            if not found:
                placed = False
                break
        if not placed:
            continue
        scene = dict(schema_version=1, category='between', index=int(index), seed=int(seed),
                     generation_attempt=attempt, initial_pose=pose, target_xy=target,
                     cylinders=cylinders, route=route,
                     pilot_geometry=dict(initial_heading_relative_gate_deg=angle_deg,
                                         start_lateral_from_gate_axis_m=-lateral,
                                         initial_target_distance_m=distance,
                                         gate_half_spacing_m=gap_half,
                                         entry_path='quadratic_bezier_tangent_to_gate_axis'))
        scene=apply_standard(scene)
        try:
            scene['initial_visibility']=validate_scene(scene)
        except ValueError:
            continue
        return scene
    raise ValueError(f'Could not generate angled gap for seed={seed}, index={index}, count={count}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--count', type=int, choices=(2,3,4), default=2)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    scene=generate(args.seed,args.index,args.count)
    content=json.dumps(scene,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(content,encoding='utf-8')
    else:
        print(content,end='')


if __name__=='__main__':
    main()

"""Near-bank demonstrations with seven distant, non-blocking distractor buoys."""
import math
import random

import numpy as np

from usv_vla.core import BUOY_SLOTS
from usv_vla.green_scene import buoy_radius
from usv_vla.green_shore import P0, P1
from usv_vla.scene_plan import _segment_distance, shore_clearance, validate_scene, xy_from
from usv_vla.world import ORIGIN


def generate(target_slot, index, seed):
    if target_slot not in (0, 4, 6):
        raise ValueError('Only the three existing round targets are supported')
    rng = random.Random(seed)
    unit = (P1-P0)/np.linalg.norm(P1-P0)
    length = float(np.linalg.norm(P1-P0))
    base_yaw = math.atan2(unit[1], unit[0])
    color = BUOY_SLOTS[target_slot][0]
    dataset = {0:'red_multi_v1', 4:'green_multi_v1', 6:'blue_multi_v1'}[target_slot]
    errors = []
    for attempt in range(300):
        distance = rng.uniform(25.2, 28.2)
        along = rng.uniform(1., length-distance-1.)
        offset = rng.uniform(3.6, 3.9)
        start = P0 + unit*along + np.array([-unit[1], unit[0]])*offset
        target = start+unit*distance
        yaw = base_yaw
        if index % 2:
            start, target = target.copy(), start.copy()
            yaw += math.pi
        sign = -1 if index % 2 else 1
        pose = [float(start[0]), float(start[1]), ORIGIN[2], yaw+rng.uniform(-.015, .015)]
        stop = xy_from(start, yaw, distance-2.1, 0.)
        buoys = [dict(slot=target_slot, x=float(target[0]), y=float(target[1]))]
        slots = [slot for slot in range(8) if slot != target_slot]
        rng.shuffle(slots)
        try:
            for slot in slots:
                radius = buoy_radius(slot)
                for _ in range(1000):
                    x = rng.uniform(3., distance-3.)
                    lateral = sign*rng.uniform(4.7, 10.)
                    pos = xy_from(start, yaw, x, lateral)
                    if min(math.dist(pos, start), math.dist(pos, stop)) > 30.5: continue
                    if max(math.dist(pos, start), math.dist(pos, stop)) > 30.5: continue
                    if shore_clearance(pos) < 2.6: continue
                    if _segment_distance(pos, start, stop) < radius+.85+1.0: continue
                    if any(math.dist(pos, [b['x'],b['y']]) < radius+buoy_radius(b['slot'])+1.0 for b in buoys): continue
                    buoys.append(dict(slot=slot, x=pos[0], y=pos[1])); break
                else:
                    raise ValueError('cannot place seven non-blocking buoys')
            bank_side = 'right' if sign > 0 else 'left'
            phrases = [f'Follow the riverbank on your {bank_side} to the {color} round buoy and stop.',
                       f'Continue along the riverbank on your {bank_side}, approach the {color} round buoy and stop 1.9 to 2.3 meters away.',
                       f'Follow the shore to the {color} round buoy and stop safely.']
            route = dict(kind='bank_follow',reference_yaw_rad=yaw,
                         waypoints=[xy_from(start,yaw,(distance-2.7)*t,0.) for t in (.35,.70,1.)],
                         gates=[], cylinder_models=[], required_obstacle_models=[], waypoint_radius_m=.85,
                         cruise_speed_mps=rng.uniform(.64,.75),maneuver_speed_mps=.54,
                         prompt_include_phase=False,instruction=phrases[index%len(phrases)],
                         bank_corridor=dict(origin=start.tolist(),yaw_rad=yaw,left_min_m=-.65,left_max_m=.65,
                                            progress_start_m=0.,progress_end_m=distance-3.),
                         shore_reference=dict(source='sydney_regatta_shore collision mesh',offset_m=offset))
            scene = dict(schema_version=1,dataset_type=dataset,category='shore',index=index,seed=seed,
                         generation_attempt=attempt,initial_pose=pose,target_slot=target_slot,target_xy=target.tolist(),
                         buoys=buoys,cylinders=[],route=route,supplement='shore_clear_10')
            scene['initial_visibility'] = validate_scene(scene)
            return scene
        except ValueError as error:
            errors.append(str(error))
    raise ValueError('Unable to generate clear shore scene: '+str(errors[-3:]))

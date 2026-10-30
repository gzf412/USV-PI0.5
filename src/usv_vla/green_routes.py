"""Generate varied green-target multi-buoy demonstrations without running Gazebo.

Plans contain privileged teacher geometry. Only the fixed instruction, images
and buoy state are model inputs. Physical execution must be checked separately.
"""
import math
import random

from usv_vla.core import BUOY_SLOTS
from usv_vla.green_scene import buoy_radius, validate_green_scene
from usv_vla.scene_plan import _segment_distance, shore_clearance, xy_from
from usv_vla.world import ORIGIN

MODES = ('direct_visible', 'direct_offscreen', 'buoy_pass', 'buoy_gap', 'mixed_multi')


def _smooth(t):
    t = max(0., min(1., t))
    return t*t*(3.-2.*t)


def _buoy_name(slot):
    color, shape = BUOY_SLOTS[slot]
    return f'{color} {shape} buoy'


def _build(mode, index, seed, rng, attempt):
    reference = ORIGIN[3]+rng.uniform(-.13, .13)
    angle = (rng.choice((-1, 1))*rng.uniform(54., 66.) if mode == 'direct_offscreen'
             else rng.uniform(-20., 20.) if mode != 'direct_visible' else rng.uniform(-19., 19.))
    pose = [ORIGIN[0]+rng.uniform(-1.4, 1.4), ORIGIN[1]+rng.uniform(.7, 2.7), ORIGIN[2], reference+math.radians(angle)]
    distance = rng.uniform(10., 27.) if mode.startswith('direct') else rng.uniform(24., 28.8)
    target = xy_from(pose, reference, distance, 0.)
    buoys = [dict(slot=4, x=target[0], y=target[1])]
    cylinders = []
    slots = [i for i in range(8) if i != 4]
    rng.shuffle(slots)
    route = dict(kind='direct', reference_yaw_rad=reference, waypoints=[], gates=[],
                 cylinder_models=[], required_obstacle_models=[], waypoint_radius_m=.85,
                 cruise_speed_mps=rng.uniform(.60, .75), maneuver_speed_mps=.50,
                 prompt_include_phase=False)
    free = index % 3 == 0
    vague = ['Go to the green round buoy and stop.',
             'Reach the green round buoy and stop safely.',
             'Avoid obstacles as needed, approach the green round buoy, slow down and stop.']
    category = 'approach_offscreen' if mode == 'direct_offscreen' else 'approach_visible'
    instruction = vague[index % len(vague)] if index % 2 == 0 else 'Approach the green round buoy, slow down and stop 1.9 to 2.3 meters away.'

    def add_buoy(slot, f, l):
        x,y = xy_from(pose, reference, f,l)
        buoys.append(dict(slot=slot,x=x,y=y))
        return f'v2_buoy_{slot+1}'

    def add_cylinder(f,l):
        x,y=xy_from(pose,reference,f,l)
        cylinders.append(dict(id=len(cylinders)+1,x=x,y=y,radius_m=.405))
        return f'usv_cylinder_{len(cylinders)}'

    side = 1 if index % 2 == 0 else -1
    if mode == 'buoy_gap':
        category = 'between'
        f = rng.uniform(9.5,12.)
        lateral = math.tan(math.radians(angle))*(f-4.)*.70
        half = rng.uniform(2.25,2.5)
        pair = [slots.pop(), slots.pop()]
        names = [add_buoy(pair[0],f,lateral+half),add_buoy(pair[1],f,lateral-half)]
        center=xy_from(pose,reference,f,lateral)
        ghalf=half-max(buoy_radius(i) for i in pair)-.85-.25
        route.update(kind='cylinder_gap',required_obstacle_models=names,
                     gates=[dict(center=center,yaw_rad=reference,left_min_m=-ghalf,left_max_m=ghalf,exit_forward_m=2.)])
        # Smooth quadratic entry tangent to the straight crossing line.
        route['waypoints']=[xy_from(pose,reference,2*(1-t)*t*(f-4.)+t*t*f,2*(1-t)*t*lateral+t*t*lateral)
                            for t in (.2,.4,.6,.8,1.)]+[xy_from(pose,reference,f+2.6,lateral)]
        if not free:
            instruction = f'Pass between the {_buoy_name(pair[0])} and the {_buoy_name(pair[1])}, then approach the green round buoy and stop.'
    elif mode in ('buoy_pass','mixed_multi'):
        category = 'around_left' if side == 1 else 'around_right'
        count = 1 if mode == 'buoy_pass' else 2+(index % 2)
        if count == 3: distance = rng.uniform(28.,28.8); target = xy_from(pose,reference,distance,0.); buoys[0].update(x=target[0],y=target[1])
        fractions = [.29,.66] if count == 2 else ([.20,.50,.80] if count == 3 else [.38])
        xs=[distance*(fraction+rng.uniform(-.009,.009)) for fraction in fractions]
        entries=[x-2. for x in xs]; exits=[x+2.5 for x in xs]
        crossings=[]; gates=[]; names=[]; identities=[]
        for i,x in enumerate(xs):
            sign=side*(-1)**i
            object_left=-sign*rng.uniform(1.23,1.32)
            if mode == 'mixed_multi' and i % 2 == 1:
                name=add_cylinder(x,object_left); label='orange cylinder'; radius=.405
            else:
                slot=slots.pop(); name=add_buoy(slot,x,object_left); label=_buoy_name(slot); radius=buoy_radius(slot)
            crossing=sign*(max(2.15,radius+.85+.4)-abs(object_left))
            crossings.append(crossing); names.append(name); identities.append((label,'left' if sign==1 else 'right'))
            center=xy_from(pose,reference,x,object_left)
            low,high=(2.,3.7) if sign==1 else (-3.7,-2.)
            gates.append(dict(center=center,yaw_rad=reference,left_min_m=low,left_max_m=high,exit_forward_m=2.))
        def left_at(x):
            if x<entries[0]: return crossings[0]*_smooth(x/entries[0])
            for j in range(count):
                if x<=exits[j]: return crossings[j]
                if j+1<count and x<entries[j+1]: return crossings[j]+(crossings[j+1]-crossings[j])*_smooth((x-exits[j])/(entries[j+1]-exits[j]))
            return crossings[-1]
        n=math.ceil(exits[-1]/.4)
        points=sorted({exits[-1]*i/n for i in range(1,n+1)}|set(entries)|set(xs)|set(exits))
        route.update(kind='cylinder_pass',side='left' if side==1 else 'right',gates=gates,
                     required_obstacle_models=names,
                     cylinder_models=[name for name in names if name.startswith('usv_cylinder_')],
                     waypoints=[xy_from(pose,reference,x,left_at(x)) for x in points])
        if not free:
            sequence=', then '.join(f'pass on the {s} side of the {label}' for label,s in identities)
            instruction=f'{sequence.capitalize()}, then approach the green round buoy and stop. Left and right refer to the initial direction to the target.'
    route['instruction']=instruction
    # Fill with separated, non-blocking buoy/cylinder distractors. Number and
    # arrangement change across episodes; all training-position distances are
    # bounded from all path vertices, not just from the initial boat position.
    target_n=1+index%8
    target_n=max(target_n,len(buoys))
    cylinder_n=1+(index//2)%4
    cylinder_n=max(cylinder_n,len(cylinders))
    last=route['waypoints'][-1] if route['waypoints'] else pose[:2]
    vx,vy=target[0]-last[0],target[1]-last[1]; norm=math.hypot(vx,vy)
    stop=[target[0]-vx/norm*2.1,target[1]-vy/norm*2.1]
    points=[pose[:2],*route['waypoints'],stop]
    def objects():
        return [(b['x'],b['y'],buoy_radius(b['slot'])) for b in buoys]+[(c['x'],c['y'],.405) for c in cylinders]
    def place(radius, is_buoy):
        for _ in range(500):
            f=rng.uniform(4.,min(distance-3.,24.)); l=rng.choice((-1,1))*rng.uniform(3.5,9.)
            x,y=xy_from(pose,reference,f,l)
            if shore_clearance([x,y])<2.6: continue
            if any(math.dist([x,y],[ox,oy]) < radius+r+1.0 for ox,oy,r in objects()): continue
            if any(_segment_distance([x,y],a,b)<radius+.85+.75 for a,b in zip(points,points[1:])): continue
            if is_buoy and any(math.dist([x,y],p)>30.8 for p in points): continue
            return x,y
        raise ValueError('Could not place a safe distractor')
    while len(buoys)<target_n:
        slot=slots.pop(); x,y=place(buoy_radius(slot),True); buoys.append(dict(slot=slot,x=x,y=y))
    while len(cylinders)<cylinder_n:
        x,y=place(.405,False); cylinders.append(dict(id=len(cylinders)+1,x=x,y=y,radius_m=.405))
    scene=dict(schema_version=1,dataset_type='green_multi_v1',category=category,index=int(index),seed=int(seed),
               generation_attempt=attempt,initial_pose=pose,target_slot=4,target_xy=target,buoys=buoys,
               cylinders=cylinders,route=route,green_mode=mode)
    scene['initial_visibility']=validate_green_scene(scene)
    return scene


def generate_scene(mode, index, seed):
    """Deterministically generate a preflighted scene; no simulation is run."""
    if mode not in MODES:
        raise ValueError(f'Unknown green scene mode: {mode}')
    rng=random.Random(int(seed)); errors=[]
    for attempt in range(500):
        try:
            return _build(mode,int(index),int(seed),rng,attempt)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(f'Could not generate {mode} scene seed={seed}, index={index}: {errors[-1]}')

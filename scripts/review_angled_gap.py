#!/usr/bin/env python3
"""Independently measure an angled-gap pilot using recorded physical poses.

Computes an interpolated gate crossing and conservative circular hull-to-object
clearance along the linear interpolation between 10 Hz poses. This is a
simulation-only bounding approximation, not an exact hull geometry measurement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from usv_vla.route import VESSEL_RADIUS_M
from usv_vla.scene_plan import initial_visibility, _segment_distance
from usv_vla.trajectory_quality import inspect_trajectory_quality


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def local(gate, xy):
    x=xy[0]-gate['center'][0]; y=xy[1]-gate['center'][1]; c=math.cos(gate['yaw_rad']); s=math.sin(gate['yaw_rad'])
    return (c*x+s*y,-s*x+c*y)


def review(episode):
    episode=Path(episode).resolve()
    metadata=json.loads((episode/'episode.json').read_text())
    rows=[json.loads(line) for line in (episode/'steps.jsonl').read_text().splitlines() if line.strip()]
    scene=metadata['scene_plan']; route=scene['route']; gates=route['gates']
    if route['kind'] != 'cylinder_gap' or len(gates)!=1 or len(route['cylinder_models'])!=2 or len(rows)<2:
        raise ValueError('Expected one cylinder-gap gate, two named cylinders and >=2 samples')
    gate=gates[0]; first=rows[0]; last=rows[-1]; initial=first['pose_xyzrpy']; xy0=initial[:2]
    task_cylinders=[]
    for name in route['cylinder_models']:
        matches=[x for x in metadata['obstacles'] if x['model']==name]
        if len(matches)!=1:
            raise ValueError(f'Missing or duplicated obstacle {name}')
        task_cylinders.append(matches[0])
    crossing=None
    for i,(a,b) in enumerate(zip(rows,rows[1:])):
        af,al=local(gate,a['pose_xyzrpy'][:2]); bf,bl=local(gate,b['pose_xyzrpy'][:2])
        if af<0<=bf:
            f=-af/(bf-af); ap=a['pose_xyzrpy']; bp=b['pose_xyzrpy']
            yaw=ap[5]+f*wrap(bp[5]-ap[5]); xy=[ap[j]+f*(bp[j]-ap[j]) for j in (0,1)]
            crossing=dict(before_index=i, interpolation_fraction=f, xy=xy, lateral_m=al+f*(bl-al),
                          heading_relative_gate_deg=math.degrees(wrap(yaw-gate['yaw_rad'])),
                          elapsed_s=(a['decision_sim_time_ns']+f*(b['decision_sim_time_ns']-a['decision_sim_time_ns'])-first['decision_sim_time_ns'])/1e9)
            break
    clearances=[]
    for c in task_cylinders:
        center=[c['x'],c['y']]
        minimum=min(_segment_distance(center,a['pose_xyzrpy'][:2],b['pose_xyzrpy'][:2]) for a,b in zip(rows,rows[1:]))
        clearances.append(dict(model=c['model'], minimum_center_distance_m=minimum,
                               conservative_minimum_hull_clearance_m=minimum-c['radius_m']-VESSEL_RADIUS_M,
                               at_crossing_hull_clearance_m=(math.dist(center,crossing['xy'])-c['radius_m']-VESSEL_RADIUS_M) if crossing else None))
    visibility=initial_visibility(scene,[initial[0],initial[1],initial[2],initial[5]])
    angle=math.degrees(wrap(initial[5]-gate['yaw_rad'])); forward,lateral=local(gate,xy0)
    target_dist=math.dist(xy0,scene['target_xy'])
    prompts=set(row['prompt'] for row in rows)
    quality=inspect_trajectory_quality(episode)
    result={'schema_version':1, 'episode':str(episode),
            'steps_sha256':hashlib.sha256((episode/'steps.jsonl').read_bytes()).hexdigest(),
            'metadata_sha256':hashlib.sha256((episode/'episode.json').read_bytes()).hexdigest(),
            'hull_clearance_method':'0.85 m bounding circle; linearly interpolated consecutive 10 Hz center positions; not exact hull distance',
            'initial_heading_relative_gate_deg':angle,'initial_lateral_from_gate_axis_m':lateral,
            'initial_forward_from_gate_m':forward,'initial_target_distance_m':target_dist,
            'crossing':crossing,'cylinders':clearances,'initial_visibility':visibility,
            'unique_prompt_count':len(prompts),'prompt':next(iter(prompts)) if len(prompts)==1 else None,
            'quality_passed':quality['passed'],'quality_issues':quality['issues'],
            'final_success':last.get('success'), 'final_route_gates_completed':last.get('route_gates_completed'),
            'issues':[]}
    issues=result['issues']
    if not 12-1e-5 <= abs(angle) <= 25+1e-5: issues.append('Initial measured heading does not fall within 12 to 25 degrees')
    if not 1. <= abs(lateral) <= 4.5: issues.append('Initial measured lateral offset outside 1 to 4.5 meters')
    if forward>=-.5: issues.append('Initial pose is not before the gate')
    if not 22.-1e-5<=target_dist<=28.+1e-5: issues.append('Initial measured target distance outside 22 to 28 meters')
    if crossing is None: issues.append('No physical forward crossing')
    elif not gate['left_min_m']<=crossing['lateral_m']<=gate['left_max_m']: issues.append('Physical crossing is outside the protected gate')
    elif abs(crossing['heading_relative_gate_deg'])>10.: issues.append('Physical gate heading differs from gate axis by over 10 degrees')
    if any(c['conservative_minimum_hull_clearance_m']<metadata['task']['minimum_clearance_m'] for c in clearances): issues.append('Conservative physical cylinder clearance below task limit')
    if len(prompts)!=1 or next(iter(prompts),None)!=route['instruction']: issues.append('Task instruction changed or contains a privileged phase')
    if not all(c['fully_visible'] for c in visibility['cylinders']): issues.append('A cylinder is not fully visible at actual initial pose')
    if not quality['passed']: issues.append('Independent trajectory quality failed')
    if not (last.get('success') is True and last.get('route_gates_completed')==1 and last.get('route_complete') is True): issues.append('Task/gate did not complete successfully')
    result['passed']=not issues
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=review(args.episode)
    data=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output: args.output.write_text(data,encoding='utf-8')
    else: print(data,end='')
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

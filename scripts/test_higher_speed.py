#!/usr/bin/env python3
"""Paired green-buoy simulation speed trials; never modifies the source batch."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from usv_vla import batch_collect as bc
from usv_vla.route import VESSEL_RADIUS_M, _segment_distance
from usv_vla.scene_plan import validate_scene
from usv_vla.trajectory_quality import inspect_trajectory_quality
from collect_green_batch import validate_recording

ROOT = bc.OUTPUTS / 'episodes'
BASE = ROOT / 'green_round_200_20261028'
SELECTED = [('direct', 1), ('buoy_pass', 3), ('buoy_gap', 4), ('mixed_multi', 5), ('shore', 12)]


def read(path):
    return json.loads(Path(path).read_text())


def measures(directory):
    rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines() if line.strip()]
    meta = read(directory/'episode.json'); summary = read(directory/'summary.json')
    pts = [np.asarray(r['pose_xyzrpy'][:2]) for r in rows]
    speed = np.asarray([math.hypot(r['state'][0], r['state'][1]) for r in rows])
    path = sum(float(np.linalg.norm(b-a)) for a,b in zip(pts,pts[1:]))
    seconds = (rows[-1]['decision_sim_time_ns']-rows[0]['decision_sim_time_ns'])/1e9
    non_target = [o for o in meta['obstacles'] if o['model'] != 'v2_buoy_5']
    clearances = [min(_segment_distance(a,b,np.asarray([o['x'],o['y']])) - VESSEL_RADIUS_M-o['radius_m']
                      for a,b in zip(pts,pts[1:])) for o in non_target]
    actions = np.asarray([r['action'] for r in rows])
    return dict(path_m=path, duration_s=seconds, mean_mps=path/seconds,
                speed_p90_mps=float(np.quantile(speed,.9)), speed_peak_mps=float(speed.max()),
                min_non_target_hull_clearance_m=min(clearances) if clearances else None,
                peak_thrust_N=float(np.abs(actions).max()), final_distance_m=summary.get('distance_m'),
                success=summary.get('success'), route_gates_completed=summary.get('route_gates_completed'),
                route_gate_count=summary.get('route_gate_count'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--cruise', type=float, default=.95)
    parser.add_argument('--maneuver', type=float, default=.80)
    parser.add_argument('--lookahead', type=float)
    parser.add_argument('--hold-until-bank-recovery', action='store_true')
    parser.add_argument('--shore-id', type=int, default=12,
                        help='Source green batch episode for the shore trial')
    parser.add_argument('--shore-detour-scale', type=float, default=1.,
                        help='Scale lateral offsets of shore trial waypoints only')
    parser.add_argument('--shore-long-arc', action='store_true',
                        help='Use an elongated smooth arc for the shore trial')
    parser.add_argument('--modes', nargs='+', choices=[item[0] for item in SELECTED],
                        default=[item[0] for item in SELECTED])
    args = parser.parse_args()
    if not .75 < args.cruise <= 1.05 or not .15 <= args.maneuver <= args.cruise:
        parser.error('Require .75 < cruise <= 1.05 and .15 <= maneuver <= cruise')
    if args.lookahead is not None and not 1.5 <= args.lookahead <= 4.:
        parser.error('Trial lookahead must be 1.5..4 meters')
    if not .7 <= args.shore_detour_scale <= 1.:
        parser.error('Trial shore detour scale must be .7..1')
    root = args.output.resolve()
    if root.parent != ROOT.resolve() or not root.name.startswith('speed_trial_'):
        parser.error('Output must be a new speed_trial_ directory under outputs/episodes')
    root.mkdir(parents=True, exist_ok=False)
    (root/'plans').mkdir()
    manifest = read(BASE/'manifest.json')
    result = dict(profile='simulation_speed_trial_v1', target='green-round',
                  commanded_cruise_mps=args.cruise, commanded_maneuver_mps=args.maneuver,
                  data_kind='scripted_simulation_trial_not_trained_vla', cases=[])
    for mode, number in SELECTED:
        if mode not in args.modes:
            continue
        if mode == 'shore':
            number = args.shore_id
        entry = next(e for e in manifest['entries'] if e['episode_id'].startswith(f'episode_{number:04d}_'))
        if mode == 'shore' and entry['mode'] != 'shore':
            raise ValueError('Selected shore source is not a shore episode')
        original = read(BASE/entry['scene'])
        scene = json.loads(json.dumps(original))
        scene.pop('collection', None)
        if mode == 'shore' and (args.shore_detour_scale != 1. or args.shore_long_arc):
            bank = scene['shore_obstacle']
            ox, oy = bank['baseline_origin']
            yaw = bank['yaw_rad']
            sx, sy = -math.sin(yaw), math.cos(yaw)
            if args.shore_long_arc:
                laterals = [(p[0]-ox)*sx+(p[1]-oy)*sy for p in scene['route']['waypoints']]
                amplitude = max(laterals, key=abs)*args.shore_detour_scale
                gate = bank['gate_forward_m']
                end = bank['detour_window_m'][1]
                entry_end, exit_start = gate-1.7, gate+2.5
            for point in scene['route']['waypoints']:
                lateral = (point[0]-ox)*sx + (point[1]-oy)*sy
                if args.shore_long_arc:
                    forward = (point[0]-ox)*math.cos(yaw)+(point[1]-oy)*math.sin(yaw)
                    if forward <= entry_end:
                        t = forward/entry_end
                    elif forward <= exit_start:
                        t = 1.
                    else:
                        t = (end-forward)/(end-exit_start)
                    t = max(0., min(1., t))
                    new_lateral = amplitude*t*t*(3.-2.*t)
                else:
                    new_lateral = args.shore_detour_scale*lateral
                point[0] += (new_lateral-lateral)*sx
                point[1] += (new_lateral-lateral)*sy
        scene['route'].update(speed_profile='simulation_speed_trial_v1', cruise_speed_mps=args.cruise,
                              maneuver_speed_mps=args.maneuver)
        if args.lookahead is not None:
            scene['route']['speed_trial_lookahead_m'] = args.lookahead
        if args.hold_until_bank_recovery and mode == 'shore':
            bank = scene['shore_obstacle']
            scene['route']['speed_trial_bank_recovery'] = dict(
                origin=bank['baseline_origin'], yaw_rad=bank['yaw_rad'],
                end_forward_m=bank['detour_window_m'][1])
        scene['speed_trial'] = dict(source_episode=entry['accepted_episode'], original_scene_sha256=entry['scene_sha256'],
                                    original_cruise_mps=original['route']['cruise_speed_mps'],
                                    original_maneuver_mps=original['route'].get('maneuver_speed_mps'))
        scene['initial_visibility'] = validate_scene(scene)
        plan = root/'plans'/f'{mode}.json'; bc._atomic_json(plan, scene)
        episode = root/mode; log = root/f'{mode}.recorder.log'
        cmd = bc.record_command(plan, episode, ffmpeg=args.ffmpeg, episode_wall_s=240.)
        cmd = ['bash', str(bc.PROJECT_ROOT/'scripts/sim.sh'), 'record', *cmd[3:]]
        started = time.monotonic()
        code = bc.run_recorder(cmd, log, 360.)
        summary = read(episode/'summary.json')
        if not summary.get('published_zero_on_exit') or not summary.get('processes_stopped'):
            raise RuntimeError(f'Simulator teardown unconfirmed for {mode}')
        case = dict(mode=mode, episode=str(episode), baseline_episode=entry['accepted_episode'],
                    baseline=measures(BASE/entry['accepted_episode']), measured=measures(episode),
                    wall_s=time.monotonic()-started, recorder_returncode=code)
        try:
            if code: raise ValueError(f'Recorder returned {code}')
            proof = validate_recording(episode, scene, args.ffmpeg, 'shore' if mode == 'shore' else mode)
            case.update(passed=True, gate_crossings=proof['gate_crossings'], shore_review=proof['shore_review'])
        except (ValueError, OSError, KeyError, TypeError) as error:
            case.update(passed=False, rejection=f'{type(error).__name__}: {error}',
                        quality=inspect_trajectory_quality(episode))
        result['cases'].append(case)
        bc._atomic_json(root/'results.json', result)
        print(json.dumps({'mode':mode, 'passed':case['passed'], 'mean_mps':case['measured']['mean_mps'],
                          'baseline_mean_mps':case['baseline']['mean_mps'], 'error':case.get('rejection')}, ensure_ascii=False), flush=True)
    print(json.dumps({'finished':len(result['cases']), 'passed':sum(c['passed'] for c in result['cases']),
                      'output':str(root)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

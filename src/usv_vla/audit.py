"""Offline checks for causal timestamps, executed commands and image integrity."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _selected_target(metadata):
    matches = [buoy for buoy in metadata["buoys"] if buoy["slot"] == metadata["task"]["target"]]
    if len(matches) != 1:
        raise ValueError("Expected exactly one buoy matching task target slot")
    return matches[0]


def audit_episode(directory, *, check_images=True):
    directory=Path(directory).resolve()
    metadata=json.loads((directory/'episode.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
    if not rows or not summary['recording_valid']:
        raise ValueError(f'Invalid/empty recording: {directory}')
    previous_action=np.zeros(2)
    previous_decision=None
    gaps=[]
    for index,row in enumerate(rows):
        if row['index']!=index:
            raise ValueError('Row index discontinuity')
        state=np.asarray(row['state'])
        action=np.asarray(row['action'])
        if state.shape!=(33,) or action.shape!=(2,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('State/action shape or finite-value error')
        if np.any(np.abs(action)>15):
            raise ValueError('Action exceeds configured thrust limit')
        np.testing.assert_allclose(state[7:9],previous_action/15,atol=1e-6)
        slots=state[9:].reshape(8,3)
        if not np.isin(slots[:,2],[0,1]).all() or np.any(slots[slots[:,2]==0,:2]!=0):
            raise ValueError('Invalid absence/presence encoding')
        for source in row['sources'].values():
            if source['source_time_ns']>row['decision_sim_time_ns']:
                raise ValueError('Future sensor source leaked into observation')
            if source['arrival_monotonic_ns']>row['decision_monotonic_ns']:
                raise ValueError('Late-arriving sensor leaked into observation')
        if row['action_publish_monotonic_ns']<row['decision_monotonic_ns']:
            raise ValueError('Action predates observation decision')
        if metadata.get('decision_schedule') == 'causal_10hz_ticks_max_50ms_lateness':
            lateness = row['decision_latest_clock_sim_time_ns']-row['decision_sim_time_ns']
            if lateness != row['decision_tick_lateness_ns'] or not 0 <= lateness <= 50_000_000:
                raise ValueError('Decision scheduling exceeded its recorded lateness budget')
        if previous_decision is not None:
            gap=(row['decision_sim_time_ns']-previous_decision)/1e9
            if gap<=0:
                raise ValueError('Non-monotonic decision time')
            gaps.append(gap)
        image=directory/row['image']
        if not image.is_file():
            raise ValueError('Missing image')
        if check_images:
            with Image.open(image) as opened:
                if opened.size!=(1280,720):
                    raise ValueError('Unexpected image dimensions')
                opened.verify()
        previous_action,previous_decision=action,row['decision_sim_time_ns']
    if summary['steps']!=len(rows):
        raise ValueError('Summary count does not match recording')
    route_replayed = False
    if metadata.get('route'):
        from usv_vla.core import TaskConfig
        from usv_vla.record import conservative_clearance
        from usv_vla.route import RouteTaskEvaluator
        from usv_vla.scene_plan import initial_visibility, validate_scene
        scene = metadata['scene_plan']
        validate_scene(scene)
        first_pose = rows[0]['pose_xyzrpy']
        visibility = initial_visibility(scene, [*first_pose[:3], first_pose[5]])
        if scene.get('dataset_type') in ('green_multi_v1', 'red_multi_v1', 'blue_multi_v1'):
            by_model = {item['model']: item for item in visibility['buoys'] + visibility['cylinders']}
            required = set(metadata['route'].get('required_obstacle_models', [])) | set(metadata['route'].get('cylinder_models', []))
            if any(model not in by_model or not by_model[model]['unobstructed_visible'] for model in required):
                raise ValueError('Required obstacles are not unobstructed in the measured initial camera view')
            if scene['category'] == 'approach_visible' and not visibility['target']['unobstructed_visible']:
                raise ValueError('Target is not unobstructed in the measured initial camera view')
            if scene['category'] == 'approach_offscreen' and not visibility['target']['entirely_outside_horizontal']:
                raise ValueError('Target is not offscreen in the measured initial camera view')
        elif metadata['route']['kind'] in ('cylinder_pass', 'cylinder_gap'):
            if not all(item['fully_visible'] for item in visibility['cylinders']):
                raise ValueError('Task cylinders are outside the measured initial camera view')
        target = _selected_target(metadata)
        dx, dy = target['x']-first_pose[0], target['y']-first_pose[1]
        import math
        if math.cos(first_pose[5])*dx+math.sin(first_pose[5])*dy <= 0:
            raise ValueError('Target is not in the initial forward half-plane')
        task = RouteTaskEvaluator(TaskConfig(**metadata['task']),
            {item['slot']:(item['x'],item['y']) for item in metadata['buoys']},
            first_pose[:2], first_pose[5], route=metadata['route'], obstacles=metadata['obstacles'])
        previous_xy = None
        for row in rows:
            xy = row['pose_xyzrpy'][:2]
            clearance = conservative_clearance(xy, previous_xy, metadata['obstacles'])
            if previous_xy is not None and abs(row['clearance_m']-clearance)>1e-5:
                raise ValueError('Saved obstacle clearance differs from the recorded trajectory')
            result = task.update(row['sources']['pose']['source_time_ns']/1e9, xy,
                float(np.linalg.norm(row['state'][:2])), clearance_m=clearance)
            for key in ('phase','success','done','route_complete','route_gates_completed'):
                if result[key] != row[key]:
                    raise ValueError(f'Independent route replay differs at row {row["index"]}: {key}')
            if row['prompt'] != task.context():
                raise ValueError('Saved instruction does not match its actual task configuration and phase')
            previous_xy = xy
        if bool(result['success']) != bool(summary['success']):
            raise ValueError('Route replay does not match summary success')
        route_replayed = True
    return dict(episode=str(directory),steps=len(rows),success=summary['success'],
                causal_alignment=True,previous_command_alignment=True,
                route_replayed=route_replayed,
                decision_gap_s=[min(gaps),max(gaps)] if gaps else [],
                state_source=metadata['state_source'],controller=metadata['controller'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',nargs='+')
    args=parser.parse_args()
    print(json.dumps([audit_episode(path) for path in args.episodes],indent=2))

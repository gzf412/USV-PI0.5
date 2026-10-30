#!/usr/bin/env python3
"""Safely replace selected red-target scenes with real multi-buoy recordings."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path

from usv_vla import batch_collect as bc
from usv_vla.red_multi_scene import generate_scene
from usv_vla.route import _gate_coordinates
from usv_vla.core import wrap
from usv_vla.scene_plan import validate_scene
import replace_complex_episodes as repl

VERSION = 'red_multi_buoy_v1'
COUNT = 20


def choose_entries(root, manifest):
    override = repl.read(root/'instruction_overrides.json')
    protected = {x['episode_id'] for x in override['entries']}
    selected = []
    for category, quotas in {'between': {'train': 6, 'validation': 2},
                             'around_left': {'train': 5, 'validation': 1},
                             'around_right': {'train': 5, 'validation': 1}}.items():
        for split, count in quotas.items():
            options = []
            for entry in manifest['entries']:
                if entry['category'] != category or entry['split'] != split or entry['episode_id'] in protected:
                    continue
                scene = repl.read(root/entry['scene'])
                distance = math.dist(scene['initial_pose'][:2], scene['target_xy'])
                options.append((distance, entry['episode_id'], entry))
            if len(options) < count:
                raise ValueError('Insufficient replaceable scenes')
            selected.extend((entry, {'initial_target_distance_m': distance, 'ranking': 'shortest within category and split'})
                            for distance, _, entry in sorted(options)[:count])
    return selected


def make_scene(entry, variant, nonce=0):
    category = entry['category']
    mode = 'buoy_gap' if category == 'between' else ('mixed_multi' if variant % 2 else 'buoy_pass')
    index = 2*variant + (1 if category == 'around_right' else 0)
    # Every sixth instruction is short; others specify the obstacle sequence.
    for offset in range(50):
        geometry_seed = bc._scene_seed(entry['seed'], f'{VERSION}-{nonce}-{mode}', offset)
        try:
            scene = generate_scene(mode, index, geometry_seed)
            scene['seed'] = entry['seed']
            scene['index'] = entry['category_index']
            scene['replacement_geometry'] = {'geometry_seed': geometry_seed, 'mode': mode, 'nonce': nonce}
            scene['collection'] = {'episode_id': entry['episode_id'], 'category': category,
                                   'category_index': entry['category_index'], 'seed': entry['seed'],
                                   'split': entry['split']}
            if scene['category'] != category:
                raise ValueError('Scene category mismatch')
            scene['initial_visibility'] = validate_scene(scene)
            return scene
        except ValueError:
            continue
    raise ValueError('Unable to generate safe red multi-buoy scene')


def review(directory):
    directory = Path(directory)
    meta = repl.read(directory/'episode.json')
    rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
    crossings, issues = [], []
    for gate in meta['route'].get('gates', []):
        found = []
        for a, b in zip(rows, rows[1:]):
            af, al = _gate_coordinates(gate, a['pose_xyzrpy'][:2])
            bf, bl = _gate_coordinates(gate, b['pose_xyzrpy'][:2])
            if af < 0 <= bf:
                t = -af/(bf-af)
                yaw = a['pose_xyzrpy'][5]+t*wrap(b['pose_xyzrpy'][5]-a['pose_xyzrpy'][5])
                found.append({'lateral_m': al+t*(bl-al), 'heading_deg': math.degrees(wrap(yaw-gate['yaw_rad']))})
        if len(found) != 1 or not gate['left_min_m'] <= found[0]['lateral_m'] <= gate['left_max_m'] or abs(found[0]['heading_deg']) > 10:
            issues.append('Invalid gate crossing or heading')
        crossings.extend(found)
    sha = lambda name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
    return {'passed': not issues, 'issues': issues, 'gate_crossings': crossings,
            'steps_sha256': sha('steps.jsonl'), 'metadata_sha256': sha('episode.json'),
            'inputs_sha256': {name: sha(name) for name in ('episode.json', 'steps.jsonl')}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=COUNT)
    p.add_argument('--replan-failed', action='store_true')
    p.add_argument('--ffmpeg', required=True)
    p.add_argument('--episode-wall-s', type=float, default=240.)
    p.add_argument('--min-free-gib', type=float, default=3.)
    args = p.parse_args()
    args.retry_failed = False
    if not 1 <= args.limit <= COUNT:
        p.error('limit must be between 1 and 20')
    root = repl.ROOT.resolve()
    path = root/f'{VERSION}_journal.json'
    repl.VERSION = VERSION
    repl.choose_entries = choose_entries
    repl.make_complex_scene = make_scene
    repl.review_gap = repl.review_multi = review
    # The standard journal constructor assumes 40 candidates.
    with (root/'.batch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = repl.read(root/'manifest.json')
        bc._validate_manifest(root, manifest, manifest['seed'])
        if path.exists():
            journal = repl.read(path)
        else:
            entries = choose_entries(root, manifest)
            if len(entries) != COUNT: raise ValueError('Selection count mismatch')
            plans = root/'plans'/VERSION
            plans.mkdir(parents=True, exist_ok=True)
            items = []
            for variant, (entry, selection) in enumerate(entries):
                scene = make_scene(entry, variant)
                relplan = str(Path('plans')/VERSION/(entry['episode_id']+'.json'))
                bc._atomic_json(root/relplan, scene)
                number = max((int(a['number']) for a in entry['attempts']), default=0)+1
                candidate = f"{entry['episode_id']}/attempt_{number:02d}"
                if repl.safe_path(root,candidate,entry['episode_id']).exists(): raise ValueError('Candidate exists')
                items.append({'episode_id':entry['episode_id'], 'stage':'prepared', 'selection':selection,
                              'old_entry':deepcopy(entry), 'variant':variant, 'new_scene':relplan,
                              'new_scene_sha256':bc._digest(scene), 'number':number, 'candidate':candidate,
                              'candidate_log':candidate+'.recorder.log', 'failed_attempts':[]})
            journal = {'version':VERSION, 'root':str(root), 'stage':'collecting', 'items':items, 'created_at':bc._now()}
            repl.save(path,journal)
        if journal['version'] != VERSION or len(journal['items']) != COUNT: raise ValueError('Invalid journal')
        ffmpeg = bc.resolve_ffmpeg(args.ffmpeg)
        if journal['stage'] == 'collecting':
            for item in journal['items'][:args.limit]:
                repl.record_item(root,path,journal,item,manifest,args,ffmpeg)
            if not all(x['stage']=='validated' for x in journal['items']):
                print(json.dumps({'validated':sum(x['stage']=='validated' for x in journal['items']), 'originals_retained':True}),flush=True)
                return
        if journal['stage'] in ('collecting','commit_pending','verifying_commit'):
            repl.commit(root,path,journal,manifest)
        if journal['stage']=='cleanup':
            repl.cleanup(root,path,journal,manifest)


if __name__ == '__main__':
    main()

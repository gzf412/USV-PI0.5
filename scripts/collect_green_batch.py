#!/usr/bin/env python3
"""Record and independently audit a new green-round, multi-buoy batch.

The old red-target batch is never modified. Plans and progress are atomic and
resumable; every accepted recording retains its two review videos. Scene
geometry is teacher-only information, not a policy observation.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
import re
import tarfile
import tempfile
from pathlib import Path
import shutil

import numpy as np

from usv_vla import batch_collect as bc
from usv_vla.audit import audit_episode
from usv_vla.core import BUOY_SLOTS, body_xy, wrap
from usv_vla.route import _gate_coordinates
from usv_vla.scene_plan import initial_visibility, validate_scene
from usv_vla.trajectory_quality import require_trajectory_quality
from usv_vla.green_routes import generate_scene as generate_route
from usv_vla.green_shore import generate_green_shore, review_green_shore
from usv_vla.green_scene import buoy_radius
from usv_vla.training_selection import all_accepted_for_training, selected, accepted_supplements

ROOT = bc.OUTPUTS / 'episodes/green_round_200_20261028'
QUOTAS = {'direct_visible': 45, 'direct_offscreen': 20, 'buoy_pass': 35,
          'buoy_gap': 35, 'mixed_multi': 25, 'shore': 40}
SEED = 20261028
DATASET_TYPE = 'green_multi_v1'
TARGET_SLOT = 4
TARGET_NAME = 'green-round'
ROOT_PREFIX = 'green_'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(root, manifest):
    manifest['updated_at'] = bc._now()
    manifest['accepted_count'] = sum(x['status'] == 'accepted' for x in manifest['entries'])
    bc._atomic_json(root / 'manifest.json', manifest)
    accepted = [e for e in manifest['entries'] if e['status'] == 'accepted']
    all_for_training = all_accepted_for_training(root)
    lists = {split + '_episodes': selected(accepted, split, all_for_training=all_for_training)
             for split in ('accepted', 'train', 'validation')}
    splits = root / 'splits'
    splits.mkdir(exist_ok=True)
    for name, rows in lists.items():
        paths = [e['accepted_episode'] for e in rows]
        if name == 'accepted_episodes' or (name == 'train_episodes' and all_for_training):
            paths += accepted_supplements(root)
        (splits / (name + '.txt')).write_text(''.join(path+'\n' for path in paths))


def make_scene(mode, index, seed):
    scene = generate_green_shore(index, seed) if mode == 'shore' else generate_route(mode, index, seed)
    if scene.get('dataset_type') != DATASET_TYPE or scene.get('target_slot') != TARGET_SLOT:
        raise ValueError('Generator returned the wrong dataset or target slot')
    scene.setdefault('collection', {})
    return scene


def create_manifest(root):
    root.mkdir(parents=True, exist_ok=True)
    plans = root / 'plans'
    plans.mkdir(exist_ok=True)
    counts = Counter()
    entries = []
    # Interleave task families for useful coverage even in early pilots.
    for index in range(max(QUOTAS.values())):
        for mode, quota in QUOTAS.items():
            if index >= quota:
                continue
            number = len(entries) + 1
            seed = bc._scene_seed(SEED, mode, index)
            split = 'validation' if index % 5 == 0 else 'train'
            episode_id = f'episode_{number:04d}_{mode}'
            scene = make_scene(mode, index, seed)
            scene['collection'] = dict(episode_id=episode_id, mode=mode, category=scene['category'],
                                       category_index=index, seed=seed, split=split)
            validate_scene(scene)
            path = plans / (episode_id + '.json')
            if path.exists() and bc._digest(read(path)) != bc._digest(scene):
                raise ValueError('Existing plan conflicts: ' + str(path))
            if not path.exists():
                bc._atomic_json(path, scene)
            entries.append(dict(episode_id=episode_id, mode=mode, index=index, seed=seed,
                                split=split, category=scene['category'], scene=str(path.relative_to(root)),
                                scene_sha256=bc._digest(scene), status='planned', attempts=[]))
            counts[mode] += 1
    if dict(counts) != QUOTAS or sum(e['split'] == 'validation' for e in entries) != 40:
        raise ValueError('Quota or split error in generated plans')
    manifest = dict(schema_version=1, dataset_type=DATASET_TYPE, target_slot=TARGET_SLOT,
                    seed=SEED, quotas=QUOTAS, root=str(root), created_at=bc._now(), entries=entries)
    save(root, manifest)
    return manifest


def validate_recording(episode, scene, ffmpeg, mode):
    """Audit real simulator data, state slots, gates, clearance and videos."""
    episode = Path(episode).resolve()
    meta, summary = read(episode / 'episode.json'), read(episode / 'summary.json')
    if (bc._digest(meta['scene_plan']) != bc._digest(scene)
            or meta['task']['target'] != scene['target_slot'] or meta['controller'] != 'scripted'):
        raise ValueError('Recorded scene, target, or controller differs from plan')
    if not all(summary.get(k) is True for k in ('done', 'success', 'route_complete', 'recording_valid',
                                                'published_zero_on_exit', 'processes_stopped')):
        raise ValueError('Recording failed or lacks safe teardown: ' + str(summary.get('termination_reason')))
    if summary.get('videos', {}).get('status') != 'completed':
        raise ValueError('Missing completed videos')
    actual = {b['slot']: b for b in meta['buoys']}
    planned = {b['slot']: b for b in scene['buoys']}
    if set(actual) != set(planned) or scene['target_slot'] not in actual or len(actual) != len(meta['buoys']):
        raise ValueError('Recorded buoy slots differ from planned slots')
    for slot, buoy in actual.items():
        if ((buoy['color'], buoy['shape']) != BUOY_SLOTS[slot]
                or math.dist([buoy['x'], buoy['y']], [planned[slot]['x'], planned[slot]['y']]) > 1e-6):
            raise ValueError('Recorded buoy identity or position differs from plan')
    expected_objects = {
        f"v2_buoy_{slot+1}": (planned[slot]['x'], planned[slot]['y'], buoy_radius(slot))
        for slot in planned
    }
    expected_objects.update({
        f"usv_cylinder_{c['id']}": (c['x'], c['y'], c['radius_m'])
        for c in scene['cylinders']
    })
    obstacles = {o['model']: o for o in meta['obstacles']}
    if len(obstacles) != len(meta['obstacles']) or set(obstacles) != set(expected_objects):
        raise ValueError('Collision obstacle models differ from planned buoys and cylinders')
    for name, (x, y, radius) in expected_objects.items():
        actual_obstacle = obstacles[name]
        if (math.dist([x, y], [actual_obstacle['x'], actual_obstacle['y']]) > 1e-6
                or abs(radius-actual_obstacle['radius_m']) > 1e-6):
            raise ValueError('Collision obstacle geometry differs from plan: '+name)
    audit = audit_episode(episode, check_images=True)
    videos = bc.inspect_videos(episode, ffmpeg)
    quality = require_trajectory_quality(episode)
    rows = [json.loads(line) for line in (episode / 'steps.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != summary['steps'] or len(rows) < 3:
        raise ValueError('Missing decision rows')
    gps_state_present = 'state_gps_v3' in rows[0]
    if gps_state_present or scene.get('dataset_type') == 'yellow_multi_v1':
        from usv_vla.export_lerobot import selected_gps_fixes
        from usv_vla.gps_state import gps_position_state
        fixes = selected_gps_fixes(episode, rows)
        for row, fix in zip(rows, fixes):
            actual_gps_state = np.asarray(row.get('state_gps_v3'), dtype=np.float64)
            expected_gps_state = gps_position_state(row['state'], fix)
            if actual_gps_state.shape != (35,) or not np.allclose(actual_gps_state, expected_gps_state, atol=1e-9, rtol=0):
                raise ValueError('35D GPS state missing or inconsistent with selected GPS measurement')
    instruction = scene['route']['instruction']
    if scene['route'].get('prompt_include_phase') is not False or any(row['prompt'] != instruction for row in rows):
        raise ValueError('Task text must remain fixed without teacher phase')
    times = np.asarray([r['decision_sim_time_ns'] for r in rows], dtype=np.int64)
    if np.any(np.abs(np.diff(times) - 100_000_000) > 100_000):
        raise ValueError('Decisions do not follow the 10 Hz grid')
    states = np.asarray([r['state'] for r in rows], dtype=float)
    actions = np.asarray([r['action'] for r in rows], dtype=float)
    if states.shape != (len(rows), 33) or actions.shape != (len(rows), 2):
        raise ValueError('Incorrect model state or action shape')
    scales = np.asarray([2, 2, 1, math.pi, math.pi, 1, 1, 1, 1] + [32, 32, 1] * 8)
    if not np.isfinite(states).all() or np.any(np.abs(states / scales) > 1 + 1e-6):
        raise ValueError('State exceeds the model engineering scales')
    if not np.isfinite(actions).all() or np.any(np.abs(actions) > 15):
        raise ValueError('Invalid thrust')
    nonzero = actions[np.max(np.abs(actions), axis=1) > .05]
    if len(nonzero) < 2 or len(np.unique(np.round(nonzero, 4), axis=0)) < 2:
        raise ValueError('Commands do not demonstrate varying feedback thrust')
    for row, state in zip(rows, states):
        pose = row['pose_xyzrpy']
        for slot in range(8):
            values = state[9 + slot*3:12 + slot*3]
            if slot not in actual:
                if not np.array_equal(values, [0., 0., 0.]):
                    raise ValueError('Absent buoy slot is not zero')
            else:
                buoy = actual[slot]
                relative = body_xy(buoy['x']-pose[0], buoy['y']-pose[1], pose[5])
                if values[2] != 1 or not np.allclose(values[:2], relative, atol=1e-4):
                    raise ValueError('Buoy state differs from actual boat-relative location')
    first = rows[0]['pose_xyzrpy']
    visibility = initial_visibility(scene, [*first[:3], first[5]])
    gates = scene['route'].get('gates', [])
    crossings = []
    for number, gate in enumerate(gates, 1):
        found = []
        for a, b in zip(rows, rows[1:]):
            af, al = _gate_coordinates(gate, a['pose_xyzrpy'][:2])
            bf, bl = _gate_coordinates(gate, b['pose_xyzrpy'][:2])
            if af < 0 <= bf:
                fraction = -af / (bf-af)
                yaw = a['pose_xyzrpy'][5] + fraction * wrap(b['pose_xyzrpy'][5]-a['pose_xyzrpy'][5])
                found.append(dict(lateral_m=al+fraction*(bl-al),
                                  heading_deg=math.degrees(wrap(yaw-gate['yaw_rad']))))
        if len(found) != 1 or not gate['left_min_m'] <= found[0]['lateral_m'] <= gate['left_max_m']:
            raise ValueError(f'Gate {number} lacks one valid physical crossing')
        if abs(found[0]['heading_deg']) > 10.:
            raise ValueError(f'Gate {number} heading exceeds the 10 degree smoothness standard')
        crossings.append(found[0])
    if mode == 'shore' and scene.get('supplement') == 'shore_clear_10':
        from collect_shore_clear_supplement import review_clear
        shore = review_clear(episode, scene)
    else:
        shore = review_green_shore(episode) if mode == 'shore' else None
    if shore is not None and (shore.get('passed') is not True or shore.get('issues')):
        raise ValueError('Bank obstacle / return review failed: ' + str(shore.get('issues')))
    result = dict(validated_at=bc._now(), scene_sha256=bc._digest(scene), audit=audit,
                  videos=videos, trajectory_quality=quality, initial_actual_visibility=visibility,
                  target_slot=scene['target_slot'], buoy_slots=sorted(actual), gate_crossings=crossings,
                  state_shape=[33], gps_state_shape=([35] if gps_state_present else None), action_shape=[2], strict_10hz_grid=True,
                  steps=len(rows), shore_review=shore,
                  steps_sha256=hash_file(episode/'steps.jsonl'),
                  payload_fingerprint=payload_fingerprint(episode, rows))
    bc._atomic_json(episode / 'batch_acceptance.json', result)
    return result


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_attempt(root, relative, episode_id):
    path = Path(relative)
    if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] != episode_id
            or not re.fullmatch(r'attempt_[0-9]{2,}', path.parts[1])):
        raise ValueError('Unsafe attempt path: ' + str(relative))
    target = root / path
    if target.is_symlink() or target.parent.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError('Unsafe symlink or attempt path: ' + str(relative))
    return target


def payload_fingerprint(episode, rows=None):
    """Bind every training image, two videos, and metadata to their bytes."""
    episode = Path(episode).resolve()
    if rows is None:
        rows = [json.loads(line) for line in (episode/'steps.jsonl').read_text().splitlines() if line.strip()]
    paths = {'episode.json', 'summary.json', 'steps.jsonl', 'videa/metadata.json',
             'videa/first_person.mp4', 'videa/aerial_global.mp4'}
    for row in rows:
        name = row['image']
        p = Path(name)
        if p.is_absolute() or len(p.parts) != 2 or p.parts[0] != 'images':
            raise ValueError('Unsafe image path: ' + str(name))
        paths.add(name)
    digest = hashlib.sha256()
    sizes = 0
    for name in sorted(paths):
        path = episode / name
        if path.is_symlink() or not path.resolve().is_relative_to(episode) or not path.is_file():
            raise ValueError('Missing or unsafe payload file: ' + name)
        size = path.stat().st_size
        if size == 0:
            raise ValueError('Empty payload file: ' + name)
        sizes += size
        digest.update((name + '\0' + str(size) + '\0' + hash_file(path) + '\n').encode())
    return dict(algorithm='sha256_paths_sizes_and_sha256_contents_v1', sha256=digest.hexdigest(),
                file_count=len(paths), image_count=len(paths)-6, bytes=sizes)


def validate_manifest(root, manifest):
    """Verify stable identities, partitions, immutable plans, and managed paths."""
    if (manifest.get('schema_version') != 1 or manifest.get('dataset_type') != DATASET_TYPE
            or manifest.get('target_slot') != TARGET_SLOT or manifest.get('seed') != SEED
            or manifest.get('quotas') != QUOTAS or manifest.get('root') != str(root)):
        raise ValueError('Incompatible green batch manifest')
    entries = manifest.get('entries')
    if not isinstance(entries, list) or len(entries) != 200:
        raise ValueError('Manifest must have 200 entries')
    modes, partitions = Counter(), Counter()
    seen_id, seen_seed, seen_scene, seen_mode_index, seen_paths = set(), set(), set(), set(), set()
    for e in entries:
        ident, mode, index = e['episode_id'], e['mode'], e['index']
        if (not isinstance(ident, str) or not re.fullmatch(r'episode_[0-9]{4}_[a-z_]+', ident)
                or mode not in QUOTAS or type(index) is not int or not 0 <= index < QUOTAS[mode]
                or (mode, index) in seen_mode_index or ident in seen_id):
            raise ValueError('Duplicate or invalid episode identity, mode, or index')
        seen_id.add(ident); seen_mode_index.add((mode,index)); modes[mode] += 1
        if e['split'] != ('validation' if index % 5 == 0 else 'train'):
            raise ValueError('Episode split changed')
        partitions[e['split']] += 1
        seed = e['seed']
        if type(seed) is not int or seed in seen_seed:
            raise ValueError('Duplicate or invalid scene seed')
        seen_seed.add(seed)
        scene_rel = Path(e['scene'])
        if (scene_rel.is_absolute() or len(scene_rel.parts) != 2 or scene_rel.parts[0] != 'plans'
                or not scene_rel.name.startswith(ident) or scene_rel.suffix != '.json'):
            raise ValueError('Invalid plan path')
        path = root / scene_rel
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or unsafe plan')
        scene = read(path)
        if bc._digest(scene) != e['scene_sha256'] or e['scene_sha256'] in seen_scene:
            raise ValueError('Plan digest changed or is duplicated')
        seen_scene.add(e['scene_sha256'])
        collect = scene.get('collection', {})
        if (scene.get('seed') != seed or scene.get('category') != e['category']
                or collect != dict(episode_id=ident, mode=mode, category=e['category'],
                                   category_index=index, seed=seed, split=e['split'])):
            raise ValueError('Plan identity differs from manifest')
        if e['status'] not in ('planned','recording','failed','accepted'):
            raise ValueError('Unknown entry status')
        if not isinstance(e['attempts'], list):
            raise ValueError('Invalid attempt history')
        for n, attempt in enumerate(e['attempts'], 1):
            relative = attempt['directory']
            safe_attempt(root, relative, ident)
            if relative != f'{ident}/attempt_{n:02d}' or relative in seen_paths:
                raise ValueError('Attempt path differs from its history')
            seen_paths.add(relative)
            if attempt['log'] != relative + '.recorder.log':
                raise ValueError('Unexpected recorder log path')
        if e['status'] == 'accepted':
            if (not e['attempts'] or e.get('accepted_episode') != e['attempts'][-1]['directory']
                    or e['attempts'][-1].get('status') != 'accepted' or not e.get('validation')):
                raise ValueError('Accepted entry lacks its accepted attempt/proof')
        elif e.get('accepted_episode'):
            raise ValueError('Nonaccepted entry may not point to accepted data')
    if dict(modes) != QUOTAS or dict(partitions) != {'validation':40, 'train':160}:
        raise ValueError('Batch mode quota or train/validation split differs')


def active_recorder(episode):
    """On Linux detect a still-live collector before touching its files."""
    wanted = str(episode.resolve())
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
            args = [x.decode(errors='replace') for x in args if x]
            if 'usv_vla.record' in args and wanted in args:
                return int(proc.name)
        except (OSError, ValueError):
            continue
    return None


def archive_failed(root, manifest, entry, attempt):
    """Verify recorder teardown, archive diagnostics, then remove failed payload."""
    relative = attempt['directory']
    episode = safe_attempt(root, relative, entry['episode_id'])
    if any(e.get('accepted_episode') == relative for e in manifest['entries']):
        raise ValueError('Refusing to remove accepted data')
    archive = root/'failed_metadata'/entry['episode_id']/(episode.name+'.tar.gz')
    if attempt.get('cleanup') == 'done':
        if episode.exists() or not archive.is_file() or hash_file(archive) != attempt.get('archive_sha256'):
            raise ValueError('Failed cleanup archive or tombstone changed')
        return
    pid = active_recorder(episode)
    if pid is not None:
        raise bc.BatchStopped(f'Recorder {pid} still active for {relative}; refusing cleanup')
    if attempt.get('cleanup') == 'archived' and not episode.exists():
        if not archive.is_file() or hash_file(archive) != attempt['archive_sha256']:
            raise ValueError('Missing or changed failure archive')
        log = root/attempt['log']
        if log.is_symlink(): raise ValueError('Unsafe recorder log')
        log.unlink(missing_ok=True)
        attempt['cleanup'] = 'done'; save(root, manifest); return
    summary = read(episode/'summary.json')
    if summary.get('published_zero_on_exit') is not True or summary.get('processes_stopped') is not True:
        raise bc.BatchStopped('No verified zero thrust and stopped simulator for '+relative)
    log = root/attempt['log']
    if log.is_symlink():
        raise ValueError('Unsafe recorder log')
    files = [(episode/name, name) for name in ('episode.json','summary.json','batch_acceptance.json') if (episode/name).is_file()]
    if log.is_file(): files.append((log,'recorder.log'))
    if not files or any(path.is_symlink() for path,_ in files):
        raise ValueError('Unsafe or missing failure metadata')
    expected = {name:hash_file(path) for path,name in files}
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        fd, temp = tempfile.mkstemp(prefix='.failed_', dir=archive.parent); os.close(fd)
        try:
            with tarfile.open(temp,'w:gz') as tar:
                for path,name in files: tar.add(path,arcname=name,recursive=False)
            with tarfile.open(temp,'r:gz') as tar:
                found = {item.name:hashlib.sha256(tar.extractfile(item).read()).hexdigest() for item in tar if item.isfile()}
            if found != expected: raise ValueError('Failure archive verification failed')
            os.replace(temp,archive)
        finally:
            Path(temp).unlink(missing_ok=True)
    else:
        with tarfile.open(archive,'r:gz') as tar:
            found = {item.name:hashlib.sha256(tar.extractfile(item).read()).hexdigest() for item in tar if item.isfile()}
        if found != expected: raise ValueError('Failure archive conflict')
    attempt.update(cleanup='archived', archive=str(archive.relative_to(root)), archive_sha256=hash_file(archive))
    save(root,manifest)
    if episode.exists(): shutil.rmtree(episode)
    log.unlink(missing_ok=True)
    attempt['cleanup'] = 'done'; save(root,manifest)


def recover_failed(root, manifest, entry, max_attempts, ffmpeg):
    attempt = entry['attempts'][-1]
    episode = safe_attempt(root, attempt['directory'], entry['episode_id'])
    if entry['status'] == 'recording' and episode.is_dir() and (episode/'summary.json').is_file() and active_recorder(episode) is None:
        # An interruption can occur after a successful run and before manifest commit.
        try:
            validation = validate_recording(episode, read(root/entry['scene']), ffmpeg, entry['mode'])
        except Exception as error:
            attempt.update(status='failed',error=f'Interrupted validation: {type(error).__name__}: {error}')
            entry['status']='failed'; save(root,manifest)
        else:
            entry.update(status='accepted',accepted_episode=attempt['directory'],validation=validation)
            attempt.update(status='accepted',finished_at=bc._now()); save(root,manifest); return
    if entry['status'] != 'failed':
        raise bc.BatchStopped('Interrupted recorder has no confirmed teardown: '+entry['episode_id'])
    archive_failed(root,manifest,entry,attempt)
    if len(entry['attempts']) >= max_attempts:
        raise bc.BatchStopped('Attempt limit reached: '+entry['episode_id'])
    retry = len(entry['attempts'])
    seed = bc._scene_seed(SEED, entry['mode']+':retry:'+str(retry), entry['index'])
    used = {x['seed'] for x in manifest['entries']} | {a.get('seed') for x in manifest['entries'] for a in x['attempts']}
    if seed in used: raise ValueError('Retry seed is not unique')
    scene = make_scene(entry['mode'],entry['index'],seed)
    scene['collection'] = dict(episode_id=entry['episode_id'],mode=entry['mode'],category=scene['category'],
                               category_index=entry['index'],seed=seed,split=entry['split'])
    validate_scene(scene)
    path = root/'plans'/(entry['episode_id']+f'_retry{retry:02d}.json')
    if path.exists() and bc._digest(read(path)) != bc._digest(scene):
        raise ValueError('Retry plan conflicts with existing file')
    if not path.exists(): bc._atomic_json(path,scene)
    attempt['superseded_scene'] = entry['scene']
    attempt['superseded_scene_sha256'] = entry['scene_sha256']
    entry.update(scene=str(path.relative_to(root)),scene_sha256=bc._digest(scene),seed=seed,
                 category=scene['category'],status='planned')
    save(root,manifest)


def summarize(root, manifest, ffmpeg):
    validate_manifest(root, manifest)
    accepted = [e for e in manifest['entries'] if e['status'] == 'accepted']
    issues = []
    for entry in accepted:
        try:
            episode = safe_attempt(root, entry['accepted_episode'], entry['episode_id'])
            proof = read(episode/'batch_acceptance.json')
            if proof != entry['validation'] or proof['scene_sha256'] != entry['scene_sha256']:
                raise ValueError('Acceptance proof changed')
            if payload_fingerprint(episode) != proof['payload_fingerprint']:
                raise ValueError('Metadata, images, video or steps content changed')
            audit = audit_episode(episode,check_images=True)
            if not audit['success'] or not audit['route_replayed']:
                raise ValueError('Independent recording replay failed')
            if bc.inspect_videos(episode,ffmpeg) != proof['videos']:
                raise ValueError('Video decode or frame count changed')
            if require_trajectory_quality(episode) != proof['trajectory_quality']:
                raise ValueError('Independent smoothness proof changed')
        except Exception as error:
            issues.append(entry['episode_id'] + ': ' + str(error))
    report = dict(dataset_type=DATASET_TYPE,complete=len(accepted)==200 and not issues,
                  target=TARGET_NAME,accepted_valid=len(accepted)-len(issues),
                  accepted_by_mode=dict(Counter(e['mode'] for e in accepted)),
                  splits=dict(Counter(e['split'] for e in accepted)),issues=issues,
                  total_decisions=sum(e.get('validation',{}).get('steps',0) for e in accepted),
                  state_source='simulator_pose_debug_not_real_localization',
                  data_kind='scripted_teacher_gazebo_demonstrations_not_trained_vla')
    all_for_training = all_accepted_for_training(root)
    supplements = accepted_supplements(root)
    report['accepted_supplement_count'] = len(supplements)
    report['training_selection'] = {'mode': 'all_accepted_for_training' if all_for_training else 'original_split',
                                    'train': len(selected(accepted, 'train', all_for_training=all_for_training)) + (len(supplements) if all_for_training else 0),
                                    'validation': len(selected(accepted, 'validation', all_for_training=all_for_training))}
    bc._atomic_json(root/'collection_summary.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--min-free-gib', type=float, default=5.)
    parser.add_argument('--max-wall-s', type=float, default=240.)
    parser.add_argument('--max-attempts', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.limit <= 200 or not math.isfinite(args.min_free_gib) or args.min_free_gib < 5 or not 1 <= args.max_attempts <= 20:
        parser.error('limit must be 1..200, free space at least 5 GiB, attempts 1..20')
    root = args.root.resolve()
    if root.parent != ROOT.parent.resolve() or not root.name.startswith(ROOT_PREFIX):
        parser.error('root must be a named target batch directly under outputs/episodes')
    root.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(bc.resolve_ffmpeg(args.ffmpeg))
    with (root/'.batch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(root/'manifest.json') if (root/'manifest.json').exists() else create_manifest(root)
        validate_manifest(root, manifest)
        for entry in manifest['entries'][:args.limit]:
            if entry['status'] == 'accepted': continue
            while entry['status'] != 'accepted':
                if entry['status'] in ('recording','failed'):
                    recover_failed(root,manifest,entry,args.max_attempts,ffmpeg)
                    if entry['status'] == 'accepted': break
                if entry['status'] != 'planned': raise ValueError('Unexpected status')
                scene = read(root/entry['scene'])
                if bc._digest(scene) != entry['scene_sha256']:
                    raise ValueError('Immutable plan changed: '+entry['episode_id'])
                if shutil.disk_usage(root).free < args.min_free_gib*bc.GIB:
                    raise bc.BatchStopped('Disk free space is below the guard')
                relative = f"{entry['episode_id']}/attempt_{len(entry['attempts'])+1:02d}"
                episode = safe_attempt(root,relative,entry['episode_id'])
                log = root/(relative+'.recorder.log')
                if episode.exists() or log.exists() or log.is_symlink():
                    raise ValueError('Refusing to overwrite an existing recording')
                log.parent.mkdir(parents=True,exist_ok=True)
                command = bc.record_command(root/entry['scene'],episode,ffmpeg=ffmpeg,episode_wall_s=args.max_wall_s)
                command = ['bash',str(bc.PROJECT_ROOT/'scripts/sim.sh'),'record',*command[3:]]
                attempt = dict(directory=relative,log=str(log.relative_to(root)),status='recording',
                               seed=entry['seed'],scene_sha256=entry['scene_sha256'],started_at=bc._now())
                entry['attempts'].append(attempt); entry['status']='recording'; save(root,manifest)
                try:
                    code = bc.run_recorder(command,log,args.max_wall_s+120)
                    if code != 0: raise ValueError(f'Recorder returned {code}')
                    validation = validate_recording(episode,scene,ffmpeg,entry['mode'])
                    entry.update(status='accepted',accepted_episode=relative,validation=validation)
                    attempt.update(status='accepted',finished_at=bc._now()); save(root,manifest)
                    print(json.dumps({'episode':entry['episode_id'],'mode':entry['mode'],
                                      'accepted':manifest['accepted_count'],'target':TARGET_NAME},ensure_ascii=False),flush=True)
                except BaseException as error:
                    entry['status']='failed'
                    attempt.update(status='failed',error=f'{type(error).__name__}: {error}',finished_at=bc._now())
                    save(root,manifest)
                    if not isinstance(error, Exception): raise
                    print(json.dumps({'episode':entry['episode_id'],'attempt':relative,
                                      'failed':attempt['error']},ensure_ascii=False),flush=True)
        report = summarize(root,manifest,ffmpeg)
        print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

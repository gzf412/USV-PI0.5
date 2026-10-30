#!/usr/bin/env python3
"""Safely recollect a plan of cylinder-side episodes, one at a time.

The manifest's accepted path remains authoritative while a candidate records.
A durable journal makes recording, manifest commit and cleanup resumable.
Run via system Python; scripts/sim.sh supplies ROS/Gazebo to the recorder.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

from usv_vla import batch_collect as bc
from usv_vla.summarize_collection import summarize

ROOT = bc.OUTPUTS / 'episodes/red_round_200_20261026'
PROJECT = bc.PROJECT_ROOT


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def safe_path(root, relative, episode_id):
    """Restrict removal and recording to a single attempt under its episode."""
    p = Path(relative)
    if p.is_absolute() or len(p.parts) != 2 or p.parts[0] != episode_id or not p.parts[1].startswith('attempt_') or not p.parts[1][8:].isdigit():
        raise ValueError(f'Unsafe attempt path: {relative}')
    target = root / p
    if target.is_symlink() or target.parent.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe attempt path: {relative}')
    return target


def turn_metric(episode, side):
    """Largest outward yaw change from start through first physical gate crossing."""
    if side not in ('left', 'right'):
        raise ValueError('Not a cylinder-side task')
    scene = read_json(Path(episode) / 'episode.json')['scene_plan']
    route = scene['route']
    if route['kind'] != 'cylinder_pass' or route['side'] != side:
        raise ValueError('Scene side does not match category')
    gate = route['gates'][0]
    rows = [json.loads(line) for line in (Path(episode) / 'steps.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) < 3:
        raise ValueError('Too few decision samples')
    cx, cy = gate['center']
    c, s = math.cos(gate['yaw_rad']), math.sin(gate['yaw_rad'])
    initial_yaw = rows[0]['pose_xyzrpy'][5]
    sign = 1 if side == 'left' else -1
    values = []
    prior = None
    crossing = None
    for i, row in enumerate(rows):
        x, y, _, _, _, yaw = row['pose_xyzrpy']
        forward = (x-cx)*c + (y-cy)*s
        delta = math.atan2(math.sin(yaw-initial_yaw), math.cos(yaw-initial_yaw))
        values.append(sign*math.degrees(delta))
        if prior is not None and prior < 0 <= forward:
            crossing = i
            break
        prior = forward
    if crossing is None:
        raise ValueError('No recorded first gate crossing')
    return max(0., max(values)), crossing


def archive_metadata(episode, log, archive):
    """Retain only small diagnostics, and verify archive bytes before cleanup."""
    names = ('episode.json', 'summary.json', 'world.json', 'batch_failure.json',
             'batch_acceptance.json', 'trajectory_quality_review.json', 'payload_cleanup.json')
    files = [(episode / n, n) for n in names if (episode / n).is_file()]
    if log is not None and log.is_file():
        files.append((log, 'recorder.log'))
    if not files:
        raise ValueError(f'No metadata to archive in {episode}')
    for path, _ in files:
        if path.is_symlink():
            raise ValueError(f'Symlink in archive input: {path}')
    expected = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        with tarfile.open(archive, 'r:gz') as tar:
            got = {item.name: hashlib.sha256(tar.extractfile(item).read()).hexdigest()
                   for item in tar.getmembers() if item.isfile()}
        if got != expected:
            raise ValueError(f'Conflicting metadata archive: {archive}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.recollect_archive_', dir=archive.parent)
    os.close(fd)
    try:
        with tarfile.open(temporary, 'w:gz') as tar:
            for path, name in files:
                tar.add(path, arcname=name, recursive=False)
        with tarfile.open(temporary, 'r:gz') as tar:
            got = {item.name: hashlib.sha256(tar.extractfile(item).read()).hexdigest()
                   for item in tar.getmembers() if item.isfile()}
        if got != expected:
            raise ValueError('Archive verification failed')
        os.replace(temporary, archive)
    finally:
        Path(temporary).unlink(missing_ok=True)


def remove_attempt(root, relative, episode_id, log_relative, manifest):
    p = safe_path(root, relative, episode_id)
    accepted = {e.get('accepted_episode') for e in manifest['entries'] if e.get('status') == 'accepted'}
    if relative in accepted:
        raise ValueError(f'Refusing to remove accepted episode {relative}')
    expected_log = str(Path(relative).parent / (Path(relative).name + '.recorder.log'))
    if log_relative is not None and log_relative != expected_log:
        raise ValueError(f'Unexpected log path: {log_relative}')
    log = root / log_relative if log_relative is not None else None
    if log is not None and log.is_symlink():
        raise ValueError('Refusing symlink log')
    if p.exists():
        shutil.rmtree(p)
    if log is not None:
        log.unlink(missing_ok=True)


def save(journal_path, journal):
    journal['updated_at'] = bc._now()
    bc._atomic_json(journal_path, journal)


def find_entry(manifest, episode_id):
    hits = [e for e in manifest['entries'] if e['episode_id'] == episode_id]
    if len(hits) != 1:
        raise ValueError(f'Unknown or duplicate episode {episode_id}')
    return hits[0]


def finish_summary(root):
    report = summarize(root)
    if not report['complete'] or report['accepted_valid'] != 200 or report['issues']:
        raise ValueError('Post-commit collection summary is incomplete; old payload retained')


def run(args):
    root = args.root.resolve()
    plan = read_json(args.plan)
    plan_rows = plan.get('entries') if isinstance(plan, dict) else None
    rows_by_id = {row['episode_id']: row for row in plan_rows} if isinstance(plan_rows, list) else {}
    ids = ([row['episode_id'] for row in plan_rows] if plan_rows is not None else
           plan if isinstance(plan, list) else plan.get('episode_ids', plan.get('recollect_episode_ids')))
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('Plan must contain a nonempty unique episode_ids list')
    plan_digest = hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()
    journal_path = args.journal.resolve()
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with (root / '.batch.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise bc.BatchStopped('Batch is already locked') from error
        manifest = read_json(root / 'manifest.json')
        bc._validate_manifest(root, manifest, manifest['seed'])
        if journal_path.exists():
            journal = read_json(journal_path)
            if (journal['plan_sha256'] != plan_digest or journal['root'] != str(root)
                    or journal.get('minimum_improvement_deg') != args.minimum_improvement_deg):
                raise ValueError('Journal does not match plan, batch or improvement threshold')
        else:
            journal = {'plan_sha256': plan_digest, 'root': str(root), 'plan': str(args.plan.resolve()),
                       'minimum_improvement_deg': args.minimum_improvement_deg, 'items': {}}
            save(journal_path, journal)
        ffmpeg = bc.resolve_ffmpeg(args.ffmpeg)
        for episode_id in ids:
            entry = find_entry(manifest, episode_id)
            item = journal['items'].get(episode_id)
            if item and item['stage'] in ('done', 'failed'):
                print(json.dumps({'episode': episode_id, 'stage': item['stage'], 'error': item.get('error')}, ensure_ascii=False), flush=True)
                continue
            side = {'around_left': 'left', 'around_right': 'right'}.get(entry['category'])
            if side is None or entry['status'] != 'accepted':
                raise ValueError(f'{episode_id} is not an accepted cylinder-side episode')
            if item is None:
                old = entry['accepted_episode']
                safe_path(root, old, episode_id)
                if rows_by_id:
                    planned = rows_by_id[episode_id]
                    for key in ('category', 'scene', 'scene_sha256', 'seed', 'split'):
                        if entry[key] != planned[key]:
                            raise ValueError(f'{episode_id}: planned {key} changed')
                    if old != planned['current_accepted_episode']:
                        raise ValueError(f'{episode_id}: planned accepted path changed')
                    old_sha = hashlib.sha256((root / old / 'steps.jsonl').read_bytes()).hexdigest()
                    if old_sha != planned['old_steps_sha256']:
                        raise ValueError(f'{episode_id}: planned old steps changed')
                number = max((int(a['number']) for a in entry['attempts']), default=0) + 1
                relative = f'{episode_id}/attempt_{number:02d}'
                candidate = safe_path(root, relative, episode_id)
                log_relative = f'{episode_id}/attempt_{number:02d}.recorder.log'
                if candidate.exists() or (root / log_relative).exists() or any(a['directory'] == relative for a in entry['attempts']):
                    raise ValueError(f'Candidate path already exists: {relative}')
                old_peak, old_crossing = turn_metric(root / old, side)
                item = {'stage': 'prepared', 'old': old, 'candidate': relative, 'candidate_log': log_relative,
                        'number': number, 'side': side, 'old_peak_deg': old_peak, 'old_crossing_index': old_crossing,
                        'scene_sha256': entry['scene_sha256'], 'seed': entry['seed'], 'split': entry['split']}
                journal['items'][episode_id] = item
                save(journal_path, journal)
            if any(entry[k] != item[k] for k in ('scene_sha256', 'seed', 'split')):
                raise ValueError(f'{episode_id}: scene, seed or split changed')
            old, relative = item['old'], item['candidate']
            candidate = safe_path(root, relative, episode_id)
            log = root / item['candidate_log']
            if item['stage'] == 'prepared':
                if entry['accepted_episode'] != old:
                    raise ValueError('Accepted path changed before recording')
                if candidate.exists() or log.exists():
                    raise ValueError('Unexpected existing candidate in prepared stage')
                if shutil.disk_usage(root).free < args.min_free_gib * bc.GIB:
                    raise bc.BatchStopped('Insufficient free disk space')
                item['stage'] = 'recording'
                save(journal_path, journal)
                command = bc.record_command(root / entry['scene'], candidate, ffmpeg=ffmpeg, episode_wall_s=args.episode_wall_s)
                command = ['bash', str(PROJECT / 'scripts/sim.sh'), 'record', *command[3:]]
                item['command'] = command
                save(journal_path, journal)
                try:
                    item['returncode'] = bc.run_recorder(command, log, args.episode_wall_s + 120)
                except bc.BatchStopped as error:
                    item['recording_error'] = str(error)
                    save(journal_path, journal)
                    raise
                save(journal_path, journal)
            if item['stage'] == 'recording':
                if entry['accepted_episode'] != old:
                    raise ValueError('Accepted path changed during recording')
                if item.get('returncode') is None:
                    raise bc.BatchStopped('Recorder exit is unknown; retain candidate and inspect/stop simulator before resuming. Do not start another collection.')
                try:
                    teardown = read_json(candidate / 'summary.json')
                except (OSError, ValueError):
                    raise bc.BatchStopped('Recorder exited without a teardown proof; retain candidate and inspect simulator before resuming.')
                if not teardown.get('processes_stopped') or not teardown.get('published_zero_on_exit'):
                    raise bc.BatchStopped('Recorder teardown is unconfirmed; retain candidate and inspect simulator before resuming.')
                try:
                    if item.get('returncode') != 0:
                        raise ValueError(f"Recorder returned {item.get('returncode')}")
                    validation = bc.validate_episode(candidate, ffmpeg=ffmpeg, expected_scene_sha256=entry['scene_sha256'])
                    new_peak, crossing = turn_metric(candidate, side)
                    if item['old_peak_deg'] - new_peak < args.minimum_improvement_deg:
                        raise ValueError(f"Initial outward turn did not improve by {args.minimum_improvement_deg:g} deg: {item['old_peak_deg']:.3f} -> {new_peak:.3f}")
                    item.update(stage='validated', new_peak_deg=new_peak, new_crossing_index=crossing,
                                validation=validation)
                    save(journal_path, journal)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    item.update(stage='failure_pending', error=f'{type(error).__name__}: {error}')
                    save(journal_path, journal)
            if item['stage'] == 'rollback_pending':
                entry.clear()
                entry.update(deepcopy(item['old_entry']))
                bc._save_state(root, manifest, selected_count=200)
                finish_summary(root)
                item['stage'] = 'failure_pending'
                save(journal_path, journal)
            if item['stage'] == 'failure_pending':
                if entry['accepted_episode'] != old:
                    raise ValueError('Accepted path changed while handling failure')
                archive = PROJECT / 'reports/history' / f'{episode_id}_attempt_{item["number"]:02d}_initial_turn_candidate.tar.gz'
                if candidate.is_dir():
                    if not any((candidate / n).is_file() for n in ('episode.json','summary.json','batch_failure.json')):
                        bc._atomic_json(candidate / 'batch_failure.json', {'error': item['error'], 'candidate': relative})
                    archive_metadata(candidate, log, archive)
                elif not any(a['directory'] == relative for a in entry['attempts']):
                    candidate.mkdir(parents=True, exist_ok=False)
                    bc._atomic_json(candidate / 'batch_failure.json', {'error': item['error'], 'candidate': relative})
                    archive_metadata(candidate, log, archive)
                attempts = [a for a in entry['attempts'] if a['directory'] == relative]
                if not attempts:
                    entry['attempts'].append({'number': item['number'], 'directory': relative, 'log': item['candidate_log'],
                                              'status': 'failed', 'error': item['error'], 'finished_at': bc._now(),
                                              'payload_cleanup': {'directory_removed': True, 'metadata_archive': str(archive)}})
                    bc._save_state(root, manifest, selected_count=200)
                remove_attempt(root, relative, episode_id, item['candidate_log'], manifest)
                item['stage'] = 'failed'
                save(journal_path, journal)
                print(json.dumps({'episode': episode_id, 'stage': 'failed', 'error': item['error']}, ensure_ascii=False), flush=True)
                continue
            if item['stage'] == 'validated':
                if 'old_entry' not in item:
                    if entry['accepted_episode'] != old:
                        raise ValueError('Missing prior-entry recovery snapshot')
                    item['old_entry'] = deepcopy(entry)
                    save(journal_path, journal)
                if entry['accepted_episode'] == old:
                    old_attempts = [a for a in entry['attempts'] if a['directory'] == old and a['status'] == 'accepted']
                    if len(old_attempts) != 1:
                        raise ValueError('Old accepted attempt is ambiguous')
                    old_attempt = old_attempts[0]
                    old_path = safe_path(root, old, episode_id)
                    archive = PROJECT / 'reports/history' / f'{episode_id}_{Path(old).name}_initial_turn_superseded.tar.gz'
                    old_log = old_attempt.get('log')
                    expected_log = str(Path(old).parent / (Path(old).name + '.recorder.log'))
                    if old_log is not None and old_log != expected_log:
                        raise ValueError('Old log path is not the expected attempt log')
                    archive_metadata(old_path, root / old_log if old_log else None, archive)
                    old_attempt.update(status='superseded', superseded_by=relative, superseded_at=bc._now(),
                                       supersession_reason='Improved unnecessary initial outward turn',
                                       payload_cleanup={'directory_removed': True, 'metadata_archive': str(archive)})
                    entry['attempts'].append({'number': item['number'], 'directory': relative, 'log': item['candidate_log'],
                                              'status': 'accepted', 'finished_at': bc._now(), 'command': item['command']})
                    bc._accept_entry(entry, relative, item['validation'])
                    bc._save_state(root, manifest, selected_count=200)
                elif entry['accepted_episode'] != relative:
                    raise ValueError('Accepted path changed during commit')
                item['stage'] = 'committed'
                save(journal_path, journal)
            if item['stage'] == 'committed':
                if entry['accepted_episode'] != relative:
                    raise ValueError('Committed candidate is no longer accepted')
                bc._save_state(root, manifest, selected_count=200)
                try:
                    finish_summary(root)
                except Exception as error:
                    item['error'] = f'Post-commit full-summary validation failed: {type(error).__name__}: {error}'
                    item['stage'] = 'rollback_pending'
                    save(journal_path, journal)
                    entry.clear()
                    entry.update(deepcopy(item['old_entry']))
                    bc._save_state(root, manifest, selected_count=200)
                    finish_summary(root)
                    item['stage'] = 'failure_pending'
                    save(journal_path, journal)
                    raise bc.BatchStopped('Full summary rejected candidate; previous accepted restored. Resume to archive candidate.') from error
                old_attempt = next(a for a in entry['attempts'] if a['directory'] == old)
                remove_attempt(root, old, episode_id, old_attempt.get('log'), manifest)
                item['stage'] = 'done'
                item.pop('validation', None)
                save(journal_path, journal)
            print(json.dumps({'episode': episode_id, 'stage': item['stage'], 'old_peak_deg': item['old_peak_deg'], 'new_peak_deg': item.get('new_peak_deg')}, ensure_ascii=False), flush=True)
        finish_summary(root)
        failed = [eid for eid in ids if journal['items'][eid]['stage'] == 'failed']
        print(json.dumps({'complete': True, 'improved': len(ids)-len(failed), 'failed': failed}, ensure_ascii=False), flush=True)
        return len(failed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--journal', type=Path, required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--ffmpeg')
    p.add_argument('--minimum-improvement-deg', type=float, default=3.)
    p.add_argument('--min-free-gib', type=float, default=3.)
    p.add_argument('--episode-wall-s', type=float, default=240.)
    args = p.parse_args()
    if args.minimum_improvement_deg < 3 or args.min_free_gib < 3 or args.episode_wall_s <= 0:
        p.error('Minimum improvement >= 3 degrees, free space >= 3 GiB and positive timeout required')
    sys.exit(1 if run(args) else 0)


if __name__ == '__main__':
    main()

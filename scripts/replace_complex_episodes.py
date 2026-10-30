#!/usr/bin/env python3
"""Safely replace simple obstacle episodes with independently checked complex routes.

Use --limit 2 for a pilot; resume without --limit after inspection. All 40 old
recordings remain accepted until every new recording passes all checks.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import random
from pathlib import Path
import shutil
import sys

from usv_vla import batch_collect as bc
from usv_vla.scene_plan import validate_scene
from pilot_angled_gap import generate as generate_gap
from pilot_multi_obstacle import make_scene as generate_multi
from review_angled_gap import review as review_gap
from review_multi_obstacle import review_episode as review_multi
from usv_vla.summarize_collection import summarize
from recollect_initial_turns import archive_metadata, remove_attempt, safe_path

ROOT = bc.OUTPUTS / 'episodes/red_round_200_20261027'
PROJECT = bc.PROJECT_ROOT
VERSION = 'complex_replacement_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    data['updated_at'] = bc._now()
    bc._atomic_json(path, data)


def entry_by_id(manifest, episode_id):
    found = [e for e in manifest['entries'] if e['episode_id'] == episode_id]
    if len(found) != 1:
        raise ValueError(f'Unknown or duplicate episode: {episode_id}')
    return found[0]


def verify_summary(root):
    report = summarize(root)
    if not report['complete'] or report['accepted_valid'] != 200 or report['issues']:
        raise ValueError('Full dataset summary failed: ' + str(report.get('issues', []))[:1500])
    return report


def choose_entries(root, manifest):
    """Rank shorter, straighter routes; preserve 80/20 split per category."""
    selected = []
    quotas = {'between': {'train': 16, 'validation': 4},
              'around_left': {'train': 8, 'validation': 2},
              'around_right': {'train': 8, 'validation': 2}}
    for category, splits in quotas.items():
        for split, quota in splits.items():
            candidates = []
            for entry in manifest['entries']:
                if entry['category'] != category or entry['split'] != split:
                    continue
                scene = read(root / entry['scene'])
                pose, target = scene['initial_pose'], scene['target_xy']
                distance = math.dist(pose[:2], target)
                angle = abs(math.degrees(math.atan2(
                    math.sin(pose[3] - scene['route']['reference_yaw_rad']),
                    math.cos(pose[3] - scene['route']['reference_yaw_rad']))))
                # Distance is the main ranking criterion; initial angle breaks
                # ties to prioritise straighter examples for replacement.
                candidates.append((distance, angle, entry['episode_id'], entry))
            if len(candidates) < quota:
                raise ValueError('Not enough eligible episodes in %s/%s' % (category, split))
            for distance, angle, _, entry in sorted(candidates)[:quota]:
                selected.append((entry, {'initial_target_distance_m': distance,
                                         'absolute_heading_offset_deg': angle,
                                         'ranking': 'shortest distance, then smallest initial heading angle within category/split'}))
    # Interleave route types to make --limit 2 exercise both validators.
    by_category = {c: [x for x in selected if x[0]['category'] == c] for c in quotas}
    output = []
    for i in range(20):
        output.append(by_category['between'][i])
        if i < 20:
            side = 'around_left' if i % 2 == 0 else 'around_right'
            output.append(by_category[side][i // 2])
    return output


def make_complex_scene(entry, variant, nonce=0):
    """Keep manifest seed identity while recording the separate geometry seed."""
    category = entry['category']
    # Balances sign, gate count, and route distances across both split groups.
    seed_rng = random.Random(entry['seed'] ^ 0x849d7271 ^ (nonce * 0x9e3779b97f4a7c15))
    failures = []
    for _ in range(100):
        geometry_seed = seed_rng.randrange(2**32)
        try:
            if category == 'between':
                scene = generate_gap(geometry_seed, variant % 2, 2 + variant % 3)
            else:
                side = 'left' if category == 'around_left' else 'right'
                gates = 2 if (variant // 2) % 2 == 0 else 3
                # Angles point toward the first-cylinder side as in pilots.
                magnitude = seed_rng.uniform(8., 14.) if gates == 2 else seed_rng.uniform(10., 15.)
                heading = magnitude if side == 'left' else -magnitude
                distance = seed_rng.uniform(26., 29.) if gates == 2 else seed_rng.uniform(28.4, 29.)
                scene = generate_multi(geometry_seed, side, heading, distance, gates)
            scene['seed'] = entry['seed']
            scene['index'] = entry['category_index']
            scene['complex_replacement'] = {'geometry_seed': geometry_seed, 'variant': variant, 'replan_nonce': nonce}
            scene['collection'] = {'episode_id': entry['episode_id'], 'category': category,
                                   'category_index': entry['category_index'], 'seed': entry['seed'],
                                   'split': entry['split']}
            scene['initial_visibility'] = validate_scene(scene)
            return scene
        except ValueError as error:
            failures.append(str(error))
    raise ValueError('Cannot generate %s with original seed %s: %s' %
                     (entry['episode_id'], entry['seed'], failures[-3:]))


def make_journal(root, path, manifest):
    entries = choose_entries(root, manifest)
    if len(entries) != 40 or any(e['status'] != 'accepted' for e, _ in entries):
        raise ValueError('Expected exactly 40 accepted obstacle episodes.')
    plan_dir = root / 'plans' / VERSION
    plan_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for variant, (entry, selection) in enumerate(entries):
        episode_id = entry['episode_id']
        old = entry['accepted_episode']
        safe_path(root, old, episode_id)
        scene = make_complex_scene(entry, variant // 2)
        relplan = str(Path('plans') / VERSION / (episode_id + '.json'))
        planfile = root / relplan
        if planfile.exists():
            if bc._digest(read(planfile)) != bc._digest(scene):
                raise ValueError(f'Conflicting existing complex plan: {planfile}')
        else:
            bc._atomic_json(planfile, scene)
        number = max((int(a['number']) for a in entry['attempts']), default=0) + 1
        rel = f'{episode_id}/attempt_{number:02d}'
        if safe_path(root, rel, episode_id).exists() or (root / (rel + '.recorder.log')).exists():
            raise ValueError(f'Candidate already exists without journal: {rel}')
        items.append({'episode_id': episode_id, 'stage': 'prepared', 'selection': selection,
                      'old_entry': deepcopy(entry), 'variant': variant // 2, 'new_scene': relplan,
                      'new_scene_sha256': bc._digest(scene), 'number': number,
                      'candidate': rel, 'candidate_log': rel + '.recorder.log',
                      'failed_attempts': []})
    journal = {'version': VERSION, 'root': str(root), 'stage': 'collecting', 'items': items,
               'created_at': bc._now()}
    save(path, journal)
    return journal


def verify_old_entry(entry, item):
    before = item['old_entry']
    for key in ('episode_id', 'category', 'seed', 'split', 'scene', 'scene_sha256', 'accepted_episode'):
        if entry.get(key) != before.get(key):
            raise ValueError(f"{item['episode_id']}: original {key} changed while collecting")


def check_plan(root, item):
    planfile = (root / item['new_scene']).resolve()
    if not planfile.is_relative_to(root.resolve()) or bc._digest(read(planfile)) != item['new_scene_sha256']:
        raise ValueError(f"Changed/unsafe plan for {item['episode_id']}")


def prepare_retry(root, item):
    number = item['number'] + 1
    rel = f"{item['episode_id']}/attempt_{number:02d}"
    if safe_path(root, rel, item['episode_id']).exists() or (root / (rel + '.recorder.log')).exists():
        raise ValueError(f'Next candidate already exists: {rel}')
    if item.get('retry_replan'):
        scene = make_complex_scene(item['old_entry'], item['variant'], nonce=number)
        relplan = str(Path('plans') / VERSION / f"{item['episode_id']}_attempt_{number:02d}.json")
        planfile = root / relplan
        if planfile.exists():
            if bc._digest(read(planfile)) != bc._digest(scene):
                raise ValueError(f'Conflicting existing retry plan: {planfile}')
        else:
            bc._atomic_json(planfile, scene)
        item.update(new_scene=relplan, new_scene_sha256=bc._digest(scene))
    item.update(stage='prepared', number=number, candidate=rel, candidate_log=rel + '.recorder.log')
    for key in ('error', 'returncode', 'command', 'validation', 'recording_error', 'retry_replan'):
        item.pop(key, None)


def retry_failed(root, journal_path, journal, item, manifest, replan):
    candidate = safe_path(root, item['candidate'], item['episode_id'])
    log = root / item['candidate_log']
    summary = read(candidate / 'summary.json')
    if not summary.get('processes_stopped') or not summary.get('published_zero_on_exit'):
        raise bc.BatchStopped('Cannot retry: candidate has no safe teardown proof.')
    archive = PROJECT / 'reports/history' / f"{item['episode_id']}_attempt_{item['number']:02d}_{VERSION}_failed.tar.gz"
    archive_metadata(candidate, log, archive)
    failed = {'number': item['number'], 'directory': item['candidate'], 'log': item['candidate_log'],
              'status': 'failed', 'error': item['error'], 'finished_at': bc._now(),
              'scene': item['new_scene'], 'scene_sha256': item['new_scene_sha256'],
              'payload_cleanup': {'directory_removed': True, 'metadata_archive': str(archive)}}
    if not any(a['directory'] == failed['directory'] for a in item['failed_attempts']):
        item['failed_attempts'].append(failed)
    item['stage'] = 'retry_cleanup'
    item['retry_replan'] = replan
    save(journal_path, journal)
    remove_attempt(root, item['candidate'], item['episode_id'], item['candidate_log'], manifest)
    prepare_retry(root, item)
    save(journal_path, journal)


def record_item(root, journal_path, journal, item, manifest, args, ffmpeg):
    entry = entry_by_id(manifest, item['episode_id'])
    verify_old_entry(entry, item)
    check_plan(root, item)
    if item['stage'] == 'validated':
        return
    if item['stage'] in ('failed', 'retry_cleanup'):
        if not (args.retry_failed or args.replan_failed):
            raise bc.BatchStopped(f"{item['episode_id']} failed: {item.get('error')}; inspect, then use --replan-failed or --retry-failed")
        if item['stage'] == 'retry_cleanup':
            # Prior retry may have stopped immediately after deletion. Recreate only
            # the journal transition, never require the removed summary again.
            remove_attempt(root, item['candidate'], item['episode_id'], item['candidate_log'], manifest)
            prepare_retry(root, item)
            save(journal_path, journal)
        else:
            retry_failed(root, journal_path, journal, item, manifest, args.replan_failed)
    candidate = safe_path(root, item['candidate'], item['episode_id'])
    log = root / item['candidate_log']
    if item['stage'] == 'prepared':
        if candidate.exists() or log.exists():
            raise ValueError(f'Unexpected existing candidate: {candidate}')
        if shutil.disk_usage(root).free < args.min_free_gib * bc.GIB:
            raise bc.BatchStopped('Insufficient free disk space')
        command = bc.record_command(root / item['new_scene'], candidate, ffmpeg=ffmpeg,
                                    episode_wall_s=args.episode_wall_s)
        command = ['bash', str(PROJECT / 'scripts/sim.sh'), 'record', *command[3:]]
        item.update(stage='recording', command=command)
        save(journal_path, journal)
        try:
            item['returncode'] = bc.run_recorder(command, log, args.episode_wall_s + 120)
        except BaseException as error:
            item['recording_error'] = f'{type(error).__name__}: {error}'
            save(journal_path, journal)
            raise
        save(journal_path, journal)
    if item['stage'] == 'recording':
        try:
            teardown = read(candidate / 'summary.json')
        except (OSError, ValueError) as error:
            raise bc.BatchStopped('Recorder teardown unknown; inspect and stop simulator before resuming.') from error
        if not teardown.get('processes_stopped') or not teardown.get('published_zero_on_exit'):
            raise bc.BatchStopped('Recorder teardown unconfirmed; do not start another collection.')
        # A successful, fully audited summary is sufficient to recover an interrupted
        # parent whose recorder completed before the return code was journalled.
        try:
            if item.get('returncode') not in (None, 0):
                raise ValueError(f"Recorder returned {item['returncode']}")
            validation = bc.validate_episode(candidate, ffmpeg=ffmpeg,
                                             expected_scene_sha256=item['new_scene_sha256'])
            review = review_gap(candidate) if entry['category'] == 'between' else review_multi(candidate)
            if not review.get('passed') or review.get('issues'):
                raise ValueError('Independent obstacle review failed: ' + str(review.get('issues')))
            validation['complex_obstacle_review'] = review
            bc._atomic_json(candidate / 'batch_acceptance.json', validation)
            item.update(stage='validated', validation=validation)
            save(journal_path, journal)
        except (ValueError, OSError, KeyError, TypeError) as error:
            item.update(stage='failed', error=f'{type(error).__name__}: {error}')
            save(journal_path, journal)
            raise bc.BatchStopped(f"{item['episode_id']} rejected: {item['error']}") from error
    print(json.dumps({'episode': item['episode_id'], 'stage': item['stage'],
                      'obstacle_review': item.get('validation', {}).get('complex_obstacle_review')}, ensure_ascii=False), flush=True)


def check_candidate(root, item):
    """Reject changed raw data, stale proof, or a now-failing independent review."""
    candidate = safe_path(root, item['candidate'], item['episode_id'])
    acceptance = read(candidate / 'batch_acceptance.json')
    if acceptance != item['validation']:
        raise ValueError('Candidate acceptance proof changed: ' + str(candidate))
    category = item['old_entry']['category']
    proof = acceptance.get('complex_obstacle_review', {})
    digest = lambda name: hashlib.sha256((candidate / name).read_bytes()).hexdigest()
    if category == 'between':
        matching = (proof.get('steps_sha256') == digest('steps.jsonl') and
                    proof.get('metadata_sha256') == digest('episode.json'))
        actual = review_gap(candidate)
    else:
        hashes = proof.get('inputs_sha256', {})
        matching = all(hashes.get(name) == digest(name) for name in ('episode.json', 'steps.jsonl'))
        actual = review_multi(candidate)
    if not matching or not actual.get('passed') or actual.get('issues') or actual != proof:
        raise ValueError('Candidate independent review or bound raw files changed: ' + str(candidate))


def commit(root, journal_path, journal, manifest):
    # All old payloads are still intact at this point. Archive their metadata
    # before switching all entries in one atomic manifest write.
    if journal['stage'] == 'collecting':
        for item in journal['items']:
            if item['stage'] != 'validated':
                raise ValueError('Cannot commit before every candidate is validated.')
            entry = entry_by_id(manifest, item['episode_id'])
            verify_old_entry(entry, item)
            check_plan(root, item)
            check_candidate(root, item)
            old = item['old_entry']['accepted_episode']
            matches = [a for a in entry['attempts'] if a['directory'] == old and a['status'] == 'accepted']
            if len(matches) != 1:
                raise ValueError(f'Old accepted attempt is ambiguous: {old}')
            old_attempt = matches[0]
            archive = PROJECT / 'reports/history' / f"{item['episode_id']}_{Path(old).name}_{VERSION}_superseded.tar.gz"
            log = old_attempt.get('log')
            expected_log = old + '.recorder.log'
            if log is not None and log != expected_log:
                raise ValueError(f'Unexpected old log: {log}')
            archive_metadata(safe_path(root, old, item['episode_id']), root / log if log else None, archive)
            item['old_archive'] = str(archive)
        journal['stage'] = 'commit_pending'
        save(journal_path, journal)
    states = [entry_by_id(manifest, x['episode_id'])['accepted_episode'] == x['candidate'] for x in journal['items']]
    if any(states) and not all(states):
        raise ValueError('Mixed complex manifest state; refusing automatic cleanup.')
    if journal['stage'] in ('commit_pending', 'verifying_commit') and not any(states):
        for item in journal['items']:
            entry = entry_by_id(manifest, item['episode_id'])
            verify_old_entry(entry, item)
            old = item['old_entry']['accepted_episode']
            old_attempt = next(a for a in entry['attempts'] if a['directory'] == old)
            old_attempt.update(status='superseded', superseded_by=item['candidate'], superseded_at=bc._now(),
                               supersession_reason='Longer, more diverse obstacle demonstration',
                               payload_cleanup={'directory_removed': True, 'metadata_archive': item['old_archive']})
            entry['attempts'].extend(item['failed_attempts'])
            entry['attempts'].append({'number': item['number'], 'directory': item['candidate'],
                                      'log': item['candidate_log'], 'status': 'accepted',
                                      'finished_at': bc._now(), 'command': item['command']})
            entry.update(scene=item['new_scene'], scene_sha256=item['new_scene_sha256'])
            bc._accept_entry(entry, item['candidate'], item['validation'])
        bc._validate_manifest(root, manifest, manifest['seed'])
        bc._save_state(root, manifest, selected_count=200)
    journal['stage'] = 'verifying_commit'
    save(journal_path, journal)
    try:
        verify_summary(root)
    except Exception as error:
        # Old payload is retained. Restore the exact previous entries in a single write.
        for item in journal['items']:
            entry = entry_by_id(manifest, item['episode_id'])
            entry.clear()
            entry.update(deepcopy(item['old_entry']))
        bc._save_state(root, manifest, selected_count=200)
        verify_summary(root)
        journal['stage'] = 'collecting'
        journal['commit_error'] = f'{type(error).__name__}: {error}'
        save(journal_path, journal)
        raise bc.BatchStopped('Full dataset validation failed; original accepted recordings restored.') from error
    journal['stage'] = 'cleanup'
    save(journal_path, journal)


def cleanup(root, journal_path, journal, manifest):
    for item in journal['items']:
        entry = entry_by_id(manifest, item['episode_id'])
        if entry['accepted_episode'] != item['candidate']:
            raise ValueError('Accepted recording changed before cleanup')
        old = item['old_entry']['accepted_episode']
        old_attempt = next(a for a in entry['attempts'] if a['directory'] == old)
        archive = Path(old_attempt['payload_cleanup']['metadata_archive'])
        if not archive.is_file():
            raise ValueError('Required old metadata archive missing')
        remove_attempt(root, old, item['episode_id'], old_attempt.get('log'), manifest)
        item['old_payload_removed'] = True
        save(journal_path, journal)
    report = verify_summary(root)
    journal.update(stage='done', completed_at=bc._now())
    save(journal_path, journal)
    print(json.dumps({'complete': True, 'accepted_valid': report['accepted_valid'],
                      'replaced_obstacle': len(journal['items'])}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--journal', type=Path)
    p.add_argument('--limit', type=int, default=40, help='Number of interleaved entries to collect (default: all 40).')
    p.add_argument('--retry-failed', action='store_true', help='Retry the same validated plan after failure.')
    p.add_argument('--replan-failed', action='store_true', help='Create a new immutable geometry after safe teardown.')
    p.add_argument('--min-free-gib', type=float, default=3.)
    p.add_argument('--episode-wall-s', type=float, default=240.)
    p.add_argument('--ffmpeg')
    args = p.parse_args()
    if not 1 <= args.limit <= 40 or args.min_free_gib < 3 or args.episode_wall_s <= 0:
        p.error('Require limit 1..40, >=3 GiB free space, and positive timeout.')
    root = args.root.resolve()
    path = (args.journal or root / f'{VERSION}_journal.json').resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with (root / '.batch.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise bc.BatchStopped('Batch is already locked.') from error
        manifest = read(root / 'manifest.json')
        bc._validate_manifest(root, manifest, manifest['seed'])
        journal = read(path) if path.exists() else make_journal(root, path, manifest)
        if journal.get('version') != VERSION or journal.get('root') != str(root) or len(journal.get('items', [])) != 40:
            raise ValueError('Journal does not match this batch/version.')
        ffmpeg = bc.resolve_ffmpeg(args.ffmpeg)
        if journal['stage'] == 'collecting':
            for item in journal['items'][:args.limit]:
                record_item(root, path, journal, item, manifest, args, ffmpeg)
            if not all(item['stage'] == 'validated' for item in journal['items']):
                print(json.dumps({'complete': False, 'validated': sum(x['stage'] == 'validated' for x in journal['items']),
                                  'old_dataset_still_accepted': True}, ensure_ascii=False), flush=True)
                return
        if journal['stage'] in ('collecting', 'commit_pending', 'verifying_commit'):
            commit(root, path, journal, manifest)
        if journal['stage'] == 'cleanup':
            cleanup(root, path, journal, manifest)
        elif journal['stage'] == 'done':
            verify_summary(root)
            print(json.dumps({'complete': True, 'replaced_obstacle': 40}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Replace the 20 short shore demonstrations with independently audited longer ones.

Run from this project with PYTHONPATH=src /usr/bin/python3. Plans and journal are
persistent. --limit 2 collects a pilot without switching the manifest; run again
without --limit to finish. The old recordings remain accepted until every new
recording passes both the normal audit and the physical shoreline review.
"""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import sys

from usv_vla import batch_collect as bc
from usv_vla.scene_plan import generate_scene, validate_scene
from usv_vla.shore_review import require_long_shore
from usv_vla.summarize_collection import summarize
from recollect_initial_turns import archive_metadata, remove_attempt, safe_path

ROOT = bc.OUTPUTS / 'episodes/red_round_200_20261027'
PROJECT = bc.PROJECT_ROOT
VERSION = 'long_shore_v1'


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


def make_journal(root, path, manifest):
    entries = [e for e in manifest['entries'] if e['category'] == 'shore']
    if len(entries) != 20 or any(e['status'] != 'accepted' for e in entries):
        raise ValueError('Expected exactly 20 accepted shore episodes.')
    plan_dir = root / 'plans' / VERSION
    plan_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for entry in entries:
        episode_id = entry['episode_id']
        old = entry['accepted_episode']
        safe_path(root, old, episode_id)
        old_scene = read(root / entry['scene'])
        scene = generate_scene('shore', entry['category_index'], entry['seed'])
        if scene['seed'] != entry['seed'] or scene['category'] != 'shore':
            raise ValueError(f'Generated scene identity differs: {episode_id}')
        scene['collection'] = deepcopy(old_scene['collection'])
        scene['route']['cruise_speed_mps'] = old_scene['route']['cruise_speed_mps']
        scene['demonstration_standard'] = deepcopy(old_scene['demonstration_standard'])
        validate_scene(scene)
        relplan = str(Path('plans') / VERSION / (episode_id + '.json'))
        planfile = root / relplan
        if planfile.exists():
            if bc._digest(read(planfile)) != bc._digest(scene):
                raise ValueError(f'Conflicting existing long-shore plan: {planfile}')
        else:
            bc._atomic_json(planfile, scene)
        number = max((int(a['number']) for a in entry['attempts']), default=0) + 1
        rel = f'{episode_id}/attempt_{number:02d}'
        if safe_path(root, rel, episode_id).exists() or (root / (rel + '.recorder.log')).exists():
            raise ValueError(f'Candidate already exists without journal: {rel}')
        items.append({'episode_id': episode_id, 'stage': 'prepared',
                      'old_entry': deepcopy(entry), 'new_scene': relplan,
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


def retry_failed(root, journal_path, journal, item, manifest):
    candidate = safe_path(root, item['candidate'], item['episode_id'])
    log = root / item['candidate_log']
    summary = read(candidate / 'summary.json')
    if not summary.get('processes_stopped') or not summary.get('published_zero_on_exit'):
        raise bc.BatchStopped('Cannot retry: candidate has no safe teardown proof.')
    archive = PROJECT / 'reports/history' / f"{item['episode_id']}_attempt_{item['number']:02d}_{VERSION}_failed.tar.gz"
    archive_metadata(candidate, log, archive)
    failed = {'number': item['number'], 'directory': item['candidate'], 'log': item['candidate_log'],
              'status': 'failed', 'error': item['error'], 'finished_at': bc._now(),
              'payload_cleanup': {'directory_removed': True, 'metadata_archive': str(archive)}}
    if not any(a['directory'] == failed['directory'] for a in item['failed_attempts']):
        item['failed_attempts'].append(failed)
    item['stage'] = 'retry_cleanup'
    save(journal_path, journal)
    remove_attempt(root, item['candidate'], item['episode_id'], item['candidate_log'], manifest)
    number = item['number'] + 1
    rel = f"{item['episode_id']}/attempt_{number:02d}"
    if safe_path(root, rel, item['episode_id']).exists() or (root / (rel + '.recorder.log')).exists():
        raise ValueError(f'Next candidate already exists: {rel}')
    item.update(stage='prepared', number=number, candidate=rel, candidate_log=rel + '.recorder.log')
    for key in ('error', 'returncode', 'command', 'validation'):
        item.pop(key, None)
    save(journal_path, journal)


def record_item(root, journal_path, journal, item, manifest, args, ffmpeg):
    entry = entry_by_id(manifest, item['episode_id'])
    verify_old_entry(entry, item)
    check_plan(root, item)
    if item['stage'] == 'validated':
        return
    if item['stage'] in ('failed', 'retry_cleanup'):
        if not args.retry_failed:
            raise bc.BatchStopped(f"{item['episode_id']} failed: {item.get('error')}; inspect, then use --retry-failed")
        if item['stage'] == 'retry_cleanup':
            # Prior retry may have stopped immediately after deletion. Recreate only
            # the journal transition, never require the removed summary again.
            remove_attempt(root, item['candidate'], item['episode_id'], item['candidate_log'], manifest)
            number = item['number'] + 1
            rel = f"{item['episode_id']}/attempt_{number:02d}"
            if safe_path(root, rel, item['episode_id']).exists() or (root / (rel + '.recorder.log')).exists():
                raise ValueError(f'Next candidate already exists: {rel}')
            item.update(stage='prepared', number=number, candidate=rel, candidate_log=rel + '.recorder.log')
            for key in ('error', 'returncode', 'command', 'validation'):
                item.pop(key, None)
            save(journal_path, journal)
        else:
            retry_failed(root, journal_path, journal, item, manifest)
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
            validation['shore_follow_review'] = require_long_shore(candidate)
            bc._atomic_json(candidate / 'batch_acceptance.json', validation)
            item.update(stage='validated', validation=validation)
            save(journal_path, journal)
        except (ValueError, OSError, KeyError, TypeError) as error:
            item.update(stage='failed', error=f'{type(error).__name__}: {error}')
            save(journal_path, journal)
            raise bc.BatchStopped(f"{item['episode_id']} rejected: {item['error']}") from error
    print(json.dumps({'episode': item['episode_id'], 'stage': item['stage'],
                      'shore_review': item.get('validation', {}).get('shore_follow_review')}, ensure_ascii=False), flush=True)


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
        raise ValueError('Mixed shore manifest state; refusing automatic cleanup.')
    if journal['stage'] in ('commit_pending', 'verifying_commit') and not any(states):
        for item in journal['items']:
            entry = entry_by_id(manifest, item['episode_id'])
            verify_old_entry(entry, item)
            old = item['old_entry']['accepted_episode']
            old_attempt = next(a for a in entry['attempts'] if a['directory'] == old)
            old_attempt.update(status='superseded', superseded_by=item['candidate'], superseded_at=bc._now(),
                               supersession_reason='Longer, closer bank-following demonstration',
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
                      'replaced_shore': len(journal['items'])}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--journal', type=Path)
    p.add_argument('--limit', type=int, default=20, help='Number of shore entries to collect (default: all 20).')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--min-free-gib', type=float, default=3.)
    p.add_argument('--episode-wall-s', type=float, default=240.)
    p.add_argument('--ffmpeg')
    args = p.parse_args()
    if not 1 <= args.limit <= 20 or args.min_free_gib < 3 or args.episode_wall_s <= 0:
        p.error('Require limit 1..20, >=3 GiB free space, and positive timeout.')
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
        if journal.get('version') != VERSION or journal.get('root') != str(root) or len(journal.get('items', [])) != 20:
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
            print(json.dumps({'complete': True, 'replaced_shore': 20}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

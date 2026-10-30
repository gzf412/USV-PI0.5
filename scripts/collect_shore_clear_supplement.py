#!/usr/bin/env python3
"""Collect 10 clear-bank, eight-buoy demonstrations for each existing batch."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil

import numpy as np

from usv_vla import batch_collect as bc
from usv_vla.core import body_xy
from usv_vla.scene_plan import shore_clearance
from usv_vla.shore_clear_scene import generate
from usv_vla.training_selection import accepted_supplements, all_accepted_for_training
from collect_green_batch import validate_recording, active_recorder

ROOT = bc.OUTPUTS/'episodes'
BATCHES = [('red_round_200_20261027',0),('green_round_200_20261028',4),('blue_round_200_20261028',6)]
VERSION = 'shore_clear_10'


def read(path):
    return json.loads(Path(path).read_text())


def save(parent, manifest):
    root = parent/VERSION
    bc._atomic_json(root/'manifest.json', manifest)
    original = read(parent/'manifest.json')
    base = [e['accepted_episode'] for e in original['entries'] if e['status']=='accepted']
    extra = accepted_supplements(parent)
    if not all_accepted_for_training(parent):
        raise ValueError('Supplement requires all-accepted training selection')
    for kind in ('train','accepted'):
        path = parent/'splits'/f'{kind}_episodes.txt'
        temporary = path.with_suffix('.txt.tmp')
        temporary.write_text(''.join(p+'\n' for p in base+extra))
        temporary.replace(path)
    path = parent/'splits'/'validation_episodes.txt'
    temporary = path.with_suffix('.txt.tmp');temporary.write_text('');temporary.replace(path)


def initialize(parent, slot):
    root = parent/VERSION
    root.mkdir(exist_ok=True)
    (root/'plans').mkdir(exist_ok=True)
    path = root/'manifest.json'
    if path.exists():
        m = read(path)
        if m.get('kind') != VERSION or m.get('target_slot') != slot or len(m.get('entries',[])) != 10:
            raise ValueError('Unexpected supplement manifest')
        return m
    entries=[]
    for index in range(10):
        ident=f'episode_{index+1:04d}_shore_clear'
        entries.append(dict(episode_id=ident,index=index,status='planned',attempts=[]))
    m=dict(schema_version=1,kind=VERSION,target_slot=slot,entries=entries,created_at=bc._now())
    save(parent,m)
    return m


def review_clear(episode, scene):
    rows=[json.loads(line) for line in (episode/'steps.jsonl').read_text().splitlines()]
    bank=scene['route']['bank_corridor']; points=[np.asarray(row['pose_xyzrpy'][:2]) for row in rows]
    if len(scene['buoys']) != 8 or len(scene['cylinders']) != 0 or scene['route']['gates']:
        raise ValueError('Expected eight buoys, no cylinders and no obstacle gate')
    distances=[shore_clearance(point) for point in points]
    spacing=[float(np.linalg.norm(b-a)) for a,b in zip(points,points[1:])]
    lower=min(min(a,b)-s/2 for a,b,s in zip(distances,distances[1:],spacing))
    if lower < 2.5 or max(distances)>4.7:
        raise ValueError('Measured trajectory is outside the near-bank corridor')
    local=[body_xy(*(p-np.asarray(bank['origin'])),bank['yaw_rad']) for p in points]
    if max(abs(y) for _,y in local)>.75 or max(x for x,_ in local)-min(x for x,_ in local)<15:
        raise ValueError('Measured path lacks long, straight near-bank travel')
    return dict(passed=True,center_shore_lower_bound_m=lower,center_shore_max_m=max(distances),
                lateral_max_m=max(abs(y) for _,y in local),progress_m=max(x for x,_ in local)-min(x for x,_ in local),
                non_target_buoys=7,cylinders=0)


def collect(parent, slot, manifest, entry, ffmpeg, max_attempts):
    root=parent/VERSION
    if entry['status']=='accepted': return
    if entry['status']=='recording':
        record=entry['attempts'][-1]
        path=root/record['directory']
        if active_recorder(path) is not None:
            raise bc.BatchStopped('Previous recorder is still active')
        if not (path/'summary.json').is_file():
            raise bc.BatchStopped('Interrupted recorder has no confirmed teardown')
        summary=read(path/'summary.json')
        if not summary.get('processes_stopped') or not summary.get('published_zero_on_exit'):
            raise bc.BatchStopped('Interrupted recorder teardown not confirmed')
        try:
            scene=read(root/record['plan'])
            proof=validate_recording(path,scene,ffmpeg,'shore_clear')
            proof['clear_shore_review']=review_clear(path,scene)
            bc._atomic_json(path/'batch_acceptance.json',proof)
            record.update(status='accepted',finished_at=bc._now())
            entry.update(status='accepted',accepted_episode=record['directory'],validation=proof)
            save(parent,manifest);return
        except (ValueError, OSError, KeyError, TypeError) as error:
            record.update(status='failed',error=str(error),finished_at=bc._now())
            entry['status']='planned';save(parent,manifest)
            archive=root/'failed_metadata'/entry['episode_id'];archive.mkdir(parents=True,exist_ok=True)
            bc._atomic_json(archive/f'attempt_{record["number"]:02d}.json',{'summary':summary,'error':str(error)})
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe failed attempt')
            shutil.rmtree(path)
    for attempt in range(len(entry['attempts'])+1,max_attempts+1):
        if shutil.disk_usage(parent).free < 5*bc.GIB:
            raise bc.BatchStopped('Disk free space below 5 GiB')
        seed=bc._scene_seed(20261029, f'clear-shore-{slot}-{entry["index"]}',attempt)
        scene=generate(slot,entry['index'],seed)
        ident=entry['episode_id']; rel=f'{ident}/attempt_{attempt:02d}'
        path=root/rel; plan=root/'plans'/f'{ident}_attempt_{attempt:02d}.json'
        if path.exists() or plan.exists(): raise ValueError('Refusing to overwrite attempt or plan')
        bc._atomic_json(plan,scene)
        record=dict(number=attempt,directory=rel,plan=str(plan.relative_to(root)),seed=seed,status='recording')
        entry['attempts'].append(record);entry['status']='recording';save(parent,manifest)
        command=bc.record_command(plan,path,ffmpeg=ffmpeg,episode_wall_s=240.)
        command=['bash',str(bc.PROJECT_ROOT/'scripts/sim.sh'),'record',*command[3:]]
        log=root/(rel+'.recorder.log');log.parent.mkdir(parents=True,exist_ok=True)
        code=bc.run_recorder(command,log,360.)
        summary=read(path/'summary.json')
        if not summary.get('processes_stopped') or not summary.get('published_zero_on_exit'):
            raise bc.BatchStopped('Simulator teardown not confirmed')
        try:
            if code: raise ValueError(f'Recorder exit {code}: {summary.get("termination_reason")}')
            proof=validate_recording(path,scene,ffmpeg,'shore_clear')
            proof['clear_shore_review']=review_clear(path,scene)
            bc._atomic_json(path/'batch_acceptance.json',proof)
            record.update(status='accepted',finished_at=bc._now())
            entry.update(status='accepted',accepted_episode=rel,validation=proof)
            save(parent,manifest)
            print(json.dumps({'target':slot,'episode':ident,'accepted':True},ensure_ascii=False),flush=True)
            return
        except (ValueError, OSError, KeyError, TypeError) as error:
            record.update(status='failed',error=str(error),finished_at=bc._now())
            entry['status']='planned';save(parent,manifest)
            print(json.dumps({'target':slot,'episode':ident,'attempt':attempt,'error':str(error)},ensure_ascii=False),flush=True)
            archive=root/'failed_metadata'/ident;archive.mkdir(parents=True,exist_ok=True)
            bc._atomic_json(archive/f'attempt_{attempt:02d}.json',{'summary':summary,'error':str(error)})
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe failed attempt')
            shutil.rmtree(path)
    raise bc.BatchStopped(f'Attempt limit reached for {entry["episode_id"]}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ffmpeg',required=True)
    parser.add_argument('--limit-per-batch',type=int,default=10)
    parser.add_argument('--max-attempts',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.limit_per_batch<=10 or not 1<=args.max_attempts<=20: parser.error('Invalid limits')
    ffmpeg=str(bc.resolve_ffmpeg(args.ffmpeg))
    batches=[]
    for name,slot in BATCHES:
        parent=ROOT/name;manifest=initialize(parent,slot);batches.append((parent,slot,manifest))
    lockpath=ROOT/'.shore_clear_supplement.lock'
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index in range(args.limit_per_batch):
            for parent,slot,manifest in batches:
                collect(parent,slot,manifest,manifest['entries'][index],ffmpeg,args.max_attempts)
    for parent,slot,manifest in batches:
        save(parent,manifest)
        print(json.dumps({'target':slot,'accepted':sum(e['status']=='accepted' for e in manifest['entries'])}),flush=True)


if __name__=='__main__':
    main()

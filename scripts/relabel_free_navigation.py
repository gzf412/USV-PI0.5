#!/usr/bin/env python3
"""Build and audit 20 goal-only training instruction overlays.

This is a post-hoc annotation, not a new recording. It preserves raw data,
manifest, original instruction, teacher route and all acceptance evidence.
Run with --write after the read-only audit to create the map and Chinese report.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

from usv_vla import batch_collect as bc
from usv_vla.audit import audit_episode
from usv_vla.export_lerobot import training_instruction
from usv_vla.trajectory_quality import require_trajectory_quality
from review_multi_obstacle import review_episode as review_multi

ROOT = bc.OUTPUTS / 'episodes/red_round_200_20261027'
VERSION = 'free_navigation_relabel_v1'
SELECTION = [
    (1, 'Reach the red round buoy and stop.', '直达，浮标可见'),
    (7, 'Navigate to the red round buoy and stop 1.9 to 2.3 meters from its center.', '直达，浮标可见'),
    (13, 'Go to the red round buoy and stop nearby.', '直达，浮标可见'),
    (43, 'Navigate to the red round buoy, then stop between 1.9 and 2.3 meters from it.', '直达，浮标可见'),
    (85, 'Head toward the red round buoy and stop near it.', '直达，浮标可见，验证集'),
    (2, 'Find the red round buoy, approach it, and stop the boat nearby.', '直达，浮标初始在画面外'),
    (8, 'Navigate safely to the red round buoy and stop 1.9 to 2.3 meters away.', '直达，浮标初始在画面外'),
    (14, 'Steer toward the red round buoy, slow down as you approach, and stop nearby.', '直达，浮标初始在画面外'),
    (50, 'Go to the red round buoy, slow down, and come to a stop near the buoy.', '直达，浮标初始在画面外'),
    (86, 'Travel toward the red round buoy and stop at a distance of 1.9 to 2.3 meters.', '直达，浮标初始在画面外，验证集'),
    (21, 'Reach the red round buoy and bring the boat to a stop.', '原示范连续绕行两个圆筒'),
    (51, 'Reach the red round buoy, slow down, and stop 1.9 to 2.3 meters away.', '原示范连续绕行三个圆筒'),
    (63, 'Head to the red round buoy and bring the boat to a stop 1.9 to 2.3 meters away.', '原示范连续绕行三个圆筒'),
    (81, 'Approach the red round buoy and come to a stop nearby.', '原示范连续绕行两个圆筒'),
    (117, 'Go to the red round buoy and stop close by.', '原示范连续绕行三个圆筒，验证集'),
    (34, 'Drive the boat to the red round buoy, slow down and stop nearby.', '原示范连续绕行两个圆筒'),
    (46, 'Bring the boat near the red round buoy and stop.', '原示范连续绕行两个圆筒'),
    (82, 'Move toward the red round buoy, safely avoiding obstacles, and stop close to it.', '原示范连续绕行三个圆筒'),
    (124, 'Proceed to the red round buoy, avoid any obstacles, and stop nearby.', '原示范连续绕行两个圆筒'),
    (28, 'Pilot the boat to the red round buoy and stop close to the buoy.', '原示范连续绕行三个圆筒，验证集'),
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    manifest = read(root / 'manifest.json')
    if len(manifest.get('entries', [])) != 200:
        raise ValueError('Unexpected manifest size')
    indexed = {}
    for entry in manifest['entries']:
        prefix = entry['episode_id'].split('_')[1]
        if prefix in indexed:
            raise ValueError('Duplicate episode index')
        indexed[prefix] = entry
    entries = []
    for index, instruction, reason in SELECTION:
        entry = indexed[f'{index:04d}']
        if entry['status'] != 'accepted':
            raise ValueError(f'Episode {index} is not accepted')
        relative = Path(entry['accepted_episode'])
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != entry['episode_id']:
            raise ValueError('Unexpected accepted episode path')
        episode = root / relative
        if episode.is_symlink() or episode.parent.is_symlink() or not episode.resolve().is_relative_to(root):
            raise ValueError('Unsafe accepted episode path')
        scene = read(root / entry['scene'])
        metadata = read(episode / 'episode.json')
        original_instruction = metadata.get('route', {}).get('instruction')
        if not isinstance(original_instruction, str) or instruction == original_instruction:
            raise ValueError('New training instruction must differ from the original: ' + entry['episode_id'])
        scene_hash = bc._digest(scene)
        if scene_hash != entry['scene_sha256'] or scene_hash != bc._digest(metadata['scene_plan']):
            raise ValueError('Scene hash mismatch: ' + entry['episode_id'])
        summary = read(episode / 'summary.json')
        if not summary.get('success') or not summary.get('recording_valid'):
            raise ValueError('Original recording failed: ' + entry['episode_id'])
        acceptance = read(episode / 'batch_acceptance.json')
        if acceptance != entry['validation']:
            raise ValueError('Acceptance proof differs from manifest: ' + entry['episode_id'])
        quality = require_trajectory_quality(episode)
        if quality != acceptance['trajectory_quality']:
            raise ValueError('Trajectory quality evidence mismatch: ' + entry['episode_id'])
        original_audit = audit_episode(episode, check_images=True)
        if not original_audit['success'] or not original_audit['route_replayed']:
            raise ValueError('Original route replay failed: ' + entry['episode_id'])
        rows = [json.loads(line) for line in (episode / 'steps.jsonl').read_text().splitlines() if line.strip()]
        if instruction == training_instruction(metadata, rows):
            raise ValueError('Relabeled task must differ from the original task: ' + entry['episode_id'])
        category = entry['category']
        gates = len(scene['route'].get('gates', []))
        if category in ('around_left', 'around_right'):
            if not scene.get('complex_replacement') or gates not in (2, 3):
                raise ValueError('Unexpected multi-obstacle route: ' + entry['episode_id'])
            review = review_multi(episode)
            if not review['passed'] or review['issues'] or review != acceptance.get('complex_obstacle_review'):
                raise ValueError('Independent obstacle review failed: ' + entry['episode_id'])
        elif category not in ('approach_visible', 'approach_offscreen') or scene['route']['kind'] != 'direct':
            raise ValueError('Unexpected direct route: ' + entry['episode_id'])
        if not all((episode / 'videa' / (view + '.mp4')).is_file() for view in ('first_person', 'aerial_global')):
            raise ValueError('Original review videos missing')
        entries.append(dict(episode_id=entry['episode_id'], episode=relative.as_posix(), split=entry['split'],
            category=category, scene_sha256=scene_hash, steps_sha256=sha256(episode / 'steps.jsonl'),
            episode_metadata_sha256=sha256(episode / 'episode.json'), instruction=instruction,
            reason=reason, teacher_gate_count=gates))
        print('审计通过：' + entry['episode_id'], flush=True)
    if len(entries) != 20 or len({e['instruction'] for e in entries}) != 20:
        raise ValueError('Need twenty distinct instructions')
    if Counter(e['split'] for e in entries) != {'train': 16, 'validation': 4}:
        raise ValueError('Unexpected split distribution')
    if Counter(e['teacher_gate_count'] for e in entries) != {0: 10, 2: 5, 3: 5}:
        raise ValueError('Unexpected teacher gate distribution')
    return dict(schema_version=1, version=VERSION,
        provenance={'type': 'post_hoc_training_instruction_overlay',
            'description': 'Training instruction labels were added after recording. Original recordings, teacher route and validation are unchanged. Original teacher route sides and gate sequence do not constrain the relabeled instruction.',
            'scene_hash_method': 'canonical JSON SHA-256 as used by batch manifest',
            'file_hash_method': 'SHA-256 of original bytes'}, entries=entries)


def chinese_report(data):
    lines = ['# 20 条目标导航任务的训练指令', '',
      '这些指令是对已有示范做的**事后训练语言标注**，没有重新驾驶、采集或改变动作、图像、原始任务和视频。',
      '10 条原示范是直接前往，地图中仍有圆筒，只是原规划路线不要求绕行；另 10 条原示范连续绕过两个或三个圆筒。',
      '任务文本不规定避障的左右侧或顺序；部分文本明确要求安全避障。原仿真教师内部路线与通过门仅用于审核既有示范，不能作为自由任务的成功标准。',
      '模型是否能根据简略指令自主选择安全路线，必须通过后续新的闭环测试判断。训练集 16 条、验证集 4 条。', '',
      '| 回合 | 划分 | 训练指令 | 原示范 | 视频 |', '|---|---|---|---|---|']
    for e in data['entries']:
        p = e['episode']
        lines.append(f"| {e['episode_id']} | {e['split']} | {e['instruction']} | {e['reason']} | [第一视角]({p}/videa/first_person.mp4) · [空中视角]({p}/videa/aerial_global.mp4) |")
    lines.extend(['', '指令映射保存在 `instruction_overrides.json`，每条绑定原回合路径、场景摘要及原始数据文件哈希。', ''])
    return '\n'.join(lines)


def atomic_text(path, text):
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.instruction_overlay_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--write', action='store_true', help='Write instruction mapping and Chinese report after successful audit')
    args = parser.parse_args()
    root = args.root.resolve()
    data = verify(root)
    target = root / 'instruction_overrides.json'
    report = root / 'FREE_NAVIGATION_REVIEW.md'
    if args.write:
        if target.exists() and read(target) != data:
            raise ValueError('Existing mapping differs; refusing to overwrite')
        atomic_text(target, json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        atomic_text(report, chinese_report(data))
    print(json.dumps({'audited': len(data['entries']), 'wrote': args.write, 'mapping': str(target), 'report': str(report)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

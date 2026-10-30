"""Training tasks exclude evaluator-only route progress and heading."""
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

from usv_vla.export_lerobot import (export, training_instruction, instruction_for_export,
                                    _file_sha256, _scene_sha256)


class FakeDataset:
    last = None

    @classmethod
    def create(cls, **kwargs):
        Path(kwargs['root']).mkdir()
        cls.last = cls()
        cls.last.frames = []
        cls.last.root = Path(kwargs['root'])
        return cls.last

    def _get_image_file_path(self, episode_index, image_key, frame_index):
        return self.root/f'{episode_index}_{frame_index}.png'

    def add_frame(self, frame):
        if '_save_image' in self.__dict__:
            self._save_image(frame['observation.images.front'],
                             self._get_image_file_path(0, 'observation.images.front', len(self.frames)))
        self.frames.append(frame)

    def save_episode(self):
        pass

    def stop_image_writer(self):
        pass


class ExportTaskTests(unittest.TestCase):
    def recording(self, directory, *, fixed=False):
        instruction = 'Pass on the left, then stop near the red buoy.'
        route = {'instruction': instruction, 'reference_yaw_rad': .3,
                 'prompt_include_phase': not fixed}
        yaw = .1
        relative = .3-yaw
        suffix = ('Left and right use the frozen initial route heading: '
                  f'sin {math.sin(relative):.3f}, cos {math.cos(relative):.3f} '
                  'relative to the initial boat heading.')
        metadata = {'route': route}
        rows = []
        for i, phase in enumerate(('cylinder_pass', 'approach')):
            prompt = instruction if fixed else f'{instruction} Current phase: {phase}. {suffix}'
            rows.append({'decision_sim_time_ns': i*100_000_000, 'phase': phase,
                         'pose_xyzrpy': [0., 0., 0., 0., 0., yaw], 'prompt': prompt,
                         'image': f'{i}.jpg', 'state': [0.]*33, 'action': [0., 0.]})
            Image.new('RGB', (2, 2)).save(directory/f'{i}.jpg')
        (directory/'episode.json').write_text(json.dumps(metadata))
        (directory/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return metadata, rows

    def test_export_uses_fixed_task_for_legacy_and_fixed_recordings(self):
        modules = {}
        for name in ('lerobot', 'lerobot.common', 'lerobot.common.datasets',
                     'lerobot.common.datasets.lerobot_dataset'):
            modules[name] = types.ModuleType(name)
        modules['lerobot.common.datasets.lerobot_dataset'].LeRobotDataset = FakeDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode1, episode2 = root/'legacy', root/'fixed'
            episode1.mkdir(); episode2.mkdir()
            metadata1, rows1 = self.recording(episode1)
            metadata2, rows2 = self.recording(episode2, fixed=True)
            saved = (episode1/'steps.jsonl').read_bytes()
            with patch.dict(sys.modules, modules), patch('usv_vla.export_lerobot.audit_episode',
                    return_value={'success': True, 'decision_gap_s': [.1, .1], 'steps': 2}):
                export([episode1, episode2], 'local/test', root/'output')
            self.assertEqual([frame['task'] for frame in FakeDataset.last.frames],
                             [metadata1['route']['instruction']]*2+[metadata2['route']['instruction']]*2)
            self.assertNotEqual(rows1[0]['prompt'], FakeDataset.last.frames[0]['task'])
            self.assertEqual(rows2[0]['prompt'], FakeDataset.last.frames[2]['task'])
            self.assertEqual((episode1/'steps.jsonl').read_bytes(), saved)

    def test_default_export_preserves_original_jpeg_bytes(self):
        modules = {name: types.ModuleType(name) for name in ('lerobot', 'lerobot.common',
                   'lerobot.common.datasets', 'lerobot.common.datasets.lerobot_dataset')}
        modules['lerobot.common.datasets.lerobot_dataset'].LeRobotDataset = FakeDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); episode = root/'recording'; episode.mkdir()
            self.recording(episode, fixed=True)
            with patch.dict(sys.modules, modules), patch('usv_vla.export_lerobot.audit_episode',
                    return_value={'success': True, 'decision_gap_s': [.1, .1], 'steps': 2}):
                report = export([episode], 'local/test', root/'output')
            self.assertEqual((root/'output/0_0.jpg').read_bytes(), (episode/'0.jpg').read_bytes())
            self.assertEqual((root/'output/0_1.jpg').read_bytes(), (episode/'1.jpg').read_bytes())
            self.assertEqual(report['image_storage'], 'original JPEG bytes embedded in parquet')

    def test_mismatched_task_heading_phase_and_missing_instruction_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata, rows = self.recording(Path(tmp))
            for change in ('wrong_task', 'wrong_phase', 'wrong_heading', 'missing_instruction', 'wrong_scene'):
                metadata_copy = json.loads(json.dumps(metadata))
                rows_copy = json.loads(json.dumps(rows))
                if change == 'wrong_task':
                    rows_copy[1]['prompt'] = rows_copy[1]['prompt'].replace('left', 'right', 1)
                elif change == 'wrong_phase':
                    rows_copy[1]['phase'] = 'cylinder_pass'
                elif change == 'wrong_heading':
                    metadata_copy['route']['reference_yaw_rad'] = .6
                elif change == 'missing_instruction':
                    del metadata_copy['route']['instruction']
                else:
                    metadata_copy['scene_plan'] = {'route': {'instruction': 'A different task.'}}
                with self.subTest(change=change), self.assertRaises(ValueError):
                    training_instruction(metadata_copy, rows_copy)
            fixed_metadata, fixed_rows = self.recording(Path(tmp), fixed=True)
            fixed_rows[1]['prompt'] += ' Current phase: approach.'
            with self.assertRaises(ValueError):
                training_instruction(fixed_metadata, fixed_rows)

    def test_twenty_bound_overrides_are_exported_and_originals_remain_unchanged(self):
        modules = {}
        for name in ('lerobot', 'lerobot.common', 'lerobot.common.datasets',
                     'lerobot.common.datasets.lerobot_dataset'):
            modules[name] = types.ModuleType(name)
        modules['lerobot.common.datasets.lerobot_dataset'].LeRobotDataset = FakeDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'batch'
            root.mkdir()
            entries, overrides, episodes, originals = [], [], [], {}
            for i in range(21):
                episode_id = f'episode_{i:04d}_test'
                episode = root/episode_id/'attempt_01'
                episode.mkdir(parents=True)
                metadata, _ = self.recording(episode, fixed=True)
                metadata['scene_plan'] = {'route': metadata['route'], 'seed': i}
                (episode/'episode.json').write_text(json.dumps(metadata))
                relative = episode.relative_to(root).as_posix()
                scene_hash = _scene_sha256(metadata['scene_plan'])
                split = 'train' if i < 16 else 'validation'
                entries.append(dict(episode_id=episode_id, accepted_episode=relative,
                                    status='accepted', split=split, scene_sha256=scene_hash))
                if i < 20:
                    overrides.append(dict(episode_id=episode_id, episode=relative, split=split,
                                          scene_sha256=scene_hash,
                                          steps_sha256=_file_sha256(episode/'steps.jsonl'),
                                          episode_metadata_sha256=_file_sha256(episode/'episode.json'),
                                          instruction=f'Go to the red buoy and stop. {i}'))
                episodes.append(episode)
                originals[episode] = ((episode/'episode.json').read_bytes(), (episode/'steps.jsonl').read_bytes())
            (root/'manifest.json').write_text(json.dumps({'entries': entries}))
            mapping = {'schema_version': 1, 'version': 'free_navigation_relabel_v1', 'entries': overrides}
            (root/'instruction_overrides.json').write_text(json.dumps(mapping))
            with patch.dict(sys.modules, modules), patch('usv_vla.export_lerobot.audit_episode',
                    return_value={'success': True, 'decision_gap_s': [.1, .1], 'steps': 2}):
                report = export(episodes, 'local/test', Path(tmp)/'output')
            self.assertEqual([frame['task'] for frame in FakeDataset.last.frames],
                             sum(([item['instruction']]*2 for item in overrides), []) +
                             ['Pass on the left, then stop near the red buoy.']*2)
            self.assertEqual(sum(p['source'] == 'posthoc_training_override'
                                 for p in report['training_instructions']), 20)
            for episode in episodes:
                self.assertEqual(((episode/'episode.json').read_bytes(), (episode/'steps.jsonl').read_bytes()),
                                 originals[episode])
            metadata = json.loads((episodes[0]/'episode.json').read_text())
            rows = [json.loads(line) for line in (episodes[0]/'steps.jsonl').read_text().splitlines()]
            for field in ('scene_sha256', 'steps_sha256', 'episode_metadata_sha256', 'split', 'episode'):
                bad = json.loads(json.dumps(mapping))
                bad['entries'][0][field] = ('episode_0000_test/attempt_02' if field == 'episode' else 'tampered')
                (root/'instruction_overrides.json').write_text(json.dumps(bad))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    instruction_for_export(episodes[0], metadata, rows)
            (root/'instruction_overrides.json').write_text(json.dumps(mapping))
            bad_rows = json.loads(json.dumps(rows))
            bad_rows[0]['prompt'] = 'fabricated recorded prompt'
            with self.assertRaises(ValueError):
                instruction_for_export(episodes[0], metadata, bad_rows)


if __name__ == '__main__':
    unittest.main()

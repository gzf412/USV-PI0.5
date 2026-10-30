"""Recovery invariants for replacement of previously accepted recordings."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/recollect_initial_turns.py'
spec = importlib.util.spec_from_file_location('recollect_initial_turns', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write_episode(path, peak):
    path.mkdir(parents=True)
    scene = {'route': {'kind': 'cylinder_pass', 'side': 'right',
                       'gates': [{'center': [3, 0], 'yaw_rad': 0}]}}
    (path / 'episode.json').write_text(json.dumps({'scene_plan': scene}))
    yaw = -peak * 3.141592653589793 / 180
    rows = [{'pose_xyzrpy': [x, 0, 0, 0, 0, angle]} for x, angle in ((0, 0), (1, yaw), (4, 0))]
    (path / 'steps.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (path / 'summary.json').write_text(json.dumps({'processes_stopped': True, 'published_zero_on_exit': True}))


class TestRecollectionRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        self.root = self.project / 'batch'
        self.root.mkdir()
        self.episode_id = 'episode_0010_around_right'
        self.old = f'{self.episode_id}/attempt_01'
        write_episode(self.root / self.old, 50)
        self.old_log = f'{self.episode_id}/attempt_01.recorder.log'
        (self.root / self.old_log).write_text('old log')
        entry = {'episode_id': self.episode_id, 'category': 'around_right', 'status': 'accepted',
                 'accepted_episode': self.old, 'attempts': [{'number': 1, 'directory': self.old,
                 'log': self.old_log, 'status': 'accepted'}], 'scene': 'plan.json',
                 'scene_sha256': 'scenehash', 'seed': 42, 'split': 'train', 'validation': {}}
        (self.root / 'manifest.json').write_text(json.dumps({'seed': 9, 'entries': [entry]}))
        self.plan = self.project / 'plan.json'
        self.plan.write_text(json.dumps({'episode_ids': [self.episode_id]}))
        self.journal = self.project / 'journal.json'
        self.args = SimpleNamespace(root=self.root, plan=self.plan, journal=self.journal, ffmpeg=None,
                                    min_free_gib=3, episode_wall_s=240, minimum_improvement_deg=3.)
        self.proof = {'trajectory_quality': {'profile': mod.bc.QUALITY_PROFILE, 'passed': True,
                                            'issues': [], 'thresholds': {}}}
        def save_state(root, manifest, **kw):
            (root / 'manifest.json').write_text(json.dumps(manifest))
        self.patches = [patch.object(mod, 'PROJECT', self.project),
                        patch.object(mod.bc, '_validate_manifest'),
                        patch.object(mod.bc, '_save_state', side_effect=save_state),
                        patch.object(mod.bc, 'resolve_ffmpeg', return_value='/ffmpeg'),
                        patch.object(mod.bc, 'validate_episode', return_value=self.proof)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def recorder(self, command, log, timeout):
        write_episode(Path(command[command.index('--output') + 1]), 30)
        Path(log).write_text('new log')
        return 0

    def manifest_entry(self):
        return json.loads((self.root / 'manifest.json').read_text())['entries'][0]

    def test_failed_candidate_keeps_old_and_does_not_retry(self):
        def poor(command, log, timeout):
            write_episode(Path(command[command.index('--output') + 1]), 49)
            Path(log).write_text('poor log')
            return 0
        with patch.object(mod.bc, 'run_recorder', side_effect=poor) as recorder, \
             patch.object(mod, 'finish_summary'):
            mod.run(self.args)
            mod.run(self.args)
        self.assertEqual(recorder.call_count, 1)
        self.assertEqual(self.manifest_entry()['accepted_episode'], self.old)
        self.assertTrue((self.root / self.old / 'steps.jsonl').exists())
        self.assertFalse((self.root / self.episode_id / 'attempt_02').exists())
        self.assertEqual(json.loads(self.journal.read_text())['items'][self.episode_id]['stage'], 'failed')

    def test_valid_replacement_deletes_only_old_after_summary(self):
        with patch.object(mod.bc, 'run_recorder', side_effect=self.recorder) as recorder, \
             patch.object(mod, 'finish_summary'):
            self.assertEqual(mod.run(self.args), 0)
            self.assertEqual(mod.run(self.args), 0)
        self.assertEqual(recorder.call_count, 1)
        self.assertFalse((self.root / self.old).exists())
        self.assertTrue((self.root / self.episode_id / 'attempt_02').exists())
        self.assertEqual(self.manifest_entry()['accepted_episode'], f'{self.episode_id}/attempt_02')
        self.assertEqual(json.loads(self.journal.read_text())['items'][self.episode_id]['stage'], 'done')

    def test_summary_failure_rolls_back_before_deleting_old(self):
        with patch.object(mod.bc, 'run_recorder', side_effect=self.recorder) as recorder, \
             patch.object(mod, 'finish_summary', side_effect=[ValueError('summary failed'), None, None]):
            with self.assertRaises(mod.bc.BatchStopped):
                mod.run(self.args)
            self.assertTrue((self.root / self.old / 'steps.jsonl').exists())
            self.assertEqual(self.manifest_entry()['accepted_episode'], self.old)
            mod.run(self.args)
        self.assertEqual(recorder.call_count, 1)
        self.assertTrue((self.root / self.old).exists())
        self.assertFalse((self.root / self.episode_id / 'attempt_02').exists())
        self.assertEqual(json.loads(self.journal.read_text())['items'][self.episode_id]['stage'], 'failed')

    def test_interrupted_recorder_preserves_candidate_and_stops(self):
        def interrupted(command, log, timeout):
            write_episode(Path(command[command.index('--output') + 1]), 30)
            Path(log).write_text('in progress')
            raise mod.bc.BatchStopped('interrupted')
        with patch.object(mod.bc, 'run_recorder', side_effect=interrupted), patch.object(mod, 'finish_summary'):
            with self.assertRaises(mod.bc.BatchStopped):
                mod.run(self.args)
            with self.assertRaises(mod.bc.BatchStopped):
                mod.run(self.args)
        self.assertTrue((self.root / self.old).exists())
        self.assertTrue((self.root / self.episode_id / 'attempt_02').exists())
        self.assertEqual(self.manifest_entry()['accepted_episode'], self.old)

    def test_cli_entrypoint_executes_run(self):
        env = dict(os.environ, PYTHONPATH=str(SCRIPT.parents[1] / 'src'))
        result = subprocess.run(['/usr/bin/python3', str(SCRIPT), '--plan', str(self.project / 'missing.json'),
                                 '--journal', str(self.journal)], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FileNotFoundError', result.stderr)
        self.assertNotIn('NameError', result.stderr)

    def test_path_guard_refuses_alias_and_parent_traversal(self):
        for value in ('../other/attempt_02', f'{self.episode_id}/../attempt_02', '/tmp/attempt_02'):
            with self.assertRaises(ValueError):
                mod.safe_path(self.root, value, self.episode_id)


if __name__ == '__main__':
    unittest.main()

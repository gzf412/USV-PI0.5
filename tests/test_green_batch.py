"""Regression checks for green-batch payload integrity and safe failure cleanup."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts/collect_green_batch.py'
spec = importlib.util.spec_from_file_location('collect_green_batch', MODULE_PATH)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class GreenBatchSafetyTests(unittest.TestCase):
    def test_payload_fingerprint_detects_image_video_and_metadata_changes(self):
        with tempfile.TemporaryDirectory() as d:
            episode = Path(d)
            (episode/'images').mkdir(); (episode/'videa').mkdir()
            names = ['episode.json','summary.json','steps.jsonl','videa/metadata.json',
                     'videa/first_person.mp4','videa/aerial_global.mp4','images/0.jpg']
            for name in names: (episode/name).write_bytes(b'initial')
            rows = [dict(image='images/0.jpg')]
            initial = batch.payload_fingerprint(episode,rows)
            for name in ['episode.json','summary.json','videa/first_person.mp4','videa/aerial_global.mp4','images/0.jpg']:
                (episode/name).write_bytes(b'changed')
                self.assertNotEqual(initial['sha256'], batch.payload_fingerprint(episode,rows)['sha256'])
                (episode/name).write_bytes(b'initial')
            with self.assertRaises(ValueError): batch.payload_fingerprint(episode,[dict(image='../outside.jpg')])

    def test_cleanup_requires_teardown_and_never_deletes_accepted_payload(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            episode = root/'episode_0001_direct_visible'/'attempt_01'; episode.mkdir(parents=True)
            attempt = dict(directory='episode_0001_direct_visible/attempt_01',
                           log='episode_0001_direct_visible/attempt_01.recorder.log',status='failed')
            entry = dict(episode_id='episode_0001_direct_visible',status='failed',attempts=[attempt])
            manifest = dict(entries=[entry])
            summary = episode/'summary.json'
            summary.write_text(json.dumps(dict(published_zero_on_exit=False,processes_stopped=True)))
            with self.assertRaises(batch.bc.BatchStopped): batch.archive_failed(root,manifest,entry,attempt)
            self.assertTrue(summary.exists())
            summary.write_text(json.dumps(dict(published_zero_on_exit=True,processes_stopped=True)))
            entry['accepted_episode'] = attempt['directory']
            with self.assertRaises(ValueError): batch.archive_failed(root,manifest,entry,attempt)
            self.assertTrue(summary.exists())
            del entry['accepted_episode']
            batch.archive_failed(root,manifest,entry,attempt)
            self.assertFalse(episode.exists())
            self.assertEqual(attempt['cleanup'], 'done')
            self.assertTrue((root/attempt['archive']).is_file())
            batch.archive_failed(root,manifest,entry,attempt)


if __name__ == '__main__': unittest.main()

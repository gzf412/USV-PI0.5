import json
from pathlib import Path
import tempfile
import unittest

from usv_vla.training_selection import all_accepted_for_training, selected


class TrainingSelectionTests(unittest.TestCase):
    def test_selection_preserves_provenance_and_uses_every_accepted_episode(self):
        rows = [{'split': 'train', 'id': 1}, {'split': 'validation', 'id': 2}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(all_accepted_for_training(root))
            self.assertEqual(selected(rows, 'train', all_for_training=False), rows[:1])
            (root/'training_selection.json').write_text(json.dumps({'mode': 'all_accepted_for_training'}))
            enabled = all_accepted_for_training(root)
            self.assertEqual(selected(rows, 'train', all_for_training=enabled), rows)
            self.assertEqual(selected(rows, 'validation', all_for_training=enabled), [])
            self.assertEqual(rows[1]['split'], 'validation')
            (root/'training_selection.json').write_text(json.dumps({'mode': 'typo'}))
            with self.assertRaises(ValueError):
                all_accepted_for_training(root)


if __name__ == '__main__':
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from usv_vla.shore_review import require_long_shore


class ShoreReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.episode = Path(self.tmp.name)
        self.metadata = {"scene_plan": {"category": "shore", "target_xy": [27, 0],
                                       "route": {"kind": "bank_follow"}}}
        self.summary = {"done": True, "success": True, "route_complete": True}
        self.rows = [
            {"pose_xyzrpy": [x, 0, 0, 0, 0, 0], "decision_sim_time_ns": i * 100000000,
             "done": i == 2, "success": i == 2, "route_complete": i == 2}
            for i, x in enumerate([0, 12, 25])
        ]

    def save(self):
        (self.episode / "episode.json").write_text(json.dumps(self.metadata))
        (self.episode / "summary.json").write_text(json.dumps(self.summary))
        (self.episode / "steps.jsonl").write_text("\n".join(map(json.dumps, self.rows)))

    def test_reports_measured_margins_and_path(self):
        # Use a realistic 10 Hz path so the continuous-segment bound is meaningful.
        self.rows = [{"pose_xyzrpy": [i / 10, 0, 0, 0, 0, 0],
                      "decision_sim_time_ns": i * 100000000,
                      "done": i == 250, "success": i == 250, "route_complete": i == 250}
                     for i in range(251)]
        self.save()
        with patch("usv_vla.shore_review.shore_clearance", side_effect=lambda xy: 3 + xy[0] / 100):
            result = require_long_shore(self.episode)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["initial_target_distance_m"], 27)
        self.assertAlmostEqual(result["recorded_path_length_m"], 25)
        self.assertAlmostEqual(result["duration_sim_s"], 25)
        self.assertAlmostEqual(result["center_shore_distance_min_m"], 3)
        self.assertAlmostEqual(result["center_shore_distance_max_m"], 3.25)
        self.assertAlmostEqual(result["hull_shore_clearance_lower_bound_m"], 2.1)

    def test_rejects_unfinished_short_or_far_and_close_to_land(self):
        self.save()
        with patch("usv_vla.shore_review.shore_clearance", return_value=2.49):
            with self.assertRaisesRegex(ValueError, "outside 2.5"):
                require_long_shore(self.episode)
        with patch("usv_vla.shore_review.shore_clearance", return_value=4.71):
            with self.assertRaisesRegex(ValueError, "outside 2.5"):
                require_long_shore(self.episode)
        self.summary["route_complete"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "did not complete"):
            require_long_shore(self.episode)
        self.summary["route_complete"] = True
        self.metadata["scene_plan"]["target_xy"] = [15, 0]
        self.save()
        with self.assertRaisesRegex(ValueError, "outside 25"):
            require_long_shore(self.episode)


if __name__ == "__main__":
    unittest.main()

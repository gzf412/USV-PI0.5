import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from usv_vla.batch_collect import (
    QUOTAS, collect, inspect_scene_and_actions, inspect_videos, make_manifest, parser,
    record_command, validate_episode,
)


def scene(category, index, seed):
    return {"category": category, "index": index, "seed": seed,
            "initial_pose": [index, 0., 0.], "target_xy": [index + 5., 0.]}


def options(root, *flags):
    return parser().parse_args(["--output-root", str(root), *flags])


def acceptance(episode, ffmpeg, **kwargs):
    return {"steps": 12, "videos": {"first_person": {}, "aerial_global": {}},
            "trajectory_quality": {"profile": "smooth_trajectory_v1", "passed": True,
                                   "issues": [], "thresholds": {"test_fixture": True}}}


class ManifestTests(unittest.TestCase):
    def test_quota_seed_split_balance_and_stratified_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(root, scene_generator=scene)
            self.assertEqual(len(manifest["entries"]), 200)
            self.assertEqual([entry["category"] for entry in manifest["entries"][:6]], list(QUOTAS))
            self.assertEqual(len({entry["seed"] for entry in manifest["entries"]}), 200)
            for category, count in QUOTAS.items():
                entries = [entry for entry in manifest["entries"] if entry["category"] == category]
                self.assertEqual(len(entries), count)
                self.assertEqual(sum(entry["split"] == "validation" for entry in entries), count // 5)
                loaded = json.loads((root / entries[0]["scene"]).read_text())
                self.assertEqual(loaded["collection"]["episode_id"], entries[0]["episode_id"])

    def test_plan_only_writes_no_recordings_and_protects_existing_batch(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary) / "batch"
            calls = []
            self.assertEqual(collect(options(root, "--plan-only", "--limit", "6"),
                                     scene_generator=scene, recorder=lambda *args: calls.append(args)), 0)
            self.assertEqual(calls, [])
            self.assertEqual(len(list((root / "plans").glob("*.json"))), 200)
            self.assertEqual(json.loads((root / "progress.json").read_text())["accepted"], 0)
            with self.assertRaisesRegex(ValueError, "already exists"):
                collect(options(root, "--plan-only"), scene_generator=scene)
            self.assertEqual(collect(options(root, "--plan-only", "--resume"), scene_generator=scene), 0)
            plan = next((root / "plans").glob("*.json"))
            plan.write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed after creation"):
                collect(options(root, "--plan-only", "--resume"), scene_generator=scene)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.encoder = patch("usv_vla.batch_collect.resolve_ffmpeg", return_value="/example/ffmpeg")
        self.encoder.start()
        self.addCleanup(self.encoder.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_fresh_attempt_paths_retry_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            calls, validations = [], []

            def recorder(command, log, timeout):
                output = Path(command[command.index("--output") + 1])
                self.assertFalse(output.exists())
                output.mkdir()
                Path(log).write_text("recorded log\n")
                calls.append(output)
                return 1 if len(calls) == 1 else 0

            def validator(episode, ffmpeg, **kwargs):
                validations.append(episode)
                return acceptance(episode, ffmpeg)

            result = collect(options(root, "--limit", "2"), scene_generator=scene,
                             recorder=recorder, validator=validator)
            self.assertEqual(result, 0)
            self.assertEqual([path.name for path in calls], ["attempt_01", "attempt_02", "attempt_01"])
            self.assertTrue((calls[0] / "batch_failure.json").is_file())
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(progress["accepted"], 2)
            self.assertFalse(progress["complete"])
            self.assertEqual(progress["attempts"], 3)
            result = collect(options(root, "--limit", "2", "--resume"), scene_generator=scene,
                             recorder=recorder, validator=validator)
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(validations), 4)
            self.assertEqual(len((root / "index.csv").read_text().splitlines()), 201)

    def test_bad_data_never_counts_as_success_and_retries_are_finite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            calls = []

            def recorder(command, *_):
                output = Path(command[command.index("--output") + 1])
                output.mkdir()
                calls.append(output)
                return 0

            def reject(*args, **kwargs):
                raise ValueError("Causal image audit failed")

            self.assertEqual(collect(options(root, "--limit", "1"), scene_generator=scene,
                                     recorder=recorder, validator=reject), 1)
            self.assertEqual(len(calls), 3)
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(progress["accepted"], 0)
            self.assertEqual(progress["failed"], 1)
            self.assertEqual(len(list(root.rglob("batch_failure.json"))), 3)
            self.assertEqual(collect(options(root, "--limit", "1", "--resume"), scene_generator=scene,
                                     recorder=recorder, validator=reject), 1)
            self.assertEqual(len(calls), 3)

    def test_disk_floor_stops_before_starting_a_simulator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            calls = []
            disk = lambda path: SimpleNamespace(free=2 * 1024 ** 3)
            self.assertEqual(collect(options(root, "--limit", "1"), scene_generator=scene,
                                     recorder=lambda *args: calls.append(args), validator=acceptance,
                                     disk_usage=disk), 1)
            self.assertEqual(calls, [])
            progress = json.loads((root / "progress.json").read_text())
            self.assertIn("Free disk", progress["stopped_reason"])
            self.assertEqual(progress["attempts"], 0)

    def test_quality_failure_can_pause_without_spending_all_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            recordings = []

            def recorder(command, *_):
                output = Path(command[command.index("--output") + 1])
                output.mkdir()
                recordings.append(output)
                return 0

            def reject(*args, **kwargs):
                raise ValueError("Trajectory smoothness failed: repeated heading reversals")

            result = collect(options(root, "--limit", "1", "--max-attempts", "6", "--stop-on-quality-failure"),
                             scene_generator=scene, recorder=recorder, validator=reject)
            self.assertEqual(result, 1)
            self.assertEqual(len(recordings), 1)
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(progress["accepted"], 0)
            self.assertEqual(progress["quality_passed"], 0)
            self.assertIn("inspect its controller", progress["stopped_reason"])
            self.assertTrue((recordings[0] / "batch_failure.json").is_file())

    def test_failed_smoothness_proof_never_counts_as_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"

            def recorder(command, *_):
                Path(command[command.index("--output") + 1]).mkdir()
                return 0

            def failed_quality(episode, ffmpeg, **kwargs):
                proof = acceptance(episode, ffmpeg)
                proof["trajectory_quality"].update(passed=False, issues=["Repeated heading reversals"])
                return proof

            self.assertEqual(collect(options(root, "--limit", "1"), scene_generator=scene,
                                     recorder=recorder, validator=failed_quality), 1)
            manifest = json.loads((root / "manifest.json").read_text())
            progress = json.loads((root / "progress.json").read_text())
            entry = manifest["entries"][0]
            self.assertEqual(entry["status"], "failed")
            self.assertIsNone(entry["accepted_episode"])
            self.assertEqual([a["status"] for a in entry["attempts"]], ["failed"] * 3)
            self.assertEqual(progress["accepted"], 0)
            self.assertEqual(progress["quality_passed"], 0)
            self.assertFalse(progress["complete"])

    def test_validator_cannot_omit_smoothness_even_if_other_checks_claim_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"

            def recorder(command, *_):
                Path(command[command.index("--output") + 1]).mkdir()
                return 0

            def missing_quality(episode, ffmpeg, **kwargs):
                proof = acceptance(episode, ffmpeg)
                proof.pop("trajectory_quality")
                return proof

            self.assertEqual(collect(options(root, "--limit", "1", "--max-attempts", "1"),
                                     scene_generator=scene, recorder=recorder, validator=missing_quality), 1)
            updated = json.loads((root / "manifest.json").read_text())
            self.assertEqual(updated["entries"][0]["status"], "failed")
            self.assertIn("smoothness approval", updated["entries"][0]["attempts"][0]["error"])

    def test_resume_supersedes_old_recording_and_uses_next_number_after_four_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            recordings = []

            def recorder(command, *_):
                output = Path(command[command.index("--output") + 1])
                output.mkdir()
                (output / "original.txt").write_text("Do not delete the recording")
                recordings.append(output)
                return 0

            self.assertEqual(collect(options(root, "--limit", "1"), scene_generator=scene,
                                     recorder=recorder, validator=acceptance), 0)
            manifest = json.loads((root / "manifest.json").read_text())
            entry = manifest["entries"][0]
            old_relative = entry["accepted_episode"]
            original_validation = entry["validation"].copy()
            original_finished_at = entry["attempts"][0]["finished_at"]
            for number in range(2, 5):
                entry["attempts"].append({"number": number, "directory": f"{entry['episode_id']}/attempt_{number:02d}",
                                          "status": "failed", "finished_at": "earlier"})
            manifest["quality_review"].update(status="partial", pending_smoothness_review_episode_ids=[entry["episode_id"]])
            (root / "manifest.json").write_text(json.dumps(manifest))

            def revalidate(episode, ffmpeg, **kwargs):
                if episode == root / old_relative:
                    raise ValueError("Trajectory smoothness: repeated heading reversals")
                return acceptance(episode, ffmpeg)

            self.assertEqual(collect(options(root, "--limit", "1", "--resume", "--max-attempts", "6"),
                                     scene_generator=scene, recorder=recorder, validator=revalidate), 0)
            updated = json.loads((root / "manifest.json").read_text())
            entry = updated["entries"][0]
            old = entry["attempts"][0]
            self.assertEqual(old["status"], "superseded")
            self.assertEqual(old["validation_before_supersession"], original_validation)
            self.assertEqual(old["finished_at"], original_finished_at)
            self.assertIn("repeated heading reversals", old["superseded_reason"])
            self.assertTrue((root / old_relative / "original.txt").is_file())
            self.assertEqual([path.name for path in recordings], ["attempt_01", "attempt_05"])
            self.assertEqual(entry["accepted_episode"], f"{entry['episode_id']}/attempt_05")
            self.assertEqual([a["number"] for a in entry["attempts"] if a["status"] == "accepted"], [5])
            self.assertEqual(entry["quality_review"]["status"], "passed")
            self.assertEqual(entry["quality_review"]["accepted_episode"], entry["accepted_episode"])
            self.assertNotIn(entry["episode_id"], updated["quality_review"]["pending_smoothness_review_episode_ids"])
            self.assertEqual(updated["quality_review"]["passed_episode_ids"], [entry["episode_id"]])
            self.assertEqual(updated["quality_review"]["passed_count"], 1)
            self.assertEqual(updated["quality_review"]["pending_count"], 199)
            self.assertEqual(entry["validation"]["trajectory_quality"]["passed"], True)

    def test_running_resume_can_only_recover_with_smoothness_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            collect(options(root, "--plan-only", "--limit", "1"), scene_generator=scene)
            manifest = json.loads((root / "manifest.json").read_text())
            entry = manifest["entries"][0]
            relative = f"{entry['episode_id']}/attempt_01"
            (root / relative).mkdir(parents=True)
            entry.update(status="running", attempts=[{"number": 1, "directory": relative, "status": "running"}])
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(collect(options(root, "--limit", "1", "--resume"), scene_generator=scene,
                                     recorder=lambda *_: self.fail("Recovered recording must not be run again"),
                                     validator=acceptance), 0)
            updated = json.loads((root / "manifest.json").read_text())
            recovered = updated["entries"][0]
            self.assertEqual(recovered["accepted_episode"], relative)
            self.assertEqual(recovered["quality_review"]["status"], "passed")
            self.assertEqual(recovered["validation"]["trajectory_quality"]["profile"], "smooth_trajectory_v1")

    def test_supersession_is_saved_before_low_disk_stop_and_old_proof_not_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"

            def recorder(command, *_):
                Path(command[command.index("--output") + 1]).mkdir()
                return 0

            collect(options(root, "--limit", "1"), scene_generator=scene,
                    recorder=recorder, validator=acceptance)

            def reject(*_args, **_kwargs):
                raise ValueError("Trajectory smoothness: rejected")

            self.assertEqual(collect(options(root, "--limit", "1", "--resume"), scene_generator=scene,
                                     recorder=lambda *_: self.fail("Low disk must stop before recollection"),
                                     validator=reject, disk_usage=lambda _: SimpleNamespace(free=2 * 1024 ** 3)), 1)
            updated = json.loads((root / "manifest.json").read_text())
            progress = json.loads((root / "progress.json").read_text())
            entry = updated["entries"][0]
            self.assertEqual(entry["attempts"][0]["status"], "superseded")
            self.assertIsNone(entry["accepted_episode"])
            self.assertNotIn("validation", entry)
            self.assertEqual(updated["quality_review"]["passed_count"], 0)
            self.assertEqual(progress["accepted"], 0)
            self.assertEqual(progress["quality_passed"], 0)
            self.assertIn("Free disk", progress["stopped_reason"])

    def test_same_system_interpreter_and_explicit_recording_parameters(self):
        command = record_command("/batch/scene.json", "/batch/episode", python="/usr/bin/python3",
                                 ffmpeg="/encoder/ffmpeg")
        self.assertEqual(command[:3], ["/usr/bin/python3", "-m", "usv_vla.record"])
        for flag, value in [("--camera-rate-hz", "10"), ("--aerial-rate-hz", "5"),
                            ("--jpeg-quality", "90"), ("--controller", "scripted"),
                            ("--scene-json", "/batch/scene.json")]:
            self.assertEqual(command[command.index(flag) + 1], value)


class AcceptanceTests(unittest.TestCase):
    def _control_fixture(self, episode):
        source_names = ("record.py", "world.py", "core.py", "route.py", "scene_plan.py", "video.py")
        scene = {"category": "approach_visible", "initial_pose": [0, 0, 0, 0],
                 "target_xy": [8, 0], "cylinders": [{"id": 1, "x": 4, "y": 4}], "route": {}}
        description = {"scene_plan": scene,
                       "buoys": [{"slot": 0, "color": "red", "shape": "round", "x": 8, "y": 0}],
                       "obstacles": [{"model": "usv_cylinder_1", "x": 4, "y": 4}],
                       "source_fingerprints": {name: "a" * 64 for name in source_names}}
        rows = [{"decision_sim_time_ns": index * 100_000_000,
                 "pose_xyzrpy": [0, 0, 0, 0, 0, 0], "action": action}
                for index, action in enumerate(([1, 1], [2, 1], [0, 0]))]
        (episode / "episode.json").write_text(json.dumps(description))
        (episode / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows, description

    def test_variable_nonzero_thrust_and_strict_grid_are_required(self):
        visibility = {"target": {"center_forward_m": 7.5, "fully_visible": True,
                                  "occluded_by_cylinder": False}, "cylinders": [{"fully_visible": True}]}
        with tempfile.TemporaryDirectory() as temporary, \
                patch("usv_vla.scene_plan.validate_scene"), \
                patch("usv_vla.scene_plan.initial_visibility", return_value=visibility):
            episode = Path(temporary)
            rows, _ = self._control_fixture(episode)
            result = inspect_scene_and_actions(episode)
            self.assertEqual(result["thrust"]["unique_nonzero_pairs_rounded_4dp"], 2)
            self.assertEqual(result["decision_gap_s"], [.1, .1])
            rows[1]["action"] = [1, 1]
            (episode / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Nonzero thrust is constant"):
                inspect_scene_and_actions(episode)
            rows[1]["action"] = [2, 1]
            rows[1]["decision_sim_time_ns"] += 1_000_000
            (episode / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "10 Hz training grid"):
                inspect_scene_and_actions(episode)

    def test_only_planned_single_red_buoy_and_initial_visible_cylinders_are_accepted(self):
        visibility = {"target": {"center_forward_m": 7.5, "fully_visible": True,
                                  "occluded_by_cylinder": False}, "cylinders": [{"fully_visible": False}]}
        with tempfile.TemporaryDirectory() as temporary, \
                patch("usv_vla.scene_plan.validate_scene"), \
                patch("usv_vla.scene_plan.initial_visibility", return_value=visibility):
            episode = Path(temporary)
            _, description = self._control_fixture(episode)
            with self.assertRaisesRegex(ValueError, "immutable batch plan"):
                inspect_scene_and_actions(episode, expected_scene_sha256="changed")
            description["buoys"].append(description["buoys"][0].copy())
            (episode / "episode.json").write_text(json.dumps(description))
            with self.assertRaisesRegex(ValueError, "exactly one red round"):
                inspect_scene_and_actions(episode)
            description["buoys"].pop()
            description["scene_plan"]["category"] = "around_left"
            (episode / "episode.json").write_text(json.dumps(description))
            with self.assertRaisesRegex(ValueError, "cylinders in the camera"):
                inspect_scene_and_actions(episode)

    def test_video_claim_without_real_decode_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            (episode / "videa").mkdir()
            (episode / "videa/metadata.json").write_text(json.dumps({
                "timeline": {"frame_count": 3}, "aerial_available": True,
                "views": {"first_person": {"covered_output_frames": 3, "missing_output_frames": 0,
                                           "width": 1280, "height": 720,
                                           "frame_map": [{"source_time_ns": 0, "sim_time_ns": 0, "age_ns": 0}] * 3}}
            }))
            (episode / "videa/first_person.mp4").write_bytes(b"fake mp4")
            with patch("usv_vla.batch_collect.subprocess.run", side_effect=RuntimeError("decoder failed")):
                with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                    inspect_videos(episode, "/encoder")

    def test_summary_flags_and_causal_audit_are_both_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            summary = {"success": True, "recording_valid": True, "videos": {"status": "completed"},
                       "processes_stopped": True, "published_zero_on_exit": True, "steps": 3}
            (episode / "summary.json").write_text(json.dumps(summary))
            audit = lambda *args, **kwargs: {"success": False}
            with self.assertRaisesRegex(ValueError, "Causal raw"):
                validate_episode(episode, ffmpeg="/encoder", audit_fn=audit)
            summary["videos"]["status"] = "failed"
            (episode / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "review videos"):
                validate_episode(episode, ffmpeg="/encoder", audit_fn=audit)

    def test_smoothness_runs_after_existing_audits_and_persists_full_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            (episode / "summary.json").write_text(json.dumps({
                "success": True, "recording_valid": True, "videos": {"status": "completed"},
                "processes_stopped": True, "published_zero_on_exit": True, "steps": 3}))
            calls = []
            quality = {"schema_version": 1, "profile": "smooth_trajectory_v1", "passed": True,
                       "issues": [], "thresholds": {"max_turn_reversals": 1}, "metrics": {"turn_reversals": 0},
                       "steps_sha256": "a" * 64, "episode_metadata_sha256": "b" * 64}

            def audit(*_args, **_kwargs):
                calls.append("causal")
                return {"success": True}

            def videos(*_args, **_kwargs):
                calls.append("videos")
                return {"first_person": {}, "aerial_global": {}}

            def layout(*_args, **_kwargs):
                calls.append("scene")
                return {"scene_sha256": "c" * 64}

            def smoothness(path):
                self.assertEqual(path, episode)
                calls.append("smoothness")
                return quality

            with patch("usv_vla.batch_collect.inspect_scene_and_actions", side_effect=layout):
                result = validate_episode(episode, ffmpeg="/encoder", audit_fn=audit,
                                          video_fn=videos, quality_fn=smoothness)
                self.assertEqual(calls, ["causal", "videos", "scene", "smoothness"])
                self.assertEqual(result["trajectory_quality"], quality)
                self.assertEqual(json.loads((episode / "batch_acceptance.json").read_text()), result)
                (episode / "batch_acceptance.json").unlink()
                quality.update(passed=False, issues=["Repeated heading reversals"])
                with self.assertRaisesRegex(ValueError, "smoothness approval"):
                    validate_episode(episode, ffmpeg="/encoder", audit_fn=audit,
                                     video_fn=videos, quality_fn=smoothness)
                self.assertFalse((episode / "batch_acceptance.json").exists())


if __name__ == "__main__":
    unittest.main()

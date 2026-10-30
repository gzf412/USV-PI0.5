import json
import math
from pathlib import Path
import tempfile
import unittest

from usv_vla.summarize_collection import QUOTAS, _hash, summarize
from usv_vla.trajectory_quality import PROFILE, inspect_trajectory_quality


FIXTURE_STEPS = 201


def write_json(path, value):
    path.write_text(json.dumps(value))


def refresh_quality_proof(directory):
    proof_path = directory / "batch_acceptance.json"
    proof = json.loads(proof_path.read_text())
    proof["trajectory_quality"] = inspect_trajectory_quality(directory)
    write_json(proof_path, proof)
    return proof["trajectory_quality"]


def fixture(root, accepted=1):
    entries = []
    for category, count in QUOTAS.items():
        for index in range(count):
            number = len(entries)
            entries.append({"episode_id": f"episode_{number:04d}_{category}", "category": category,
                            "split": "validation" if index % 5 == 4 else "train", "seed": number,
                            "scene_sha256": f"planned-{number}", "status": "pending",
                            "accepted_episode": None, "attempts": []})
    for number, entry in enumerate(entries[:accepted]):
        accept(root, entry, number)
    write_json(root / "manifest.json", {"entries": entries})
    return entries


def accept(root, entry, offset):
    relative = f"{entry['episode_id']}/attempt_01"
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / "videa").mkdir()
    scene = {"seed": entry["seed"], "category": entry["category"], "target_xy": [offset + 8., 0.],
             "initial_pose": [offset, 0, .1, 0], "cylinders": [{"id": 1}],
             "route": {"kind": "direct", "cruise_speed_mps": .5, "instruction": "Approach red buoy."}}
    if entry["category"] == "shore":
        scene["route"]["instruction"] = "Follow the riverbank on your left."
    scene_hash = _hash(scene)
    fingerprints = {name: "a" * 64 for name in ("record.py", "world.py", "core.py", "route.py", "scene_plan.py", "video.py")}
    description = {"scene_plan": scene, "controller": "scripted", "is_trained_vla": False,
                   "source_fingerprints": fingerprints,
                   "buoys": [{"slot": 0, "color": "red", "shape": "round", "x": offset + 8., "y": 0}],
                   "task": {"stop_min_m": 1.9, "stop_max_m": 2.3, "stop_speed_mps": .2, "hold_s": 2.}}
    rows = []
    for index in range(FIXTURE_STEPS):
        fraction = index / (FIXTURE_STEPS - 1)
        progress = fraction - math.sin(2 * math.pi * fraction) / (2 * math.pi)
        x = offset + 5.9 * progress
        speed = 5.9 / 20 * (1 - math.cos(2 * math.pi * fraction))
        force = 2 * math.sin(math.pi * fraction) ** 2
        action = [force, force + .1 * math.sin(2 * math.pi * fraction) ** 2]
        state = [0.] * 33
        state[0], state[6], state[11] = speed, 1., 1.
        rows.append({"decision_sim_time_ns": index * 100_000_000, "pose_xyzrpy": [x, 0., .1, 0., 0., 0.],
                     "state": state, "action": action, "clearance_m": .5,
                     "done": index == FIXTURE_STEPS - 1})
    (directory / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {"success": True, "recording_valid": True, "published_zero_on_exit": True,
               "processes_stopped": True, "videos": {"status": "completed"}, "steps": FIXTURE_STEPS,
               "distance_m": 2.1, "speed_mps": 0., "hold_s": 2., "route_complete": True,
               "raw_counts": {"image": 4, "aerial_image": 2}}
    proof = {"scene_sha256": scene_hash, "steps": FIXTURE_STEPS, "source_fingerprints": fingerprints,
             "audit": {"success": True, "causal_alignment": True, "previous_command_alignment": True,
                       "route_replayed": True, "steps": FIXTURE_STEPS},
             "strict_10hz_grid": True, "initial_actual_visibility": {
                 "target": {"fully_visible": entry["category"] != "approach_offscreen",
                            "occluded_by_cylinder": False,
                            "entirely_outside_horizontal": entry["category"] == "approach_offscreen"},
                 "cylinders": [{"fully_visible": True}]}, "videos": {}}
    for view in ("first_person", "aerial_global"):
        path = directory / "videa" / f"{view}.mp4"
        path.write_bytes(b"predecoded-test-video-proof")
        proof["videos"][view] = {"bytes": path.stat().st_size, "decoded_frames": FIXTURE_STEPS, "width": 1280, "height": 720}
    write_json(directory / "episode.json", description)
    write_json(directory / "summary.json", summary)
    write_json(directory / "batch_acceptance.json", proof)
    write_json(directory / "videa/metadata.json", {"timeline": {"frame_count": FIXTURE_STEPS}, "aerial_available": True})
    quality = refresh_quality_proof(directory)
    if not quality["passed"]:
        raise AssertionError(quality["issues"])
    entry.update(status="accepted", accepted_episode=relative, scene_sha256=scene_hash,
                 attempts=[{"directory": relative, "status": "accepted"}])
    return directory


class SummaryTests(unittest.TestCase):
    def test_partial_summary_excludes_failed_attempts_and_lists_only_accepted_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root)
            entries[0]["attempts"].insert(0, {"directory": "old_metadata_attempt", "status": "failed",
                                             "error": "Recorded scene differs from the immutable batch plan."})
            old = root / "old_metadata_attempt"
            old.mkdir()
            write_json(old / "summary.json", {"success": True, "recording_valid": True})
            cleanup = {"episode_id": "episode_0153_around_left", "directory": "episode_0153_around_left/attempt_01",
                       "removed_at_utc": "2026-10-26T12:14:10+00:00", "steps_discarded": 0, "reason": "interrupted"}
            write_json(root / "manifest.json", {"entries": entries, "cleanup_history": [cleanup]})
            result = summarize(root)
            self.assertFalse(result["complete"])
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["accepted_valid"], 1)
            self.assertEqual(result["total_decisions"], FIXTURE_STEPS)
            self.assertEqual(result["video_count"], 2)
            self.assertEqual(result["fixed_source_grid_schedule_episodes"], 0)
            self.assertIsNone(result["maximum_decision_tick_lateness_s"])
            self.assertEqual(result["issues"], [])
            self.assertEqual(result["cleanup_history"], [cleanup])
            self.assertEqual(result["excluded_metadata_hash_attempts"], 1)
            self.assertEqual((root / "splits/accepted_episodes.txt").read_text(), entries[0]["accepted_episode"] + "\n")
            self.assertNotIn("old_metadata_attempt", (root / "splits/train_episodes.txt").read_text())
            self.assertIn("未完成", (root / "README.md").read_text())
            self.assertIn("episode_0153_around_left", (root / "README.md").read_text())
            self.assertIn("丢弃 0 个决策样本", (root / "README.md").read_text())

    def test_duplicate_physical_trajectory_is_excluded_despite_unique_scene_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root, accepted=2)
            first = root / entries[0]["accepted_episode"]
            second = root / entries[1]["accepted_episode"]
            # Keep unique scene metadata, but copy the actual measured pose/time/action.
            rows = [json.loads(line) for line in (second / "steps.jsonl").read_text().splitlines()]
            first_rows = [json.loads(line) for line in (first / "steps.jsonl").read_text().splitlines()]
            for row, original in zip(rows, first_rows):
                row["pose_xyzrpy"] = original["pose_xyzrpy"]
            (second / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            description = json.loads((second / "episode.json").read_text())
            description["buoys"][0]["x"] = 8.
            write_json(second / "episode.json", description)
            refresh_quality_proof(second)
            result = summarize(root)
            self.assertEqual(result["accepted_valid"], 1)
            self.assertTrue(any("Duplicate accepted trajectory_sha256" in issue for issue in result["issues"]))

    def test_nonred_buoy_slot_is_excluded_from_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root)
            directory = root / entries[0]["accepted_episode"]
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            rows[0]["state"][14] = 1.
            (directory / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            refresh_quality_proof(directory)
            result = summarize(root)
            self.assertEqual(result["accepted_valid"], 0)
            self.assertTrue(any("non-red buoy state slots" in issue for issue in result["issues"]))
            self.assertEqual((root / "splits/accepted_episodes.txt").read_text(), "")

    def test_fixed_source_grid_keeps_and_bounds_decision_start_clock_lateness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root)
            directory = root / entries[0]["accepted_episode"]
            description = json.loads((directory / "episode.json").read_text())
            description["decision_schedule"] = "causal_10hz_ticks_max_50ms_lateness"
            write_json(directory / "episode.json", description)
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            for row in rows:
                row["decision_latest_clock_sim_time_ns"] = row["decision_sim_time_ns"] + 20_000_000
                row["decision_tick_lateness_ns"] = 20_000_000
            (directory / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            refresh_quality_proof(directory)
            result = summarize(root)
            self.assertEqual(result["maximum_decision_tick_lateness_s"], .02)
            self.assertEqual(result["episodes"][0]["maximum_decision_tick_lateness_s"], .02)
            self.assertEqual(result["fixed_source_grid_schedule_episodes"], 1)
            self.assertNotIn("maximum_execution_tick_lateness_s", result)
            self.assertIn("known at decision start", result["decision_timing_note"])
            self.assertIn("action_publish_monotonic_ns", result["decision_timing_note"])
            self.assertIn("决策开始时已知的最新仿真 clock", (root / "README.md").read_text())
            self.assertIn("action_publish_monotonic_ns", (root / "README.md").read_text())
            rows[1]["decision_tick_lateness_ns"] = 60_000_000
            (directory / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            refresh_quality_proof(directory)
            result = summarize(root)
            self.assertEqual(result["accepted_valid"], 0)
            self.assertTrue(any("decision tick lateness" in issue for issue in result["issues"]))

    def test_complete_200_quota_split_and_unique_trajectories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root, accepted=200)
            result = summarize(root)
            self.assertTrue(result["complete"], result["issues"])
            self.assertEqual(result["accepted_by_category"], QUOTAS)
            self.assertEqual(result["accepted_by_split"], {"train": 160, "validation": 40})
            self.assertEqual(result["unique_pose_time_action_trajectory_hashes"], 200)
            self.assertEqual(result["video_count"], 400)
            self.assertEqual(result["total_decisions"], FIXTURE_STEPS * 200)
            self.assertEqual(result["smoothness_profile"], PROFILE)
            self.assertEqual(result["smoothness_passed_count"], 200)
            self.assertEqual(result["smoothness_failed_episodes"], [])
            self.assertEqual(result["smoothness_pending_episodes"], [])
            self.assertEqual(len((root / "splits/train_episodes.txt").read_text().splitlines()), 160)
            self.assertEqual(len((root / "splits/validation_episodes.txt").read_text().splitlines()), 40)

            # A previously task-successful batch must become incomplete if one
            # raw trajectory acquires repeated turns, even with its old passing
            # proof still on disk.  Use gentle yaw rates/accelerations so this
            # specifically exercises the heading-reversal quality requirement.
            directory = root / entries[0]["accepted_episode"]
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            for index, row in enumerate(rows):
                time_s = index * .1
                yaw = .15 * math.sin(2 * math.pi * time_s / 5)
                speed = math.hypot(*row["state"][:2])
                row["pose_xyzrpy"][5] = yaw
                row["state"][0] = speed * math.cos(yaw)
                row["state"][1] = -speed * math.sin(yaw)
                row["state"][2] = .15 * (2 * math.pi / 5) * math.cos(2 * math.pi * time_s / 5)
            (directory / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertTrue(json.loads((directory / "summary.json").read_text())["success"])
            result = summarize(root)
            self.assertFalse(result["complete"])
            self.assertEqual(result["accepted_valid"], 199)
            self.assertEqual(result["smoothness_passed_count"], 199)
            self.assertEqual(len(result["smoothness_failed_episodes"]), 1)
            self.assertTrue(any("heading revers" in issue for issue in result["smoothness_failed_episodes"][0]["issues"]))
            self.assertNotIn(entries[0]["accepted_episode"], (root / "splits/accepted_episodes.txt").read_text().splitlines())

    def test_missing_legacy_quality_proof_is_pending_and_not_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root)
            directory = root / entries[0]["accepted_episode"]
            path = directory / "batch_acceptance.json"
            proof = json.loads(path.read_text())
            proof.pop("trajectory_quality")
            write_json(path, proof)
            (root / "QUALITY_REVIEW.md").write_text("# Independent quality review\n")

            result = summarize(root)

            self.assertFalse(result["complete"])
            self.assertEqual(result["accepted_valid"], 0)
            self.assertEqual(result["smoothness_passed_count"], 0)
            self.assertEqual(len(result["smoothness_pending_episodes"]), 1)
            self.assertEqual(result["smoothness_failed_episodes"], [])
            self.assertTrue(result["smoothness_pending_episodes"][0]["recomputed"]["passed"])
            self.assertTrue(any("proof is missing" in issue for issue in result["issues"]))
            self.assertEqual((root / "splits/accepted_episodes.txt").read_text(), "")
            readme = (root / "README.md").read_text()
            self.assertIn("任务成功只是验收的一部分", readme)
            self.assertIn("[QUALITY_REVIEW.md](QUALITY_REVIEW.md)", readme)
            self.assertIn("等待有效证明 1 条", readme)

    def test_reasonable_large_initial_turn_is_not_mistaken_for_heading_oscillation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root, accepted=0)
            entry = next(item for item in entries if item["category"] == "approach_offscreen")
            directory = accept(root, entry, 50)
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            for index, row in enumerate(rows):
                # A slow, monotonic 70 degree initial alignment to the target.
                fraction = min(index / 100, 1.)
                yaw = math.radians(70) * (1 - fraction) ** 2
                speed = math.hypot(*row["state"][:2])
                row["pose_xyzrpy"][5] = yaw
                row["state"][0] = speed * math.cos(yaw)
                row["state"][1] = -speed * math.sin(yaw)
                row["state"][2] = -2 * math.radians(70) * (1 - fraction) / 10 if fraction < 1 else 0.
            (directory / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            description = json.loads((directory / "episode.json").read_text())
            description["scene_plan"]["initial_pose"][3] = math.radians(70)
            write_json(directory / "episode.json", description)
            entry["scene_sha256"] = _hash(description["scene_plan"])
            proof_path = directory / "batch_acceptance.json"
            proof = json.loads(proof_path.read_text())
            proof["scene_sha256"] = entry["scene_sha256"]
            write_json(proof_path, proof)
            write_json(root / "manifest.json", {"entries": entries})
            quality = refresh_quality_proof(directory)
            self.assertTrue(quality["passed"], quality["issues"])

            result = summarize(root)

            self.assertEqual(result["accepted_valid"], 1, result["issues"])
            self.assertEqual(result["smoothness_passed_count"], 1)
            self.assertAlmostEqual(result["episodes"][0]["trajectory_quality"]["metrics"]["yaw_total_variation_deg"], 70.)
            self.assertEqual(result["smoothness_failed_episodes"], [])

    def test_stale_proof_rejects_changed_raw_trajectory_even_when_it_remains_smooth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root)
            directory = root / entries[0]["accepted_episode"]
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            rows[50]["action"][0] += .05
            (directory / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertTrue(inspect_trajectory_quality(directory)["passed"])

            result = summarize(root)

            self.assertEqual(result["accepted_valid"], 0)
            self.assertEqual(len(result["smoothness_pending_episodes"]), 1)
            self.assertTrue(any("does not match" in issue and "steps_sha256" in issue for issue in result["issues"]))
            refresh_quality_proof(directory)
            self.assertEqual(summarize(root)["accepted_valid"], 1)

    def test_superseded_success_is_not_included_in_training_or_valid_quality_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root)
            relative = "old_successful_attempt"
            directory = root / relative
            directory.mkdir()
            write_json(directory / "summary.json", {"success": True, "recording_valid": True})
            entries[0]["attempts"].insert(0, {"directory": relative, "status": "superseded",
                                               "superseded_by": entries[0]["accepted_episode"],
                                               "supersession_reason": "Repeated heading corrections."})
            write_json(root / "manifest.json", {"entries": entries})

            result = summarize(root)

            self.assertEqual(result["accepted_valid"], 1)
            self.assertEqual(result["smoothness_passed_count"], 1)
            self.assertEqual(result["excluded_superseded_attempts"], 1)
            self.assertTrue(result["excluded_attempts"][0]["physical_task_success"])
            self.assertEqual(result["excluded_attempts"][0]["reason"], "Repeated heading corrections.")
            self.assertNotIn(relative, (root / "splits/accepted_episodes.txt").read_text())
            self.assertIn("被替换的旧示范即使任务成功也不列入训练", (root / "README.md").read_text())

    def test_cleaned_failed_payload_keeps_complete_batch_and_failure_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = fixture(root, accepted=200)
            cleanup = {"episode_id": "episode_0153_around_left", "directory": "interrupted_attempt",
                       "removed_at_utc": "2026-10-26T12:14:10+00:00", "steps_discarded": 0,
                       "reason": "interrupted"}
            for number, entry in enumerate(entries[:9]):
                relative = f"{entry['episode_id']}/failed_attempt"
                directory = root / relative
                directory.mkdir()
                tombstone = {"removed_at_utc": "2026-10-26T14:00:00+00:00",
                             "reason": "failed attempt payload cleanup", "removed_bytes": 1000 + number}
                write_json(directory / "episode.json", {"seed": entry["seed"]})
                write_json(directory / "summary.json", {"success": False, "recording_valid": False})
                write_json(directory / "world.json", {"category": entry["category"]})
                write_json(directory / "batch_failure.json", {"error": "Failed route audit."})
                write_json(directory / "payload_cleanup.json", tombstone)
                entry["attempts"].insert(0, {"directory": relative, "status": "failed",
                                              "error": "Failed route audit.", "payload_cleanup": tombstone})
            write_json(root / "manifest.json", {"entries": entries, "cleanup_history": [cleanup]})

            result = summarize(root)

            self.assertTrue(result["complete"], result["issues"])
            self.assertEqual(result["accepted_valid"], 200)
            self.assertEqual(result["video_count"], 400)
            self.assertEqual(len(result["excluded_attempts"]), 9)
            self.assertEqual(sum(len(entry["attempts"]) for entry in entries), 209)
            self.assertEqual(result["cleanup_history"], [cleanup])
            for row, entry in zip(result["excluded_attempts"], entries[:9]):
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["reason"], "Failed route audit.")
                self.assertEqual(row["payload_cleanup"], entry["attempts"][0]["payload_cleanup"])
                self.assertFalse(row["physical_task_success"])
                self.assertFalse(row["recording_valid"])
                self.assertNotIn(row["directory"], (root / "splits/accepted_episodes.txt").read_text().splitlines())
            readme = (root / "README.md").read_text()
            self.assertIn("其中 9 次保留失败原因配置，原始数据已清理。", readme)
            self.assertNotIn("这些尝试保留日志和原始记录", readme)
            self.assertIn("丢弃 0 个决策样本", readme)


if __name__ == "__main__":
    unittest.main()

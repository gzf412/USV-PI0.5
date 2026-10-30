"""Collect a reproducible batch of independent, audited red-round-buoy episodes.

Run through scripts/sim.sh so the system Python inherits the Gazebo environment.
No episode is accepted from a copied trajectory or from summary flags alone.
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from usv_vla.audit import audit_episode
from usv_vla.paths import OUTPUTS, PROJECT_ROOT
from usv_vla.video import resolve_ffmpeg


QUOTAS = {
    "approach_visible": 50,
    "approach_offscreen": 30,
    "around_left": 30,
    "around_right": 30,
    "between": 40,
    "shore": 20,
}
SCHEMA_VERSION = 1
DEFAULT_SEED = 20261026
GIB = 1024 ** 3
QUALITY_PROFILE = "smooth_trajectory_v1"


class BatchStopped(RuntimeError):
    """A batch cannot safely continue, but its manifest remains resumable."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _scene_seed(seed, category, index):
    value = hashlib.sha256(f"{seed}:{category}:{index}".encode()).digest()
    return int.from_bytes(value[:4], "big")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def make_manifest(root, seed=DEFAULT_SEED, scene_generator=None):
    """Write 200 immutable independent scene plans, interleaved by category.

    The first six entries cover all six categories, making --limit 6 a useful pilot.
    Splits are attached to whole scene episodes, never individual frames.
    """
    if scene_generator is None:
        from usv_vla.scene_plan import generate_scene
        scene_generator = generate_scene
    root = Path(root)
    plans = root / "plans"
    plans.mkdir()
    entries = []
    for index in range(max(QUOTAS.values())):
        for category, count in QUOTAS.items():
            if index >= count:
                continue
            number = len(entries) + 1
            episode_id = f"episode_{number:04d}_{category}"
            scene_seed = _scene_seed(seed, category, index)
            scene = scene_generator(category, index, scene_seed)
            # A constrained fresh replacement may use its own documented RNG
            # seed. Keep the manifest and recorded scene seed in agreement.
            scene_seed = int(scene.get("seed", scene_seed))
            split = "validation" if index % 5 == 4 else "train"
            scene["collection"] = {"episode_id": episode_id, "category": category,
                                   "category_index": index, "seed": scene_seed,
                                   "split": split}
            relative = f"plans/{episode_id}.json"
            _atomic_json(root / relative, scene)
            entries.append({"episode_id": episode_id, "category": category,
                            "category_index": index, "seed": scene_seed, "split": split,
                            "scene": relative, "scene_sha256": _digest(scene),
                            "status": "pending", "attempts": [], "accepted_episode": None})
    return {"schema_version": SCHEMA_VERSION, "created_at": _now(), "seed": seed,
            "target_accepted": sum(QUOTAS.values()), "quotas": dict(QUOTAS),
            "root": str(root.resolve()), "entries": entries}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_manifest(root, manifest, seed):
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("seed") != seed
            or manifest.get("quotas") != QUOTAS or len(manifest.get("entries", [])) != 200):
        raise ValueError("Existing manifest is incompatible with this seed or batch schema.")
    identifiers = set()
    seeds = set()
    for entry in manifest["entries"]:
        if entry["episode_id"] in identifiers or entry["seed"] in seeds:
            raise ValueError("Duplicate episode or scene seed in the manifest.")
        identifiers.add(entry["episode_id"])
        seeds.add(entry["seed"])
        scene_path = (root / entry["scene"]).resolve()
        if not scene_path.is_relative_to(root.resolve()):
            raise ValueError("Scene plan escapes its batch directory.")
        if _digest(_read_json(scene_path)) != entry["scene_sha256"]:
            raise ValueError(f"Scene plan was changed after creation: {scene_path}")


def record_command(scene, output, *, python=sys.executable, ffmpeg=None, episode_wall_s=240):
    command = [python, "-m", "usv_vla.record", "--scene-json", str(scene),
               "--controller", "scripted", "--camera-rate-hz", "10",
               "--aerial-rate-hz", "5", "--jpeg-quality", "90",
               "--max-wall-s", str(episode_wall_s), "--output", str(output)]
    if ffmpeg:
        command += ["--ffmpeg", str(ffmpeg)]
    return command


def _stop_recorder(process):
    """Give record's signal handler time to zero thrust and stop its own Gazebo."""
    if process.poll() is not None:
        return
    for signum, grace in ((signal.SIGINT, 60), (signal.SIGTERM, 15)):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)
    raise BatchStopped("Recorder did not complete safe teardown; inspect its Gazebo before resuming.")


def run_recorder(command, log_path, timeout_s):
    """Run exactly one fresh simulator, retaining its complete stdout/stderr log."""
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=PROJECT_ROOT, start_new_session=True)
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _stop_recorder(process)
            raise BatchStopped(f"Recorder exceeded {timeout_s:g}s; inspect teardown before resuming.")
        except BaseException:
            _stop_recorder(process)
            raise


def inspect_videos(episode, ffmpeg):
    """Fully decode both MP4s and compare real decoded frame counts to metadata."""
    episode = Path(episode)
    metadata = _read_json(episode / "videa/metadata.json")
    expected = metadata["timeline"]["frame_count"]
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        raise ValueError("Invalid expected video frame count.")
    if not metadata.get("aerial_available"):
        raise ValueError("Aerial footage was not recorded.")
    result = {}
    for view in ("first_person", "aerial_global"):
        path = episode / "videa" / f"{view}.mp4"
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty {view} MP4.")
        detail = metadata["views"][view]
        if detail.get("covered_output_frames") != expected or detail.get("missing_output_frames") != 0:
            raise ValueError(f"Incomplete frame coverage for {view}.")
        if detail.get("width") != 1280 or detail.get("height") != 720:
            raise ValueError(f"Unexpected {view} video dimensions.")
        frame_map = detail.get("frame_map", [])
        if len(frame_map) != expected or any(
                item["source_time_ns"] > item["sim_time_ns"] or item["age_ns"] > 300_000_000
                for item in frame_map):
            raise ValueError(f"Causal video source alignment failed for {view}.")
        decoded = subprocess.run(
            [ffmpeg, "-v", "error", "-xerror", "-nostdin", "-i", str(path),
             "-map", "0:v:0", "-an", "-progress", "pipe:1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120, check=True,
        )
        frames = [int(line.split("=", 1)[1]) for line in decoded.stdout.splitlines()
                  if line.startswith("frame=")]
        if not frames or frames[-1] != expected:
            raise ValueError(f"Decoded {view} frame count does not match metadata: {frames[-1:]}, {expected}.")
        result[view] = {"decoded_frames": frames[-1], "bytes": path.stat().st_size,
                        "width": detail["width"], "height": detail["height"]}
    return result


def inspect_scene_and_actions(episode, *, expected_scene_sha256=None):
    """Verify this was the planned layout and retain measurable command diversity."""
    from usv_vla.scene_plan import initial_visibility, validate_scene

    description = _read_json(episode / "episode.json")
    scene = description["scene_plan"]
    scene_hash = _digest(scene)
    if expected_scene_sha256 is not None and scene_hash != expected_scene_sha256:
        raise ValueError("Recorded scene differs from the immutable batch plan.")
    validate_scene(scene)
    buoys = description["buoys"]
    multi_red = scene.get('dataset_type') == 'red_multi_v1'
    target = next((b for b in buoys if b['slot'] == 0), None)
    if (target is None or target['color'] != 'red' or target['shape'] != 'round'
            or (not multi_red and len(buoys) != 1)):
        raise ValueError("Batch requires exactly one red round buoy and no other buoys.")
    if math.dist([target["x"], target["y"]], scene["target_xy"]) > 1e-8:
        raise ValueError("Recorded target differs from the planned buoy position.")
    if multi_red:
        actual = {b['slot']: b for b in buoys}
        planned = {b['slot']: b for b in scene['buoys']}
        if set(actual) != set(planned) or len(actual) != len(buoys):
            raise ValueError('Recorded buoy identities differ from scene plan.')
        for slot, buoy in actual.items():
            if math.dist([buoy['x'], buoy['y']], [planned[slot]['x'], planned[slot]['y']]) > 1e-6:
                raise ValueError('Recorded buoy position differs from plan.')
    cylinders = {row["model"]: row for row in description["obstacles"]
                 if row["model"].startswith("usv_cylinder_")}
    if len(cylinders) != len(scene["cylinders"]) or not 1 <= len(cylinders) <= 4:
        raise ValueError("Recorded cylinder count differs from the planned 1..4.")
    for item in scene["cylinders"]:
        actual = cylinders[f"usv_cylinder_{item['id']}"]
        if math.dist([actual["x"], actual["y"]], [item["x"], item["y"]]) > 1e-8:
            raise ValueError("Recorded cylinder differs from its planned position.")
    rows = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) < 3:
        raise ValueError("Too few decisions to verify control variation.")
    times = np.asarray([row["decision_sim_time_ns"] for row in rows], dtype=np.int64)
    gaps = np.diff(times)
    if np.any(np.abs(gaps - 100_000_000) > 100_000):
        raise ValueError("Decisions do not follow the 10 Hz training grid (100 microsecond tolerance).")
    pose = rows[0]["pose_xyzrpy"]
    visibility = initial_visibility(scene, [pose[0], pose[1], pose[2], pose[5]])
    if visibility["target"].get("center_forward_m", -1) <= 0:
        raise ValueError("Target must be ahead at the first actual observation.")
    category = scene["category"]
    if category == "approach_visible" and (
            not visibility["target"]["fully_visible"] or visibility["target"]["occluded_by_cylinder"]):
        raise ValueError("Visible-target episode did not start with an unobscured visible target.")
    if category == "approach_offscreen" and not visibility["target"].get("entirely_outside_horizontal"):
        raise ValueError("Offscreen episode did not start with the entire target outside the image.")
    if category in ("around_left", "around_right", "between") and not multi_red and not all(
            item["fully_visible"] for item in visibility["cylinders"]):
        raise ValueError("Cylinder route did not start with its cylinders in the camera image.")
    if multi_red:
        from usv_vla.core import body_xy
        objects = {b['slot']: b for b in buoys}
        views = {o['model']: o for o in visibility['buoys'] + visibility['cylinders']}
        if any(not views[name]['unobstructed_visible'] for name in scene['route'].get('required_obstacle_models', [])):
            raise ValueError('Required obstacle was not visible at the measured initial pose.')
        for row in rows:
            pose, state = row['pose_xyzrpy'], row['state']
            for slot in range(8):
                values = state[9+slot*3:12+slot*3]
                if slot not in objects:
                    if not np.array_equal(values, [0., 0., 0.]):
                        raise ValueError('Absent buoy slot is not zero.')
                else:
                    b = objects[slot]
                    relative = body_xy(b['x']-pose[0], b['y']-pose[1], pose[5])
                    if values[2] != 1 or not np.allclose(values[:2], relative, atol=1e-4):
                        raise ValueError('Recorded buoy state differs from boat-relative location.')
    actions = np.asarray([row["action"] for row in rows], dtype=float)
    if actions.shape != (len(rows), 2) or not np.isfinite(actions).all():
        raise ValueError("Invalid thrust values.")
    moving = actions[np.max(np.abs(actions), axis=1) > .05]
    if len(moving) < 2 or len(np.unique(np.round(moving, 4), axis=0)) < 2:
        raise ValueError("Nonzero thrust is constant; this is not a varied feedback demonstration.")
    fingerprints = description.get("source_fingerprints")
    if not isinstance(fingerprints, dict) or not all(
            name in fingerprints and isinstance(fingerprints[name], str) and len(fingerprints[name]) == 64
            for name in ("record.py", "world.py", "core.py", "route.py", "scene_plan.py", "video.py")):
        raise ValueError("Collection source fingerprints are incomplete.")
    thrust = {"units": "N", "sample_count": len(actions), "nonzero_sample_count": len(moving),
              "unique_pairs_rounded_4dp": len(np.unique(np.round(actions, 4), axis=0)),
              "unique_nonzero_pairs_rounded_4dp": len(np.unique(np.round(moving, 4), axis=0)),
              "standard_deviation_left_right_N": np.std(actions, axis=0).tolist(),
              "minimum_left_right_N": np.min(actions, axis=0).tolist(),
              "maximum_left_right_N": np.max(actions, axis=0).tolist(),
              "peak_absolute_left_right_N": np.max(np.abs(actions), axis=0).tolist(),
              "saturation_fraction_left_right": np.mean(np.abs(actions) >= 14.999, axis=0).tolist(),
              "nonzero_saturation_fraction_left_right": np.mean(np.abs(moving) >= 14.999, axis=0).tolist()}
    return {"scene_sha256": scene_hash, "source_fingerprints": fingerprints,
            "demonstration_standard": scene.get("demonstration_standard"),
            "initial_actual_visibility": visibility, "thrust": thrust,
            "strict_10hz_grid": True, "decision_gap_s": [float(gaps.min()) / 1e9, float(gaps.max()) / 1e9]}


def _require_quality_proof(proof):
    """Do not accept a validator result that omitted or failed smoothness."""
    if (not isinstance(proof, dict) or proof.get("profile") != QUALITY_PROFILE
            or proof.get("passed") is not True or proof.get("issues") != []
            or not isinstance(proof.get("thresholds"), dict)):
        raise ValueError("Trajectory smoothness approval is missing or failed.")
    return proof


def validate_episode(episode, *, ffmpeg, expected_scene_sha256=None,
                     audit_fn=audit_episode, video_fn=inspect_videos, quality_fn=None):
    episode = Path(episode)
    summary = _read_json(episode / "summary.json")
    if not summary.get("success") or not summary.get("recording_valid"):
        raise ValueError(f"Task or raw recording failed: {summary.get('termination_reason')}")
    if summary.get("videos", {}).get("status") != "completed":
        raise ValueError("Both review videos must complete before acceptance.")
    if not summary.get("processes_stopped") or not summary.get("published_zero_on_exit"):
        raise ValueError("Recorder did not confirm zero thrust and stopped simulator.")
    audit = audit_fn(episode, check_images=True)
    if not audit.get("success"):
        raise ValueError("Causal raw episode audit rejected task success.")
    videos = video_fn(episode, ffmpeg)
    layout = inspect_scene_and_actions(episode, expected_scene_sha256=expected_scene_sha256)
    if quality_fn is None:
        from usv_vla.trajectory_quality import require_trajectory_quality
        quality_fn = require_trajectory_quality
    quality = _require_quality_proof(quality_fn(episode))
    result = {"validated_at": _now(), "audit": audit, "videos": videos,
              "steps": summary["steps"], "termination_reason": summary.get("termination_reason"),
              "trajectory_quality": quality, **layout}
    if layout.get("demonstration_standard"):
        from usv_vla.collection_standard import inspect_standard
        alignment = inspect_standard(episode)
        if not alignment["passed"]:
            raise ValueError("Trajectory smoothness failed: early alignment: "+"; ".join(alignment["issues"]))
        result["navigation_standard"] = alignment
    _atomic_json(episode / "batch_acceptance.json", result)
    return result


def _quality_passed(entry):
    review = entry.get("quality_review", {})
    if (entry["status"] != "accepted" or not entry.get("accepted_episode")
            or review.get("status") != "passed" or review.get("profile") != QUALITY_PROFILE
            or review.get("accepted_episode") != entry["accepted_episode"]):
        return False
    try:
        _require_quality_proof(entry.get("validation", {}).get("trajectory_quality"))
    except ValueError:
        return False
    return True


def _refresh_quality_review(manifest):
    """Recompute current approvals instead of retaining an old pending list."""
    passed = [entry["episode_id"] for entry in manifest["entries"] if _quality_passed(entry)]
    approved = set(passed)
    pending = [entry["episode_id"] for entry in manifest["entries"] if entry["episode_id"] not in approved]
    review = manifest.setdefault("quality_review", {})
    review.update(profile=QUALITY_PROFILE, updated_at_utc=_now(),
                  status="passed" if not pending else ("partial" if passed else "pending"),
                  task_acceptance_is_not_demonstration_quality_approval=True,
                  passed_count=len(passed), pending_count=len(pending),
                  passed_episode_ids=passed, pending_smoothness_review_episode_ids=pending,
                  note="Current smoothness approvals are recomputed from each accepted episode and its validation proof.")
    return len(passed), len(pending)


def _accept_entry(entry, relative, validation):
    _require_quality_proof(validation.get("trajectory_quality"))
    entry.update(status="accepted", accepted_episode=relative, validation=validation,
                 quality_review={"profile": QUALITY_PROFILE, "status": "passed",
                                 "accepted_episode": relative, "validated_at": validation.get("validated_at", _now())})


def _supersede_accepted(entry, error):
    """Retain the rejected recording and proof, but remove its active acceptance."""
    previous = entry["accepted_episode"]
    matches = [attempt for attempt in entry["attempts"] if attempt["directory"] == previous]
    if len(matches) != 1:
        raise BatchStopped("Accepted episode does not identify exactly one tracked attempt.")
    attempt = matches[0]
    reason = f"Resume validation rejected: {type(error).__name__}: {error}"
    attempt.update(status="superseded", validation_before_supersession=deepcopy(entry.get("validation", {})),
                   superseded_at=_now(), superseded_reason=reason, reason=reason)
    attempt.setdefault("finished_at", attempt["superseded_at"])
    entry.update(status="failed", accepted_episode=None, resume_rejection=str(error),
                 quality_review={"profile": QUALITY_PROFILE, "status": "pending", "rejected_episode": previous,
                                 "reason": str(error), "updated_at": attempt["superseded_at"]})
    entry.pop("validation", None)


def _validated(validator, episode, ffmpeg, scene_sha256):
    validation = validator(episode, ffmpeg=ffmpeg, expected_scene_sha256=scene_sha256)
    _require_quality_proof(validation.get("trajectory_quality"))
    return validation


def _save_state(root, manifest, *, selected_count, stopped_reason=None):
    entries = manifest["entries"]
    accepted = [entry for entry in entries if entry["status"] == "accepted"]
    quality_passed, quality_pending = _refresh_quality_review(manifest)
    progress = {"updated_at": _now(), "target_accepted": manifest["target_accepted"],
                "selected_count": selected_count, "accepted": len(accepted),
                "failed": sum(entry["status"] == "failed" for entry in entries),
                "pending": sum(entry["status"] == "pending" for entry in entries),
                "running": sum(entry["status"] == "running" for entry in entries),
                "accepted_by_category": dict(Counter(entry["category"] for entry in accepted)),
                "accepted_by_split": dict(Counter(entry["split"] for entry in accepted)),
                "attempts": sum(len(entry["attempts"]) for entry in entries),
                "quality_passed": quality_passed, "quality_pending": quality_pending,
                "complete": len(accepted) == manifest["target_accepted"] and quality_pending == 0,
                "stopped_reason": stopped_reason}
    manifest["updated_at"] = progress["updated_at"]
    _atomic_json(root / "manifest.json", manifest)
    _atomic_json(root / "progress.json", progress)
    descriptor, temporary = tempfile.mkstemp(prefix=".index.", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            fields = ["episode_id", "category", "split", "status", "seed", "accepted_episode", "steps", "attempts"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for entry in entries:
                row = {key: entry.get(key) for key in fields}
                row.update(steps=entry.get("validation", {}).get("steps", ""), attempts=len(entry["attempts"]))
                writer.writerow(row)
        os.replace(temporary, root / "index.csv")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return progress


def collect(args, *, scene_generator=None, recorder=run_recorder, validator=validate_episode,
            disk_usage=shutil.disk_usage):
    """Persist progress after every attempt. Failed attempts never increase accepted."""
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".batch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BatchStopped("Another collector is already using this batch directory.") from error
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            if not args.resume:
                raise ValueError("This batch already exists; use --resume or choose a new output root.")
            manifest = _read_json(manifest_path)
            _validate_manifest(root, manifest, args.seed)
        else:
            if any(path.name != ".batch.lock" for path in root.iterdir()):
                raise ValueError("Output root is not empty and has no manifest; choose a new directory.")
            manifest = make_manifest(root, args.seed, scene_generator)
        selected = manifest["entries"][:args.limit] if args.limit else manifest["entries"]
        _save_state(root, manifest, selected_count=len(selected))
        if args.plan_only:
            print(json.dumps({"manifest": str(manifest_path), "planned": len(manifest["entries"]),
                              "selected": len(selected), "plan_only": True}), flush=True)
            return 0
        binary = resolve_ffmpeg(args.ffmpeg)
        stopped_reason = None
        try:
            for entry in selected:
                if entry["status"] == "accepted":
                    try:
                        validation = _validated(validator, root / entry["accepted_episode"], binary,
                                                entry["scene_sha256"])
                    except Exception as error:
                        _supersede_accepted(entry, error)
                        _save_state(root, manifest, selected_count=len(selected))
                    else:
                        _accept_entry(entry, entry["accepted_episode"], validation)
                        _save_state(root, manifest, selected_count=len(selected))
                        print(json.dumps({"episode": entry["episode_id"], "status": "skipped_verified"}), flush=True)
                        continue
                elif entry["status"] == "running":
                    # A prior interrupted attempt is retained and rechecked, never overwritten.
                    previous = entry["attempts"][-1]
                    try:
                        validation = _validated(validator, root / previous["directory"], binary,
                                                entry["scene_sha256"])
                    except Exception as error:
                        previous.update(status="interrupted", error=str(error), finished_at=_now())
                        entry["status"] = "failed"
                    else:
                        previous.update(status="accepted", finished_at=_now())
                        _accept_entry(entry, previous["directory"], validation)
                        _save_state(root, manifest, selected_count=len(selected))
                        continue
                while entry["status"] != "accepted" and len(entry["attempts"]) < args.max_attempts:
                    free = disk_usage(root).free
                    if free < args.min_free_gib * GIB:
                        raise BatchStopped(f"Free disk {free / GIB:.2f} GiB is below {args.min_free_gib:g} GiB.")
                    attempt_index = len(entry["attempts"]) + 1
                    relative = f"{entry['episode_id']}/attempt_{attempt_index:02d}"
                    episode = root / relative
                    episode.parent.mkdir(exist_ok=True)
                    if episode.exists():
                        raise BatchStopped(f"Untracked attempt directory already exists; inspect it: {episode}")
                    log = episode.parent / f"attempt_{attempt_index:02d}.recorder.log"
                    attempt = {"number": attempt_index, "directory": relative,
                               "log": str(log.relative_to(root)), "status": "running", "started_at": _now()}
                    entry["attempts"].append(attempt)
                    entry["status"] = "running"
                    _save_state(root, manifest, selected_count=len(selected))
                    command = record_command(root / entry["scene"], episode, ffmpeg=binary,
                                             episode_wall_s=args.episode_wall_s)
                    attempt["command"] = command
                    begin = time.monotonic()
                    try:
                        returncode = recorder(command, log, args.episode_wall_s + 120)
                        attempt["returncode"] = returncode
                        if returncode != 0:
                            raise ValueError(f"Recorder exited with code {returncode}; see {log}")
                        validation = _validated(validator, episode, binary, entry["scene_sha256"])
                        _accept_entry(entry, relative, validation)
                        attempt["status"] = "accepted"
                    except BatchStopped:
                        raise
                    except Exception as error:
                        entry["status"] = "failed"
                        attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
                        episode.mkdir(exist_ok=True)
                        _atomic_json(episode / "batch_failure.json", {**attempt, "scene": entry["scene"]})
                        if args.stop_on_quality_failure and str(error).startswith("Trajectory smoothness failed:"):
                            raise BatchStopped(f"Trajectory quality rejected {entry['episode_id']}; inspect its controller before retrying: {error}") from error
                    finally:
                        attempt.update(finished_at=_now(), wall_s=time.monotonic() - begin)
                        progress = _save_state(root, manifest, selected_count=len(selected))
                        print(json.dumps({"episode": entry["episode_id"], "category": entry["category"],
                                          "attempt": attempt_index, "status": attempt["status"],
                                          "accepted": progress["accepted"], "target": manifest["target_accepted"],
                                          "failed": progress["failed"], "error": attempt.get("error")},
                                         ensure_ascii=False), flush=True)
        except (BatchStopped, KeyboardInterrupt) as error:
            stopped_reason = str(error) or "Interrupted by user"
            print(json.dumps({"stopped": stopped_reason}, ensure_ascii=False), flush=True)
        finally:
            progress = _save_state(root, manifest, selected_count=len(selected), stopped_reason=stopped_reason)
        return 0 if all(_quality_passed(entry) for entry in selected) and not stopped_reason else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path,
                        default=OUTPUTS / "episodes/red_round_200_20261027")
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--limit", type=int, help="Select the first N interleaved plans; 6 covers every category")
    result.add_argument("--max-attempts", type=int, default=3, help="Total attempts per scene, including earlier runs")
    result.add_argument("--stop-on-quality-failure", action="store_true",
                        help="Pause after a new trajectory fails smoothness instead of repeating the same controller")
    result.add_argument("--min-free-gib", type=float, default=3.)
    result.add_argument("--episode-wall-s", type=float, default=240.)
    result.add_argument("--ffmpeg")
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 200:
        cli.error("--limit must be between 1 and 200")
    if not 1 <= args.max_attempts <= 10:
        cli.error("--max-attempts must be between 1 and 10")
    if args.min_free_gib < 3 or args.episode_wall_s <= 0:
        cli.error("--min-free-gib must be at least 3; --episode-wall-s must be positive")
    try:
        raise SystemExit(collect(args))
    except (ValueError, OSError, BatchStopped) as error:
        cli.exit(1, f"Batch collection failed: {error}\n")


if __name__ == "__main__":
    main()

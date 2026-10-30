"""Summarize accepted simulation demonstrations without rerunning video decoding.

Only accepted, internally consistent episodes enter the emitted split lists.
Decode evidence comes from each immutable episode's batch_acceptance.json.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from usv_vla.trajectory_quality import PROFILE as SMOOTHNESS_PROFILE, inspect_trajectory_quality
from usv_vla.training_selection import all_accepted_for_training, selected, accepted_supplements


QUOTAS = {"approach_visible": 50, "approach_offscreen": 30, "around_left": 30,
          "around_right": 30, "between": 40, "shore": 20}
FIXED_SCHEDULE = "causal_10hz_ticks_max_50ms_lateness"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _safe_episode(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Accepted episode path must be relative to the batch root.")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Accepted episode path escapes the batch root.")
    return result


def _distribution(values):
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Cannot summarize nonfinite measurements.")
    return {"min": float(data.min()), "mean": float(data.mean()), "median": float(np.median(data)),
            "p90": float(np.quantile(data, .9)), "max": float(data.max())}


def _smoothness_review(root, entry):
    """Recompute quality from raw data; a historical success flag is insufficient."""
    review = {"episode_id": entry["episode_id"], "directory": entry["accepted_episode"],
              "category": entry["category"], "status": "failed", "issues": [],
              "profile": SMOOTHNESS_PROFILE, "recomputed": None, "proof_matches": False}
    try:
        directory = _safe_episode(root, entry["accepted_episode"])
        quality = inspect_trajectory_quality(directory)
        review["recomputed"] = quality
        if (quality.get("profile") != SMOOTHNESS_PROFILE or quality.get("passed") is not True
                or quality.get("issues")):
            review["issues"] = quality.get("issues") or ["Independent trajectory smoothness inspection did not pass."]
            return review
        review["status"] = "pending"
        proof = _json(directory / "batch_acceptance.json").get("trajectory_quality")
        if not isinstance(proof, dict):
            review["issues"] = ["Trajectory smoothness acceptance proof is missing; legacy task success is not sufficient."]
            return review
        # The inspector is deterministic. Check the raw-file fingerprints,
        # inspector implementation, thresholds, and every computed metric so a
        # passing proof copied from an earlier trajectory cannot certify new data.
        keys = ("schema_version", "profile", "passed", "issues", "thresholds", "metrics",
                "category", "route_kind", "steps_sha256", "episode_metadata_sha256", "inspector_sha256")
        missing = [key for key in keys if key not in quality or key not in proof]
        changed = [key for key in keys if key not in missing and _canonical(proof[key]) != _canonical(quality[key])]
        if missing or changed:
            review["issues"] = ["Trajectory smoothness proof does not match the current raw trajectory or inspector: "
                                + ", ".join(missing + changed) + "."]
            return review
        review.update(status="passed", proof_matches=True, issues=[])
    except (OSError, ValueError, KeyError, TypeError) as error:
        review["issues"] = [f"Trajectory smoothness inspection unavailable: {type(error).__name__}: {error}"]
    return review


def _smoothness_distributions(reviews):
    values = defaultdict(list)

    def collect(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                collect(f"{prefix}.{key}" if prefix else key, item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value):
                values[prefix].append(value)

    for review in reviews:
        quality = review.get("recomputed")
        if isinstance(quality, dict):
            collect("", quality.get("metrics", {}))
    return {key: _distribution(items) for key, items in sorted(values.items())}


class ActionStatistics:
    def __init__(self):
        self.count = 0
        self.sum = np.zeros(2)
        self.squares = np.zeros(2)
        self.minimum = np.full(2, np.inf)
        self.maximum = np.full(2, -np.inf)
        self.nonzero = 0
        self.unique = set()
        self.unique_nonzero = set()
        self.saturated = np.zeros(2, dtype=np.int64)

    def add(self, actions):
        actions = np.asarray(actions, dtype=float)
        self.count += len(actions)
        self.sum += actions.sum(axis=0)
        self.squares += np.square(actions).sum(axis=0)
        self.minimum = np.minimum(self.minimum, actions.min(axis=0))
        self.maximum = np.maximum(self.maximum, actions.max(axis=0))
        self.saturated += (np.abs(actions) >= 14.999).sum(axis=0)
        rounded = [tuple(row) for row in np.round(actions, 4)]
        self.unique.update(rounded)
        moving = np.max(np.abs(actions), axis=1) > .05
        self.nonzero += int(moving.sum())
        self.unique_nonzero.update(pair for pair, yes in zip(rounded, moving) if yes)

    def result(self):
        if not self.count:
            return {"units": "N", "samples": 0}
        mean = self.sum / self.count
        return {"units": "N", "samples": self.count, "nonzero_samples": self.nonzero,
                "mean_left_right": mean.tolist(),
                "std_left_right": np.sqrt(np.maximum(0, self.squares / self.count - mean ** 2)).tolist(),
                "min_left_right": self.minimum.tolist(), "max_left_right": self.maximum.tolist(),
                "unique_pairs_rounded_4dp": len(self.unique),
                "unique_nonzero_pairs_rounded_4dp": len(self.unique_nonzero),
                "nonzero_unique_ratio": len(self.unique_nonzero) / self.nonzero if self.nonzero else 0,
                "saturation_fraction_left_right": (self.saturated / self.count).tolist()}


def _episode(root, entry):
    directory = _safe_episode(root, entry["accepted_episode"])
    description = _json(directory / "episode.json")
    summary = _json(directory / "summary.json")
    proof = _json(directory / "batch_acceptance.json")
    scene = description["scene_plan"]
    if entry["scene_sha256"] != _hash(scene) or proof["scene_sha256"] != entry["scene_sha256"]:
        raise ValueError("Accepted scene hash does not match its plan and audit proof.")
    if scene["seed"] != entry["seed"] or scene["category"] != entry["category"]:
        raise ValueError("Accepted scene seed or category differs from the manifest.")
    if not (summary["success"] and summary["recording_valid"] and summary["published_zero_on_exit"]
            and summary["processes_stopped"] and summary["videos"]["status"] == "completed"):
        raise ValueError("Accepted episode no longer has a valid successful summary.")
    audit = proof["audit"]
    if not all(audit.get(key) for key in ("success", "causal_alignment", "previous_command_alignment", "route_replayed")):
        raise ValueError("Required independent route and causal audit proof is missing.")
    if not proof.get("strict_10hz_grid"):
        raise ValueError("Strict 10 Hz audit proof is missing.")
    if description.get("controller") != "scripted" or description.get("is_trained_vla") is not False:
        raise ValueError("Expected scripted simulator demonstrations, not model evaluation data.")
    buoys = description["buoys"]
    multi_red = scene.get('dataset_type') == 'red_multi_v1'
    target_buoy = next((b for b in buoys if b['slot'] == 0), None)
    if target_buoy is None or (target_buoy['color'], target_buoy['shape']) != ('red', 'round') or (not multi_red and len(buoys) != 1):
        raise ValueError("Accepted layout must contain only the red round buoy.")
    if multi_red and ({b['slot'] for b in buoys} != {b['slot'] for b in scene['buoys']} or len({b['slot'] for b in buoys}) != len(buoys)):
        raise ValueError('Recorded multi-buoy identities differ from the scene.')
    if not 1 <= len(scene["cylinders"]) <= 4:
        raise ValueError("Accepted layout requires 1..4 cylinders.")
    rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines() if line.strip()]
    if not rows or len(rows) != summary["steps"] or len(rows) != proof["steps"] or len(rows) != audit["steps"]:
        raise ValueError("Decision count does not match summary and acceptance proof.")
    states = np.asarray([row["state"] for row in rows], dtype=float)
    actions = np.asarray([row["action"] for row in rows], dtype=float)
    poses = np.asarray([row["pose_xyzrpy"] for row in rows], dtype=float)
    if (states.shape != (len(rows), 33) or actions.shape != (len(rows), 2)
            or poses.shape != (len(rows), 6) or not all(np.isfinite(a).all() for a in (states, actions, poses))):
        raise ValueError("Invalid state, action or measured pose arrays.")
    if np.any(np.abs(actions) > 15) or not np.all(states[:, 11] == 1) or (not multi_red and not np.all(states[:, 12:] == 0)):
        raise ValueError("Force bound exceeded or non-red buoy state slots are not absent.")
    if multi_red:
        from usv_vla.core import body_xy
        by_slot = {b['slot']: b for b in buoys}
        for pose, state in zip(poses, states):
            for slot in range(8):
                values = state[9+3*slot:12+3*slot]
                if slot not in by_slot:
                    if not np.array_equal(values, [0., 0., 0.]):
                        raise ValueError('Absent buoy slot must be zero.')
                else:
                    b = by_slot[slot]
                    expected = body_xy(b['x']-pose[0], b['y']-pose[1], pose[5])
                    if values[2] != 1 or not np.allclose(values[:2], expected, atol=1e-4):
                        raise ValueError('Multi-buoy state does not match measured pose.')
    times = np.asarray([row["decision_sim_time_ns"] for row in rows], dtype=np.int64)
    gaps = np.diff(times)
    if len(gaps) and np.any(np.abs(gaps - 100_000_000) > 100_000):
        raise ValueError("Actual decisions violate the strict 10 Hz grid.")
    lateness = []
    schedule = description.get("decision_schedule")
    if schedule == FIXED_SCHEDULE:
        for row in rows:
            latest = row["decision_latest_clock_sim_time_ns"]
            late = row["decision_tick_lateness_ns"]
            if (not isinstance(latest, int) or not isinstance(late, int)
                    or latest - row["decision_sim_time_ns"] != late or not 0 <= late <= 50_000_000):
                raise ValueError("Fixed-source-grid decision tick lateness must match the latest clock known at decision start and be <=50ms.")
            lateness.append(late / 1e9)
    first = poses[0]
    dx, dy = target_buoy["x"] - first[0], target_buoy["y"] - first[1]
    forward = math.cos(first[5]) * dx + math.sin(first[5]) * dy
    left = -math.sin(first[5]) * dx + math.cos(first[5]) * dy
    if forward <= 0:
        raise ValueError("Target was not ahead at the first actual pose.")
    visibility = proof["initial_actual_visibility"]
    target = visibility["target"]
    category = entry["category"]
    if category == "approach_visible" and (not target["fully_visible"] or target["occluded_by_cylinder"]):
        raise ValueError("Visible-target acceptance proof is inconsistent.")
    if category == "approach_offscreen" and not target.get("entirely_outside_horizontal"):
        raise ValueError("Offscreen-target acceptance proof is inconsistent.")
    if category in ("around_left", "around_right", "between") and not multi_red and not all(
            item["fully_visible"] for item in visibility["cylinders"]):
        raise ValueError("Required cylinder initial-visibility proof is inconsistent.")
    video_metadata = _json(directory / "videa/metadata.json")
    expected_frames = video_metadata["timeline"]["frame_count"]
    if not isinstance(expected_frames, int) or isinstance(expected_frames, bool) or expected_frames <= 0:
        raise ValueError("Video frame count must be a positive integer.")
    if not video_metadata.get("aerial_available"):
        raise ValueError("Aerial video metadata is missing.")
    video_bytes = 0
    for view in ("first_person", "aerial_global"):
        path = directory / "videa" / f"{view}.mp4"
        detail = proof["videos"][view]
        if (not path.is_file() or path.stat().st_size != detail["bytes"] or detail["decoded_frames"] != expected_frames
                or detail["width"] != 1280 or detail["height"] != 720):
            raise ValueError("MP4 file does not match its existing full-decode proof.")
        video_bytes += path.stat().st_size
    trajectory = hashlib.sha256()
    for row in rows:
        trajectory.update(_canonical([row["decision_sim_time_ns"], row["pose_xyzrpy"], row["action"]]))
        trajectory.update(b"\n")
    distance = math.dist(poses[-1, :2], [target_buoy["x"], target_buoy["y"]])
    speed = float(np.linalg.norm(states[-1, :2]))
    if abs(distance - summary["distance_m"]) > 1e-5 or abs(speed - summary["speed_mps"]) > 1e-5:
        raise ValueError("Final physical distance/speed differ from the successful summary.")
    config = description["task"]
    if not config["stop_min_m"] <= distance <= config["stop_max_m"] or speed > config["stop_speed_mps"]:
        raise ValueError("Accepted final stop is outside its required band or speed limit.")
    if summary["hold_s"] < config["hold_s"] - 1e-6 or not summary["route_complete"]:
        raise ValueError("Accepted route or low-speed dwell is incomplete.")
    local_actions = ActionStatistics()
    local_actions.add(actions)
    local_statistics = local_actions.result()
    if local_statistics["unique_nonzero_pairs_rounded_4dp"] < 2:
        raise ValueError("Nonzero command variation is absent.")
    fingerprint = description.get("source_fingerprints")
    if not fingerprint or proof.get("source_fingerprints") != fingerprint:
        raise ValueError("Collection source fingerprint proof is inconsistent.")
    bank_side = None
    instruction = scene["route"]["instruction"]
    if category == "shore":
        bank_side = "left" if "riverbank on your left" in instruction else "right" if "riverbank on your right" in instruction else "unknown"
    detail = {"episode_id": entry["episode_id"], "directory": entry["accepted_episode"],
              "category": category, "split": entry["split"], "seed": entry["seed"],
              "scene_sha256": entry["scene_sha256"], "trajectory_sha256": trajectory.hexdigest(),
              "steps": len(rows), "simulation_span_s": float(times[-1] - times[0]) / 1e9,
              "initial_distance_m": math.hypot(dx, dy), "initial_bearing_deg": math.degrees(math.atan2(left, forward)),
              "cylinders": len(scene["cylinders"]), "bank_side": bank_side,
              "final_distance_m": distance, "final_speed_mps": speed, "hold_s": summary["hold_s"],
              "minimum_clearance_m": min(row["clearance_m"] for row in rows),
              "route_replayed": True, "strict_10hz_grid": True, "initial_visibility_verified": True,
              "decision_schedule": schedule, "maximum_decision_tick_lateness_s": max(lateness) if lateness else None,
              "video_decode_proofs": 2, "video_decoded_frames_each": expected_frames,
              "video_bytes": video_bytes, "raw_counts": summary.get("raw_counts", {}),
              "thrust": local_statistics, "source_fingerprint_group": _hash(fingerprint)}
    return detail, actions, fingerprint


def _disk_bytes(root):
    size, count = 0, 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            try:
                size += path.stat().st_size
                count += 1
            except FileNotFoundError:
                pass  # A live recorder can still be creating files in unaccepted attempts.
    return {"bytes": size, "GiB": size / 1024 ** 3, "files": count}


def _write_readme(root, report):
    counts = report["accepted_by_category"]
    state = "已完成" if report["complete"] else "采集中 / 未完成"
    cleaned_attempts = sum(bool(row.get("payload_cleanup")) for row in report["excluded_attempts"])
    lines = [f"# 红色圆形浮标采集：{state}", "",
             f"已验收 **{report['accepted_valid']} / 200 回合**，共 **{report['total_decisions']} 个决策样本**。"
             f" 仿真任务跨度合计 {report['total_simulation_span_s']:.1f} 秒。", "",
             f"任务成功只是验收的一部分：每条原始轨迹还必须独立通过 `{report['smoothness_profile']}` 平顺性检查，"
             "并与保存的质量证明一致。缺少证明、证明过期或有反复转向等质量问题的回合不进入训练/验证清单。",
             f"平顺性证明有效 {report['smoothness_passed_count']} 条，未通过 {len(report['smoothness_failed_episodes'])} 条，"
             f"等待有效证明 {len(report['smoothness_pending_episodes'])} 条。", "",
             "这是脚本反馈控制器在 Gazebo 中采集的示范数据，艇状态来自仿真真值；没有执行 VLA 模型训练。", "",
             "| 任务 | 已验收 / 计划 |", "| --- | ---: |"]
    if report.get('accepted_supplement_count'):
        lines[1:1] = ['', f"另有 {report['accepted_supplement_count']} 条已验收的沿岸无阻挡补充回合；"
                           '上面的决策数仅统计原始 200 条。']
    if (root / "QUALITY_REVIEW.md").is_file():
        lines[1:1] = ["", "质量复查与重采记录：[QUALITY_REVIEW.md](QUALITY_REVIEW.md)。"]
    lines += [f"| {category} | {counts.get(category, 0)} / {count} |" for category, count in QUOTAS.items()]
    lines += ["", f"两路回看视频共 {report['video_count']} 份，引用每回合已有的完整解码证明，本次汇总没有重复解码。",
              f"当前训练 / 验证回合：{report['training_selection']['train']} / {report['training_selection']['validation']}。"
              " 原始清单中的采集分组仅作为历史来源保留。", "",
              "原始数据路径清单： [全部已验收](splits/accepted_episodes.txt)、[训练](splits/train_episodes.txt)、"
              "[验证](splits/validation_episodes.txt)。清单中的路径相对于本目录，未复制图像或动作数据。", "",
              f"未纳入训练的尝试：{len(report['excluded_attempts'])} 次，其中旧元数据哈希问题 {report['excluded_metadata_hash_attempts']} 次，"
              f"被后续回合替换（superseded）{report['excluded_superseded_attempts']} 次。被替换的旧示范即使任务成功也不列入训练。"]
    if cleaned_attempts:
        lines[-1] += f" 其中 {cleaned_attempts} 次保留失败原因配置，原始数据已清理。"
        retained_attempts = len(report["excluded_attempts"]) - cleaned_attempts
        if retained_attempts:
            lines[-1] += f" 其余 {retained_attempts} 次保留日志和原始记录。"
        lines[-1] += " 失败、被替换和质量未验收的尝试不能与已验收回合混合导出。"
    else:
        lines[-1] += " 未清理的原始记录仅供复查，不能与已验收回合混合导出。"
    if report["cleanup_history"]:
        lines += ["", "中断清理记录来自 `manifest.cleanup_history`：", ""]
        accepted_ids = {row["episode_id"] for row in report["episodes"]}
        for cleanup in report["cleanup_history"]:
            outcome = "随后重新采集并独立验收。" if cleanup["episode_id"] in accepted_ids else "等待重新采集和独立验收。"
            lines.append(f"- {cleanup['episode_id']}：中断尝试 `{cleanup['directory']}` "
                         f"已删除（丢弃 {cleanup['steps_discarded']} 个决策样本），"
                         f"{outcome}")
    lines += ["", "## 回看示例", ""]
    for category in QUOTAS:
        example = next((row for row in report["episodes"] if row["category"] == category), None)
        if example:
            path = example["directory"]
            lines.append(f"- {category}：[第一视角]({path}/videa/first_person.mp4)、[空中视角]({path}/videa/aerial_global.mp4)。")
    lines += ["", "每回合的 `batch_acceptance.json` 保存因果审计、独立路线回放、初始可见性、推力变化、平顺性和视频解码证据；"
              "`collection_summary.json` 保存覆盖范围、统计、轨迹指纹及本次从原始数据重新计算的平顺性结果。"]
    if report["maximum_decision_tick_lateness_s"] is not None:
        lines += ["", f"规则源时刻调度覆盖 {report['fixed_source_grid_schedule_episodes']} 个回合，"
                  f"决策开始时已知的最新仿真 clock 相对源 tick 最多晚 {report['maximum_decision_tick_lateness_s'] * 1000:.1f} ms（上限 50 ms）。"
                  "训练观测源时刻为规则 10 Hz；`action_publish_monotonic_ns` 单独记录发布时的单调墙钟。"
                  "这些字段不提供精确的仿真发布或执行时刻，不能据此宣称满足精确执行 deadline 或没有抖动。"]
    if report["issues"]:
        lines += ["", "## 待处理检查", ""] + [f"- {issue}" for issue in report["issues"]]
    if not report["complete"]:
        lines += ["", "当前是部分数据快照；完成后重新运行 `python -m usv_vla.summarize_collection 本目录 --require-complete`。"]
    _atomic_text(root / "README.md", "\n".join(lines) + "\n")


def summarize(root):
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "manifest.json")
    issues, episodes, excluded, smoothness_reviews = [], [], [], []
    global_actions = ActionStatistics()
    category_actions = defaultdict(ActionStatistics)
    seen = {"directory": set(), "seed": set(), "scene_sha256": set(), "trajectory_sha256": set()}
    fingerprint_groups = {}
    for entry in manifest["entries"]:
        accepted_directory = entry.get("accepted_episode") if entry["status"] == "accepted" else None
        for attempt in entry.get("attempts", []):
            if attempt["directory"] == accepted_directory and attempt["status"] == "accepted":
                continue
            row = {"episode_id": entry["episode_id"], "directory": attempt["directory"],
                   "status": attempt["status"],
                   "reason": attempt.get("error") or attempt.get("supersession_reason") or attempt.get("reason"),
                   "superseded_by": attempt.get("superseded_by"), "superseded_at": attempt.get("superseded_at"),
                   "payload_cleanup": attempt.get("payload_cleanup")}
            try:
                prior = _json(_safe_episode(root, attempt["directory"]) / "summary.json")
                row.update(physical_task_success=prior.get("success"), recording_valid=prior.get("recording_valid"))
            except (OSError, ValueError):
                pass
            excluded.append(row)
        if entry["status"] != "accepted":
            continue
        review = _smoothness_review(root, entry)
        smoothness_reviews.append(review)
        if review["status"] != "passed":
            reason = "; ".join(str(value) for value in review["issues"])
            issues.append(f"{entry['episode_id']}: Trajectory smoothness {review['status']}: {reason}")
            excluded.append({"episode_id": entry["episode_id"], "directory": accepted_directory,
                             "status": f"smoothness_{review['status']}", "reason": reason,
                             "payload_cleanup": None, "accepted_in_manifest": True})
            continue
        try:
            detail, actions, fingerprints = _episode(root, entry)
            detail["trajectory_quality"] = review["recomputed"]
            detail["smoothness_profile"] = SMOOTHNESS_PROFILE
            detail["smoothness_passed"] = True
            for key in seen:
                if detail[key] in seen[key]:
                    raise ValueError(f"Duplicate accepted {key}: {detail[key]}")
            if detail["split"] not in ("train", "validation"):
                raise ValueError("Unknown scene split.")
            for key in seen:
                seen[key].add(detail[key])
            episodes.append(detail)
            global_actions.add(actions)
            category_actions[entry["category"]].add(actions)
            group = detail["source_fingerprint_group"]
            fingerprint_groups.setdefault(group, {"source_fingerprints": fingerprints, "episodes": []})["episodes"].append(detail["episode_id"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"{entry['episode_id']}: {type(error).__name__}: {error}")
            excluded.append({"episode_id": entry["episode_id"], "directory": accepted_directory,
                             "status": "invalid_accepted", "reason": str(error),
                             "payload_cleanup": None, "accepted_in_manifest": True})
    counts = dict(Counter(row["category"] for row in episodes))
    splits = dict(Counter(row["split"] for row in episodes))
    planned_categories = dict(Counter(entry["category"] for entry in manifest["entries"]))
    planned_splits = dict(Counter(entry["split"] for entry in manifest["entries"]))
    if len(manifest["entries"]) != 200 or planned_categories != QUOTAS or planned_splits != {"train": 160, "validation": 40}:
        issues.append("Manifest does not contain the required 200 category quotas and 160/40 scene split.")
    planned_seeds = [entry["seed"] for entry in manifest["entries"]]
    planned_hashes = [entry["scene_sha256"] for entry in manifest["entries"]]
    if len(set(planned_seeds)) != len(planned_seeds) or len(set(planned_hashes)) != len(planned_hashes):
        issues.append("Duplicate planned scene seeds or hashes.")
    complete = len(episodes) == 200 and counts == QUOTAS and splits == {"train": 160, "validation": 40} and not issues
    metrics = ("initial_distance_m", "initial_bearing_deg", "final_distance_m", "final_speed_mps", "hold_s", "minimum_clearance_m")
    raw_counts = Counter()
    for row in episodes:
        raw_counts.update(row["raw_counts"])
    report = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "batch_directory": str(root), "complete": complete, "status": "complete" if complete else "partial",
              "data_kind": "scripted_teacher_gazebo_demonstrations_not_trained_vla",
              "state_source": "simulator_pose_debug_not_real_localization",
              "smoothness_profile": SMOOTHNESS_PROFILE,
              "smoothness_evaluated_count": len(smoothness_reviews),
              "smoothness_passed_count": sum(row["status"] == "passed" for row in smoothness_reviews),
              "smoothness_failed_episodes": [row for row in smoothness_reviews if row["status"] == "failed"],
              "smoothness_pending_episodes": [row for row in smoothness_reviews if row["status"] == "pending"],
              "smoothness_status_counts": dict(Counter(row["status"] for row in smoothness_reviews)),
              "smoothness_metrics": _smoothness_distributions(smoothness_reviews),
              "smoothness_metrics_by_category": {
                  category: _smoothness_distributions([row for row in smoothness_reviews if row["category"] == category])
                  for category in QUOTAS},
              "smoothness_reviews": smoothness_reviews,
              "smoothness_validation": "Independently recomputed from current raw steps and episode metadata; deterministic metrics, thresholds, raw-file and inspector fingerprints must match the saved smooth_trajectory_v1 proof before inclusion in split lists.",
              "manifest_status_counts": dict(Counter(entry["status"] for entry in manifest["entries"])),
              "accepted_valid": len(episodes), "target_accepted": 200, "accepted_by_category": counts,
              "accepted_by_split": splits, "total_decisions": sum(row["steps"] for row in episodes),
              "total_simulation_span_s": sum(row["simulation_span_s"] for row in episodes),
              "total_raw_counts": dict(raw_counts), "cylinder_count_histogram": dict(Counter(row["cylinders"] for row in episodes)),
              "shore_side_histogram": dict(Counter(row["bank_side"] for row in episodes if row["bank_side"])),
              "metrics": {key: _distribution([row[key] for row in episodes]) for key in metrics},
              "metrics_by_category": {category: {key: _distribution([row[key] for row in episodes if row["category"] == category])
                                                 for key in metrics} for category in counts},
              "actions": global_actions.result(),
              "actions_by_category": {key: value.result() for key, value in category_actions.items()},
              "per_episode_nonzero_unique_ratio": _distribution([row["thrust"]["nonzero_unique_ratio"] for row in episodes]),
              "unique_scene_hashes": len(seen["scene_sha256"]), "unique_seeds": len(seen["seed"]),
              "unique_pose_time_action_trajectory_hashes": len(seen["trajectory_sha256"]),
              "route_replayed_count": sum(row["route_replayed"] for row in episodes),
              "strict_10hz_count": sum(row["strict_10hz_grid"] for row in episodes),
              "initial_visibility_verified_count": sum(row["initial_visibility_verified"] for row in episodes),
              "fixed_source_grid_schedule_episodes": sum(row["decision_schedule"] == FIXED_SCHEDULE for row in episodes),
              "maximum_decision_tick_lateness_s": max(
                  (row["maximum_decision_tick_lateness_s"] for row in episodes
                   if row["maximum_decision_tick_lateness_s"] is not None), default=None),
              "decision_timing_note": "Training observations use regular 10 Hz causal source ticks. For fixed-grid schedule episodes, the latest simulation clock known at decision start is retained with its 0..50ms lateness relative to the source tick. action_publish_monotonic_ns separately records the publication wall clock; these fields do not provide exact simulation publication/execution times or establish an exact execution deadline or zero jitter.",
              "video_count": sum(row["video_decode_proofs"] for row in episodes),
              "video_validation": "Existing full-decode proofs verified against current file byte sizes and frame metadata; no repeat decoding.",
              "video_bytes": sum(row["video_bytes"] for row in episodes), "source_fingerprint_groups": fingerprint_groups,
              "excluded_attempts": excluded,
              "cleanup_history": manifest.get("cleanup_history", []),
              "artifact_cleanup_history": manifest.get("artifact_cleanup_history", []),
              "excluded_superseded_attempts": sum(row["status"] == "superseded" for row in excluded),
              "excluded_metadata_hash_attempts": sum("immutable" in (row["reason"] or "").lower() for row in excluded),
              "issues": issues, "episodes": episodes, "disk_usage_snapshot": _disk_bytes(root)}
    all_for_training = all_accepted_for_training(root)
    supplements = accepted_supplements(root)
    report['training_selection'] = {'mode': 'all_accepted_for_training' if all_for_training else 'original_split',
                                    'train': len(episodes)+len(supplements) if all_for_training else splits.get('train', 0),
                                    'validation': 0 if all_for_training else splits.get('validation', 0)}
    report['accepted_supplement_count'] = len(supplements)
    for split in ("train", "validation", "accepted"):
        rows = selected(episodes, split, all_for_training=all_for_training)
        paths = [row['directory'] for row in rows]
        if split == 'accepted' or (split == 'train' and all_for_training):
            paths += supplements
        _atomic_text(root / "splits" / f"{split}_episodes.txt", "".join(path + "\n" for path in paths))
    _atomic_text(root / "collection_summary.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _write_readme(root, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        result = summarize(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Collection summary failed: {error}\n")
    print(json.dumps({"root": result["batch_directory"], "status": result["status"],
                      "accepted": result["accepted_valid"], "decisions": result["total_decisions"],
                      "videos": result["video_count"], "issues": result["issues"]}, ensure_ascii=False))
    return 1 if result["issues"] or args.require_complete and not result["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

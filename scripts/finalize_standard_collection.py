#!/usr/bin/env python3
"""Verify the new 200 demonstrations, then optionally remove superseded data.

The default dry run refreshes the normal collection summary and split lists,
but creates no archive or cleanup report and removes nothing.  Pass --apply
only after collection has stopped.  Deletion targets are deliberately fixed;
the model directory, source tree and current accepted recordings are excluded.
"""

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from usv_vla import batch_collect as bc
from usv_vla.collection_standard import PROFILE, inspect_standard
from usv_vla.summarize_collection import _distribution, summarize


NEW_ROOT = PROJECT / "outputs/episodes/red_round_200_20261027"
OLD_ROOTS = tuple(PROJECT / name for name in (
    "outputs/episodes/red_round_200_20261026",
    "outputs/experiments/speed_1p5_20261027",
    "outputs/experiments/speed_2x_20261027",
))
ARCHIVE = PROJECT / "reports/history/pre_standard_collection_20261027.tar.gz"
REPORT = PROJECT / "reports/finalized_standard_collection_20261027.json"
SMALL_METADATA_LIMIT = 32 * 1024 * 1024
FAILED_STATUSES = frozenset(("failed", "interrupted", "superseded"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guarded_path(path, *, allow_missing=False):
    """Reject aliases, parent symlinks and any path outside this project."""
    path = Path(path)
    if not path.is_absolute() or path == PROJECT or not path.is_relative_to(PROJECT):
        raise ValueError(f"Path is outside the permitted project subtree: {path}")
    current = PROJECT
    for part in path.relative_to(PROJECT).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in cleanup path: {current}")
    if path.resolve() != path:
        raise ValueError(f"Cleanup path does not resolve exactly: {path}")
    if not allow_missing and not path.exists():
        raise ValueError(f"Required path is missing: {path}")
    return path


def tree_stats(path):
    """Never follow links; allow only three known old runtime resource links."""
    guarded_path(path, allow_missing=True)
    result = {"bytes": 0, "allocated_bytes": 0, "files": 0, "directories": 0}
    if not path.exists():
        return result
    if not path.is_dir():
        raise ValueError(f"Expected a dataset directory: {path}")
    legacy_links = {OLD_ROOTS[2] / "runtime" / name: PROJECT / name
                    for name in ("assets", "configs", "sim_runtime")}
    for directory, dirs, files in os.walk(path, followlinks=False):
        result["directories"] += 1
        for name in dirs + files:
            item = Path(directory) / name
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode):
                expected = legacy_links.get(item) if path == OLD_ROOTS[2] else None
                if expected is None or not expected.is_dir() or item.resolve() != expected:
                    raise ValueError(f"Symlink inside dataset tree: {item}")
                result["files"] += 1
                result["bytes"] += info.st_size
                result["allocated_bytes"] += info.st_blocks * 512
                continue
            if stat.S_ISREG(info.st_mode):
                result["files"] += 1
                result["bytes"] += info.st_size
                result["allocated_bytes"] += info.st_blocks * 512
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Special file inside dataset tree: {item}")
    return result


def attempt_path(root, relative, episode_id):
    relative = Path(relative)
    if (relative.is_absolute() or len(relative.parts) != 2
            or relative.parts[0] != episode_id
            or not relative.parts[1].startswith("attempt_")
            or not relative.parts[1][8:].isdigit()):
        raise ValueError(f"Unexpected attempt directory: {relative}")
    return guarded_path(root / relative, allow_missing=True)


def verify_collection(root):
    """Require both the full existing audit and fresh navigation measurements."""
    summary = summarize(root)
    if (summary.get("complete") is not True or summary.get("accepted_valid") != 200
            or summary.get("target_accepted") != 200 or summary.get("issues") != []
            or summary.get("video_count") != 400):
        raise ValueError("The new collection is not complete with 200 valid episodes and 400 videos")
    manifest = read_json(root / "manifest.json")
    reviews, accepted, fingerprints = [], set(), []
    for entry in manifest["entries"]:
        if entry["status"] != "accepted":
            raise ValueError("Every new manifest entry must be accepted")
        relative = entry["accepted_episode"]
        directory = attempt_path(root, relative, entry["episode_id"])
        if relative in accepted or not directory.is_dir():
            raise ValueError(f"Missing or duplicate accepted directory: {relative}")
        accepted.add(relative)
        measured = inspect_standard(directory)
        proof = read_json(directory / "batch_acceptance.json").get("navigation_standard")
        if (measured.get("profile") != PROFILE or measured.get("passed") is not True
                or measured.get("issues") != [] or not isinstance(proof, dict)
                or bc._digest(proof) != bc._digest(measured)):
            raise ValueError(f"Current navigation proof does not match raw data: {relative}")
        # Explicit raw hashes make the deletion prerequisite auditable even if
        # the inspector's result format gains further fields in the future.
        if (measured.get("steps_sha256") != file_hash(directory / "steps.jsonl")
                or measured.get("metadata_sha256") != file_hash(directory / "episode.json")):
            raise ValueError(f"Raw data changed during navigation inspection: {relative}")
        manifest_proof = entry.get("validation", {}).get("navigation_standard")
        if manifest_proof is not None and bc._digest(manifest_proof) != bc._digest(measured):
            raise ValueError(f"Manifest navigation proof differs from the raw episode: {relative}")
        reviews.append({"episode_id": entry["episode_id"], "directory": relative,
                        "category": entry["category"], **measured})
        fingerprints.append([relative, measured["steps_sha256"], measured["metadata_sha256"]])
    if len(accepted) != 200:
        raise ValueError("Exactly 200 distinct accepted recordings are required")
    return summary, manifest, reviews, accepted, bc._digest(sorted(fingerprints))


def rejected_attempts(root, manifest, accepted):
    result, known = [], set()
    for entry in manifest["entries"]:
        for attempt in entry.get("attempts", []):
            relative = attempt["directory"]
            directory = attempt_path(root, relative, entry["episode_id"])
            if relative in known:
                raise ValueError(f"Duplicate manifest attempt path: {relative}")
            known.add(relative)
            if relative in accepted:
                if attempt["status"] != "accepted":
                    raise ValueError(f"Accepted recording has conflicting attempt status: {relative}")
                continue
            if attempt["status"] not in FAILED_STATUSES:
                raise ValueError(f"An unfinished or unknown attempt remains: {relative}")
            log = root / relative
            log = log.parent / (log.name + ".recorder.log")
            if attempt.get("log") is not None and attempt["log"] != str(log.relative_to(root)):
                raise ValueError(f"Unexpected recorder log path: {attempt['log']}")
            guarded_path(log, allow_missing=True)
            if log.exists() and not log.is_file():
                raise ValueError(f"Recorder log is not a regular file: {log}")
            result.append((entry, attempt, directory, log))
    for directory in root.glob("episode_*/attempt_*"):
        if directory.is_dir() and str(directory.relative_to(root)) not in known:
            raise ValueError(f"Unreferenced attempt directory requires inspection: {directory}")
    return result


def metadata_inputs(failed):
    """An allowlist: no raw steps, image, video, logs or runtime payloads."""
    files = {}
    old_batch = OLD_ROOTS[0]
    for name in ("manifest.json", "collection_summary.json", "README.md", "index.csv",
                 "progress.json", "QUALITY_REVIEW.md"):
        path = old_batch / name
        if path.is_file():
            files[str(path.relative_to(PROJECT))] = path
    for path in (old_batch / "plans").glob("*.json"):
        if path.is_file():
            files[str(path.relative_to(PROJECT))] = path
    for root in OLD_ROOTS[1:]:
        for path in list(root.glob("*.json")) + [root / "README.md"]:
            if path.is_file():
                files[str(path.relative_to(PROJECT))] = path
    for _, _, directory, _ in failed:
        for name in ("episode.json", "summary.json", "batch_failure.json", "batch_acceptance.json"):
            path = directory / name
            if path.is_file():
                files[str(path.relative_to(PROJECT))] = path
    for path in files.values():
        guarded_path(path)
        if path.stat().st_size > SMALL_METADATA_LIMIT:
            raise ValueError(f"Metadata exceeds the small-file archive limit: {path}")
    return files


def verify_archive(path):
    contents = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts
                    or member.name in contents or member.size > SMALL_METADATA_LIMIT):
                raise ValueError("Unexpected or unsafe metadata archive member")
            stream = archive.extractfile(member)
            contents[member.name] = hashlib.sha256(stream.read()).hexdigest()
    return contents


def archive_metadata(inputs, previous):
    guarded_path(ARCHIVE, allow_missing=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    expected = {name: file_hash(path) for name, path in inputs.items()}
    if ARCHIVE.exists():
        archived = verify_archive(ARCHIVE)
        # Already removed roots contribute no current inputs on a resumed run.
        # The durable report retains the complete original member hash map.
        original = previous.get("metadata_archive", {}).get("members")
        if original is None or archived != original:
            raise ValueError("Existing archive lacks a matching verified cleanup report")
        if any(archived.get(name) != digest for name, digest in expected.items()):
            raise ValueError("Current metadata differs from its existing archive")
        expected = original
        digest = file_hash(ARCHIVE)
        if digest != previous["metadata_archive"]["sha256"]:
            raise ValueError("Metadata archive file hash changed")
    else:
        if previous.get("metadata_archive"):
            raise ValueError("A previously verified cleanup archive is missing")
        fd, temporary = tempfile.mkstemp(prefix=".standard_archive_", dir=ARCHIVE.parent)
        os.close(fd)
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                for name, path in sorted(inputs.items()):
                    archive.add(path, arcname=name, recursive=False)
            if verify_archive(temporary) != expected:
                raise ValueError("Metadata archive content hashes do not match source files")
            digest = file_hash(temporary)
            os.replace(temporary, ARCHIVE)
            if file_hash(ARCHIVE) != digest:
                raise ValueError("Metadata archive file hash changed after atomic placement")
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"path": str(ARCHIVE.relative_to(PROJECT)), "sha256": digest,
            "bytes": ARCHIVE.stat().st_size, "members": expected,
            "member_count": len(expected), "verified_at_utc": bc._now()}


def standard_distributions(reviews):
    result = {}
    for name, getter in (
        ("cruise_speed_mps", lambda row: row["cruise_speed_mps"]),
        ("initial_outward_peak_deg", lambda row: row.get("initial_outward_peak_deg")),
        ("abeam_heading_relative_route_deg", lambda row: (row.get("crossing") or {}).get("heading_relative_route_deg")),
        ("absolute_abeam_heading_deg", lambda row: abs(row["crossing"]["heading_relative_route_deg"]) if row.get("crossing") else None),
        ("abeam_left_m", lambda row: (row.get("crossing") or {}).get("left_m")),
    ):
        values = [value for row in reviews if (value := getter(row)) is not None]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Nonfinite standard measurement")
        result[name] = {"count": len(values), "distribution": _distribution(values)}
    return result


def save_report(report):
    guarded_path(REPORT, allow_missing=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at_utc"] = bc._now()
    bc._atomic_json(REPORT, report)


def removal_record(path, kind, previous):
    stats = tree_stats(path)
    prior = next((row for row in previous.get("removals", []) if row["path"] == str(path)), None)
    if prior and not path.exists():
        return {**prior, "status": "removed" if prior.get("existed") else "already_absent"}
    if prior and prior.get("status") in ("removing", "removed"):
        # rmtree may have stopped partway through.  Retain the original byte
        # count rather than reporting only what remains after an interruption.
        return {**prior, "status": "planned", "remaining_before_resume": stats}
    return {"path": str(path), "kind": kind, "existed": path.exists(),
            "status": "planned", **stats}


def finalize(*, apply=False):
    for path in (NEW_ROOT, *OLD_ROOTS):
        guarded_path(path, allow_missing=path != NEW_ROOT)
    if any(NEW_ROOT == old or NEW_ROOT.is_relative_to(old) or old.is_relative_to(NEW_ROOT) for old in OLD_ROOTS):
        raise ValueError("New and superseded dataset paths overlap")
    tree_stats(NEW_ROOT)  # No accepted or rejected path may contain a symlink.
    with ExitStack() as stack:
        for root in (NEW_ROOT, OLD_ROOTS[0]):
            lock_path = guarded_path(root / ".batch.lock", allow_missing=root != NEW_ROOT)
            if lock_path.exists():
                lock = stack.enter_context(lock_path.open("r+"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(f"Collection is still running or locked: {root}") from error
        summary, manifest, reviews, accepted, accepted_hash = verify_collection(NEW_ROOT)
        failed = rejected_attempts(NEW_ROOT, manifest, accepted)
        previous = read_json(REPORT) if guarded_path(REPORT, allow_missing=True).is_file() else {}
        if previous and (previous.get("batch_directory") != str(NEW_ROOT)
                         or previous.get("accepted_set_sha256") != accepted_hash):
            raise ValueError("Existing cleanup report belongs to different accepted recordings")
        removals = [removal_record(root, "superseded_dataset", previous) for root in OLD_ROOTS]
        removals += [removal_record(directory, "rejected_attempt", previous) for _, _, directory, _ in failed]
        inputs = metadata_inputs(failed)
        report = {"schema_version": 1, "batch_directory": str(NEW_ROOT), "profile": PROFILE,
                  "mode": "apply" if apply else "dry_run", "status": "validated",
                  "accepted_valid": 200, "video_count": summary["video_count"],
                  "total_decisions": summary["total_decisions"],
                  "total_simulation_span_s": summary["total_simulation_span_s"],
                  "accepted_set_sha256": accepted_hash, "navigation_standard_passed": len(reviews),
                  "navigation_standard_metrics": standard_distributions(reviews),
                  "navigation_standard_reviews": reviews, "removals": removals,
                  "failed_recorder_logs": [str(log) for _, _, _, log in failed if log.exists()],
                  "metadata_files_to_archive": len(inputs), "issues": []}
        if not apply:
            return report
        archive = archive_metadata(inputs, previous)
        report.update(metadata_archive=archive, status="archive_verified")
        save_report(report)  # Verified evidence is durable before any deletion.
        for row in removals:
            path = Path(row["path"])
            guarded_path(path, allow_missing=True)
            if row["kind"] == "superseded_dataset" and path not in OLD_ROOTS:
                raise ValueError("Deletion is outside the three fixed superseded roots")
            if row["kind"] == "rejected_attempt" and str(path.relative_to(NEW_ROOT)) in accepted:
                raise ValueError("Refusing to delete a current accepted recording")
            if path.exists():
                tree_stats(path)
                row["status"] = "removing"
                save_report(report)
                shutil.rmtree(path)
                row.update(status="removed", removed_at_utc=bc._now())
            elif row["status"] == "planned":
                row["status"] = "already_absent"
            save_report(report)
        for _, attempt, directory, log in failed:
            guarded_path(log, allow_missing=True)
            log_bytes = (log.stat().st_size if log.exists() else
                         attempt.get("payload_cleanup", {}).get("recorder_log_removal_bytes", 0))
            log.unlink(missing_ok=True)
            record = next(row for row in removals if row["path"] == str(directory))
            attempt["payload_cleanup"] = {
                **attempt.get("payload_cleanup", {}), "directory_removed": True,
                "directory_removed_at_utc": bc._now(), "directory_removal_bytes": record["bytes"],
                "recorder_log_removed": True, "recorder_log_removal_bytes": log_bytes,
                "metadata_archive": str(ARCHIVE.relative_to(PROJECT)), "metadata_archive_base": "project_root",
                "cleanup_report": str(REPORT.relative_to(PROJECT)),
                "reason_for_directory_removal": "New 200-episode batch passed task, smoothness and measured early-alignment checks."}
            bc._atomic_json(NEW_ROOT / "manifest.json", manifest)
        final_summary = summarize(NEW_ROOT)
        if not final_summary["complete"] or final_summary["accepted_valid"] != 200 or final_summary["issues"]:
            raise ValueError("Post-cleanup summary failed; inspect the preserved current recordings")
        report.update(status="complete", removed_directory_count=sum(row["existed"] for row in removals),
                      removed_bytes=sum(row["bytes"] for row in removals if row["existed"]),
                      removed_file_count=sum(row["files"] for row in removals if row["existed"]),
                      failed_recorder_log_bytes=sum(a.get("payload_cleanup", {}).get("recorder_log_removal_bytes", 0)
                                                    for _, a, _, _ in failed),
                      completed_at_utc=bc._now())
        save_report(report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Archive metadata and remove only verified obsolete data")
    mode.add_argument("--dry-run", action="store_true", help="Verify and list cleanup without archiving or deleting (default)")
    args = parser.parse_args()
    try:
        report = finalize(apply=args.apply)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        parser.exit(1, f"Finalization refused: {type(error).__name__}: {error}\n")
    print(json.dumps({key: report[key] for key in (
        "mode", "status", "accepted_valid", "video_count", "total_decisions",
        "navigation_standard_passed", "navigation_standard_metrics", "removals")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

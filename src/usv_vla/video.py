"""Export review videos on the episode's simulation clock.

The aerial camera is for human review only. This module never changes the
observations, action records, or training image views.
"""

import argparse
import bisect
from fractions import Fraction
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time

from PIL import Image


class VideoExportError(RuntimeError):
    """A review video could not be produced without hiding a recording fault."""


MAX_FRAME_AGE_NS = 300_000_000
MAX_OUTPUT_FRAMES = 1_000_000


def _executable(value):
    candidate = shutil.which(str(value))
    if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
        return str(Path(candidate).resolve())
    raise VideoExportError(f"FFmpeg executable is unavailable: {value}")


def resolve_ffmpeg(ffmpeg=None):
    """Find FFmpeg without importing model-environment packages in Gazebo Python."""
    selected = ffmpeg or os.environ.get("USV_FFMPEG")
    if selected:
        return _executable(selected)
    found = shutil.which("ffmpeg")
    if found:
        return _executable(found)
    interpreters = dict.fromkeys(filter(None, (
        os.environ.get("USV_VIDEO_PYTHON"), os.environ.get("USV_MODEL_PYTHON"), sys.executable,
    )))
    failures = []
    for interpreter in interpreters:
        try:
            result = subprocess.run(
                [interpreter, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
                capture_output=True, text=True, timeout=15, check=True,
            )
            return _executable(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, VideoExportError) as error:
            failures.append(f"{interpreter}: {type(error).__name__}")
    raise VideoExportError(
        "FFmpeg was not found. Set USV_FFMPEG to an executable, or set "
        "USV_VIDEO_PYTHON / USV_MODEL_PYTHON to an environment containing imageio_ffmpeg. "
        + "; ".join(failures)
    )


def _rows(path):
    if not path.is_file():
        raise VideoExportError(f"Required recording index is missing: {path}")
    result = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if len(line) > 1_000_000:
                raise VideoExportError(f"Oversized JSON row at {path}:{index}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise VideoExportError(f"Invalid JSON at {path}:{index}: {error}") from error
            if not isinstance(row, dict):
                raise VideoExportError(f"Expected JSON object at {path}:{index}")
            result.append(row)
    if not result:
        raise VideoExportError(f"Recording index is empty: {path}")
    return result


def _timestamp(value, context):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VideoExportError(f"Expected nonnegative integer nanoseconds in {context}: {value!r}")
    return value


def _safe_image(episode, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise VideoExportError(f"Image must be relative to its episode: {value!r}")
    image = (episode / value).resolve()
    if not image.is_relative_to(episode):
        raise VideoExportError(f"Image escapes episode directory: {value}")
    if not image.is_file():
        raise VideoExportError(f"Recorded image is missing: {image}")
    return image


def _timeline(episode, fps):
    try:
        rate = Fraction(str(fps))
    except (ValueError, ZeroDivisionError) as error:
        raise VideoExportError(f"Invalid video fps: {fps!r}") from error
    if rate <= 0 or rate > 120:
        raise VideoExportError("Video fps must be greater than zero and at most 120.")
    steps = _rows(episode / "steps.jsonl")
    times = [_timestamp(row.get("decision_sim_time_ns"), "steps.jsonl") for row in steps]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise VideoExportError("Decision simulation timestamps must increase strictly.")
    start, end = times[0], times[-1]
    count = (end - start) * rate.numerator // (1_000_000_000 * rate.denominator) + 1
    if count > MAX_OUTPUT_FRAMES:
        raise VideoExportError(f"Refusing oversized video: {count} frames.")
    ticks = [start + i * 1_000_000_000 * rate.denominator // rate.numerator for i in range(count)]
    return rate, ticks, dict(
        clock="simulation_time", start_sim_time_ns=start, end_sim_time_ns=end,
        decision_count=len(times), frame_count=count, fps=float(rate),
        fps_fraction=f"{rate.numerator}/{rate.denominator}",
        last_frame_sim_time_ns=ticks[-1], recorded_span_s=(end - start) / 1e9,
        encoded_duration_s=float(Fraction(count, 1) / rate),
        endpoint_rule="Inclusive samples at start + n/fps through the last decision; final sample displays for one frame.",
    )


def _select_frames(episode, index_name, ticks):
    rows = _rows(episode / index_name)
    by_time = {}
    for row in rows:
        timestamp = _timestamp(row.get("source_time_ns"), index_name)
        by_time[timestamp] = row
    ordered = sorted(by_time)
    mapping = []
    checked = set()
    for index, target in enumerate(ticks):
        position = bisect.bisect_right(ordered, target) - 1
        if position < 0:
            raise VideoExportError(f"{index_name}: no camera frame at or before {target} ns.")
        source = ordered[position]
        age = target - source
        if age > MAX_FRAME_AGE_NS:
            raise VideoExportError(
                f"{index_name}: frame gap at {target} ns; latest source is {age / 1e9:.3f}s old "
                f"(limit {MAX_FRAME_AGE_NS / 1e9:.3f}s)."
            )
        relative = by_time[source].get("image")
        if not isinstance(relative, str):
            raise VideoExportError(f"Image must be relative to its episode: {relative!r}")
        if relative not in checked:
            _safe_image(episode, relative)
            checked.add(relative)
        mapping.append(dict(frame_index=index, sim_time_ns=target, source_time_ns=source,
                            age_ns=age, image=relative))
    selected_sources = {row["source_time_ns"] for row in mapping}
    inside = [source for source in ordered if ticks[0] <= source <= ticks[-1]]
    stats = dict(
        source_index=index_name, source_rows=len(rows), unique_source_timestamps=len(ordered),
        duplicate_source_timestamps=len(rows) - len(ordered),
        source_frames_in_timeline=len(inside), selected_unique_frames=len(selected_sources),
        raw_frames_not_selected_in_timeline=len(set(inside) - selected_sources),
        repeated_output_frames=len(mapping) - len(selected_sources),
        max_source_age_s=max(row["age_ns"] for row in mapping) / 1e9,
        first_source_time_ns=mapping[0]["source_time_ns"],
        last_source_time_ns=mapping[-1]["source_time_ns"],
        covered_output_frames=len(mapping), missing_output_frames=0,
        frame_map=mapping,
    )
    return mapping, stats


def _rgb(episode, relative):
    path = _safe_image(episode, relative)
    try:
        with Image.open(path) as source:
            source.load()
            rgb = source.convert("RGB")
            if rgb.width % 2 or rgb.height % 2:
                raise VideoExportError(f"Video image dimensions must be even for yuv420p: {path}")
            return rgb.size, rgb.tobytes()
    except (OSError, ValueError) as error:
        raise VideoExportError(f"Cannot decode recorded image {path}: {error}") from error


def _write_frame(process, pixels, deadline):
    descriptor = process.stdin.fileno()
    view = memoryview(pixels)
    while view:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VideoExportError("FFmpeg encoding timed out.")
        if process.poll() is not None:
            raise VideoExportError(f"FFmpeg exited during encoding ({process.returncode}).")
        _, writable, _ = select.select([], [descriptor], [], min(remaining, 1.0))
        if not writable:
            continue
        try:
            written = os.write(descriptor, view[:1_048_576])
        except BlockingIOError:
            continue
        except BrokenPipeError as error:
            raise VideoExportError("FFmpeg closed its input during encoding.") from error
        view = view[written:]


def _encode(episode, mapping, rate, ffmpeg, output):
    size, pixels = _rgb(episode, mapping[0]["image"])
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{size[0]}x{size[1]}",
            "-framerate", f"{rate.numerator}/{rate.denominator}", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    deadline = time.monotonic() + max(60, min(1800, len(mapping) * .2 + 30))
    with tempfile.TemporaryFile() as errors:
        process = None
        try:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            os.set_blocking(process.stdin.fileno(), False)
            last_image = mapping[0]["image"]
            for frame in mapping:
                if frame["image"] != last_image:
                    next_size, pixels = _rgb(episode, frame["image"])
                    if next_size != size:
                        raise VideoExportError(f"Camera resolution changed during recording: {frame['image']}")
                    last_image = frame["image"]
                _write_frame(process, pixels, deadline)
            process.stdin.close()
            process.wait(timeout=max(1, deadline - time.monotonic()))
            if process.returncode:
                raise VideoExportError(f"FFmpeg failed with exit code {process.returncode}.")
            if not output.is_file() or output.stat().st_size == 0:
                raise VideoExportError("FFmpeg returned no MP4 output.")
        except (OSError, subprocess.SubprocessError, VideoExportError) as error:
            if process is not None:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            errors.seek(0, os.SEEK_END)
            length = errors.tell()
            errors.seek(max(0, length - 8192))
            detail = errors.read().decode("utf-8", errors="replace").strip()
            raise VideoExportError(f"{error}" + (f" FFmpeg: {detail}" if detail else "")) from error
    return dict(width=size[0], height=size[1], pixel_format="yuv420p", codec="h264",
                bytes=output.stat().st_size)


def export_episode_videos(episode, ffmpeg=None, fps=10, require_aerial=True, overwrite=False):
    """Produce synchronized review MP4s; fail rather than invent an aerial view.

    Returns the same JSON-serializable metadata saved to ``videa/metadata.json``.
    Existing exports are protected unless ``overwrite=True`` is explicit.
    """
    episode = Path(episode).resolve()
    if not episode.is_dir():
        raise VideoExportError(f"Episode directory does not exist: {episode}")
    rate, ticks, timeline = _timeline(episode, fps)
    views = [("first_person", "frames.jsonl")]
    aerial_exists = (episode / "aerial_frames.jsonl").is_file()
    if aerial_exists:
        views.append(("aerial_global", "aerial_frames.jsonl"))
    elif require_aerial:
        raise VideoExportError(
            "No aerial_frames.jsonl was recorded. An aerial video cannot be reconstructed from front images. "
            "Use --allow-missing-aerial only to export first_person.mp4 for a historical episode."
        )
    selections = {name: _select_frames(episode, index, ticks) for name, index in views}
    binary = resolve_ffmpeg(ffmpeg)
    destination = episode / "videa"
    destination.mkdir(exist_ok=True)
    lock = destination / ".export.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise VideoExportError(f"An export is already active, or its stale lock needs inspection: {lock}") from error
    os.close(descriptor)
    try:
        existing = [destination / f"{name}.mp4" for name, _ in views] + [destination / "metadata.json"]
        if not overwrite and any(path.exists() for path in existing):
            raise VideoExportError(f"Review exports already exist in {destination}; use --overwrite to replace them.")
        metadata = dict(
            schema_version=1, episode_directory=str(episode), timeline=timeline,
            selection_rule="Latest source_time_ns <= frame simulation timestamp; never select a future image.",
            max_allowed_source_age_s=MAX_FRAME_AGE_NS / 1e9,
            aerial_available=aerial_exists,
            aerial_role="Human review only; excluded from the VLA training observations.",
            training_observations_unchanged=True,
            ffmpeg=binary, views={},
        )
        episode_description = episode / "episode.json"
        if episode_description.is_file():
            try:
                metadata["aerial_camera"] = json.loads(episode_description.read_text()).get("aerial_camera")
            except (OSError, json.JSONDecodeError) as error:
                raise VideoExportError(f"Cannot read episode camera metadata: {error}") from error
        with tempfile.TemporaryDirectory(prefix=".export-", dir=destination) as temporary:
            temporary = Path(temporary)
            for name, _ in views:
                mapping, stats = selections[name]
                stats = dict(stats)
                stats.update(_encode(episode, mapping, rate, binary, temporary / f"{name}.mp4"))
                stats["file"] = f"{name}.mp4"
                metadata["views"][name] = stats
            (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            for name, _ in views:
                os.replace(temporary / f"{name}.mp4", destination / f"{name}.mp4")
            os.replace(temporary / "metadata.json", destination / "metadata.json")
        return metadata
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--ffmpeg", help="Explicit FFmpeg executable; alternatively set USV_FFMPEG.")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--allow-missing-aerial", action="store_true",
                        help="Export front camera only for a historical episode with no aerial recording.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = export_episode_videos(args.episode, args.ffmpeg, args.fps,
                                       not args.allow_missing_aerial, args.overwrite)
    except VideoExportError as error:
        parser.exit(1, f"Video export failed: {error}\n")
    print(json.dumps({"videa": str(args.episode.resolve() / "videa"),
                      "frames": result["timeline"]["frame_count"],
                      "views": list(result["views"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

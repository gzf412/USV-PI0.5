import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from PIL import Image

from usv_vla.video import (
    VideoExportError, _select_frames, _timeline, export_episode_videos, resolve_ffmpeg,
)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(directory, aerial=True):
    write_rows(directory / "steps.jsonl", [
        {"decision_sim_time_ns": timestamp} for timestamp in (0, 100_000_000, 200_000_000)
    ])
    colors = [(220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 220, 20)]
    timestamps = (0, 50_000_000, 150_000_000, 200_000_000)
    for folder, index in [("images", "frames.jsonl")] + (
        [("aerial_images", "aerial_frames.jsonl")] if aerial else []
    ):
        (directory / folder).mkdir()
        rows = []
        for timestamp, color in zip(timestamps, colors):
            relative = f"{folder}/{timestamp}.png"
            Image.new("RGB", (16, 16), color).save(directory / relative)
            rows.append({"source_time_ns": timestamp, "image": relative})
        write_rows(directory / index, rows)
    (directory / "episode.json").write_text(json.dumps({"aerial_camera": {"role": "review_only"}}))


class VideoTimeTests(unittest.TestCase):
    def test_latest_causal_frame_and_common_sim_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            fixture(episode)
            rate, ticks, metadata = _timeline(episode, 10)
            frames, stats = _select_frames(episode, "frames.jsonl", ticks)
            aerial, _ = _select_frames(episode, "aerial_frames.jsonl", ticks)
            self.assertEqual(ticks, [0, 100_000_000, 200_000_000])
            self.assertEqual([frame["source_time_ns"] for frame in frames], [0, 50_000_000, 200_000_000])
            self.assertEqual([frame["sim_time_ns"] for frame in frames],
                             [frame["sim_time_ns"] for frame in aerial])
            self.assertTrue(all(frame["source_time_ns"] <= frame["sim_time_ns"] for frame in frames))
            self.assertEqual(stats["raw_frames_not_selected_in_timeline"], 1)
            self.assertEqual(metadata["recorded_span_s"], .2)
            self.assertEqual(metadata["encoded_duration_s"], .3)

    def test_repeated_frames_reported_and_long_gap_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            fixture(episode)
            write_rows(episode / "frames.jsonl", [{"source_time_ns": 0, "image": "images/0.png"}])
            _, ticks, _ = _timeline(episode, 10)
            _, stats = _select_frames(episode, "frames.jsonl", ticks)
            self.assertEqual(stats["repeated_output_frames"], 2)
            with self.assertRaisesRegex(VideoExportError, "frame gap"):
                _select_frames(episode, "frames.jsonl", [400_000_000])

    def test_no_future_or_missing_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            fixture(episode)
            with self.assertRaisesRegex(VideoExportError, "no camera frame"):
                _select_frames(episode, "frames.jsonl", [-1])
            (episode / "images/0.png").unlink()
            with self.assertRaisesRegex(VideoExportError, "image is missing"):
                _select_frames(episode, "frames.jsonl", [0])

    def test_directory_escape_and_invalid_clock_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            fixture(episode)
            write_rows(episode / "frames.jsonl", [{"source_time_ns": 0, "image": "../escape.png"}])
            with self.assertRaisesRegex(VideoExportError, "escapes episode"):
                _select_frames(episode, "frames.jsonl", [0])
            write_rows(episode / "steps.jsonl", [{"decision_sim_time_ns": 10}, {"decision_sim_time_ns": 10}])
            with self.assertRaisesRegex(VideoExportError, "increase strictly"):
                _timeline(episode, 10)

    def test_historical_aerial_is_not_invented(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            fixture(episode, aerial=False)
            with self.assertRaisesRegex(VideoExportError, "cannot be reconstructed"):
                export_episode_videos(episode)
            self.assertFalse((episode / "videa").exists())

    def test_explicit_encoder_must_exist(self):
        with self.assertRaisesRegex(VideoExportError, "unavailable"):
            resolve_ffmpeg("/unavailable/ffmpeg")


class VideoEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ffmpeg = resolve_ffmpeg()
        except VideoExportError as error:
            raise unittest.SkipTest(str(error))

    def test_real_mp4_encoding_decoding_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            fixture(episode)
            result = export_episode_videos(episode, self.ffmpeg)
            front = episode / "videa/first_person.mp4"
            aerial = episode / "videa/aerial_global.mp4"
            self.assertGreater(front.stat().st_size, 0)
            self.assertGreater(aerial.stat().st_size, 0)
            self.assertEqual(result["timeline"]["frame_count"], 3)
            self.assertEqual(result["views"]["first_person"]["codec"], "h264")
            self.assertTrue(result["training_observations_unchanged"])
            self.assertEqual(result["aerial_camera"], {"role": "review_only"})
            decoded = subprocess.run([
                self.ffmpeg, "-v", "error", "-i", str(front), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ], capture_output=True, timeout=15, check=True).stdout
            frame_bytes = 16 * 16 * 3
            self.assertEqual(len(decoded), 3 * frame_bytes)
            expected = [(220, 20, 20), (20, 220, 20), (220, 220, 20)]
            for index, color in enumerate(expected):
                pixel = decoded[index * frame_bytes:index * frame_bytes + 3]
                self.assertTrue(all(abs(actual - target) < 10 for actual, target in zip(pixel, color)))
            before = hashlib.sha256(front.read_bytes()).hexdigest()
            with self.assertRaisesRegex(VideoExportError, "already exist"):
                export_episode_videos(episode, self.ffmpeg)
            self.assertEqual(hashlib.sha256(front.read_bytes()).hexdigest(), before)
            self.assertFalse((episode / "videa/.export.lock").exists())
            export_episode_videos(episode, self.ffmpeg, overwrite=True)

    def test_corrupt_camera_image_leaves_no_partial_published_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            fixture(episode)
            (episode / "aerial_images/50000000.png").write_bytes(b"not an image")
            with self.assertRaisesRegex(VideoExportError, "Cannot decode"):
                export_episode_videos(episode, self.ffmpeg)
            self.assertFalse((episode / "videa/first_person.mp4").exists())
            self.assertFalse((episode / "videa/aerial_global.mp4").exists())
            self.assertFalse((episode / "videa/metadata.json").exists())
            self.assertFalse((episode / "videa/.export.lock").exists())

    def test_allow_missing_aerial_exports_only_recorded_front(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            fixture(episode, aerial=False)
            result = export_episode_videos(episode, self.ffmpeg, require_aerial=False)
            self.assertEqual(list(result["views"]), ["first_person"])
            self.assertFalse(result["aerial_available"])
            self.assertFalse((episode / "videa/aerial_global.mp4").exists())

    def test_encoder_failure_is_reported_and_export_can_be_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            fixture(episode)
            failing_encoder = episode / "failed_encoder"
            failing_encoder.write_text("#!/bin/sh\necho 'intentional encoder failure' >&2\nexit 7\n")
            failing_encoder.chmod(0o700)
            with self.assertRaisesRegex(VideoExportError, "FFmpeg"):
                export_episode_videos(episode, failing_encoder)
            self.assertFalse((episode / "videa/.export.lock").exists())
            self.assertFalse((episode / "videa/metadata.json").exists())
            result = export_episode_videos(episode, self.ffmpeg)
            self.assertEqual(result["timeline"]["frame_count"], 3)


if __name__ == "__main__":
    unittest.main()

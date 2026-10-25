import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from transcribe_video import (
    extract_audio,
    format_srt_timestamp,
    main,
    mux_tamil_audio,
    remove_subtitle_tracks,
    transcribe_audio,
    translate_segments,
    write_transcription,
)


class FormatSrtTimestampTests(unittest.TestCase):
    def test_formats_hours_minutes_seconds_and_milliseconds(self) -> None:
        self.assertEqual(format_srt_timestamp(3_723.456), "01:02:03,456")

    def test_rounds_milliseconds_and_carries_into_next_second(self) -> None:
        self.assertEqual(format_srt_timestamp(1.9996), "00:00:02,000")

    def test_clamps_negative_time_to_zero(self) -> None:
        self.assertEqual(format_srt_timestamp(-1), "00:00:00,000")


class AudioExtractionTests(unittest.TestCase):
    def test_reports_missing_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.touch()

            with patch("transcribe_video.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "was not found"):
                    extract_audio(video_path, Path(temp_dir) / "video.wav")


class SubtitleTrackRemovalTests(unittest.TestCase):
    def test_copies_video_without_subtitle_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            output_path = Path(temp_dir) / "video_no_subtitles.mp4"
            video_path.touch()

            def fake_ffmpeg(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                Path(command[-1]).write_bytes(b"video")
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                patch("transcribe_video.shutil.which", return_value="/usr/bin/ffmpeg"),
                patch("transcribe_video.subprocess.run", side_effect=fake_ffmpeg) as run,
            ):
                result = remove_subtitle_tracks(video_path, output_path)

            self.assertEqual(result, output_path)
            command = run.call_args.args[0]
            self.assertIn("-map", command)
            self.assertIn("-0:s", command)
            self.assertTrue(output_path.is_file())

    def test_rejects_output_path_that_would_overwrite_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.touch()

            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                remove_subtitle_tracks(video_path, video_path)


class TranscriptionCompatibilityTests(unittest.TestCase):
    def test_reports_pyav_metadata_errors_incompatibility(self) -> None:
        class IncompatibleModel:
            def __init__(self, *_: object, **__: object) -> None:
                pass

            def transcribe(self, *_: object, **__: object) -> None:
                raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")

        fake_faster_whisper = types.SimpleNamespace(WhisperModel=IncompatibleModel)
        with patch.dict(sys.modules, {"faster_whisper": fake_faster_whisper}):
            with self.assertRaisesRegex(
                RuntimeError, "python3 -m pip install --upgrade"
            ):
                transcribe_audio(
                    Path("audio.wav"),
                    model_name="small",
                    language="en",
                    device="cpu",
                    compute_type="int8",
                )


class TamilDubbingTests(unittest.TestCase):
    def test_translation_returns_tamil_for_each_segment(self) -> None:
        genai_module = types.ModuleType("google.genai")
        types_module = types.ModuleType("google.genai.types")
        errors_module = types.ModuleType("google.genai.errors")
        errors_module.APIError = type("APIError", (Exception,), {})
        types_module.GenerateContentConfig = Mock(return_value=object())
        fake_client = Mock()
        fake_client.models.generate_content.return_value.text = '["வணக்கம்"]'
        genai_module.Client = Mock(return_value=fake_client)
        google_module = types.ModuleType("google")
        google_module.genai = genai_module

        with patch.dict(
            sys.modules,
            {
                "google": google_module,
                "google.genai": genai_module,
                "google.genai.types": types_module,
                "google.genai.errors": errors_module,
            },
        ):
            translated = translate_segments(
                [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                api_key="test-key",
                model_name="test-model",
            )

        self.assertEqual(translated[0]["translation"], "வணக்கம்")
        fake_client.models.generate_content.assert_called_once()

    def test_reports_unavailable_gemini_model_without_traceback(self) -> None:
        class FakeAPIError(Exception):
            code = 404
            message = "Model was not found"

        genai_module = types.ModuleType("google.genai")
        types_module = types.ModuleType("google.genai.types")
        errors_module = types.ModuleType("google.genai.errors")
        errors_module.APIError = FakeAPIError
        types_module.GenerateContentConfig = Mock(return_value=object())
        fake_client = Mock()
        fake_client.models.generate_content.side_effect = FakeAPIError()
        genai_module.Client = Mock(return_value=fake_client)
        google_module = types.ModuleType("google")
        google_module.genai = genai_module

        with patch.dict(
            sys.modules,
            {
                "google": google_module,
                "google.genai": genai_module,
                "google.genai.types": types_module,
                "google.genai.errors": errors_module,
            },
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Gemini model 'old-model' is unavailable.*--translation-model",
            ):
                translate_segments(
                    [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                    api_key="test-key",
                    model_name="old-model",
                )

    def test_mux_maps_video_and_replaces_audio_with_aligned_tamil_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            video_path = folder / "source.mp4"
            clip_path = folder / "voice.mp3"
            output_path = folder / "dubbed.mp4"
            video_path.touch()
            clip_path.touch()

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if "ffprobe" in command[0]:
                    return subprocess.CompletedProcess(command, 0, "3.0\n", "")
                Path(command[-1]).write_bytes(b"mp4")
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                patch("transcribe_video.require_binary", side_effect=lambda name: name),
                patch("transcribe_video.subprocess.run", side_effect=fake_run) as run,
            ):
                result = mux_tamil_audio(
                    video_path,
                    output_path,
                    [{"start": 1.25, "end": 2.0, "translation": "வணக்கம்"}],
                    [clip_path],
                )

            self.assertEqual(result, output_path)
            ffmpeg_command = run.call_args.args[0]
            self.assertIn("0:v:0", ffmpeg_command)
            self.assertIn("[aout]", ffmpeg_command)
            filter_graph = ffmpeg_command[ffmpeg_command.index("-filter_complex") + 1]
            self.assertIn("adelay=delays=1250:all=1", filter_graph)
            self.assertIn("copy", ffmpeg_command)
            self.assertTrue(output_path.is_file())

    def test_dubbing_requires_api_key_before_processing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main(["video.mp4", "--dub-tamil"]), 1)

    def test_rejects_tts_model_for_translation(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                main(
                    [
                        "video.mp4",
                        "--dub-tamil",
                        "--translation-model",
                        "gemini-3.8-flash-tts",
                    ]
                ),
                1,
            )


class WriteTranscriptionTests(unittest.TestCase):
    def test_writes_json_srt_and_plain_text(self) -> None:
        metadata = {"language": "en", "language_probability": 0.99, "duration": 2.0}
        segments = [{"start": 0.0, "end": 1.25, "text": "Hello"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, srt_path, text_path = write_transcription(
                Path(temp_dir), "video", metadata, segments
            )

            self.assertEqual(json.loads(json_path.read_text())["segments"], segments)
            self.assertEqual(
                srt_path.read_text(),
                "1\n00:00:00,000 --> 00:00:01,250\nHello\n",
            )
            self.assertEqual(text_path.read_text(), "Hello\n")


if __name__ == "__main__":
    unittest.main()

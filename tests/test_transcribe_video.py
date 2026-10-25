import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transcribe_video import extract_audio, format_srt_timestamp, write_transcription


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
                with self.assertRaisesRegex(RuntimeError, "FFmpeg was not found"):
                    extract_audio(video_path, Path(temp_dir) / "video.wav")


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

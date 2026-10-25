"""Extract a video's audio and transcribe it with Faster-Whisper."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extract mono, 16 kHz PCM audio from a video using the FFmpeg CLI."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg was not found. Install it and make sure the `ffmpeg` "
            "command is available on your PATH."
        )

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or "No error details were provided."
        raise RuntimeError(f"FFmpeg could not extract audio:\n{details}")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not create a usable audio file: {audio_path}")

    return audio_path


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp, rounding to the nearest millisecond."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02}:{minutes:02}:{whole_seconds:02},"
        f"{milliseconds:03}"
    )


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str | None,
    device: str,
    compute_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Transcribe audio and return metadata and timestamped segments."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Faster-Whisper is not installed. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), language=language)
    timestamped_segments = [
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        for segment in segments
    ]
    metadata = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }
    return metadata, timestamped_segments


def write_transcription(
    output_dir: Path,
    stem: str,
    metadata: dict[str, Any],
    segments: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Write timestamped JSON, SRT subtitles, and plain-text transcript files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    srt_path = output_dir / f"{stem}.srt"
    text_path = output_dir / f"{stem}.txt"

    json_path.write_text(
        json.dumps(
            {"metadata": metadata, "segments": segments},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    srt_entries = [
        (
            f"{index}\n"
            f"{format_srt_timestamp(segment['start'])} --> "
            f"{format_srt_timestamp(segment['end'])}\n"
            f"{segment['text']}\n"
        )
        for index, segment in enumerate(segments, start=1)
    ]
    srt_path.write_text("\n".join(srt_entries), encoding="utf-8")
    text_path.write_text(
        "\n".join(segment["text"] for segment in segments) + "\n",
        encoding="utf-8",
    )
    return json_path, srt_path, text_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract audio from a video and transcribe it locally."
    )
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for the WAV and transcript files (default: ./output)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Faster-Whisper model size or local model path (default: small)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Audio language code, or 'auto' to detect it (default: en)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Faster-Whisper compute type (default: int8)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    language = None if args.language.lower() == "auto" else args.language
    stem = args.video.stem

    try:
        audio_path = extract_audio(args.video, args.output_dir / f"{stem}.wav")
        metadata, segments = transcribe_audio(
            audio_path,
            model_name=args.model,
            language=language,
            device=args.device,
            compute_type=args.compute_type,
        )
        output_paths = write_transcription(
            args.output_dir, stem, metadata, segments
        )
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Extracted audio: {audio_path}")
    print(f"Detected language: {metadata['language']}")
    print("Transcripts:")
    for path in output_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

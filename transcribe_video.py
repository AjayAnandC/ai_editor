"""Extract a video's audio and transcribe it with Faster-Whisper."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise RuntimeError(
            f"{name.upper()} was not found. Install it and make sure the `{name}` "
            "command is available on your PATH."
        )
    return binary


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extract mono, 16 kHz PCM audio from a video using the FFmpeg CLI."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffmpeg = require_binary("ffmpeg")

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


def remove_subtitle_tracks(video_path: Path, output_path: Path) -> Path:
    """Copy a video while excluding selectable subtitle streams."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffmpeg = require_binary("ffmpeg")
    if video_path.resolve() == output_path.resolve():
        raise ValueError("Subtitle-free output must not overwrite the input video.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0",
            "-map",
            "-0:s",
            "-c",
            "copy",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or "No error details were provided."
        raise RuntimeError(f"FFmpeg could not remove subtitle tracks:\n{details}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"FFmpeg did not create a usable video file: {output_path}"
        )
    return output_path


def translate_segments(
    segments: list[dict[str, Any]],
    api_key: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """Translate transcript segments to conversational Tamil with Gemini."""
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError as exc:
        raise RuntimeError(
            "The Gemini SDK is not installed. Run "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    client = genai.Client(api_key=api_key)
    translated_segments: list[dict[str, Any]] = []
    batch_size = 30
    for start in range(0, len(segments), batch_size):
        batch = segments[start : start + batch_size]
        prompt = (
            "Translate each English transcript segment into concise, natural "
            "spoken Tamil suitable for dubbing. Preserve meaning, tone, names, "
            "and numbers. Return only a JSON array of Tamil strings, with "
            "exactly one string per input in the same order.\n\n"
            + json.dumps([segment["text"] for segment in batch], ensure_ascii=False)
        )
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
        except errors.APIError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    f"Gemini model '{model_name}' is unavailable for this API key. "
                    "Choose a model available to your account with "
                    "`--translation-model MODEL_NAME`."
                ) from exc
            if exc.code in (401, 403):
                raise RuntimeError(
                    "Gemini rejected the API key or denied access. Check that "
                    "GEMINI_API_KEY is valid and has Gemini API access."
                ) from exc
            if exc.code == 429:
                raise RuntimeError(
                    "Gemini rate or usage limit reached. Check the API quota "
                    "and try again later."
                ) from exc
            raise RuntimeError(
                f"Gemini API request failed (HTTP {exc.code}): {exc.message}"
            ) from exc
        try:
            translations = json.loads(response.text or "")
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                "Gemini returned an invalid translation response. Please retry."
            ) from exc
        if (
            not isinstance(translations, list)
            or len(translations) != len(batch)
            or any(
                not isinstance(text, str) or not text.strip()
                for text in translations
            )
        ):
            raise RuntimeError(
                "Gemini returned an unexpected number or format of Tamil "
                "translations. Please retry."
            )
        translated_segments.extend(
            {**segment, "translation": translation.strip()}
            for segment, translation in zip(batch, translations)
        )
    return translated_segments


async def _synthesize_segments(
    segments: list[dict[str, Any]],
    output_dir: Path,
    voice: str,
) -> list[Path]:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError(
            "Edge TTS is not installed. Run "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    clip_paths: list[Path] = []
    for index, segment in enumerate(segments, start=1):
        clip_path = output_dir / f"segment_{index:05}.mp3"
        await edge_tts.Communicate(segment["translation"], voice).save(
            str(clip_path)
        )
        if not clip_path.is_file() or clip_path.stat().st_size == 0:
            raise RuntimeError(f"Edge TTS did not create a usable audio clip: {clip_path}")
        clip_paths.append(clip_path)
    return clip_paths


def synthesize_segments(
    segments: list[dict[str, Any]],
    output_dir: Path,
    voice: str,
) -> list[Path]:
    """Synthesize translated segments to MP3 clips with Edge TTS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_synthesize_segments(segments, output_dir, voice))


def mux_tamil_audio(
    video_path: Path,
    output_path: Path,
    segments: list[dict[str, Any]],
    clip_paths: list[Path],
) -> Path:
    """Replace the video's audio with timestamp-aligned Tamil speech in an MP4."""
    if len(segments) != len(clip_paths):
        raise ValueError("Every translated segment must have one synthesized audio clip.")
    if not segments:
        raise ValueError("Cannot create dubbed video because no speech was transcribed.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if video_path.resolve() == output_path.resolve():
        raise ValueError("Dubbed output must not overwrite the input video.")

    ffmpeg = require_binary("ffmpeg")
    ffprobe = require_binary("ffprobe")
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        details = probe.stderr.strip() or "No error details were provided."
        raise RuntimeError(f"FFprobe could not read video duration:\n{details}")
    try:
        duration = float(probe.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("FFprobe returned an invalid video duration.") from exc
    if duration <= 0:
        raise RuntimeError("The input video has an invalid duration.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = [
        f"anullsrc=r=48000:cl=stereo,atrim=duration={duration},"
        "asetpts=PTS-STARTPTS[silence]"
    ]
    mix_inputs = ["[silence]"]
    command = [ffmpeg, "-y", "-i", str(video_path)]
    for index, (segment, clip_path) in enumerate(zip(segments, clip_paths), start=1):
        command.extend(["-i", str(clip_path)])
        delay_ms = max(0, round(float(segment["start"]) * 1000))
        filters.append(
            f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS,"
            f"adelay=delays={delay_ms}:all=1[voice{index}]"
        )
        mix_inputs.append(f"[voice{index}]")
    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest:"
        f"dropout_transition=0:normalize=0,atrim=duration={duration},"
        "asetpts=PTS-STARTPTS[aout]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            str(duration),
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or "No error details were provided."
        raise RuntimeError(f"FFmpeg could not create the Tamil-dubbed MP4:\n{details}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not create a usable video file: {output_path}")
    return output_path


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
    try:
        segments, info = model.transcribe(str(audio_path), language=language)
        timestamped_segments = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
            for segment in segments
        ]
    except TypeError as exc:
        if "metadata_errors" not in str(exc) or "unexpected keyword" not in str(exc):
            raise
        raise RuntimeError(
            "The installed PyAV version is incompatible with Faster-Whisper. "
            "From the ai_editor folder, run "
            '`python3 -m pip install --upgrade "av>=11,<19"` and retry.'
        ) from exc
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
    parser.add_argument(
        "--remove-subtitles",
        action="store_true",
        help="Also create a video copy without selectable subtitle tracks",
    )
    parser.add_argument(
        "--dub-tamil",
        action="store_true",
        help="Translate to Tamil and export an MP4 with Tamil audio",
    )
    parser.add_argument(
        "--translation-model",
        default="gemini-3.8-flash",
        help="Gemini model used for translation (default: gemini-3.8-flash)",
    )
    parser.add_argument(
        "--voice",
        default="ta-IN-PallaviNeural",
        help="Edge TTS voice used for Tamil speech",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    language = None if args.language.lower() == "auto" else args.language
    stem = args.video.stem
    if args.dub_tamil and args.translation_model.lower().endswith("-tts"):
        print(
            "Error: --translation-model must be a text-generation model, not a "
            "TTS model. Use `--translation-model gemini-3.8-flash`; the `-tts` "
            "model family is for audio generation.",
            file=sys.stderr,
        )
        return 1
    api_key = os.environ.get("GEMINI_API_KEY")
    if args.dub_tamil and not api_key:
        print(
            "Error: Set GEMINI_API_KEY in your environment to use --dub-tamil.",
            file=sys.stderr,
        )
        return 1

    try:
        subtitle_free_video = None
        if args.remove_subtitles:
            subtitle_free_video = remove_subtitle_tracks(
                args.video,
                args.output_dir / f"{stem}_no_subtitles{args.video.suffix}",
            )
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
        tamil_output = None
        if args.dub_tamil:
            translated_segments = translate_segments(
                segments,
                api_key=api_key or "",
                model_name=args.translation_model,
            )
            tamil_segments = [
                {**segment, "text": segment["translation"]}
                for segment in translated_segments
            ]
            output_paths = (
                *output_paths,
                *write_transcription(
                    args.output_dir, f"{stem}_ta", metadata, tamil_segments
                ),
            )
            with tempfile.TemporaryDirectory(
                prefix=f"{stem}_tamil_", dir=args.output_dir
            ) as temp_dir:
                clip_paths = synthesize_segments(
                    translated_segments,
                    Path(temp_dir),
                    voice=args.voice,
                )
                tamil_output = mux_tamil_audio(
                    args.video,
                    args.output_dir / f"{stem}_tamil.mp4",
                    translated_segments,
                    clip_paths,
                )
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Extracted audio: {audio_path}")
    print(f"Detected language: {metadata['language']}")
    if tamil_output is not None:
        print(f"Tamil-dubbed video: {tamil_output}")
    if subtitle_free_video is not None:
        print(f"Video without selectable subtitle tracks: {subtitle_free_video}")
    print("Transcripts:")
    for path in output_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

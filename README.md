# AI Editor: video transcription

This first prototype implements the audio extraction and speech-to-text steps
for a local English-to-Tamil dubbing workflow. It extracts mono 16 kHz WAV audio
with FFmpeg, transcribes it with Faster-Whisper, and saves timestamped JSON and
SRT output plus a plain-text transcript. Translation, Tamil speech synthesis,
and video remuxing are not included yet.

## Prerequisites

- Python 3.10 or newer
- FFmpeg installed and available as `ffmpeg` on your `PATH`
- Faster-Whisper dependencies:

```bash
python3 -m pip install -r requirements.txt
```

On macOS, install FFmpeg with `brew install ffmpeg`. On Debian or Ubuntu, use
`sudo apt install ffmpeg`. On Windows, install an FFmpeg build and add its
`bin` directory to `PATH`.

## Usage

Run from this folder:

```bash
python3 transcribe_video.py /path/to/video.mp4
```

By default, the script writes `video.wav`, `video.json`, `video.srt`, and
`video.txt` into `./output`. The JSON contains the detected language and
timestamped segment text.

Choose another output directory, model, or language:

```bash
python3 transcribe_video.py /path/to/video.mp4 \
  --output-dir ./work \
  --model small \
  --language en
```

Set `--language auto` to let Faster-Whisper detect the language. The first run
may download the selected model. The default CPU/int8 settings work on
CPU-only systems; use `--device cuda` on a supported CUDA setup.

## Tests

```bash
python3 -m unittest discover -s tests
```

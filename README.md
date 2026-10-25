# AI Editor: Tamil video dubbing

The tool can transcribe an English video, translate its timed speech into
conversational Tamil with Gemini, synthesize Tamil speech with Edge TTS, and
export an MP4 with the Tamil voice replacing the original audio. It also saves
the extracted WAV and English/Tamil transcript files. Translation needs an
internet connection and sends transcript text to the Gemini API; Edge TTS also
uses an online service.

## Prerequisites

- Python 3.10 or newer
- FFmpeg installed and available as `ffmpeg` on your `PATH`
- Faster-Whisper dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The requirements keep PyAV below version 19 because newer versions no longer
accept an argument used by Faster-Whisper's audio decoder. If you see a
`metadata_errors` TypeError, update the installed dependency with:

```bash
python3 -m pip install --upgrade "av>=11,<19"
```

On macOS, install FFmpeg with `brew install ffmpeg`. On Debian or Ubuntu, use
`sudo apt install ffmpeg`. On Windows, install an FFmpeg build and add its
`bin` directory to `PATH`.

## Create a Tamil-dubbed MP4

Set your Gemini API key in the terminal without putting it in the command
history or a project file. Type the key when prompted, then press Enter:

```bash
read -s GEMINI_API_KEY
export GEMINI_API_KEY
```

Run the complete pipeline:

```bash
python3 transcribe_video.py /path/to/video.mp4 --dub-tamil
```

For example:

```bash
python3 transcribe_video.py "$HOME/Downloads/my_video.mp4" --dub-tamil
```

The final file will be `output/my_video_tamil.mp4`. The output folder also
contains the extracted WAV and English/Tamil JSON, SRT, and text transcripts.
The original video file is not changed. The final MP4 keeps the video stream,
replaces the original audio with the generated Tamil speech, and does not carry
over selectable subtitle tracks. It does not remove subtitles burned into the
picture. Segment start times are used for alignment; translated speech may
overlap or be clipped at the end of the video if it runs longer than the source.

You can select another Gemini translation model or Tamil TTS voice:

```bash
python3 transcribe_video.py /path/to/video.mp4 --dub-tamil \
  --translation-model gemini-3.8-flash \
  --voice ta-IN-ValluvarNeural
```

Use a text-generation model such as `gemini-3.8-flash` for
`--translation-model`. Models with a `-tts` suffix generate speech and are not
translation models; the Tamil voice is selected separately with `--voice`.

Keep your API key private. If it was exposed, revoke it with the provider and
create a replacement. Do not paste API keys into source files or commit them.

## Transcription-only usage

To only extract and transcribe without translation or dubbing:

```bash
python3 transcribe_video.py /path/to/video.mp4
```

The script writes `video.wav`, `video.json`, `video.srt`, and `video.txt` into
`./output`. The JSON contains the detected language and timestamped segment text.

Choose another output directory, model, or language:

```bash
python3 transcribe_video.py /path/to/video.mp4 \
  --output-dir ./work \
  --model small \
  --language en
```

To create a separate copy without selectable subtitle tracks:

```bash
python3 transcribe_video.py /path/to/video.mp4 --remove-subtitles
```

The copy is written to `./output/video_no_subtitles.mp4`, alongside the
transcription files. FFmpeg copies the existing video/audio streams without
re-encoding and excludes subtitle streams. This does not remove captions or
subtitles that are permanently burned into the video picture.

Set `--language auto` to let Faster-Whisper detect the language. The first run
may download the selected model. The default CPU/int8 settings work on
CPU-only systems; use `--device cuda` on a supported CUDA setup.

## Tests

```bash
python3 -m unittest discover -s tests
```

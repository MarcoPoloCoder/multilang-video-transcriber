# MuLtiLang-viDeo-transcRibeR

This tool extracts audio from videos and generates high-quality transcripts and subtitle files using OpenAI's Whisper speech recognition model.

## Features

- Extracts audio from video files automatically
- Transcribes audio using state-of-the-art Whisper speech recognition
- Generates subtitle files in multiple formats:
  - Plain text transcript (.txt)
  - SubRip subtitle format (.srt)
  - WebVTT subtitle format (.vtt)
- Multiple model sizes available (tiny, base, small, medium, large)
- Progress indicators and rich console output

## Installation

### Prerequisites

1. Python 3.8 or higher
2. FFmpeg installed on your system

### Install FFmpeg

#### On Ubuntu/Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

#### On macOS with Homebrew

```bash
brew install ffmpeg
```

#### On Windows

Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your PATH.

### Python Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

The first time you run the script, it will download the Whisper model file.

## Usage

### Basic Usage

```bash
python video_transcriber.py path/to/your/video.mp4
```

This will:

1. Extract the audio from your video
2. Transcribe it using Whisper's base model
3. Generate subtitle files in the "output" directory

### Advanced Options

```bash
python video_transcriber.py path/to/your/video.mp4 --output custom_dir --model medium
```

#### Arguments

- `video_path`: Path to the video file (required)
- `--output, -o`: Output directory for generated files (default: "output")
- `--model, -m`: Whisper model size to use (choices: tiny, base, small, medium, large)

## Model Selection Guide

- **tiny**: Fastest, lowest accuracy (good for testing)
- **base**: Good balance of speed and accuracy for most content
- **small**: Better accuracy, still relatively fast
- **medium**: High accuracy, moderate processing time
- **large**: Best accuracy, slowest processing time

The larger models provide better recognition quality but require more RAM and processing time.

## Output Files

For an input file `video.mp4`, the script generates:

- `video.txt`: Complete transcript in plain text
- `video.srt`: Subtitle file in SubRip format (works with most video players)
- `video.vtt`: Subtitle file in WebVTT format (works well with HTML5 video)
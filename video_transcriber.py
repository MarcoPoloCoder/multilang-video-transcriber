import argparse
import os
import tempfile
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import ffmpeg
from rich.console import Console
from rich.progress import Progress

console = Console()

def extract_audio(video_path, output_path=None):
    """Extract audio from video file using ffmpeg"""
    if output_path is None:
        # Create a temporary file with .wav extension
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"{Path(video_path).stem}_audio.wav")
    
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar='16k')
            .run(quiet=True, overwrite_output=True)
        )
        return output_path
    except ffmpeg.Error as e:
        console.print(f"[bold red]Error extracting audio:[/bold red] {e.stderr.decode()}")
        return None

def load_whisper_model(model_size="small"):
    """Load Whisper model from HuggingFace"""
    model_id = f"openai/whisper-{model_size}"
    
    with console.status(f"[bold green]Loading Whisper {model_size} model from HuggingFace...", spinner="dots"):
        # Check for CUDA availability
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        console.print(f"[dim]Using device: {device}[/dim]")
        
        # Load model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        return pipe

def transcribe_audio(pipe, audio_path):
    """Transcribe audio using Whisper HuggingFace model"""
    console.print(f"[bold green]Transcribing audio...[/bold green]")
    
    result = pipe(audio_path)
    
    # Extract transcript and chunks with timestamps
    transcript = result["text"]
    chunks = result["chunks"] if "chunks" in result else []
    
    # If no chunks are returned (older transformers versions), create segments from the text
    if not chunks and hasattr(pipe, "tokenizer"):
        from transformers.pipelines.automatic_speech_recognition import chunk_iter
        chunks = []
        words = transcript.split()
        chunk_size = max(1, len(words) // 10)  # Simple chunking
        
        for i, chunk in enumerate(chunk_iter(words, chunk_size)):
            chunk_text = " ".join(chunk)
            chunks.append({
                "text": chunk_text,
                "timestamp": (i * 5.0, (i + 1) * 5.0)  # Approximate timestamps
            })
    
    # Convert chunks to segments format
    segments = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict) and "timestamp" in chunk:
            start, end = chunk["timestamp"]
            text = chunk["text"]
        else:
            # Fallback if chunk format is unexpected
            text = str(chunk)
            start, end = i * 5.0, (i + 1) * 5.0
            
        segments.append({
            "id": i,
            "start": start,
            "end": end,
            "text": text
        })
    
    return {"text": transcript, "segments": segments}

def format_srt(segments):
    """Convert segments to SRT format"""
    srt_content = ""
    
    for i, segment in enumerate(segments, start=1):
        # Get start and end time in SRT format (HH:MM:SS,mmm)
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        
        # Add subtitle entry
        srt_content += f"{i}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{segment['text'].strip()}\n\n"
    
    return srt_content

def format_vtt(segments):
    """Convert segments to WebVTT format"""
    vtt_content = "WEBVTT\n\n"
    
    for i, segment in enumerate(segments, start=1):
        # Get start and end time in WebVTT format (HH:MM:SS.mmm)
        start_time = format_timestamp(segment["start"], format_type="vtt")
        end_time = format_timestamp(segment["end"], format_type="vtt")
        
        # Add subtitle entry
        vtt_content += f"{i}\n"
        vtt_content += f"{start_time} --> {end_time}\n"
        vtt_content += f"{segment['text'].strip()}\n\n"
    
    return vtt_content

def format_timestamp(seconds, format_type="srt"):
    """Convert seconds to SRT (HH:MM:SS,mmm) or WebVTT (HH:MM:SS.mmm) timestamp format"""
    if seconds is None:
        return "00:00:00,000" if format_type == "srt" else "00:00:00.000"

    hours = int(int(seconds) / 3600)
    minutes = int((int(seconds) % 3600) / 60)
    seconds = int(seconds) % 60
    
    if format_type == "srt":
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{int((seconds - int(seconds)) * 1000):03d}"
    else:  # WebVTT format
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{int((seconds - int(seconds)) * 1000):03d}"

def save_transcript(transcript, segments, output_dir, filename):
    """Save transcript in various formats"""
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(filename)[0]
    
    # Save raw transcript (txt)
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    # Save SRT subtitles
    srt_path = os.path.join(output_dir, f"{base_filename}.srt")
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(format_srt(segments))
    
    # Save WebVTT subtitles
    vtt_path = os.path.join(output_dir, f"{base_filename}.vtt")
    with open(vtt_path, 'w', encoding='utf-8') as f:
        f.write(format_vtt(segments))
    
    return {
        'txt': txt_path,
        'srt': srt_path,
        'vtt': vtt_path
    }

def main():
    parser = argparse.ArgumentParser(description="Extract and generate subtitles from a video file using HuggingFace models")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", "-o", default="output", help="Output directory for subtitle files")
    parser.add_argument("--model", "-m", default="small", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use (larger = more accurate but slower)")
    args = parser.parse_args()
    
    video_path = args.video_path
    output_dir = args.output
    
    if not os.path.exists(video_path):
        console.print(f"[bold red]Error:[/bold red] Video file not found: {video_path}")
        return
    
    # Extract audio from video
    console.print(f"[bold]Processing video:[/bold] {os.path.basename(video_path)}")
    audio_path = extract_audio(video_path)
    if not audio_path:
        return
    
    # Load model and transcribe audio
    pipe = load_whisper_model(args.model)
    result = transcribe_audio(pipe, audio_path)
    
    # Save transcript and subtitles
    filename = os.path.basename(video_path)
    output_files = save_transcript(result["text"], result["segments"], output_dir, filename)
    
    # Display results
    console.print(f"\n[bold green]✓ Transcription complete![/bold green]")
    console.print(f"[bold]Files generated:[/bold]")
    console.print(f"  • Full transcript: [cyan]{output_files['txt']}[/cyan]")
    console.print(f"  • SRT subtitles: [cyan]{output_files['srt']}[/cyan]")
    console.print(f"  • WebVTT subtitles: [cyan]{output_files['vtt']}[/cyan]")
    
    # Clean up temporary audio file
    if audio_path.startswith(tempfile.gettempdir()):
        os.remove(audio_path)
        console.print(f"[dim]Temporary audio file removed[/dim]")

if __name__ == "__main__":
    main()
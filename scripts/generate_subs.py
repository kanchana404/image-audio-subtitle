#!/usr/bin/env python3
"""
generate_subs.py — transcribe an audio file to SRT subtitles using faster-whisper.

Usage:
  python scripts/generate_subs.py --audio assets/audio/bg.mp3 --out captions/subs.srt --lang en --model small

Requirements:
  pip install faster-whisper
  FFmpeg must be available on PATH (already installed in your setup)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    td = timedelta(seconds=seconds)
    # timedelta -> H:MM:SS.microseconds; we need HH:MM:SS,mmm
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments, out_path: str) -> None:
    lines = []
    index = 1
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start = float(seg.start)
        end = float(seg.end)
        lines.append(str(index))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
        index += 1
    data = "\n".join(lines).strip() + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio to SRT using faster-whisper.")
    parser.add_argument("--audio", required=True, help="Path to input audio file (e.g., MP3, WAV).")
    parser.add_argument("--out", required=True, help="Path to output SRT file.")
    parser.add_argument("--lang", default="en", help="Language code (default: en). Use 'auto' to auto-detect.")
    parser.add_argument(
        "--model", default="small", help="Whisper model size: tiny, base, small, medium, large-v3 (default: small)."
    )
    parser.add_argument(
        "--vad", action="store_true", help="Enable voice activity detection for cleaner segments (optional)."
    )

    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        raise SystemExit(f"Audio file not found: {args.audio}")

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise SystemExit(
            "faster-whisper is not installed. Install with:\n"
            "  python -m pip install faster-whisper\n"
            "Then re-run this command."
        ) from exc

    # Prefer CPU-friendly defaults; users with GPU can adjust env/args later
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    # Transcribe; vad filtering can help reduce noise/non-speech
    segments, _info = model.transcribe(
        args.audio,
        language=(None if args.lang == "auto" else args.lang),
        vad_filter=args.vad,
        vad_parameters={"min_silence_duration_ms": 300},
    )

    # segments is a generator; collect into list to write srt
    segments_list = list(segments)
    if not segments_list:
        raise SystemExit("No speech detected; SRT not created.")

    write_srt(segments_list, args.out)
    print(f"Wrote subtitles: {args.out}")


if __name__ == "__main__":
    main()




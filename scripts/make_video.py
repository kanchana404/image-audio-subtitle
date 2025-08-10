#!/usr/bin/env python3
"""
make_video.py — build a slideshow video from images + background audio using FFmpeg only.

No external Python packages are required. FFmpeg must be available on PATH.

Usage example:
  python make_video.py \
    --images_dir assets/images \
    --audio assets/audio/bg.mp3 \
    --out out/out_nosubs.mp4 \
    --w 1920 --h 1080 \
    --slide 4.0 --xfade 0.75 \
    --fps 30 --crf 19 --preset veryfast

Add-ons:
  Use --fit_audio to auto-scale slide duration so the total video duration matches the audio length.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from typing import List
from datetime import timedelta


def natural_sorted(paths: List[str]) -> List[str]:
    """Return case-insensitive filename sort.

    Works well when files are named with zero-padding (001.png, 002.png, ...).
    """
    return sorted(paths, key=lambda p: p.lower())


def build_ffmpeg_command(
    image_paths: List[str],
    audio_path: str,
    subs_path: str | None,
    out_path: str,
    width: int,
    height: int,
    slide_dur: float,
    xfade: float,
    fps: int,
    crf: int,
    preset: str,
    threads: int,
    fadein: float,
    fadeout: float,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """Construct an ffmpeg command that:
    - turns images into a letterboxed slideshow of size width x height
    - crossfades between slides by xfade seconds
    - loops/trims audio to match video length and applies gentle fades
    - optionally burns subtitles (SRT/ASS) into the video frame
    - outputs H.264 + AAC MP4
    """

    num_images = len(image_paths)
    if num_images == 0:
        raise ValueError("No images provided")

    # Total duration formula for N equal-length clips with crossfades:
    # total = N*L - (N-1)*xfade
    total_duration = num_images * slide_dur - (num_images - 1) * xfade
    total_duration = max(total_duration, 0.05)

    # ffmpeg inputs: one per image (looped stills), then the audio
    cmd: List[str] = [
        ffmpeg_bin,
        "-y",
    ]

    # Add image inputs
    for img in image_paths:
        cmd += [
            "-loop", "1",  # loop a single frame
            "-framerate", str(fps),
            "-t", f"{slide_dur}",
            "-i", img,
        ]

    # Add audio input (looped) if provided
    have_audio = bool(audio_path)
    if have_audio:
        cmd += [
            "-stream_loop", "-1",  # loop infinitely until cut by -shortest or trim
            "-i", audio_path,
        ]

    # Build filter_complex
    filter_parts: List[str] = []

    # Per-image: scale to fit, letterbox to exact WxH, set fps and pixel format
    for idx in range(num_images):
        filter_parts.append(
            f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"format=yuv420p,fps={fps}[v{idx}]"
        )

    # Chain xfade transitions if more than one image
    if num_images == 1:
        last_v_label = "v0"
    else:
        prev = "v0"
        for i in range(1, num_images):
            offset = i * (slide_dur - xfade)
            out_label = f"vx{i}"
            filter_parts.append(
                f"[{prev}][v{i}]xfade=transition=fade:duration={xfade}:offset={offset:.6f}[{out_label}]"
            )
            prev = out_label
        last_v_label = prev

    # Optional audio processing
    if have_audio:
        audio_input_index = num_images  # last input is audio
        # Trim to exact total duration, apply gentle fades in/out
        # st for fade out is (total_duration - fadeout), but not below 0
        fadeout_start = max(0.0, total_duration - max(0.0, fadeout))
        filter_parts.append(
            f"[{audio_input_index}:a]atrim=0:{total_duration:.6f},asetpts=N/SR/TB,"
            f"afade=t=in:st=0:d={max(0.0, fadein):.6f},"
            f"afade=t=out:st={fadeout_start:.6f}:d={max(0.0, fadeout):.6f}[aout]"
        )

    # Optional subtitles burn-in. We apply after transitions on the composed stream.
    if subs_path:
        # Prefer a relative, forward-slashed path for ffmpeg filter parsing on Windows.
        try:
            rel = os.path.relpath(os.path.abspath(subs_path), start=os.getcwd())
        except Exception:
            rel = subs_path
        rel = rel.replace("\\", "/")
        out_label = "vsub"
        # charenc guards against UTF-8 subtitle files
        filter_parts.append(
            f"[{last_v_label}]subtitles=filename='{rel}':charenc=UTF-8[{out_label}]"
        )
        last_v_label = out_label

    filter_complex = ";".join(filter_parts)

    # Mapping and codecs
    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{last_v_label}]",
    ]
    if have_audio:
        cmd += ["-map", "[aout]"]

    cmd += [
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", str(preset),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", str(threads),
    ]

    # Audio codec if present and ensure we cut to the shortest (video) stream length
    if have_audio:
        cmd += ["-c:a", "aac", "-shortest"]

    cmd += [out_path]
    return cmd


def get_media_duration_seconds(path: str, ffprobe_bin: str = "ffprobe") -> float:
    """Return media duration in seconds using ffprobe. Raises on failure."""
    probe_cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, check=True
        )
        val = result.stdout.strip()
        return float(val)
    except Exception as exc:
        raise SystemExit(f"Failed to probe duration with ffprobe for: {path}") from exc


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt_from_segments(segments, out_path: str) -> int:
    lines: List[str] = []
    index = 1
    for seg in segments:
        text = (getattr(seg, "text", None) or "").strip()
        if not text:
            continue
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", start))
        lines.append(str(index))
        lines.append(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
        index += 1
    data = "\n".join(lines).strip() + ("\n" if lines else "")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(data)
    return index - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a slideshow video from images + bg audio using FFmpeg.")
    parser.add_argument("--images_dir", required=True, help="Directory containing images (sorted by filename).")
    parser.add_argument("--audio", required=True, help="Background audio file (e.g., mp3).")
    parser.add_argument("--out", required=True, help="Output MP4 path.")
    parser.add_argument("--subs", default=None, help="Optional path to SRT/ASS subtitles to burn in.")
    parser.add_argument("--w", type=int, default=1920, help="Output width (default: 1920).")
    parser.add_argument("--h", type=int, default=1080, help="Output height (default: 1080).")
    parser.add_argument("--slide", type=float, default=4.0, help="Seconds each image stays (default: 4.0).")
    parser.add_argument("--xfade", type=float, default=0.75, help="Crossfade seconds between images (default: 0.75).")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    parser.add_argument("--crf", type=int, default=19, help="x264 CRF quality (lower=better, default: 19).")
    parser.add_argument("--preset", default="medium", help="x264 preset (ultrafast..placebo, default: medium).")
    parser.add_argument("--threads", type=int, default=4, help="FFmpeg threads (default: 4).")
    parser.add_argument("--fadein", type=float, default=0.5, help="Audio fade-in seconds (default: 0.5).")
    parser.add_argument("--fadeout", type=float, default=0.5, help="Audio fade-out seconds (default: 0.5).")
    parser.add_argument(
        "--fit_audio",
        action="store_true",
        help="Auto-adjust slide duration so total video duration matches audio length.",
    )
    parser.add_argument("--auto_subs", action="store_true", help="Auto-generate SRT from audio with faster-whisper.")
    parser.add_argument("--subs_lang", default="en", help="Subtitle language (use 'auto' to detect).")
    parser.add_argument("--whisper_model", default="small", help="Whisper model size (tiny/base/small/medium/large-v3).")
    parser.add_argument("--ffmpeg_bin", default="ffmpeg", help="Path to ffmpeg binary (optional).")
    parser.add_argument("--ffprobe_bin", default="ffprobe", help="Path to ffprobe binary (optional).")

    args = parser.parse_args()

    if args.slide <= 0:
        raise SystemExit("--slide must be > 0")
    if args.xfade < 0:
        raise SystemExit("--xfade must be >= 0")

    if shutil.which(args.ffmpeg_bin) is None:
        raise SystemExit("ffmpeg not found in PATH. Please install FFmpeg and try again.")
    if shutil.which(args.ffprobe_bin) is None:
        raise SystemExit("ffprobe not found in PATH. Please install FFmpeg (includes ffprobe).")

    # Collect images
    if not os.path.isdir(args.images_dir):
        raise SystemExit(f"Images directory not found: {args.images_dir}")

    image_paths = [
        p for p in glob.glob(os.path.join(args.images_dir, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    image_paths = natural_sorted(image_paths)

    if not image_paths:
        raise SystemExit(f"No images found in: {args.images_dir}")

    if not os.path.isfile(args.audio):
        raise SystemExit(f"Audio file not found: {args.audio}")

    # Generate subtitles if requested
    if args.auto_subs:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise SystemExit(
                "Auto subtitles require faster-whisper. Install with:\n  python -m pip install faster-whisper"
            ) from exc

        model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(
            args.audio,
            language=(None if args.subs_lang == "auto" else args.subs_lang),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        auto_path = os.path.join("captions", "auto.srt")
        count = _write_srt_from_segments(list(segments), auto_path)
        if count == 0:
            raise SystemExit("No speech detected; cannot build subtitles.")
        args.subs = auto_path
    elif args.subs:
        if not os.path.isfile(args.subs):
            raise SystemExit(f"Subtitles file not found: {args.subs}")

    # Ensure output folder exists
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Optionally fit slideshow duration to audio duration by scaling slide length
    if args.fit_audio:
        audio_dur = get_media_duration_seconds(args.audio, ffprobe_bin=args.ffprobe_bin)
        num_images = len(image_paths)
        # total = N*L - (N-1)*xfade  =>  L = (total + (N-1)*xfade)/N
        desired_slide = (audio_dur + (num_images - 1) * args.xfade) / max(1, num_images)
        # Ensure valid overlap by adjusting xfade down if needed
        if args.xfade >= desired_slide:
            args.xfade = max(0.0, desired_slide - 0.05)
        args.slide = max(0.05, desired_slide)

    # Final guard after potential adjustment
    if args.xfade >= args.slide:
        print(
            "Warning: crossfade is too large for slide duration; lowering xfade to slide - 0.05s.",
            file=sys.stderr,
        )
        args.xfade = max(0.0, args.slide - 0.05)

    # Build and run ffmpeg command
    cmd = build_ffmpeg_command(
        image_paths=image_paths,
        audio_path=args.audio,
        subs_path=args.subs,
        out_path=args.out,
        width=args.w,
        height=args.h,
        slide_dur=args.slide,
        xfade=args.xfade,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        threads=args.threads,
        fadein=args.fadein,
        fadeout=args.fadeout,
        ffmpeg_bin=args.ffmpeg_bin,
    )

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        # Surface the exact ffmpeg command for easier debugging
        printable = " ".join(cmd)
        raise SystemExit(f"ffmpeg failed with exit code {exc.returncode}.\nCommand:\n{printable}") from exc


if __name__ == "__main__":
    main()

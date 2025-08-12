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
    # Visual effects
    kenburns: bool = False,
    kenburns_direction: str = "auto",  # auto | zoom_in | zoom_out | pan_lr | pan_rl | pan_tb | pan_bt
    kenburns_strength: float = 1.1,     # max zoom factor (e.g., 1.1 = 10%)
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

    # Per-image: scale to fit and optionally apply Ken Burns (zoom/pan) before letterboxing
    # We generate one composed clip per image labeled [v{idx}]
    for idx in range(num_images):
        if kenburns:
            # Number of frames to output for this still
            frames = max(1, int(round(slide_dur * fps)))
            zmax = max(1.0, float(kenburns_strength))
            zincr = max(0.0, (zmax - 1.0) / max(1, frames - 1))

            # Choose direction per image index when auto
            if kenburns_direction == "auto":
                pattern = ["zoom_in", "pan_lr", "pan_rl", "zoom_out"][idx % 4]
            else:
                pattern = kenburns_direction

            if pattern == "zoom_in":
                z_expr = f"min(zoom+{zincr:.6f},{zmax:.6f})"
                x_expr = "(iw-ow)/2"
                y_expr = "(ih-oh)/2"
            elif pattern == "zoom_out":
                # Start at zmax and decrease
                # zoompan initializes zoom=1 by default; emulate decreasing by computing from frame index 'on'
                z_expr = f"max({zmax:.6f}-{zincr:.6f}*on,1.0)"
                x_expr = "(iw-ow)/2"
                y_expr = "(ih-oh)/2"
            elif pattern == "pan_lr":
                z_expr = "1.0"
                x_expr = f"(iw-ow)*on/{max(1, frames - 1)}"
                y_expr = "(ih-oh)/2"
            elif pattern == "pan_rl":
                z_expr = "1.0"
                x_expr = f"(iw-ow)*(1 - on/{max(1, frames - 1)})"
                y_expr = "(ih-oh)/2"
            elif pattern == "pan_tb":
                z_expr = "1.0"
                x_expr = "(iw-ow)/2"
                y_expr = f"(ih-oh)*on/{max(1, frames - 1)}"
            elif pattern == "pan_bt":
                z_expr = "1.0"
                x_expr = "(iw-ow)/2"
                y_expr = f"(ih-oh)*(1 - on/{max(1, frames - 1)})"
            else:
                # Fallback to gentle zoom-in
                z_expr = f"min(zoom+{zincr:.6f},{zmax:.6f})"
                x_expr = "(iw-ow)/2"
                y_expr = "(ih-oh)/2"

            # Build a single filter chain string
            filter_parts.append(
                f"[{idx}:v]"
                f"scale={width}*{zmax:.6f}:{height}*{zmax:.6f}:force_original_aspect_ratio=increase,"
                f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={width}x{height},"
                f"fps={fps},format=yuv420p[v{idx}]"
            )
        else:
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


def _write_ass_two_word_from_segments(segments, out_path: str, width: int, height: int) -> int:
    """Write an ASS subtitle file where words appear in pairs, centered slightly below center,
    with simple fade-in and karaoke-like sequential reveal per word.

    Since word timestamps may not be available, we distribute time evenly across words in a segment.
    """
    # ASS header with a single centered style; we'll position with \pos to y ~ 70% of height
    lines: List[str] = []
    lines.append("[Script Info]")
    lines.append(f"PlayResX: {width}")
    lines.append(f"PlayResY: {height}")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("WrapStyle: 2")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    # White text, black outline, subtle shadow
    lines.append("Style: CenterBig, Arial, 64, &H00FFFFFF, &H000000FF, &H00000000, &H64000000, -1, 0, 0, 0, 100, 100, 0, 0, 1, 5, 2, 2, 30, 30, 100, 1")
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    def fmt(t: float) -> str:
        return _format_srt_timestamp(max(0.0, t)).replace(",", ".")  # ASS expects HH:MM:SS.xx

    y = int(round(height * 0.68))
    x = int(round(width / 2))

    count = 0
    for seg in segments:
        raw = (getattr(seg, "text", None) or "").strip()
        if not raw:
            continue
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", start))
        dur = max(0.0, end - start)
        words = [w for w in raw.split() if w]
        if not words or dur <= 0:
            continue

        # Pair words: [w0 w1], [w2 w3], ...
        pairs: List[List[str]] = []
        i = 0
        while i < len(words):
            pairs.append(words[i:i+2])
            i += 2

        pair_dur = dur / len(pairs)
        for j, pair in enumerate(pairs):
            p_start = start + j * pair_dur
            p_end = start + (j + 1) * pair_dur
            # Distribute within-pair duration over 1 or 2 words for karaoke effect
            k_total_cs = max(1, int(round((p_end - p_start) * 100)))  # centiseconds
            if len(pair) == 1:
                k1 = k_total_cs
                ass_text = f"{{\\pos({x},{y})\\fad(80,80)}}{{\\k{k1}}}{pair[0]}"
            else:
                k1 = max(1, int(round(k_total_cs * 0.5)))
                k2 = max(1, k_total_cs - k1)
                ass_text = f"{{\\pos({x},{y})\\fad(80,80)}}{{\\k{k1}}}{pair[0]} {{\\k{k2}}}{pair[1]}"

            lines.append(
                f"Dialogue: 0,{fmt(p_start)},{fmt(p_end)},CenterBig,,0000,0000,0000,,{ass_text}"
            )
            count += 1

    data = "\n".join(lines).strip() + ("\n" if lines else "")
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig") as f:
        f.write(data)
    return count


def _extract_words_from_segment(seg) -> List[tuple[str, float, float]]:
    """Return list of (word, start, end) if available, else []"""
    words = []
    try:
        for w in getattr(seg, "words", []) or []:
            txt = (getattr(w, "word", None) or "").strip()
            if not txt:
                continue
            ws = float(getattr(w, "start", getattr(seg, "start", 0.0)))
            we = float(getattr(w, "end", getattr(seg, "end", ws)))
            words.append((txt, ws, we))
    except Exception:
        return []
    return words


def _write_ass_three_word_from_segments(segments, out_path: str, width: int, height: int) -> int:
    """Write ASS where exactly three words are shown at a time using per-word timestamps.

    Words appear centered slightly below center with gentle fade-in. Each event spans the
    exact time from the first word's start to the third word's end for that trio.
    """
    lines: List[str] = []
    lines.append("[Script Info]")
    lines.append(f"PlayResX: {width}")
    lines.append(f"PlayResY: {height}")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("WrapStyle: 2")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    lines.append("Style: CenterBig, Arial, 64, &H00FFFFFF, &H000000FF, &H00000000, &H64000000, -1, 0, 0, 0, 100, 100, 0, 0, 1, 5, 2, 2, 30, 30, 100, 1")
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    def fmt_ass(t: float) -> str:
        # ASS uses h:mm:ss.cs (centiseconds)
        srt = _format_srt_timestamp(max(0.0, t))
        hms, ms = srt.split(",")
        cs = int(round(int(ms) / 10))
        return f"{hms}.{cs:02d}"

    y = int(round(height * 0.68))
    x = int(round(width / 2))

    total_events = 0
    for seg in segments:
        words = _extract_words_from_segment(seg)
        if not words:
            continue
        # Build groups of 3 words
        i = 0
        while i < len(words):
            trio = words[i:i+3]
            if not trio:
                break
            trio_start = trio[0][1]
            trio_end = trio[-1][2]
            if trio_end <= trio_start:
                i += 3
                continue
            # Karaoke timing: allocate k values by actual word durations in centiseconds
            k_parts: List[str] = []
            display_words: List[str] = []
            for (txt, ws, we) in trio:
                dur_cs = max(1, int(round((we - ws) * 100)))
                k_parts.append(f"{{\\k{dur_cs}}}")
                display_words.append(txt)
            text = " ".join(a + b for a, b in zip(k_parts, display_words))
            ass_text = f"{{\\pos({x},{y})\\fad(80,80)}}{text}"
            lines.append(
                f"Dialogue: 0,{fmt_ass(trio_start)},{fmt_ass(trio_end)},CenterBig,,0000,0000,0000,,{ass_text}"
            )
            total_events += 1
            i += 3

    data = "\n".join(lines).strip() + ("\n" if lines else "")
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig") as f:
        f.write(data)
    return total_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a slideshow video from images + bg audio using FFmpeg.")
    parser.add_argument("--images_dir", required=True, help="Directory containing images (sorted by filename).")
    parser.add_argument("--audio", required=True, help="Background audio file (e.g., mp3).")
    parser.add_argument("--out", required=True, help="Output MP4 path.")
    parser.add_argument("--subs", default=None, help="Optional path to SRT/ASS subtitles to burn in.")
    parser.add_argument("--w", type=int, default=1920, help="Output width (default: 1920).")
    parser.add_argument("--h", type=int, default=1080, help="Output height (default: 1080).")
    parser.add_argument("--shorts_preset", action="store_true", help="Use 9:16 vertical (1080x1920) for YouTube Shorts.")
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
    parser.add_argument("--auto_subs", action="store_true", help="Auto-generate subtitles from audio with faster-whisper.")
    parser.add_argument(
        "--subs_style",
        default="srt",
        choices=["srt", "ass_2word", "ass_3word_sync"],
        help=(
            "Auto subtitles style: 'srt' (default), 'ass_2word' (two-word animated), or "
            "'ass_3word_sync' (three words at a time using exact word timings)."
        ),
    )
    parser.add_argument("--subs_lang", default="en", help="Subtitle language (use 'auto' to detect).")
    parser.add_argument("--whisper_model", default="small", help="Whisper model size (tiny/base/small/medium/large-v3).")
    parser.add_argument("--ffmpeg_bin", default="ffmpeg", help="Path to ffmpeg binary (optional).")
    parser.add_argument("--ffprobe_bin", default="ffprobe", help="Path to ffprobe binary (optional).")
    # Visual effects
    parser.add_argument("--kenburns", action="store_true", help="Enable gentle pan/zoom on still images.")
    parser.add_argument(
        "--kenburns_direction",
        default="auto",
        choices=["auto", "zoom_in", "zoom_out", "pan_lr", "pan_rl", "pan_tb", "pan_bt"],
        help="Ken Burns style (default: auto cycle).",
    )
    parser.add_argument("--kenburns_strength", type=float, default=1.1, help="Max zoom factor (e.g., 1.1 = 10%).")

    args = parser.parse_args()
    # Shorts preset overrides if not explicitly changed
    if args.shorts_preset:
        if args.w == 1920 and args.h == 1080:
            args.w, args.h = 1080, 1920

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
        want_word_ts = args.subs_style == "ass_3word_sync"
        segments, _info = model.transcribe(
            args.audio,
            language=(None if args.subs_lang == "auto" else args.subs_lang),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=want_word_ts,
        )
        seg_list = list(segments)
        if args.subs_style == "ass_2word":
            auto_path = os.path.join("captions", "auto_two_word.ass")
            count = _write_ass_two_word_from_segments(seg_list, auto_path, args.w, args.h)
            if count == 0:
                raise SystemExit("No speech detected; cannot build subtitles.")
            args.subs = auto_path
        elif args.subs_style == "ass_3word_sync":
            auto_path = os.path.join("captions", "auto_three_word.ass")
            count = _write_ass_three_word_from_segments(seg_list, auto_path, args.w, args.h)
            if count == 0:
                raise SystemExit("No speech detected; cannot build subtitles.")
            args.subs = auto_path
        else:
            auto_path = os.path.join("captions", "auto.srt")
            count = _write_srt_from_segments(seg_list, auto_path)
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
        kenburns=args.kenburns,
        kenburns_direction=args.kenburns_direction,
        kenburns_strength=args.kenburns_strength,
    )

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        # Surface the exact ffmpeg command for easier debugging
        printable = " ".join(cmd)
        raise SystemExit(f"ffmpeg failed with exit code {exc.returncode}.\nCommand:\n{printable}") from exc


if __name__ == "__main__":
    main()

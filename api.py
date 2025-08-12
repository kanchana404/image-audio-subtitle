#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field

# Import helpers from our slideshow builder
from scripts.make_video import (
    build_ffmpeg_command,
    get_media_duration_seconds,
    _write_srt_from_segments,
)

app = FastAPI(title="Slideshow Render API", version="1.0")


class RenderRequest(BaseModel):
    image_urls: List[HttpUrl] = Field(..., description="Array of image URLs in desired order")
    audio_url: HttpUrl = Field(..., description="Audio file URL (any common format; will be converted to mp3)")

    # Video options
    w: int = 1920
    h: int = 1080
    slide: float = 4.0
    xfade: float = 0.75
    fps: int = 30
    crf: int = 19
    preset: str = "medium"
    threads: int = 4
    fadein: float = 0.5
    fadeout: float = 0.5

    # Behavior flags
    fit_audio: bool = Field(False, description="Adjust slide duration to match total audio length")
    auto_subs: bool = Field(False, description="Generate subtitles from audio using faster-whisper and burn them")
    subs_lang: str = Field("en", description="Subtitle language code, or 'auto' for auto-detect")
    whisper_model: str = Field("small", description="Whisper model size: tiny/base/small/medium/large-v3")
    subs_url: Optional[HttpUrl] = Field(None, description="Optional URL to an existing SRT/ASS subtitles file to burn")
    subs_style: str = Field(
        "srt",
        description="When auto_subs is true, choose 'srt', 'ass_2word', or 'ass_3word_sync' (requires word_timestamps)",
    )

    # Presets
    shorts_preset: bool = Field(False, description="Use 9:16 1080x1920 vertical output suitable for Shorts")

    # Visual effects
    kenburns: bool = Field(False, description="Enable gentle pan/zoom on still images")
    kenburns_direction: str = Field(
        "auto",
        description="Ken Burns style: auto|zoom_in|zoom_out|pan_lr|pan_rl|pan_tb|pan_bt",
    )
    kenburns_strength: float = Field(1.1, description="Max zoom factor, e.g., 1.1 = 10%")

    # Optional explicit ffmpeg/ffprobe paths
    ffmpeg_bin: Optional[str] = None
    ffprobe_bin: Optional[str] = None


def ensure_bin(path: Optional[str], default_name: str) -> str:
    if path:
        return path
    return default_name


env_jobs_dir = os.path.join(os.getcwd(), "jobs")
os.makedirs(env_jobs_dir, exist_ok=True)
app.mount("/jobs", StaticFiles(directory=env_jobs_dir), name="jobs")


def _run_ffmpeg(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed (exit {exc.returncode}). Command: {' '.join(cmd)}")


def _download_to(path: str, url: str) -> None:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download: {url}") from exc


@app.post("/render")
def render(req: RenderRequest, request: Request):
    if len(req.image_urls) == 0:
        raise HTTPException(status_code=400, detail="image_urls cannot be empty")

    ffmpeg_bin = ensure_bin(req.ffmpeg_bin, "ffmpeg")
    ffprobe_bin = ensure_bin(req.ffprobe_bin, "ffprobe")

    # Prepare a working directory per job
    job_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_root = os.path.join(env_jobs_dir, job_id)
    images_dir = os.path.join(job_root, "assets", "images")
    captions_dir = os.path.join(job_root, "captions")
    out_dir = os.path.join(job_root, "out")
    tmp_dir = os.path.join(job_root, "tmp")

    for d in (images_dir, captions_dir, out_dir, tmp_dir):
        os.makedirs(d, exist_ok=True)

    # Download audio
    src_audio = os.path.join(tmp_dir, "audio_src")
    _download_to(src_audio, str(req.audio_url))

    # Convert audio to mp3 named bg.mp3
    audio_mp3 = os.path.join(job_root, "assets", "audio", "bg.mp3")
    os.makedirs(os.path.dirname(audio_mp3), exist_ok=True)
    _run_ffmpeg([
        ffmpeg_bin,
        "-y",
        "-i", src_audio,
        "-vn",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "192k",
        "-c:a", "libmp3lame",
        audio_mp3,
    ])

    # Download images and convert/rename to 1.png, 2.png, ...
    image_paths: List[str] = []
    for idx, url in enumerate(req.image_urls, start=1):
        tmp_path = os.path.join(tmp_dir, f"img_{idx}")
        _download_to(tmp_path, str(url))
        out_png = os.path.join(images_dir, f"{idx}.png")
        _run_ffmpeg([ffmpeg_bin, "-y", "-i", tmp_path, out_png])
        image_paths.append(out_png)

    # Optionally auto-generate subtitles
    subs_path: Optional[str] = None
    if req.subs_url:
        # Download provided subtitles file (expects .srt or .ass)
        ext = os.path.splitext(str(req.subs_url))[-1].lower() or ".srt"
        if ext not in (".srt", ".ass"):
            ext = ".srt"
        subs_path = os.path.join(captions_dir, f"external{ext}")
        _download_to(subs_path, str(req.subs_url))
    elif req.auto_subs:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Auto subtitles require faster-whisper. Install it on the server.")
        model = WhisperModel(req.whisper_model, device="cpu", compute_type="int8")
        want_word_ts = getattr(req, "subs_style", "srt") == "ass_3word_sync"
        segments, _info = model.transcribe(
            audio_mp3,
            language=(None if req.subs_lang == "auto" else req.subs_lang),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=want_word_ts,
        )
        seg_list = list(segments)
        style = getattr(req, "subs_style", "srt")
        if style == "ass_2word":
            # Write ASS two-word animated
            from scripts.make_video import _write_ass_two_word_from_segments
            subs_path = os.path.join(captions_dir, "auto_two_word.ass")
            count = _write_ass_two_word_from_segments(seg_list, subs_path, req.w, req.h)
        elif style == "ass_3word_sync":
            from scripts.make_video import _write_ass_three_word_from_segments
            subs_path = os.path.join(captions_dir, "auto_three_word.ass")
            count = _write_ass_three_word_from_segments(seg_list, subs_path, req.w, req.h)
        else:
            # Write SRT
            subs_path = os.path.join(captions_dir, "auto.srt")
            count = _write_srt_from_segments(seg_list, subs_path)
        if count == 0:
            subs_path = None

    # If requested, fit slides to audio length
    slide = req.slide
    xfade = req.xfade
    if req.fit_audio:
        audio_dur = get_media_duration_seconds(audio_mp3, ffprobe_bin=ffprobe_bin)
        n = len(image_paths)
        desired_slide = (audio_dur + (n - 1) * xfade) / max(1, n)
        if xfade >= desired_slide:
            xfade = max(0.0, desired_slide - 0.05)
        slide = max(0.05, desired_slide)
    if xfade >= slide:
        xfade = max(0.0, slide - 0.05)

    # Shorts-friendly override
    if req.shorts_preset and req.w == 1920 and req.h == 1080:
        req.w, req.h = 1080, 1920

    # Build ffmpeg command and render
    out_path = os.path.join(out_dir, "video.mp4")
    cmd = build_ffmpeg_command(
        image_paths=image_paths,
        audio_path=audio_mp3,
        subs_path=subs_path,
        out_path=out_path,
        width=req.w,
        height=req.h,
        slide_dur=slide,
        xfade=xfade,
        fps=req.fps,
        crf=req.crf,
        preset=req.preset,
        threads=req.threads,
        fadein=req.fadein,
        fadeout=req.fadeout,
        ffmpeg_bin=ffmpeg_bin,
        kenburns=req.kenburns,
        kenburns_direction=req.kenburns_direction,
        kenburns_strength=req.kenburns_strength,
    )

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Render failed (exit {exc.returncode}). Command: {' '.join(cmd)}")

    # Build a public download URL served via StaticFiles
    base_url = str(request.base_url).rstrip("/")
    public_path = f"/jobs/{job_id}/out/video.mp4"
    download_url = f"{base_url}{public_path}"

    return JSONResponse({
        "status": "ok",
        "job_id": job_id,
        "download_url": download_url
    })



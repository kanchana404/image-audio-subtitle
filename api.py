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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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

    # Optional explicit ffmpeg/ffprobe paths
    ffmpeg_bin: Optional[str] = None
    ffprobe_bin: Optional[str] = None


def ensure_bin(path: Optional[str], default_name: str) -> str:
    if path:
        return path
    return default_name


env_jobs_dir = os.path.join(os.getcwd(), "jobs")
os.makedirs(env_jobs_dir, exist_ok=True)


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
def render(req: RenderRequest):
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
    if req.auto_subs:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Auto subtitles require faster-whisper. Install it on the server.")
        model = WhisperModel(req.whisper_model, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(
            audio_mp3,
            language=(None if req.subs_lang == "auto" else req.subs_lang),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        # Write SRT
        subs_path = os.path.join(captions_dir, "auto.srt")
        count = _write_srt_from_segments(list(segments), subs_path)
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
    )

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Render failed (exit {exc.returncode}). Command: {' '.join(cmd)}")

    # Return the video file
    filename = f"render_{job_id}.mp4"
    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=filename,
    )



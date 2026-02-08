#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


# =========================
# CONFIG (all tunables here)
# =========================
DEFAULT_CHUNK_DIR = "videos/chunk-000"
DEFAULT_VIDEO_RETARGET_DIR = "video_retarget"
DEFAULT_FRAME_RETARGET_DIR = "frame_retarget"

DEFAULT_STRIDE = 1
DEFAULT_HZ = 10.0

EPISODE_PREFIX = "episode_"
EPISODE_EXT = ".mp4"

SAMPLE_PAD = 6
FRAME_PAD = 6

# H.264 encoding config
H264_CRF = 18              # lower = higher quality, larger file. 18~23 typical
H264_PRESET = "veryfast"   # ultrafast/superfast/veryfast/faster/fast/medium/slow...
H264_PIX_FMT = "yuv420p"   # most compatible
H264_AUDIO = False         # most datasets have no audio; keep False to avoid surprises


def _run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def has_ffmpeg() -> bool:
    code, _ = _run(["ffmpeg", "-version"])
    return code == 0


def list_episode_videos(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    vids = sorted(input_dir.glob(f"{EPISODE_PREFIX}*{EPISODE_EXT}"))
    return [p for p in vids if p.is_file()]


def parse_episode_index(p: Path) -> int:
    name = p.name
    if not (name.startswith(EPISODE_PREFIX) and name.endswith(EPISODE_EXT)):
        raise ValueError(f"Unexpected filename: {name}")
    mid = name[len(EPISODE_PREFIX):-len(EPISODE_EXT)]
    return int(mid)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def transcode_to_h264(src: Path, dst: Path, overwrite: bool) -> None:
    """
    Transcode src video to H.264 MP4 at dst.
    """
    ensure_dir(dst.parent)

    if dst.exists() and not overwrite:
        return

    # -y to overwrite, else ffmpeg will prompt and hang
    overwrite_flag = "-y" if overwrite else "-n"

    cmd = [
        "ffmpeg",
        overwrite_flag,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", H264_PRESET,
        "-crf", str(H264_CRF),
        "-pix_fmt", H264_PIX_FMT,
    ]

    if not H264_AUDIO:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k"]

    # Ensure MP4 is streamable and more compatible
    cmd += ["-movflags", "+faststart", str(dst)]

    code, out = _run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg transcode failed:\nsrc={src}\ndst={dst}\n{out}")


def extract_frames_ffmpeg(video_path: Path, out_dir: Path, stride: int, overwrite: bool) -> None:
    ensure_dir(out_dir)
    if overwrite:
        # remove old jpgs only
        for f in out_dir.glob("*.jpg"):
            f.unlink()

    tmp_pattern = str(out_dir / f"__tmp_%0{FRAME_PAD}d.jpg")
    select_expr = f"not(mod(n\\,{stride}))"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"select='{select_expr}'",
        "-vsync", "vfr",
        "-q:v", "2",
        tmp_pattern,
    ]
    code, out = _run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg frame extract failed on {video_path}\n{out}")

    tmp_files = sorted(out_dir.glob("__tmp_*.jpg"))
    for i, f in enumerate(tmp_files):
        new_name = f"{i:0{FRAME_PAD}d}.jpg"
        target = out_dir / new_name
        if target.exists():
            target.unlink()
        f.rename(target)


def extract_frames_opencv(video_path: Path, out_dir: Path, stride: int, overwrite: bool) -> None:
    ensure_dir(out_dir)
    if overwrite:
        for f in out_dir.glob("*.jpg"):
            f.unlink()

    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "OpenCV not installed (and ffmpeg not used for frames).\n"
            "Install opencv-python:\n  pip install opencv-python\n"
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            out_path = out_dir / f"{saved:0{FRAME_PAD}d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root path")
    ap.add_argument("--camera_view", required=True, help="Camera view subdir under videos/chunk-000/")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Extract 1 frame every N frames")
    ap.add_argument("--Hz", type=float, default=DEFAULT_HZ, help="Nominal video Hz (default 10). Kept for compatibility.")
    ap.add_argument("--overwrite_video", action="store_true", help="Overwrite existing retargeted H.264 video")
    ap.add_argument("--overwrite_frames", action="store_true", help="Overwrite existing extracted frames")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    in_dir = root / DEFAULT_CHUNK_DIR / args.camera_view

    if not has_ffmpeg():
        raise RuntimeError("ffmpeg is required for H.264 transcoding but not found in PATH.")

    vids = list_episode_videos(in_dir)
    if not vids:
        raise RuntimeError(f"No videos found under: {in_dir}")

    out_video_root = root / DEFAULT_VIDEO_RETARGET_DIR
    out_frame_root = root / DEFAULT_FRAME_RETARGET_DIR
    ensure_dir(out_video_root)
    ensure_dir(out_frame_root)

    print(f"[INFO] root={root}")
    print(f"[INFO] input_dir={in_dir}")
    print(f"[INFO] videos={len(vids)}")
    print(f"[INFO] stride={args.stride}  Hz={args.Hz}")
    print(f"[INFO] video_retarget={out_video_root}")
    print(f"[INFO] frame_retarget={out_frame_root}")
    print(f"[INFO] transcode=H.264 libx264 preset={H264_PRESET} crf={H264_CRF} pix_fmt={H264_PIX_FMT}")

    # For frames: use ffmpeg (already available); OpenCV fallback kept but unlikely needed.
    use_ffmpeg_for_frames = True

    for vp in vids:
        ep = parse_episode_index(vp)
        sample_name = f"sample_{ep:0{SAMPLE_PAD}d}"

        # 1) transcode video to H.264
        sample_video_dir = out_video_root / sample_name
        dst_video = sample_video_dir / f"Frame_{ep:0{FRAME_PAD}d}{EPISODE_EXT}"
        transcode_to_h264(vp, dst_video, overwrite=args.overwrite_video)

        # 2) extract frames (keep as before)
        sample_frame_dir = out_frame_root / sample_name
        # If already has jpgs and not overwriting, skip
        if sample_frame_dir.exists() and any(sample_frame_dir.glob("*.jpg")) and not args.overwrite_frames:
            print(f"[SKIP] frames exist: {sample_frame_dir}")
        else:
            if args.overwrite_frames and sample_frame_dir.exists():
                # keep folder, clear inside inside extractor
                pass
            if use_ffmpeg_for_frames:
                extract_frames_ffmpeg(vp, sample_frame_dir, args.stride, overwrite=args.overwrite_frames)
            else:
                extract_frames_opencv(vp, sample_frame_dir, args.stride, overwrite=args.overwrite_frames)

        print(f"[OK] {sample_name}: h264 -> {dst_video.name}, frames -> {sample_frame_dir}")

    print("[DONE]")


if __name__ == "__main__":
    main()

'''
python retarget_videos_and_frames.py \
  --root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz \
  --camera_view observation.images.head_realsense_color \
  --stride 100
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple


# =========================
# CONFIG (all tunables here)
# =========================
DEFAULT_CHUNK_DIR = "videos/chunk-000"

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
    """Transcode src video to H.264 MP4 at dst."""
    ensure_dir(dst.parent)
    if dst.exists() and not overwrite:
        return

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

    cmd += ["-movflags", "+faststart", str(dst)]

    code, out = _run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg transcode failed:\nsrc={src}\ndst={dst}\n{out}")


def extract_frames_ffmpeg(video_path: Path, out_dir: Path, stride: int, overwrite: bool) -> None:
    ensure_dir(out_dir)
    if overwrite:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root path")
    ap.add_argument("--camera_view", required=True, help="Camera view subdir under videos/chunk-000/")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Extract 1 frame every N frames")
    ap.add_argument("--Hz", type=float, default=DEFAULT_HZ, help="Nominal video Hz (default 10). Kept for compatibility.")
    ap.add_argument("--overwrite_video", action="store_true", help="Overwrite existing retargeted H.264 video")
    ap.add_argument("--overwrite_frames", action="store_true", help="Overwrite existing extracted frames")
    ap.add_argument(
        "--out_tag",
        default="right",
        help="Suffix tag for output dirs, e.g. right/head/left. "
            "Outputs: video_retarget_{tag}, frame_retarget_{tag}",
    )
    # NEW: control switches
    ap.add_argument("--do_video", action="store_true", help="Do H.264 transcoding")
    ap.add_argument("--do_frames", action="store_true", help="Do frame extraction")
    ap.add_argument(
        "--use_video_for_frames",
        action="store_true",
        help="If enabled, extract frames from retargeted H.264 video (when available) instead of original.",
    )

    args = ap.parse_args()

    # Default behavior: if neither flag set, do both (backward compatible)
    do_video = args.do_video or (not args.do_video and not args.do_frames)
    do_frames = args.do_frames or (not args.do_video and not args.do_frames)

    root = Path(args.root).resolve()
    in_dir = root / DEFAULT_CHUNK_DIR / args.camera_view

    if (do_video or do_frames) and not has_ffmpeg():
        raise RuntimeError("ffmpeg is required but not found in PATH.")

    vids = list_episode_videos(in_dir)
    if not vids:
        raise RuntimeError(f"No videos found under: {in_dir}")

    out_video_root = root / f"video_retarget_{args.out_tag}"
    out_frame_root = root / f"frame_retarget_{args.out_tag}"
    if do_video:
        ensure_dir(out_video_root)
    if do_frames:
        ensure_dir(out_frame_root)

    print(f"[INFO] root={root}")
    print(f"[INFO] input_dir={in_dir}")
    print(f"[INFO] videos={len(vids)}")
    print(f"[INFO] stride={args.stride}  Hz={args.Hz}")
    print(f"[INFO] do_video={do_video}  do_frames={do_frames}  use_video_for_frames={args.use_video_for_frames}")
    if do_video:
        print(f"[INFO] video_retarget={out_video_root}")
        print(f"[INFO] transcode=H.264 libx264 preset={H264_PRESET} crf={H264_CRF} pix_fmt={H264_PIX_FMT}")
    if do_frames:
        print(f"[INFO] frame_retarget={out_frame_root}")

    for vp in vids:
        ep = parse_episode_index(vp)
        sample_name = f"sample_{ep:0{SAMPLE_PAD}d}"

        # Paths
        sample_video_dir = out_video_root / sample_name
        dst_video = sample_video_dir / f"Frame_{ep:0{FRAME_PAD}d}{EPISODE_EXT}"
        sample_frame_dir = out_frame_root / sample_name

        # 1) video
        if do_video:
            transcode_to_h264(vp, dst_video, overwrite=args.overwrite_video)

        # 2) frames
        if do_frames:
            # If already has jpgs and not overwriting, skip
            if sample_frame_dir.exists() and any(sample_frame_dir.glob("*.jpg")) and not args.overwrite_frames:
                print(f"[SKIP] frames exist: {sample_frame_dir}")
            else:
                # choose source for frame extraction
                frame_src = dst_video if (args.use_video_for_frames and dst_video.exists()) else vp
                extract_frames_ffmpeg(frame_src, sample_frame_dir, args.stride, overwrite=args.overwrite_frames)

        # log
        msg = f"[OK] {sample_name}:"
        if do_video:
            msg += f" h264 -> {dst_video.name};"
        if do_frames:
            msg += f" frames -> {sample_frame_dir}"
        print(msg)

    print("[DONE]")


if __name__ == "__main__":
    main()

"""
Examples:

observation.images.head_realsense_color
observation.images.left_hand_realsense_color
observation.images.right_hand_realsense_color

# default (backward compatible): do both video + frames
python retarget_videos_and_frames.py \
  --root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old \
  --camera_view observation.images.right_hand_realsense_color \
  --stride 100

# only transcode video
python retarget_videos_and_frames.py \
  --root /path/to/ds \
  --camera_view observation.images.right_hand_realsense_color \
  --do_video

# only extract frames (from original video)
python retarget_videos_and_frames.py \
  --root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old \
  --camera_view observation.images.head_realsense_color \
  --do_frames \
  --stride 1 \
  --out_tag head

# extract frames from retargeted H.264 if exists
python retarget_videos_and_frames.py \
  --root /path/to/ds \
  --camera_view observation.images.right_hand_realsense_color \
  --do_video --do_frames \
  --use_video_for_frames \
  --stride 100
"""

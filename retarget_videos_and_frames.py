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


def find_datasets_batch(batch_root: Path, pattern: str) -> List[Path]:
    """Find all matching dataset dirs (assumed to be {DATASET_NAME}/{DATASET_NAME}{pattern})"""
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch root not found: {batch_root}")
    
    # Find all dirs matching pattern at depth 2
    matches = []
    for item in batch_root.iterdir():
        if item.is_dir():
            for subitem in item.iterdir():
                if subitem.is_dir() and subitem.name.endswith(pattern):
                    matches.append(subitem)
    return sorted(matches)


def process_single_dataset(
    root: Path,
    camera_view: str,
    stride: int,
    out_tag: str,
    do_video: bool,
    do_frames: bool,
    overwrite_video: bool,
    overwrite_frames: bool,
    use_video_for_frames: bool,
) -> None:
    """Process a single dataset"""
    in_dir = root / DEFAULT_CHUNK_DIR / camera_view

    vids = list_episode_videos(in_dir)
    if not vids:
        print(f"[WARN] No videos found under: {in_dir}")
        return

    out_video_root = root / f"video_retarget_{out_tag}"
    out_frame_root = root / f"frame_retarget_{out_tag}"
    if do_video:
        ensure_dir(out_video_root)
    if do_frames:
        ensure_dir(out_frame_root)

    for vp in vids:
        ep = parse_episode_index(vp)
        sample_name = f"sample_{ep:0{SAMPLE_PAD}d}"

        # Paths
        sample_video_dir = out_video_root / sample_name
        dst_video = sample_video_dir / f"Frame_{ep:0{FRAME_PAD}d}{EPISODE_EXT}"
        sample_frame_dir = out_frame_root / sample_name

        # 1) video
        if do_video:
            transcode_to_h264(vp, dst_video, overwrite=overwrite_video)

        # 2) frames
        if do_frames:
            # If already has jpgs and not overwriting, skip
            if sample_frame_dir.exists() and any(sample_frame_dir.glob("*.jpg")) and not overwrite_frames:
                print(f"[SKIP] frames exist: {sample_frame_dir}")
            else:
                # choose source for frame extraction
                frame_src = dst_video if (use_video_for_frames and dst_video.exists()) else vp
                extract_frames_ffmpeg(frame_src, sample_frame_dir, stride, overwrite=overwrite_frames)

        # log
        msg = f"[OK] {sample_name}:"
        if do_video:
            msg += f" h264 -> {dst_video.name};"
        if do_frames:
            msg += f" frames -> {sample_frame_dir}"
        print(msg)


def main():
    ap = argparse.ArgumentParser()
    
    # Single dataset mode (original)
    ap.add_argument("--root", help="Dataset root path (single dataset mode)")
    
    # Batch mode
    ap.add_argument(
        "--batch-root",
        help="Batch processing root (e.g. /path/to/datasets). Finds all dirs matching --pattern",
    )
    ap.add_argument(
        "--pattern",
        default="_old",
        help="Pattern to match dataset dirs in batch mode (default: '_old')",
    )
    ap.add_argument(
        "--cameras",
        nargs="+",
        default=[
            "observation.images.head_realsense_color",
            "observation.images.left_hand_realsense_color",
            "observation.images.right_hand_realsense_color",
        ],
        help="Camera views to process (space-separated, default: all 3)",
    )
    
    # Common args
    ap.add_argument("--camera_view", help="Camera view subdir under videos/chunk-000/ (single mode)")
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

    # Validate mode
    batch_mode = args.batch_root is not None
    single_mode = args.root is not None
    
    if batch_mode and single_mode:
        raise ValueError("Cannot specify both --root and --batch-root. Choose one mode.")
    if not batch_mode and not single_mode:
        raise ValueError("Must specify either --root (single mode) or --batch-root (batch mode)")
    
    if single_mode and not args.camera_view:
        raise ValueError("--camera_view required in single mode")

    # Default behavior: if neither flag set, do both (backward compatible)
    do_video = args.do_video or (not args.do_video and not args.do_frames)
    do_frames = args.do_frames or (not args.do_video and not args.do_frames)

    if (do_video or do_frames) and not has_ffmpeg():
        raise RuntimeError("ffmpeg is required but not found in PATH.")

    # ==================== BATCH MODE ====================
    if batch_mode:
        batch_root = Path(args.batch_root).resolve()
        datasets = find_datasets_batch(batch_root, args.pattern)
        
        if not datasets:
            raise RuntimeError(f"No datasets found matching pattern '{args.pattern}' under {batch_root}")
        
        print(f"[BATCH] Found {len(datasets)} dataset(s):")
        for ds in datasets:
            print(f"  - {ds}")
        print()
        
        total_ok = 0
        total_fail = 0
        for dataset_root in datasets:
            print(f"\n{'='*70}")
            print(f"[BATCH] Processing: {dataset_root.name}")
            print(f"[BATCH] Root: {dataset_root}")
            print(f"{'='*70}")
            
            for camera in args.cameras:
                print(f"\n[BATCH] Camera: {camera}")
                try:
                    process_single_dataset(
                        root=dataset_root,
                        camera_view=camera,
                        stride=args.stride,
                        out_tag=args.out_tag,
                        do_video=do_video,
                        do_frames=do_frames,
                        overwrite_video=args.overwrite_video,
                        overwrite_frames=args.overwrite_frames,
                        use_video_for_frames=args.use_video_for_frames,
                    )
                    total_ok += 1
                except Exception as e:
                    print(f"[ERROR] Failed to process {dataset_root.name} / {camera}")
                    print(f"[ERROR] {e}")
                    total_fail += 1
        
        print(f"\n{'='*70}")
        print(f"[BATCH] Summary: {total_ok} OK, {total_fail} FAILED")
        print(f"{'='*70}")
        return

    # ==================== SINGLE MODE ====================
    root = Path(args.root).resolve()
    in_dir = root / DEFAULT_CHUNK_DIR / args.camera_view

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

    process_single_dataset(
        root=root,
        camera_view=args.camera_view,
        stride=args.stride,
        out_tag=args.out_tag,
        do_video=do_video,
        do_frames=do_frames,
        overwrite_video=args.overwrite_video,
        overwrite_frames=args.overwrite_frames,
        use_video_for_frames=args.use_video_for_frames,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()

"""
Examples:

Camera views:
  - observation.images.head_realsense_color
  - observation.images.left_hand_realsense_color
  - observation.images.right_hand_realsense_color

============ SINGLE DATASET MODE ============

# Default (backward compatible): do both video + frames
python retarget_videos_and_frames.py \
  --root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot/Teleop_251022_GrapeCleanbgWaist_Anonymous_10Hz/Teleop_251022_GrapeCleanbgWaist_Anonymous_10Hz_old \
  --camera_view observation.images.head_realsense_color \
  --stride 10 \
  --use_video_for_frames \
  --out_tag head

# Only transcode video
python retarget_videos_and_frames.py \
  --root /path/to/ds \
  --camera_view observation.images.right_hand_realsense_color \
  --do_video

# Only extract frames (from original video)
python retarget_videos_and_frames.py \
  --root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old \
  --camera_view observation.images.head_realsense_color \
  --do_frames \
  --stride 1 \
  --out_tag head

# Extract frames from retargeted H.264 if exists
python retarget_videos_and_frames.py \
  --root /path/to/ds \
  --camera_view observation.images.right_hand_realsense_color \
  --do_video --do_frames \
  --use_video_for_frames \
  --stride 100

============ BATCH MODE (NEW!) ============

# Batch process all datasets ending with "_old" (all 3 cameras)
# Processes all matching dirs under /path/to/batch/root
python retarget_videos_and_frames.py \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot \
  --pattern _old \
  --stride 10 \
  --use_video_for_frames \
  --out_tag head

# Batch with custom pattern
python retarget_videos_and_frames.py \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot \
  --pattern _new \
  --do_frames \
  --stride 1 \
  --out_tag head

# Batch with specific cameras only
python retarget_videos_and_frames.py \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot \
  --pattern _old \
  --cameras observation.images.head_realsense_color observation.images.right_hand_realsense_color \
  --stride 10 \
  --out_tag multi_camera

# Batch: only transcode video (faster)
python retarget_videos_and_frames.py \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot \
  --pattern _old \
  --do_video \
  --out_tag head

Batch mode will:
  1. Auto-discover all matching dataset dirs
  2. Process each camera view in each dataset
  3. Catch and report errors without stopping
  4. Print summary at the end
"""

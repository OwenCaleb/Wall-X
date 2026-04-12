#!/bin/bash
# -*- coding: utf-8 -*-

# 批量处理所有数据集的便捷脚本
# 用法: bash batch_retarget.sh [OPTIONS]

set -euo pipefail

# ============ CONFIG (可修改) ============
BATCH_ROOT="/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot"
PATTERN="_old"
STRIDE=10
OUT_TAG="head"
USE_VIDEO_FOR_FRAMES=true

# 处理模式: both (默认), video_only, frames_only
MODE="both"

# 摄像头视图 (默认全部3个)
CAMERAS=(
  "observation.images.head_realsense_color"
  "observation.images.left_hand_realsense_color"
  "observation.images.right_hand_realsense_color"
)

# ========== 解析命令行参数 ==========
while [[ $# -gt 0 ]]; do
  case $1 in
    --batch-root)
      BATCH_ROOT="$2"
      shift 2
      ;;
    --pattern)
      PATTERN="$2"
      shift 2
      ;;
    --stride)
      STRIDE="$2"
      shift 2
      ;;
    --out-tag)
      OUT_TAG="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"  # both, video_only, frames_only
      shift 2
      ;;
    --no-use-video-for-frames)
      USE_VIDEO_FOR_FRAMES=false
      shift
      ;;
    --camera)
      # 清除默认摄像头，添加指定摄像头
      CAMERAS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        CAMERAS+=("$1")
        shift
      done
      ;;
    --help)
      cat <<EOF
批量处理数据集脚本

用法:
  bash batch_retarget.sh [OPTIONS]

参数:
  --batch-root PATH          批处理根目录 (默认: $BATCH_ROOT)
  --pattern PATTERN          匹配数据集的pattern (默认: $PATTERN)
  --stride N                 帧提取步长 (默认: $STRIDE)
  --out-tag TAG              输出目录后缀 (默认: $OUT_TAG)
  --mode MODE                处理模式: both, video_only, frames_only (默认: $MODE)
  --no-use-video-for-frames  不使用转码视频来提取帧（用原始视频）
  --camera CAM1 CAM2 ...     指定要处理的摄像头视图 (默认: 全部3个)
  --help                     显示本帮助信息

示例:
  # 默认配置：处理所有_old数据集，所有摄像头，stride=10
  bash batch_retarget.sh

  # 仅处理视频转码（更快）
  bash batch_retarget.sh --mode video_only

  # 仅处理head摄像头
  bash batch_retarget.sh --camera observation.images.head_realsense_color

  # 自定义batch根目录和pattern
  bash batch_retarget.sh --batch-root /path/to/datasets --pattern _new

  # 组合：处理frames，stride=1，只有head和right
  bash batch_retarget.sh \\
    --mode frames_only \\
    --stride 1 \\
    --camera observation.images.head_realsense_color observation.images.right_hand_realsense_color
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ========== 构建Python命令 ==========
PYTHON_CMD="python retarget_videos_and_frames.py"
PYTHON_CMD="$PYTHON_CMD --batch-root '$BATCH_ROOT'"
PYTHON_CMD="$PYTHON_CMD --pattern '$PATTERN'"
PYTHON_CMD="$PYTHON_CMD --stride $STRIDE"
PYTHON_CMD="$PYTHON_CMD --out_tag '$OUT_TAG'"

# 添加摄像头参数
CAMERAS_STR="${CAMERAS[*]}"
PYTHON_CMD="$PYTHON_CMD --cameras $CAMERAS_STR"

# 添加模式参数
case $MODE in
  video_only)
    PYTHON_CMD="$PYTHON_CMD --do_video"
    ;;
  frames_only)
    PYTHON_CMD="$PYTHON_CMD --do_frames"
    ;;
  both)
    PYTHON_CMD="$PYTHON_CMD --do_video --do_frames"
    ;;
  *)
    echo "Invalid mode: $MODE"
    exit 1
    ;;
esac

# 添加use_video_for_frames参数
if [ "$USE_VIDEO_FOR_FRAMES" = true ]; then
  PYTHON_CMD="$PYTHON_CMD --use_video_for_frames"
fi

# ========== 显示配置 & 运行 ==========
echo "=========================================="
echo "批处理配置:"
echo "=========================================="
echo "批根目录:    $BATCH_ROOT"
echo "Pattern:    $PATTERN"
echo "Stride:     $STRIDE"
echo "Out Tag:    $OUT_TAG"
echo "模式:       $MODE"
echo "摄像头:     ${CAMERAS[*]}"
echo "使用转码视频提取帧: $USE_VIDEO_FOR_FRAMES"
echo ""
echo "执行命令:"
echo "  $PYTHON_CMD"
echo "=========================================="
echo ""

# 执行
eval "$PYTHON_CMD"

echo ""
echo "=========================================="
echo "✓ 批处理完成！"
echo "=========================================="

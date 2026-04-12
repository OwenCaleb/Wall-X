#!/bin/bash
# 快速预检查：显示会被处理的数据集和摄像头，不实际运行处理

BATCH_ROOT="${1:-/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot}"
PATTERN="${2:-_old}"

echo "================================"
echo "批处理预检查"
echo "================================"
echo "批根目录: $BATCH_ROOT"
echo "Pattern:  $PATTERN"
echo ""

if [ ! -d "$BATCH_ROOT" ]; then
  echo "❌ 错误：目录不存在：$BATCH_ROOT"
  exit 1
fi

echo "发现的数据集:"
echo "============"

DATASET_COUNT=0
TOTAL_DATASETS=0

for parent_dir in "$BATCH_ROOT"/*/; do
  parent_name=$(basename "$parent_dir")
  for dataset_dir in "$parent_dir"*/; do
    dataset_name=$(basename "$dataset_dir")
    if [[ "$dataset_name" == *"$PATTERN"* ]]; then
      TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
      echo ""
      echo "[数据集 #$TOTAL_DATASETS] $dataset_name"
      
      chunk_dir="$dataset_dir/videos/chunk-000"
      if [ -d "$chunk_dir" ]; then
        echo "  摄像头视图:"
        for camera_dir in "$chunk_dir"/*/; do
          camera_name=$(basename "$camera_dir")
          video_count=$(find "$camera_dir" -maxdepth 1 -name "episode_*.mp4" 2>/dev/null | wc -l)
          echo "    - $camera_name ($video_count个视频)"
        done
      else
        echo "  ⚠️  chunk-000不存在"
      fi
    fi
  done
done

echo ""
echo "================================"
echo "汇总: 共发现 $TOTAL_DATASETS 个数据集"
echo "================================"

if [ $TOTAL_DATASETS -eq 0 ]; then
  echo "❌ 没有找到匹配的数据集"
  exit 1
fi

echo ""
echo "要开始处理，运行:"
echo "  bash batch_retarget.sh"
echo ""
echo "或查看帮助:"
echo "  bash batch_retarget.sh --help"

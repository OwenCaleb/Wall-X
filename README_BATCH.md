# 批量处理系统 - 使用指南

## 📋 文件清单

你的 `/mnt/nas_ssd/workspace/wenboli/projects/Wall-X` 目录现在包含：

| 文件 | 说明 |
|------|------|
| **retarget_videos_and_frames.py** | 核心处理脚本（已升级支持批处理） |
| **batch_retarget.sh** | 便捷批处理脚本 ⭐ 推荐使用 |
| **check_batch_datasets.sh** | 预检查脚本（查看会处理哪些数据集） |
| **BATCH_RETARGET.md** | 完整说明文档 |

---

## 🚀 快速开始（3步）

### Step 1: 查看要处理的数据集
```bash
cd /mnt/nas_ssd/workspace/wenboli/projects/Wall-X
bash check_batch_datasets.sh
```

输出示例：
```
发现的数据集:
[数据集 #1] Teleop_251022_GrapeCleanbgWaist_Anonymous_10Hz_old
  摄像头视图:
    - observation.images.head_realsense_color (59个视频)
    - observation.images.left_hand_realsense_color (59个视频)
    - observation.images.right_hand_realsense_color (59个视频)
[数据集 #2] Teleop_251023_GrapeCleanbgWaist_Anonymous_10Hz_old
  ...
汇总: 共发现 15 个数据集
```

### Step 2: 开始批处理（最简单的方式）
```bash
bash batch_retarget.sh
```

这会自动处理所有15个数据集的所有3个摄像头，执行：
- 视频转码为H.264
- 帧提取（stride=10）

### Step 3: 监控进度

在另一个终端查看实时日志：
```bash
# 观察处理过程
tail -f logs/processing.log  # 如果有的话

# 或者查看生成的输出
ls -lh /path/to/dataset/video_retarget_head/
ls -lh /path/to/dataset/frame_retarget_head/
```

---

## 📊 对比：旧vs新工作流

### 旧方式（逐个处理）
```bash
# 需要运行15次，每次手动修改路径
python retarget_videos_and_frames.py \
  --root /path/to/dataset1_old \
  --camera_view observation.images.head_realsense_color \
  --stride 10 --use_video_for_frames --out_tag head

# dataset2, dataset3... (重复14次) ❌ 耗时、易错
```

### 新方式（批量处理）
```bash
# 一条命令处理所有数据集、所有摄像头 ✅
bash batch_retarget.sh
```

**时间节省**: 2-4小时变成一条命令，后台运行

---

## 🎯 常见场景

### 场景1：只需要head摄像头（不需要左右手）
```bash
bash batch_retarget.sh --camera observation.images.head_realsense_color
```
- 时间：原来的 1/3
- 磁盘用量：原来的 1/3

### 场景2：仅转码视频（先收集所有转码后的视频）
```bash
bash batch_retarget.sh --mode video_only
```
然后之后再单独提取帧：
```bash
bash batch_retarget.sh --mode frames_only
```

### 场景3：自定义stride（不同的帧间隔）
```bash
# stride=1: 每帧都提取（最详细）
bash batch_retarget.sh --stride 1

# stride=5: 每5帧提取1帧（中等）
bash batch_retarget.sh --stride 5

# stride=20: 每20帧提取1帧（稀疏）
bash batch_retarget.sh --stride 20
```

### 场景4：处理新数据（不同目录）
```bash
bash batch_retarget.sh \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g2/lerobot \
  --pattern _new
```

---

## 🔧 高级选项

```bash
# 查看所有可用选项
bash batch_retarget.sh --help
```

完整选项列表：
```
--batch-root PATH              批处理根目录
--pattern PATTERN              匹配数据集的pattern (默认: _old)
--stride N                     帧提取步长 (默认: 10)
--out-tag TAG                  输出目录后缀 (默认: head)
--mode MODE                    处理模式: both(默认), video_only, frames_only
--no-use-video-for-frames      使用原始视频提取帧（不用转码视频）
--camera CAM1 CAM2 ...         指定摄像头视图
--help                         显示帮助
```

---

## ✨ 新增功能说明

### 1. 自动发现数据集
```python
# Python脚本内部实现
find_datasets_batch(batch_root, pattern)
# 自动找到所有匹配 {pattern} 的目录
```

### 2. 多摄像头并行处理
```bash
# 对每个数据集的每个摄像头独立处理
# 逻辑: for dataset in datasets: for camera in cameras: process(dataset, camera)
```

### 3. 错误恢复
```
[BATCH] Processing: Dataset2
[ERROR] Failed to process Dataset2 / camera1
[BATCH] Processing: Dataset3  # 继续处理
...
[BATCH] Summary: 42 OK, 3 FAILED  # 最后汇总
```

### 4. 进度跟踪
- 每个数据集显示开始和完成
- 实时显示处理的视频数量和状态
- 最后显示总体统计

---

## 📈 性能估算

### 15个数据集，3个摄像头，平均60个视频/摄像头的情况：

| 模式 | 时间 | 输出 |
|------|------|------|
| 视频转码 (video_only) | ~2小时 | ~600 x H.264视频 (2-3TB) |
| 帧提取 (frames_only) | ~1小时 | ~600万 个jpg帧 |
| 完整处理 (both) | ~3小时 | 视频+帧 |
| 仅head摄像头 | ~1小时 | 1/3体量 |

💡 **建议**: 
- 先运行 `video_only` (后台2小时)
- 然后运行 `frames_only` (并行处理)
- 或同时开多个终端处理不同摄像头

---

## 🐛 故障排查

### 问题1：找不到数据集
```bash
# 检查路径和pattern是否正确
bash check_batch_datasets.sh --help

# 或直接指定路径
bash check_batch_datasets.sh /path/to/data _old
```

### 问题2：处理到一半中断
```bash
# 脚本会保存进度（已处理的.DONE标记）
# 重新运行时会跳过已完成的样本

# 如果想从头开始（重新处理）：
# 1. 删除输出目录
# 2. 或添加 --overwrite_video/--overwrite_frames

# Python直接模式示例：
python retarget_videos_and_frames.py \
  --batch-root ... \
  --overwrite_video \
  ...
```

### 问题3：占用磁盘空间过多
```bash
# 分阶段处理：只转码关键摄像头

# 仅head摄像头
bash batch_retarget.sh \
  --camera observation.images.head_realsense_color \
  --do_video \
  --do_frames

# 其他摄像头延后处理
bash batch_retarget.sh \
  --camera observation.images.left_hand_realsense_color \
  --do_video \
  --do_frames
```

---

## 📞 获取帮助

### 详细文档
```bash
cat BATCH_RETARGET.md
```

### 脚本帮助
```bash
bash batch_retarget.sh --help
python retarget_videos_and_frames.py --help
```

### 检查数据集
```bash
bash check_batch_datasets.sh
```

---

## ✅ 下一步

1. **立即开始**: `bash batch_retarget.sh`
2. **预检查**: `bash check_batch_datasets.sh` (看看会处理什么)
3. **详细了解**: `cat BATCH_RETARGET.md`
4. **自定义参数**: `bash batch_retarget.sh --help`

---

**享受批量处理带来的效率提升！** 🎉

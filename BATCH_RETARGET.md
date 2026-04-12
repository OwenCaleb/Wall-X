# 批量处理视频和帧 - 快速指南

## 概述

`retarget_videos_and_frames.py` 脚本现已支持**批处理模式**，可以一次性处理多个数据集，无需逐一运行命令。

有两种使用方式：
1. **脚本模式** (推荐): `bash batch_retarget.sh` - 预置参数，开箱即用
2. **Python直接模式**: `python retarget_videos_and_frames.py --batch-root ...` - 更灵活

---

## 快速开始

### 最简单：处理所有默认数据集
```bash
cd /mnt/nas_ssd/workspace/wenboli/projects/Wall-X
bash batch_retarget.sh
```

这会：
- 找到所有 `/wallx/data/g1/lerobot/*/..._old` 目录（共15个）
- 对每个数据集的**3个摄像头视图**（head, left_hand, right_hand）
- 执行视频转码 + 帧提取（stride=10）
- 生成 `video_retarget_head` 和 `frame_retarget_head` 目录

---

## 使用方式

### 方式1：bash脚本（推荐）

#### 仅处理视频转码（更快，不提取帧）
```bash
bash batch_retarget.sh --mode video_only
```

#### 仅处理帧提取（基于已转码的视频）
```bash
bash batch_retarget.sh --mode frames_only
```

#### 只处理特定摄像头
```bash
# 只有head摄像头
bash batch_retarget.sh --camera observation.images.head_realsense_color

# head和right摄像头
bash batch_retarget.sh --camera \
  observation.images.head_realsense_color \
  observation.images.right_hand_realsense_color
```

#### 自定义参数
```bash
# 自定义batch根目录和stride
bash batch_retarget.sh \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot \
  --pattern _new \
  --stride 1 \
  --out-tag custom

# 组合多个参数
bash batch_retarget.sh \
  --mode frames_only \
  --stride 1 \
  --out-tag frame1 \
  --camera observation.images.head_realsense_color observation.images.right_hand_realsense_color
```

#### 查看帮助
```bash
bash batch_retarget.sh --help
```

---

### 方式2：直接用Python（更灵活）

#### 对比参数
```bash
# 原来的单数据集方式（现在仍然支持）
python retarget_videos_and_frames.py \
  --root /path/to/single/dataset \
  --camera_view observation.images.head_realsense_color \
  --stride 10 \
  --use_video_for_frames \
  --out_tag head

# 新的批处理方式（处理多个数据集）
python retarget_videos_and_frames.py \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot \
  --pattern _old \
  --stride 10 \
  --use_video_for_frames \
  --out_tag head
```

#### 常用批处理命令

**完整处理（视频+帧）**
```bash
python retarget_videos_and_frames.py \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot \
  --pattern _old \
  --do_video \
  --do_frames \
  --use_video_for_frames \
  --stride 10 \
  --out_tag head
```

**只转码视频（快速）**
```bash
python retarget_videos_and_frames.py \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot \
  --pattern _old \
  --do_video \
  --out_tag head
```

**处理后再提取帧**
```bash
python retarget_videos_and_frames.py \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot \
  --pattern _old \
  --do_frames \
  --use_video_for_frames \
  --stride 10 \
  --out_tag head
```

**只处理指定摄像头**
```bash
python retarget_videos_and_frames.py \
  --batch-root /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot \
  --pattern _old \
  --cameras observation.images.head_realsense_color \
  --do_video --do_frames \
  --stride 10 \
  --out_tag head
```

---

## 输出结构

对于每个数据集，生成的目录结构为：

```
<dataset_root>/
├── videos/
│   └── chunk-000/
│       ├── observation.images.head_realsense_color/     （原始视频）
│       ├── observation.images.left_hand_realsense_color/
│       └── observation.images.right_hand_realsense_color/
├── video_retarget_head/       （转码为H.264）
│   ├── sample_000000/
│   │   └── Frame_000000.mp4
│   ├── sample_000001/
│   │   └── Frame_000001.mp4
│   └── ...
└── frame_retarget_head/       （提取的帧）
    ├── sample_000000/
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   └── ...
    ├── sample_000001/
    └── ...
```

---

## 性能提示

### 分两阶段处理（推荐大规模处理）

**第1阶段：视频转码**
```bash
bash batch_retarget.sh --mode video_only
```
- 时间：~1-2小时（15个数据集）
- 磁盘IO：视频读写
- 可后台运行

**第2阶段：帧提取**
```bash
bash batch_retarget.sh --mode frames_only
```
- 时间：~30分钟（取决于stride）
- 磁盘IO：视频读、帧写
- 已有转码视频，直接从那里提取帧

### 仅处理必要摄像头

```bash
# 如只需要head摄像头，可节省2/3的时间
bash batch_retarget.sh --camera observation.images.head_realsense_color
```

### 调整视频质量

在 `retarget_videos_and_frames.py` 中修改：
```python
H264_CRF = 18  # 范围: 0-51, 默认18
# 更低 = 更高质量（更大文件）
# 建议: 18-23
H264_PRESET = "veryfast"  # 预设速度
# 选项: ultrafast/superfast/veryfast/faster/fast/medium/slow/slower/veryslow
```

---

## 错误处理

批处理会：
- 自动捕获单个数据集/摄像头的错误
- 继续处理其他数据集
- 最后显示汇总（成功/失败数量）

示例输出：
```
[BATCH] Summary: 42 OK, 3 FAILED
```

检查详细错误，查看运行时的错误消息。

---

## 常见问题

### Q: 处理了一半，想重新开始？
A: 
```bash
# 添加 --overwrite_video 或 --overwrite_frames 变量
# 修改batch_retarget.sh，或直接用Python：
python retarget_videos_and_frames.py \
  --batch-root ... \
  --overwrite_video \  # 重新转码
  --overwrite_frames \ # 重新提取帧
  ...
```

### Q: 只想处理某个特定数据集？
A: 回到单数据集模式：
```bash
python retarget_videos_and_frames.py \
  --root /path/to/specific/dataset \
  --camera_view observation.images.head_realsense_color \
  ...
```

### Q: 每个视频有多久？如何估算总时间？
A: 
- 视频转码：~5-15分钟/视频（H.264@crf18）
- 帧提取：~2-5分钟/视频（stride=10）
- 15个数据集×60个视频×2步 = 总计2-4小时

---

## 文件备索

| 文件 | 功能 |
|------|------|
| `retarget_videos_and_frames.py` | 核心处理脚本（支持单/批处理两种模式） |
| `batch_retarget.sh` | 便捷shell脚本（推荐使用） |
| `BATCH_RETARGET.md` | 本说明文档 |

---

## 更新日志

### v2.0 (最新)
- ✅ 添加批处理模式 `--batch-root`
- ✅ 自动发现匹配的数据集目录
- ✅ 支持多摄像头处理
- ✅ 便捷shell脚本 `batch_retarget.sh`
- ✅ 错误恢复和汇总统计

### v1.0
- 单数据集处理模式
- 视频转码 + 帧提取

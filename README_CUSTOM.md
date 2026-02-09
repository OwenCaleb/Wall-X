# README_CUSTOM

This document summarizes the incremental development work completed in this workspace.

## Overview
The recent changes focus on training stability, loss correctness with mixed data branches, and adding VQA/CoT data support with controllable sampling.

## Incremental Changes
- Training loop timer safety: avoid stopping unstarted timers when NaN loss is detected, and guard `set_epoch` when sampler is absent.
- Action normalization safety: avoid division by zero when `delta == 0` during min-max normalization.
- Mixed-branch flow loss fix: when a batch mixes action and non-action samples, filter flow-side tensors to avoid shape mismatch.
- VQA and CoT data branches:
  - Added VQA and CoT prompt assembly in the text builder.
  - Added VQA/CoT ratios and VQA type filtering in config and data pipeline.
  - Loaded metadata from `qa_labels.parquet` and `tasks_high_level.parquet`.
- VQA oversampling (strict mixing):
  - Built a VQA-only dataset view and a strict distributed mix sampler.
  - Ensured sampler length is consistent across ranks.
  - Epoch definition: one full pass over the main dataset, with extra VQA samples injected by ratio.

## New/Updated Config Knobs
These are added to the data config to control VQA/CoT behavior:
- `generate_vqa_ratio`
- `generate_cot_ratio`
- `vqa_types` (optional list for type filtering)
- `vqa_mix_weight_vqa`
- `vqa_mix_weight_main`

## Metadata Files Used
Expected under the dataset `meta/` directory:
- `subtasks.parquet` (subtask labels)
- `tasks_high_level.parquet` (high-level task text and CoT labels)
- `qa_labels.parquet` (VQA question/answer by episode/frame)

## Sampler Behavior (Multi-GPU)
- When VQA mix weights are enabled and VQA samples exist, a strict mix sampler is used.
- It avoids cross-rank duplication and keeps all ranks aligned in length.
- If no valid VQA samples exist after filtering, the sampler falls back to standard `DistributedSampler`.

## Known Symptoms and Debug Notes
- NCCL timeouts typically indicate rank imbalance or a per-rank stall. If this recurs:
  - Try `num_workers=0` to rule out DataLoader worker hangs.
  - Temporarily disable VQA mixing by setting `vqa_mix_weight_vqa=0`.
  - Enable NCCL tracing with `TORCH_DISTRIBUTED_DEBUG=DETAIL` and `TORCH_NCCL_TRACE_BUFFER_SIZE`.

## Files Touched (Primary)
- `wall_x/trainer/qwen_vl_act_trainer.py`
- `wall_x/model/action_head.py`
- `wall_x/model/vla_mixin.py`
- `wall_x/data/utils.py`
- `wall_x/data/config.py`
- `wall_x/data/load_lerobot_dataset.py`

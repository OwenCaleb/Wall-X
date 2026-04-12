"""
Counterfactual task-conditioned behavior test: segment-level text cyclic shift (A B C -> B C A).
"""

import argparse
import copy
import json
import os
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

from wall_x.data.load_lerobot_dataset import load_test_dataset
from wall_x.model.action_head import Normalizer
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["data"]["model_type"] = config.get("model_type")
    return config


def resolve_runtime_dataset_key(lerobot_cfg: Dict[str, Any]) -> str:
    root = str(lerobot_cfg.get("root", "")).rstrip("/")
    if root:
        return os.path.basename(root).replace(".", "_")
    return str(lerobot_cfg.get("repo_id", "dataset")).replace(".", "_")


def resolve_dataset_name(lerobot_cfg: Dict[str, Any]) -> str:
    root = str(lerobot_cfg.get("root", "")).rstrip("/")
    if root:
        parent_name = os.path.basename(os.path.dirname(root))
        if parent_name:
            return parent_name
        return os.path.basename(root)
    return str(lerobot_cfg.get("repo_id", "dataset"))


def build_multidataset_normalizers(config: Dict[str, Any], lerobot_configs: List[Dict[str, Any]]):
    action_statistic_dof = {}
    for cfg_item in lerobot_configs:
        p = cfg_item.get("norm_stats_path", None)
        if not p:
            continue
        if not os.path.exists(p):
            raise FileNotFoundError(f"norm_stats_path not found: {p}")

        stats = json.load(open(p, "r"))
        runtime_key = resolve_runtime_dataset_key(cfg_item)

        if runtime_key in stats and isinstance(stats[runtime_key], dict):
            action_statistic_dof[runtime_key] = stats[runtime_key]
        elif len(stats) == 1:
            only_key = next(iter(stats.keys()))
            action_statistic_dof[runtime_key] = stats[only_key]
        else:
            repo_id = str(cfg_item.get("repo_id", ""))
            if repo_id in stats and isinstance(stats[repo_id], dict):
                action_statistic_dof[runtime_key] = stats[repo_id]
            else:
                raise ValueError(
                    f"Cannot resolve stats key for dataset {runtime_key} from {p}. "
                    f"Available keys: {list(stats.keys())}"
                )

    if len(action_statistic_dof) == 0:
        raise ValueError("No action statistics built from lerobot_configs")

    normalizer_action = Normalizer(
        action_statistic_dof,
        config["dof_config"],
        min_key=config.get("min_key", "min"),
        delta_key=config.get("delta_key", "delta"),
    )
    normalizer_propri = Normalizer(
        action_statistic_dof,
        config["agent_pos_config"],
        min_key=config.get("min_key", "min"),
        delta_key=config.get("delta_key", "delta"),
    )
    return normalizer_action, normalizer_propri


def parse_episodes(lerobot_cfg: Dict[str, Any], fallback: List[int]) -> List[int]:
    cfg_episodes = lerobot_cfg.get("episodes", None)
    if cfg_episodes is None:
        return fallback
    if isinstance(cfg_episodes, int):
        return [int(cfg_episodes)]
    return [int(x) for x in cfg_episodes]


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def ensure_batch_on_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        try:
            batch = batch.to(device)
        except Exception:
            pass
    batch = move_to_device(batch, device)
    if isinstance(batch, dict):
        for key in ["input_ids", "attention_mask", "pixel_values", "action_chunk", "dof_mask"]:
            if key in batch and torch.is_tensor(batch[key]):
                if batch[key].device != device:
                    batch[key] = batch[key].to(device)
    return batch


def extract_subtask_instruction(text: str) -> Optional[str]:
    """Extract instruction from text for segment identification."""
    m = re.search(r'"subtask_instruction"\s*:\s*"([^"]*)"', text)
    if m:
        s = m.group(1).strip()
        return s if s else None
    m = re.search(r'"subtask_generation"\s*:\s*"([^"]*)"', text)
    if m:
        s = m.group(1).strip()
        return s if s else None
    m = re.search(r"subtask_instruction\s*:\s*(.*)", text)
    if m:
        s = m.group(1).strip()
        return s if s else None
    m = re.search(r"subtask_generation\s*:\s*(.*)", text)
    if m:
        s = m.group(1).strip()
        return s if s else None

    # Action-path fallback: parse user-side instruction in prompt text.
    m = re.search(
        r"Instruction:\s*(.*?)(?:\nPredict the next action in robot action\.|\nPredict the next action in language\.|\nAnswer the question based on the observation\.|\nOutput thought and subtask\.|<\|im_end\|>)",
        text,
        flags=re.DOTALL,
    )
    if m:
        s = m.group(1).strip()
        if s:
            return s

    # Last fallback for debug-like text blobs containing assistant target strings.
    m = re.search(r"Place the .*? basket\.", text)
    if m:
        s = m.group(0).strip()
        return s if s else None

    return None


def build_out_dir(config: Dict[str, Any]) -> str:
    save_path = str(config.get("save_path", ""))
    save_tail = os.path.basename(save_path.rstrip("/")) if save_path else "output"
    out_dir = os.path.join(
        "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/save_path_dir",
        save_tail,
        "different_task_test",
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def build_segment_shift_plan(dataset) -> Dict[str, Any]:
    """
    Build segment-level text cyclic shift: A B C -> B C A.
    Segment defined by consecutive frames with same instruction.
    Returns: frame_index -> source_frame_index for text shift.
    """
    n_items = len(dataset)
    if n_items == 0:
        return {"segments": [], "frame_to_source_idx": [], "num_segments": 0}

    instructions = [extract_subtask_instruction(str(dataset[i]["text"])) for i in range(n_items)]

    segments = []
    seg_start = 0
    cur_instr = instructions[0]
    for i in range(1, n_items):
        if instructions[i] != cur_instr:
            segments.append({"start": seg_start, "end": i - 1, "instr": cur_instr})
            seg_start = i
            cur_instr = instructions[i]
    segments.append({"start": seg_start, "end": n_items - 1, "instr": cur_instr})

    frame_to_source_idx = [-1] * n_items

    if len(segments) >= 2:
        for s_idx, seg in enumerate(segments):
            next_seg = segments[(s_idx + 1) % len(segments)]
            src_start = next_seg["start"]
            for frame_i in range(seg["start"], seg["end"] + 1):
                frame_to_source_idx[frame_i] = src_start

    return {
        "segments": segments,
        "frame_to_source_idx": frame_to_source_idx,
        "num_segments": len(segments),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_pairs_per_dataset", type=int, default=80)
    parser.add_argument("--origin_action_dim", type=int, default=19)
    parser.add_argument("--fallback_episode", type=int, default=0)
    args = parser.parse_args()

    model_path = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/models/wallx/wall-oss-flow-v0.1-copy"
    config_path = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/workspace/lerobot_example/config_qact_custom.yml"
    config = load_config(config_path)

    config["data"]["generate_vqa_ratio"] = 0
    config["data"]["generate_cot_ratio"] = 0
    config["data"]["generate_subtask_ratio"] = 0

    lerobot_configs = config.get("data", {}).get("lerobot_configs", [])
    if not lerobot_configs:
        raise ValueError("No lerobot_configs found in config['data']")

    out_dir = build_out_dir(config)
    print(f"Using model_path: {model_path}")
    print(f"Output dir: {out_dir}")

    normalizer_action, normalizer_propri = build_multidataset_normalizers(config, lerobot_configs)

    model = Qwen2_5_VLMoEForAction.from_pretrained(model_path, train_config=config)
    model.set_normalizer(copy.deepcopy(normalizer_action), copy.deepcopy(normalizer_propri))
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)
    model.to_bfloat16_for_selected_params()

    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    action_dim = 20 if predict_mode == "diffusion" else args.origin_action_dim

    with torch.no_grad():
        for cfg_idx, lerobot_cfg in enumerate(lerobot_configs):
            dataset_name = resolve_dataset_name(lerobot_cfg)
            dataset_key = resolve_runtime_dataset_key(lerobot_cfg)
            episodes = parse_episodes(lerobot_cfg, [args.fallback_episode])
            detail_path = os.path.join(out_dir, f"A({safe_name(dataset_name)}).jsonl")

            print(f"\n[{cfg_idx + 1}/{len(lerobot_configs)}] dataset={dataset_name}, episodes={episodes}")

            ds_config = copy.deepcopy(config)
            ds_norm_stats = lerobot_cfg.get("norm_stats_path", None)
            if ds_norm_stats is None:
                raise ValueError(f"norm_stats_path missing for dataset {dataset_name}")
            ds_config["norm_stats_path"] = ds_norm_stats

            dataset = load_test_dataset(
                ds_config,
                lerobot_cfg,
                normalizer_action,
                normalizer_propri,
                seed=42,
                episodes=episodes,
            )

            plan = build_segment_shift_plan(dataset)
            num_segments = plan["num_segments"]
            frame_to_source_idx = plan["frame_to_source_idx"]

            dataloader = dataset.get_dataloader()
            total_frames = len(dataloader)
            n_items = len(dataset)

            if n_items < 2 or total_frames == 0 or num_segments < 2:
                print(f"Skip {dataset_name}: n_items={n_items}, total_frames={total_frames}, num_segments={num_segments}")
                skip_record = {
                    "dataset": dataset_name,
                    "dataset_key": dataset_key,
                    "episodes": episodes,
                    "skip": True,
                    "skip_reason": (
                        "insufficient_segments"
                        if num_segments < 2
                        else "empty_or_too_short_dataset"
                    ),
                    "n_items": n_items,
                    "total_frames": total_frames,
                    "num_segments": num_segments,
                }
                with open(detail_path, "w") as f:
                    f.write(json.dumps(skip_record, ensure_ascii=False) + "\n")
                print(f"Saved detail records: {detail_path}")
                continue

            detail_records = []
            tested = 0
            sum_delta = 0.0
            sum_delta_first = 0.0
            sum_rel = 0.0
            skip_invalid_source = 0

            gt_traj = torch.full((total_frames, args.origin_action_dim), float("nan"))
            pred_base_traj = torch.full((total_frames, args.origin_action_dim), float("nan"))
            pred_cf_traj = torch.full((total_frames, args.origin_action_dim), float("nan"))

            pbar = tqdm(total=total_frames, desc=f"cf-{dataset_name}")
            for i, batch in enumerate(dataloader):
                if tested >= args.max_pairs_per_dataset:
                    pbar.update(1)
                    break

                runtime_horizon = int(batch["action_chunk"].shape[1])
                if not (i % runtime_horizon == 0 and i + runtime_horizon < total_frames):
                    pbar.update(1)
                    continue

                src_idx = frame_to_source_idx[i] if i < len(frame_to_source_idx) else -1
                if src_idx < 0:
                    skip_invalid_source += 1
                    pbar.update(1)
                    continue

                item_cur = dataset[i]
                item_cf = dataset[src_idx]

                batch_base = batch
                batch_cf_list = [item_cf]
                batch_cf = dataloader.collate_fn(batch_cf_list)

                batch_base = ensure_batch_on_device(batch_base, device)
                batch_cf = ensure_batch_on_device(batch_cf, device)

                out_base = model(
                    **batch_base,
                    action_dim=action_dim,
                    action_horizon=runtime_horizon,
                    mode="predict",
                    predict_mode=predict_mode,
                )["predict_action"][0, :, : args.origin_action_dim]

                out_cf = model(
                    **batch_cf,
                    action_dim=action_dim,
                    action_horizon=runtime_horizon,
                    mode="predict",
                    predict_mode=predict_mode,
                )["predict_action"][0, :, : args.origin_action_dim]

                gt_norm = batch_base["action_chunk"][0, :, : args.origin_action_dim]
                dof_mask = batch_base["dof_mask"][0:1].to(gt_norm.dtype)
                gt = model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_norm.unsqueeze(0),
                    [dataset_key],
                    dof_mask,
                )[0]

                start = i
                end = i + runtime_horizon
                pred_base_traj[start:end] = out_base.detach().cpu()
                pred_cf_traj[start:end] = out_cf.detach().cpu()
                gt_traj[start:end] = gt.detach().cpu()

                delta = (out_base - out_cf).abs()
                mean_abs_delta = float(delta.mean().item())
                first_step_delta = float(delta[0].mean().item())
                gt_scale = float(gt.abs().mean().item()) + 1e-6
                rel_delta_to_gt = mean_abs_delta / gt_scale

                tested += 1
                sum_delta += mean_abs_delta
                sum_delta_first += first_step_delta
                sum_rel += rel_delta_to_gt

                instr_cur = extract_subtask_instruction(item_cur["text"])
                instr_cf = extract_subtask_instruction(item_cf["text"])
                detail_records.append(
                    {
                        "dataset": dataset_name,
                        "frame_index": i,
                        "source_frame_index": src_idx,
                        "runtime_horizon": runtime_horizon,
                        "instruction_original": instr_cur,
                        "instruction_counterfactual": instr_cf,
                        "mean_abs_delta": mean_abs_delta,
                        "first_step_abs_delta": first_step_delta,
                        "relative_delta_to_gt_scale": rel_delta_to_gt,
                    }
                )

                pbar.update(1)

            pbar.close()
            print(
                f"Skip stats: invalid_source={skip_invalid_source}, tested={tested}"
            )

            if tested > 0:
                gt_traj_np = gt_traj.numpy()
                pred_base_np = pred_base_traj.numpy()
                pred_cf_np = pred_cf_traj.numpy()

                fig, axs = plt.subplots(
                    args.origin_action_dim,
                    1,
                    figsize=(15, 5 * args.origin_action_dim),
                    sharex=True,
                )
                if args.origin_action_dim == 1:
                    axs = [axs]

                timesteps = gt_traj_np.shape[0]
                x = range(timesteps)
                fig.suptitle(
                    f"Counterfactual Comparison | {dataset_name} | episodes={episodes}",
                    fontsize=14,
                )

                for d in range(args.origin_action_dim):
                    axs[d].plot(x, gt_traj_np[:, d], label="Ground Truth")
                    axs[d].plot(x, pred_base_np[:, d], label="Prediction (Original)")
                    axs[d].plot(x, pred_cf_np[:, d], label="Prediction (Counterfactual)")
                    axs[d].set_ylabel(f"Action Dim {d + 1}")
                    axs[d].legend()
                    axs[d].grid(True)

                axs[-1].set_xlabel("Timestep")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                vis_path = os.path.join(out_dir, f"A({safe_name(dataset_name)}).png")
                plt.savefig(vis_path)
                plt.close()
                print(f"Saved plot: {vis_path}")

                # Save per-dataset detail records
                with open(detail_path, "w") as f:
                    for rec in detail_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"Saved detail records: {detail_path}")

    print("\nDone.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()

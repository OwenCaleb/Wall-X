import argparse
import copy
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

from wall_x.data.load_lerobot_dataset import load_test_dataset
from wall_x.model.action_head import Normalizer
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["data"]["model_type"] = config.get("model_type")
    return config


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty integer list")
    return vals


def resolve_dataset_name(lerobot_cfg: Dict) -> str:
    root = str(lerobot_cfg.get("root", "")).rstrip("/")
    if root:
        parent_name = os.path.basename(os.path.dirname(root))
        if parent_name:
            return parent_name
        return os.path.basename(root)
    return str(lerobot_cfg.get("repo_id", "dataset"))


def resolve_runtime_dataset_key(lerobot_cfg: Dict) -> str:
    root = str(lerobot_cfg.get("root", "")).rstrip("/")
    if root:
        return os.path.basename(root).replace(".", "_")
    return str(lerobot_cfg.get("repo_id", "dataset")).replace(".", "_")


def resolve_episodes(lerobot_cfg: Dict, fallback: List[int]) -> List[int]:
    cfg_episodes = lerobot_cfg.get("episodes", None)
    if cfg_episodes is None:
        return fallback
    if isinstance(cfg_episodes, int):
        return [int(cfg_episodes)]
    return [int(x) for x in cfg_episodes]


def build_output_dir(config: Dict) -> str:
    save_path = str(config.get("save_path", ""))
    save_tail = os.path.basename(save_path.rstrip("/")) if save_path else "output"
    out_dir = os.path.join(
        "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/save_path_dir",
        save_tail,
        "horizon_test",
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_multidataset_normalizers(config: Dict, lerobot_configs: List[Dict]):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=str, default="4,32,64,128,256")
    parser.add_argument("--origin_action_dim", type=int, default=19)
    parser.add_argument("--default_episodes", type=str, default="0")
    args = parser.parse_args()

    horizons = parse_int_list(args.horizons)
    default_episodes = parse_int_list(args.default_episodes)

    model_path = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/models/wallx/wall-oss-flow-v0.1-copy"
    config_path = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/workspace/lerobot_example/config_qact_custom.yml"
    config = load_config(config_path)

    # Horizon test should stay on action prediction path.
    config["data"]["generate_vqa_ratio"] = 0
    config["data"]["generate_cot_ratio"] = 0
    config["data"]["generate_subtask_ratio"] = 0

    lerobot_configs = config.get("data", {}).get("lerobot_configs", [])
    if not lerobot_configs:
        raise ValueError("No lerobot_configs found in config['data']")

    out_dir = build_output_dir(config)
    print(f"Using model_path: {model_path}")
    print(f"Output dir: {out_dir}")
    print(f"Testing horizons: {horizons}")

    normalizer_action, normalizer_propri = build_multidataset_normalizers(config, lerobot_configs)

    model = Qwen2_5_VLMoEForAction.from_pretrained(model_path, train_config=config)
    model.set_normalizer(copy.deepcopy(normalizer_action), copy.deepcopy(normalizer_propri))
    model.eval()
    model = model.to("cuda")
    model.to_bfloat16_for_selected_params()

    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    action_dim = 20 if predict_mode == "diffusion" else args.origin_action_dim

    summary_rows = []

    with torch.no_grad():
        for idx_cfg, lerobot_cfg in enumerate(lerobot_configs):
            dataset_name = resolve_dataset_name(lerobot_cfg)
            dataset_runtime_key = resolve_runtime_dataset_key(lerobot_cfg)
            selected_episodes = resolve_episodes(lerobot_cfg, default_episodes)

            print(
                f"\n[{idx_cfg + 1}/{len(lerobot_configs)}] dataset={dataset_name}, episodes={selected_episodes}"
            )

            ds_norm_stats = lerobot_cfg.get("norm_stats_path", None)
            if ds_norm_stats is None:
                raise ValueError(f"norm_stats_path missing for dataset {dataset_name}")

            # Build one dataloader per horizon so action token slots and batch tensors
            # are generated consistently for that horizon.
            dataloader_by_h = {}
            total_frames_by_h = {}
            for h in horizons:
                ds_config_h = copy.deepcopy(config)
                ds_config_h["norm_stats_path"] = ds_norm_stats
                ds_config_h["data"]["action_horizon"] = int(h)

                dataset_h = load_test_dataset(
                    ds_config_h,
                    lerobot_cfg,
                    normalizer_action,
                    normalizer_propri,
                    seed=42,
                    episodes=selected_episodes,
                )
                dataloader_h = dataset_h.get_dataloader()
                dataloader_by_h[h] = dataloader_h
                total_frames_by_h[h] = len(dataloader_h)

            print(
                "Horizon frame counts:",
                {int(h): int(total_frames_by_h[h]) for h in horizons},
                flush=True,
            )

            non_empty = [h for h in horizons if total_frames_by_h[h] > 0]
            if len(non_empty) == 0:
                print(f"Skip {dataset_name}: all horizon dataloaders are empty")
                continue

            common_frames = min(total_frames_by_h[h] for h in non_empty)
            if common_frames == 0:
                print(f"Skip {dataset_name}: common frame length is zero")
                continue

            ref_h = non_empty[0]
            gt_traj = torch.full((common_frames, args.origin_action_dim), float("nan"))
            pred_traj_by_h = {
                h: torch.full((common_frames, args.origin_action_dim), float("nan"))
                for h in non_empty
            }

            # Ground truth from reference horizon dataloader (same dataset/episode).
            for frame_idx, batch in tqdm(
                enumerate(dataloader_by_h[ref_h]), total=common_frames, desc=f"gt-{dataset_name}"
            ):
                if frame_idx >= common_frames:
                    break
                runtime_horizon = int(batch["action_chunk"].shape[1])
                if frame_idx % runtime_horizon != 0 or frame_idx + 1 >= common_frames:
                    continue

                batch = batch.to("cuda")
                gt_action_chunk = batch["action_chunk"][:, :, : args.origin_action_dim]
                dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
                denormalized_gt = model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [dataset_runtime_key],
                    dof_mask,
                ).squeeze(0)

                gt_len = min(runtime_horizon, common_frames - frame_idx)
                gt_traj[frame_idx : frame_idx + gt_len] = denormalized_gt[:gt_len].detach().cpu()

            for h in non_empty:
                for frame_idx, batch in tqdm(
                    enumerate(dataloader_by_h[h]), total=common_frames, desc=f"h={h}-{dataset_name}"
                ):
                    if frame_idx >= common_frames:
                        break
                    runtime_horizon = int(batch["action_chunk"].shape[1])
                    if frame_idx % runtime_horizon != 0 or frame_idx + 1 >= common_frames:
                        continue

                    batch = batch.to("cuda")
                    outputs = model(
                        **batch,
                        action_dim=action_dim,
                        action_horizon=runtime_horizon,
                        mode="predict",
                        predict_mode=predict_mode,
                    )
                    pred = outputs["predict_action"][0, :, : args.origin_action_dim].detach().cpu()
                    pred_len = min(pred.shape[0], common_frames - frame_idx)
                    pred_traj_by_h[h][frame_idx : frame_idx + pred_len] = pred[:pred_len]

            gt_np = gt_traj.numpy()
            pred_np_by_h = {h: t.numpy() for h, t in pred_traj_by_h.items()}
            timesteps = gt_traj.shape[0]

            fig, axs = plt.subplots(
                args.origin_action_dim,
                1,
                figsize=(16, 5 * args.origin_action_dim),
                sharex=True,
            )
            if args.origin_action_dim == 1:
                axs = [axs]

            fig.suptitle(
                f"Horizon Test | {dataset_name} | episodes={selected_episodes}",
                fontsize=14,
            )

            x = range(timesteps)
            for d in range(args.origin_action_dim):
                axs[d].plot(x, gt_np[:, d], label="Ground Truth", linewidth=1.5)
                for h in non_empty:
                    axs[d].plot(x, pred_np_by_h[h][:, d], label=f"Pred(h={h})", linewidth=1.1)
                axs[d].set_ylabel(f"Action Dim {d + 1}")
                axs[d].grid(True)
                axs[d].legend(loc="upper right")

            axs[-1].set_xlabel("Timestep")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            out_file = os.path.join(out_dir, f"H({dataset_name}).png")
            plt.savefig(out_file)
            plt.close()
            print(f"Saved plot: {out_file}")

            # Compute MAE over valid (non-NaN) positions for each horizon.
            for h in non_empty:
                diff = torch.from_numpy(pred_np_by_h[h] - gt_np).abs()
                valid = ~torch.isnan(diff)
                mae = float(diff[valid].mean().item()) if valid.any() else float("nan")
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "episodes": str(selected_episodes),
                        "horizon": h,
                        "mae": mae,
                    }
                )

    summary_path = os.path.join(out_dir, "horizon_test_summary.jsonl")
    with open(summary_path, "w") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

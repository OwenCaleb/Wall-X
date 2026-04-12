import argparse
import copy
import json
import os

import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

from wall_x.data.load_lerobot_dataset import load_test_dataset
from wall_x.model.action_head import Normalizer
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["data"]["model_type"] = config.get("model_type")
    return config


def parse_episode_list(text):
    if text is None or text == "":
        return None
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def resolve_dataset_name(lerobot_cfg):
    root = str(lerobot_cfg.get("root", "")).rstrip("/")
    if root:
        parent_name = os.path.basename(os.path.dirname(root))
        if parent_name:
            return parent_name
        return os.path.basename(root)
    return str(lerobot_cfg.get("repo_id", "dataset"))


def resolve_runtime_dataset_key(lerobot_cfg):
    # Must match dataset_name produced in load_lerobot_dataset.py
    root = str(lerobot_cfg.get("root", "")).rstrip("/")
    if root:
        return os.path.basename(root).replace(".", "_")
    return str(lerobot_cfg.get("repo_id", "dataset")).replace(".", "_")


def resolve_episodes(lerobot_cfg, global_episodes, default_episodes):
    # Priority: per-dataset episodes > global episodes in data config > script default
    cfg_episodes = lerobot_cfg.get("episodes", None)
    if cfg_episodes is not None:
        if isinstance(cfg_episodes, int):
            return [int(cfg_episodes)]
        return [int(x) for x in cfg_episodes]

    if global_episodes is not None:
        if isinstance(global_episodes, int):
            return [int(global_episodes)]
        return [int(x) for x in global_episodes]

    return default_episodes


def build_output_dir(config):
    save_path = str(config.get("save_path", ""))
    save_tail = os.path.basename(save_path.rstrip("/")) if save_path else "output"
    out_dir = os.path.join(
        "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/save_path_dir", save_tail
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_multidataset_normalizers(config, lerobot_configs):
    action_statistic_dof = {}
    for cfg_item in lerobot_configs:
        p = cfg_item.get("norm_stats_path", None)
        if not p:
            continue
        if not os.path.exists(p):
            raise FileNotFoundError(f"norm_stats_path not found: {p}")

        stats = json.load(open(p, "r"))
        runtime_key = resolve_runtime_dataset_key(cfg_item)

        # Expected stats file format is typically {"g1custom": {...}}.
        # We remap it to runtime dataset key to match dataloader's dataset_name.
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
    parser.add_argument("--pred_horizon", type=int, default=32)
    parser.add_argument("--origin_action_dim", type=int, default=19)
    parser.add_argument(
        "--default_episodes",
        type=str,
        default="0",
        help="Fallback episodes when dataset config does not set episodes, e.g. '0,1,2'",
    )
    args = parser.parse_args()

    model_path = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/models/wallx/wall-oss-flow-v0.1-copy"
    path = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/workspace/lerobot_example/config_qact_custom.yml"
    config = load_config(path)

    # For open-loop action plotting, force pure action path in dataset text builder.
    # Do not rely on training-time text task ratios.
    config["data"]["generate_vqa_ratio"] = 0
    config["data"]["generate_cot_ratio"] = 0
    config["data"]["generate_subtask_ratio"] = 0

    data_cfg = config.get("data", {})
    lerobot_configs = data_cfg.get("lerobot_configs", [])
    if not lerobot_configs:
        raise ValueError("No lerobot_configs found in config['data']")

    default_episodes = parse_episode_list(args.default_episodes)
    if default_episodes is None or len(default_episodes) == 0:
        default_episodes = [0]

    global_episodes = data_cfg.get("episodes", None)
    out_dir = build_output_dir(config)

    print(f"Using model_path: {model_path}")
    print(f"Output dir: {out_dir}")

    normalizer_action, normalizer_propri = build_multidataset_normalizers(
        config,
        lerobot_configs,
    )

    model = Qwen2_5_VLMoEForAction.from_pretrained(model_path, train_config=config)
    model.set_normalizer(copy.deepcopy(normalizer_action), copy.deepcopy(normalizer_propri))
    model.eval()
    model = model.to("cuda")
    model.to_bfloat16_for_selected_params()

    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    action_dim = 20 if predict_mode == "diffusion" else args.origin_action_dim

    for idx_cfg, lerobot_cfg in enumerate(lerobot_configs):
        dataset_name = resolve_dataset_name(lerobot_cfg)
        dataset_runtime_key = resolve_runtime_dataset_key(lerobot_cfg)
        selected_episodes = resolve_episodes(lerobot_cfg, global_episodes, default_episodes)
        print(
            f"\n[{idx_cfg + 1}/{len(lerobot_configs)}] dataset={dataset_name}, episodes={selected_episodes}"
        )

        # load_test_dataset still expects top-level norm_stats_path in config;
        # set it per dataset here for compatibility.
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
            episodes=selected_episodes,
        )
        dataloader = dataset.get_dataloader()

        total_frames = len(dataloader)
        if total_frames == 0:
            print(f"Skip {dataset_name}: empty dataloader")
            continue

        gt_traj = torch.zeros((total_frames, args.origin_action_dim))
        pred_traj = torch.zeros((total_frames, args.origin_action_dim))

        warned_horizon_mismatch = False
        for frame_idx, batch in tqdm(
            enumerate(dataloader), total=total_frames, desc=f"predict-{dataset_name}"
        ):
            # Keep horizon aligned with collated batch tensors (action_chunk / dof_mask).
            # In this codebase test dataloader horizon is often 21 by default.
            runtime_horizon = int(batch["action_chunk"].shape[1])
            if not warned_horizon_mismatch and runtime_horizon != int(args.pred_horizon):
                print(
                    f"[WARN] {dataset_name}: pred_horizon={args.pred_horizon} differs from runtime_horizon={runtime_horizon}. "
                    f"Using runtime_horizon={runtime_horizon} for prediction.",
                    flush=True,
                )
                warned_horizon_mismatch = True

            if frame_idx % runtime_horizon == 0 and frame_idx + runtime_horizon < total_frames:
                batch = batch.to("cuda")
                with torch.no_grad():
                    outputs = model(
                        **batch,
                        action_dim=action_dim,
                        action_horizon=runtime_horizon,
                        mode="predict",
                        predict_mode=predict_mode,
                    )
                    pred_traj[frame_idx : frame_idx + runtime_horizon] = (
                        outputs["predict_action"][:, :, : args.origin_action_dim]
                        .detach()
                        .cpu()
                        .squeeze(0)
                    )

                gt_action_chunk = batch["action_chunk"][:, :, : args.origin_action_dim]
                dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
                denormalized_gt = model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [dataset_runtime_key],
                    dof_mask,
                ).squeeze(0)
                gt_traj[frame_idx : frame_idx + runtime_horizon] = denormalized_gt.detach().cpu()

        gt_traj_np = gt_traj.numpy()
        pred_traj_np = pred_traj.numpy()
        timesteps = gt_traj.shape[0]

        fig, axs = plt.subplots(
            args.origin_action_dim,
            1,
            figsize=(15, 5 * args.origin_action_dim),
            sharex=True,
        )
        fig.suptitle(
            f"Action Comparison | {dataset_name} | episodes={selected_episodes}",
            fontsize=14,
        )

        for d in range(args.origin_action_dim):
            axs[d].plot(range(timesteps), gt_traj_np[:, d], label="Ground Truth")
            axs[d].plot(range(timesteps), pred_traj_np[:, d], label="Prediction")
            axs[d].set_ylabel(f"Action Dim {d + 1}")
            axs[d].legend()
            axs[d].grid(True)

        axs[-1].set_xlabel("Timestep")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_file = os.path.join(out_dir, f"A({dataset_name}).png")
        plt.savefig(out_file)
        plt.close()
        print(f"Saved plot: {out_file}")


if __name__ == "__main__":
    main()

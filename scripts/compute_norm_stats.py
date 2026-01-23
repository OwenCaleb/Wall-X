import yaml
import torch
import tqdm
import numpy as np
import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from wall_x.data.load_lerobot_dataset import KEY_MAPPINGS
import normalize
"""
Compute (mean/std) norm stats for state/action and save as normalize.save() format.

Key idea:
- For legacy LeRobot datasets, state/action columns already exist (KEY_MAPPINGS points to them).
- For your custom dataset that stores high-dimensional signals as many `observation.ts.*` columns,
  we use a *ModalityAwareLeRobotDataset* (subclass of LeRobotDataset) to inject unified keys
  (e.g. "state" / "action") based on modality.json, without breaking LeRobotDataset behaviors.
"""


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["data"]["model_type"] = config.get("model_type")
    return config
def resolve_modality_json(args, lerobot_config: dict) -> str | None:
    """
    Priority:
      1) CLI --modality_json
      2) config: data.lerobot_config.modality_json
      3) config: data.lerobot_config.modality_path (older naming)
    """
    if args.modality_json is not None and len(args.modality_json) > 0:
        return args.modality_json

    for k in ["modality_json", "modality_path", "modality"]:
        v = lerobot_config.get(k, None)
        if isinstance(v, str) and len(v) > 0:
            return v

    return None


def build_delta_timestamps(repo_id: str, root: str, action_horizon: int, modality_json_path: str | None):
    """
    delta_timestamps 的意义（直观解释）：
    - LeRobotDataset 支持“在 __getitem__ 时，帮你把未来 t=1..H 的 action 也取出来”
    - 你给它一个 dict: { column_name: [0.0, 0.1, 0.2, ...] }，它就会在同一 episode 内根据 timestamp
      计算这些未来帧的索引，并返回一个 shape [H, D] 的张量。

    我们这里的用法：
    - 如果提供了 modality.json，我们把里面 action 的 original_key 都加进 delta_timestamps
      这样 LeRobotDataset 会直接返回这些 original_key 对应的 [H, D]，后面 wrapper 拼接更快更稳。
    """
    meta = LeRobotDatasetMetadata(repo_id, root)
    fps = meta.fps

    if modality_json_path is None:
        return None

    modality = json.loads(Path(modality_json_path).read_text())
    action_orig_keys = [cfg["original_key"] for cfg in modality.get("action", {}).values()]

    # offset seconds: [0/fps, 1/fps, ..., (H-1)/fps]
    offsets = [t / fps for t in range(int(action_horizon))]
    return {k: offsets for k in action_orig_keys}
def load_lerobot_dataset(repo_id: str, root: str, action_horizon: int, modality_json_path: str | None, args):
    # --- build delta_timestamps (optional) ---
    delta_timestamps = build_delta_timestamps(
        repo_id=repo_id,
        root=root,
        action_horizon=action_horizon,
        modality_json_path=modality_json_path,
    )

    # --- dataset ---
    if modality_json_path is not None:
        from wall_x.data.modality_wrapper import ModalityAwareLeRobotDataset

        dataset = ModalityAwareLeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
            modality_json=modality_json_path,
            action_horizon=action_horizon,
            # IMPORTANT: keep legacy mapping interface
            state_key=KEY_MAPPINGS[repo_id]["state"],
            action_key=KEY_MAPPINGS[repo_id]["action"],
        )
    else:
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )

    num_batches = len(dataset) // args.batch_size

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        generator=generator,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    return data_loader, num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modality_json", type=str, default=None)
    parser.add_argument("--config", type=str, default="/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/workspace/lerobot_example/config_qact_custom.yml")
    parser.add_argument("--output_dir", type=str, default="/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/norm_stats")
    args = parser.parse_args()

    config = load_config(args.config)
    lerobot_config = config["data"]["lerobot_config"]

    repo_id = lerobot_config.get("repo_id", None)
    root = lerobot_config.get("root", None)
    assert repo_id is not None, "repo_id is required"
    assert root is not None, "root is required"

    action_horizon = int(config["data"].get("action_horizon", 32))
    modality_json_path = resolve_modality_json(args, lerobot_config)

    data_loader, num_batches = load_lerobot_dataset(repo_id, root, action_horizon, modality_json_path, args)

    # compute stats on unified mapping keys
    keys = ["state", "action"]
    stats = {k: normalize.RunningStats() for k in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for k in keys:
            mapped_key = KEY_MAPPINGS[repo_id][k]
            stats[k].update(np.asarray(batch[mapped_key]))

    norm_stats = {KEY_MAPPINGS[repo_id][k]: stats[k].get_statistics() for k in keys}

    out_dir = Path(args.output_dir) / repo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing stats to: {out_dir}")
    normalize.save(str(out_dir), norm_stats)
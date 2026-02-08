#!/usr/bin/env python3

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from wall_x.data.load_lerobot_dataset import KEY_MAPPINGS

import torch

def write_json(path: Path, data: Dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compute_action_statistics(
    action_data_by_robot: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Dict]]:
    """
    Compute statistics (min, q01, q99, max) for each action type and dimension.

    Args:
        action_data_by_robot: Dict[robot_id][action_type] -> list of arrays/lists

    Returns:
        Dict[robot_id][action_type] -> {
            "min": [min for each dim],
            "q01": [quantile 1% for each dim],
            "q99": [quantile 99% for each dim],
            "max": [max for each dim],
            "delta": [max - min for each dim]
            "delta_q99_q01": [q99 - q01 for each dim]
        }
    """
    stats = {}

    for robot_id, action_data in action_data_by_robot.items():
        stats[robot_id] = {}

        for action_type, values_list in action_data.items():
            if not values_list:
                continue

            # Convert to numpy array: shape (num_samples, num_dims)
            try:
                values_array = np.array(values_list)
                if values_array.size == 0:
                    continue

                # Handle both 1D and 2D cases
                if values_array.ndim == 1:
                    values_array = values_array.reshape(-1, 1)
                elif values_array.ndim == 2:
                    pass
                else:
                    logging.warning(
                        f"Unexpected shape for {robot_id}/{action_type}: {values_array.shape}"
                    )
                    continue

                # Compute statistics for each dimension
                min_vals = np.min(values_array, axis=0).tolist()
                max_vals = np.max(values_array, axis=0).tolist()
                q01_vals = np.quantile(values_array, 0.01, axis=0).tolist()
                q99_vals = np.quantile(values_array, 0.99, axis=0).tolist()
                delta_vals = (np.array(max_vals) - np.array(min_vals)).tolist()
                delta_q99_q01_vals = (np.array(q99_vals) - np.array(q01_vals)).tolist()

                stats[robot_id][action_type] = {
                    "min": min_vals,
                    "q01": q01_vals,
                    "q99": q99_vals,
                    "max": max_vals,
                    "delta": delta_vals,
                    "delta_q99_q01": delta_q99_q01_vals,
                }

            except Exception as e:
                logging.warning(
                    f"Error computing statistics for {robot_id}/{action_type}: {e}"
                )
                continue

    return stats


def load_lerobot_dataset(
    repo_id: str,
    trajectory_keys: Dict,
    base_dir: Path,
) -> None:

    # Load local or remote dataset
    dataset = LeRobotDataset(base_dir)

    # Iterate through all data
    frames: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    all_features = dataset.features
    non_image_columns = [col for col in all_features if "image" not in col]

    print(f"Reading the following fields:{non_image_columns}")
    fast_dataset = dataset.hf_dataset.select_columns(non_image_columns)
    
    from pathlib import Path
    from typing import Any, Dict, Union
    def load_mapping_json(path: Union[str, Path]) -> Dict[str, Any]:
        """MVP: read JSON file and return as dict."""
        return json.loads(Path(path).read_text(encoding="utf-8"))
    
    # 1) 读 modality.json
    base_dir = Path(base_dir)
    modality_json = load_mapping_json(base_dir / "meta" / "modality.json")

    # 2) 找到 state/action 对应的 section（通常就是 "state"/"action"，但你用 KEY_MAPPINGS 做一层映射）
    state_section_key = KEY_MAPPINGS[repo_id]["state"]
    action_section_key = KEY_MAPPINGS[repo_id]["action"]

    state_dict = modality_json[state_section_key]   # e.g. modality_json["state"]
    action_dict = modality_json[action_section_key] # e.g. modality_json["action"]
    
    # 3) 收集所有要读的 columns（original_key 去重）
    cols = set()
    for spec in state_dict.values():
        cols.add(spec["original_key"])
    for spec in action_dict.values():
        cols.add(spec["original_key"])
    
    cols = sorted(cols)
    print(f"Reading columns ({len(cols)}): {cols}")

    final_dataset = dataset.hf_dataset.select_columns(cols)
    
    for i in tqdm(range(len(final_dataset))):
        sample = final_dataset[i]

        # ---- 1) 拼 state 向量（按 modality.json 顺序）----
        state_parts = []
        for spec in state_dict.values():
            x = sample[spec["original_key"]]
            x = torch.as_tensor(x)
            if "start" in spec and "end" in spec:
                x = x[spec["start"]:spec["end"]]
            state_parts.append(x.reshape(-1))
        propri = torch.cat(state_parts, dim=0) if state_parts else torch.empty(0)

        # ---- 2) 拼 action 向量（按 modality.json 顺序）----
        action_parts = []
        for spec in action_dict.values():
            x = sample[spec["original_key"]]
            x = torch.as_tensor(x)
            if "start" in spec and "end" in spec:
                x = x[spec["start"]:spec["end"]]
            action_parts.append(x.reshape(-1))
        action = torch.cat(action_parts, dim=0) if action_parts else torch.empty(0)

        # ---- 3) 沿用原 trajectory_keys 逻辑 ----
        for key, action_keys in trajectory_keys.items():
            for action_key, action_range in action_keys.items():
                if key == "action": 
                    if "dummy" in action_key: # 匹配到20维度
                        frames[repo_id][action_key].append([0.0] * int(action_range[1] - action_range[0]))
                    else:
                        frames[repo_id][action_key].append(
                            action[action_range[0]:action_range[1]].cpu().numpy().tolist()
                        )
                else:
                    if "dummy" in action_key: # 匹配到20维度
                        frames[repo_id][action_key].append([0.0] * int(action_range[1] - action_range[0]))
                    else:
                        frames[repo_id][action_key].append(
                            propri[action_range[0]:action_range[1]].cpu().numpy().tolist()
                        )
    return frames


def compute_action_normalizer(
    repo_id: str, trajectory_keys: Dict, base_dir: Path, output_dir: Path
) -> None:
    """
    Compute action normalizer statistics for all robot_ids.
    """
    logging.info("Starting action normalizer computation...")

    frames = load_lerobot_dataset(repo_id, trajectory_keys, base_dir)

    # Compute statistics
    stats = compute_action_statistics(frames)

    # Save statistics for each robot_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # for robot_id, robot_stats in stats.items():
    #     output_file = output_dir / f"{robot_id}_action_stats.json"
    #     write_json(output_file, robot_stats)
    #     logging.info(f"Saved action statistics for {robot_id} to {output_file}")

    # Also save a combined file
    combined_output = output_dir / "all_robots_action_stats.json"
    write_json(combined_output, stats)
    logging.info(f"Saved combined action statistics to {combined_output}")


def main() -> None:

    repo_id = "g1custom"  # your dataset name
    data_root_path = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_refactorized"
    output_stats_dir = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/norm_stats"
    trajectory_keys = {  # your dataset keys
        "action": {
            "observation.ts.actions_joint_robot_position": [0, 17],
            "observation.ts.actions_joint_gripper_position": [17, 19],
            # "dummy_action_joint": [19,20], # 匹配到20维度 不需要 已经硬编码
        },
        "propri": {
            "observation.ts.observations_joint_robot_position": [0, 17],
            "observation.ts.observations_joint_gripper_position": [17, 19],
            # "dummy_state_joint": [19,20], # 匹配到20维度 不需要 已经硬编码
        },
    }

    compute_action_normalizer(
        repo_id, trajectory_keys, data_root_path, output_stats_dir
    )
    logging.info("Action normalizer computation completed.")


if __name__ == "__main__":
    main()
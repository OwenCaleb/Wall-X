#!/usr/bin/env python3

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from wall_x.data.load_lerobot_dataset import KEY_MAPPINGS


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
            "delta": [max - min for each dim],
            "delta_q99_q01": [q99 - q01 for each dim]
        }
    """
    stats = {}

    for robot_id, action_data in action_data_by_robot.items():
        stats[robot_id] = {}

        for action_type, values_list in action_data.items():
            if not values_list:
                continue

            try:
                values_array = np.array(values_list)
                if values_array.size == 0:
                    continue

                if values_array.ndim == 1:
                    values_array = values_array.reshape(-1, 1)
                elif values_array.ndim == 2:
                    pass
                else:
                    logging.warning(
                        f"Unexpected shape for {robot_id}/{action_type}: {values_array.shape}"
                    )
                    continue

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
) -> Dict[str, Dict[str, List]]:
    """
    Load one local LeRobot dataset and collect action/proprio data according to modality.json.
    """
    dataset = LeRobotDataset(base_dir)

    frames: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    def load_mapping_json(path: Path) -> Dict:
        return json.loads(path.read_text(encoding="utf-8"))

    base_dir = Path(base_dir)
    modality_json_path = base_dir / "meta" / "modality.json"
    modality_json = load_mapping_json(modality_json_path)

    state_section_key = KEY_MAPPINGS[repo_id]["state"]
    action_section_key = KEY_MAPPINGS[repo_id]["action"]

    state_dict = modality_json[state_section_key]
    action_dict = modality_json[action_section_key]

    cols = set()
    for spec in state_dict.values():
        cols.add(spec["original_key"])
    for spec in action_dict.values():
        cols.add(spec["original_key"])

    cols = sorted(cols)
    print(f"Reading columns ({len(cols)}): {cols}")

    final_dataset = dataset.hf_dataset.select_columns(cols)

    for i in tqdm(range(len(final_dataset)), desc=base_dir.name):
        sample = final_dataset[i]

        # ---- 1) 拼 state 向量（按 modality.json 顺序）----
        state_parts = []
        for spec in state_dict.values():
            x = sample[spec["original_key"]]
            x = torch.as_tensor(x)
            if "start" in spec and "end" in spec:
                x = x[spec["start"] : spec["end"]]
            state_parts.append(x.reshape(-1))
        propri = torch.cat(state_parts, dim=0) if state_parts else torch.empty(0)

        # ---- 2) 拼 action 向量（按 modality.json 顺序）----
        action_parts = []
        for spec in action_dict.values():
            x = sample[spec["original_key"]]
            x = torch.as_tensor(x)
            if "start" in spec and "end" in spec:
                x = x[spec["start"] : spec["end"]]
            action_parts.append(x.reshape(-1))
        action = torch.cat(action_parts, dim=0) if action_parts else torch.empty(0)

        # ---- 3) 沿用原 trajectory_keys 逻辑 ----
        for key, action_keys in trajectory_keys.items():
            for action_key, action_range in action_keys.items():
                if key == "action":
                    if "dummy" in action_key:
                        frames[repo_id][action_key].append(
                            [0.0] * int(action_range[1] - action_range[0])
                        )
                    else:
                        frames[repo_id][action_key].append(
                            action[action_range[0] : action_range[1]]
                            .cpu()
                            .numpy()
                            .tolist()
                        )
                else:
                    if "dummy" in action_key:
                        frames[repo_id][action_key].append(
                            [0.0] * int(action_range[1] - action_range[0])
                        )
                    else:
                        frames[repo_id][action_key].append(
                            propri[action_range[0] : action_range[1]]
                            .cpu()
                            .numpy()
                            .tolist()
                        )

    return frames


def compute_action_normalizer(
    repo_id: str,
    trajectory_keys: Dict,
    base_dir: Path,
    output_file: Path,
) -> None:
    """
    Compute action normalizer statistics for one dataset and save to output_file.
    """
    logging.info(f"Starting action normalizer computation for: {base_dir}")

    frames = load_lerobot_dataset(repo_id, trajectory_keys, base_dir)
    stats = compute_action_statistics(frames)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_file, stats)

    logging.info(f"Saved action statistics to {output_file}")


def find_valid_annotated_datasets(root_dir: Path) -> List[Path]:
    """
    找到所有同时存在:
      - <A>/<A>_annotated
      - <A>/<A>_annotated_v3.0
    的 annotated 数据集目录，返回 <A>/<A>_annotated 路径列表。
    """
    valid_dirs = []

    for dataset_root in sorted(root_dir.iterdir()):
        if not dataset_root.is_dir():
            continue

        dataset_name = dataset_root.name
        annotated_dir = dataset_root / f"{dataset_name}_annotated"
        annotated_v3_dir = dataset_root / f"{dataset_name}_annotated_v3.0"

        if not (annotated_dir.is_dir() and annotated_v3_dir.is_dir()):
            continue

        modality_json = annotated_dir / "meta" / "modality.json"
        if not modality_json.is_file():
            logging.warning(
                f"Skip {dataset_name}: missing modality.json at {modality_json}"
            )
            continue

        valid_dirs.append(annotated_dir)

    return valid_dirs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    repo_id = "g1custom"
    data_root_dir = Path(
        "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1/lerobot"
    )
    output_stats_dir = Path(
        "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/norm_stats"
    )

    trajectory_keys = {
        "action": {
            "observation.ts.actions_joint_robot_position": [0, 17],
            "observation.ts.actions_joint_gripper_position": [17, 19],
            # "dummy_action_joint": [19,20],
        },
        "propri": {
            "observation.ts.observations_joint_robot_position": [0, 17],
            "observation.ts.observations_joint_gripper_position": [17, 19],
            # "dummy_state_joint": [19,20],
        },
    }

    valid_annotated_dirs = find_valid_annotated_datasets(data_root_dir)
    logging.info(f"Found {len(valid_annotated_dirs)} valid datasets to process.")

    for annotated_dir in valid_annotated_dirs:
        dataset_name = annotated_dir.parent.name
        output_file = output_stats_dir / f"{dataset_name}.json"

        try:
            logging.info(f"Processing dataset: {dataset_name}")
            compute_action_normalizer(
                repo_id=repo_id,
                trajectory_keys=trajectory_keys,
                base_dir=annotated_dir,
                output_file=output_file,
            )
        except Exception as e:
            logging.exception(f"Failed on dataset {dataset_name}: {e}")

    logging.info("All action normalizer computations completed.")


if __name__ == "__main__":
    main()
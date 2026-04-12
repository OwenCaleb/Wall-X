import argparse
import base64
import copy
import json
import os
import urllib.request
from io import BytesIO

import yaml
from PIL import Image

from wall_x.data.load_lerobot_dataset import load_test_dataset
from wall_x.model.action_head import Normalizer


DEFAULT_URL = "http://127.0.0.1:8000/vqa"
DEFAULT_CONFIG = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/workspace/lerobot_example/config_qact_custom.yml"
DEFAULT_EVAL_OUT = "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/save_path_dir/vqa_subtask_eval.jsonl"
DEFAULT_IMAGES = [
    "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old/frame_retarget/sample_000000/000000.jpg",
    "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old/frame_retarget_left/sample_000000/000000.jpg",
    "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old/frame_retarget_right/sample_000000/000000.jpg",
]
DEFAULT_VIEW_NAMES = ["front view", "left wrist view", "right wrist view"]


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def pil_from_any(image_obj):
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")

    try:
        import torch

        if torch.is_tensor(image_obj):
            arr = image_obj.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype("uint8")
            return Image.fromarray(arr).convert("RGB")
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(image_obj, np.ndarray):
            arr = image_obj
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != "uint8":
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype("uint8")
                else:
                    arr = arr.clip(0, 255).astype("uint8")
            return Image.fromarray(arr).convert("RGB")
    except Exception:
        pass

    raise ValueError(f"Unsupported image type for encoding: {type(image_obj)}")


def encode_pil_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_images_base64(image_paths: list) -> list:
    return [encode_image_base64(path) for path in image_paths]


def encode_images_from_objects(image_list: list) -> list:
    return [encode_pil_base64(pil_from_any(x)) for x in image_list]


def post_vqa(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["data"]["model_type"] = config.get("model_type")
    return config


def resolve_runtime_dataset_key(lerobot_cfg: dict) -> str:
    root = str(lerobot_cfg.get("root", "")).rstrip("/")
    if root:
        return os.path.basename(root).replace(".", "_")
    return str(lerobot_cfg.get("repo_id", "dataset")).replace(".", "_")


def resolve_dataset_name(lerobot_cfg: dict) -> str:
    root = str(lerobot_cfg.get("root", "")).rstrip("/")
    if root:
        parent_name = os.path.basename(os.path.dirname(root))
        if parent_name:
            return parent_name
        return os.path.basename(root)
    return str(lerobot_cfg.get("repo_id", "dataset"))


def build_multidataset_normalizers(config: dict, lerobot_configs: list):
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


def extract_prompt_and_gt(full_text: str):
    marker = "<|im_start|>assistant\n"
    idx = full_text.find(marker)
    if idx < 0:
        return None, ""
    prompt = full_text[: idx + len(marker)]
    tail = full_text[idx + len(marker) :]
    end_marker = "<|im_end|>"
    eidx = tail.find(end_marker)
    gt = tail[:eidx].strip() if eidx >= 0 else tail.strip()
    return prompt, gt


def run_dataset_subtask_eval(args):
    config = load_config(args.config)
    config["data"]["generate_vqa_ratio"] = 0
    config["data"]["generate_cot_ratio"] = 0
    config["data"]["generate_subtask_ratio"] = 1

    lerobot_configs = config.get("data", {}).get("lerobot_configs", [])
    if not lerobot_configs:
        raise SystemExit("No lerobot_configs found in config")

    normalizer_action, normalizer_propri = build_multidataset_normalizers(config, lerobot_configs)

    out_path = args.out_jsonl or os.path.join(
        "/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/save_path_dir",
        "vqa_subtask_eval.jsonl",
    )

    records = []
    for lerobot_cfg in lerobot_configs:
        dataset_name = resolve_dataset_name(lerobot_cfg)
        ds_config = copy.deepcopy(config)
        ds_config["norm_stats_path"] = lerobot_cfg.get("norm_stats_path")

        dataset = load_test_dataset(
            ds_config,
            lerobot_cfg,
            normalizer_action,
            normalizer_propri,
            seed=42,
            episodes=lerobot_cfg.get("episodes", [0]),
        )

        n = min(len(dataset), int(args.max_samples_per_dataset))
        print(f"dataset={dataset_name}, eval_samples={n}")
        for i in range(n):
            item = dataset[i]
            prompt, gt = extract_prompt_and_gt(str(item["text"]))
            if not prompt:
                continue

            payload = {
                "task_type": "subtask",
                "raw_prompt": prompt,
                "image_base64_list": encode_images_from_objects(item["image_inputs"]),
                "generation_params": {"max_new_tokens": args.max_new_tokens},
            }
            resp = post_vqa(args.url, payload)
            pred = str(resp.get("answer", "")).strip()

            rec = {
                "dataset": dataset_name,
                "frame_index": int(item.get("frame_index", i)),
                "gt": gt,
                "pred": pred,
                "exact_match": int(gt.strip() == pred.strip()),
            }
            records.append(rec)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved GT/Pred to: {out_path}")
    if records:
        em = sum(r["exact_match"] for r in records)
        print(f"Exact match: {em / len(records):.4f} ({em}/{len(records)})")


def run_interactive(args):
    image_paths = []
    if args.images:
        image_paths = [p for p in args.images.split(",") if p]
    elif args.image:
        image_paths = [args.image]
    else:
        image_paths = list(DEFAULT_IMAGES)

    payload_base = {
        "image_base64_list": encode_images_base64(image_paths),
        "task_type": args.task_type,
        "vqa_type": args.vqa_type,
        "generation_params": {"max_new_tokens": args.max_new_tokens},
    }
    if args.view_names:
        payload_base["view_names"] = [v for v in args.view_names.split(",") if v]

    print("Interactive mode started. Type ':quit' to exit.")
    while True:
        line = input("prompt> ").strip()
        if not line:
            continue
        if line in {":q", ":quit", "quit", "exit"}:
            break

        payload = dict(payload_base)
        if args.task_type == "subtask":
            payload["instruction"] = line
            payload["question"] = ""
        else:
            payload["question"] = line
            payload["instruction"] = args.instruction

        print(post_vqa(args.url, payload))


def run_single(args):
    question = args.question
    if args.task_type in ["vqa", "cot"] and not question:
        question = args.instruction
    if args.task_type in ["vqa", "cot"] and not question:
        raise SystemExit("--question is required for vqa/cot")
    if args.task_type == "subtask" and not args.instruction:
        raise SystemExit("--instruction is required for subtask")

    image_paths = []
    if args.images:
        image_paths = [p for p in args.images.split(",") if p]
    elif args.image:
        image_paths = [args.image]
    else:
        image_paths = list(DEFAULT_IMAGES)

    payload = {
        "image_base64_list": encode_images_base64(image_paths),
        "task_type": args.task_type,
        "instruction": args.instruction,
        "question": question,
        "vqa_type": args.vqa_type,
        "generation_params": {"max_new_tokens": args.max_new_tokens},
    }
    if args.view_names:
        payload["view_names"] = [v for v in args.view_names.split(",") if v]

    print(post_vqa(args.url, payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--images", type=str, default=",".join(DEFAULT_IMAGES))
    parser.add_argument("--view_names", type=str, default=",".join(DEFAULT_VIEW_NAMES))
    parser.add_argument("--task_type", type=str, default="subtask")
    parser.add_argument("--instruction", type=str, default="pick all objects in to the boxes.")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--vqa_type", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument("--interactive", action="store_true", default=True)
    parser.add_argument("--dataset_subtask_eval", action="store_true")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--max_samples_per_dataset", type=int, default=32)
    parser.add_argument("--out_jsonl", type=str, nargs="?", const=DEFAULT_EVAL_OUT, default="")
    args = parser.parse_args()

    if args.dataset_subtask_eval:
        run_dataset_subtask_eval(args)
        return

    if args.interactive:
        run_interactive(args)
        return

    run_single(args)


if __name__ == "__main__":
    main()

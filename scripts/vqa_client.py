import argparse
import base64
import json
import urllib.request

'''

COT TASK
    python scripts/vqa_client.py \
    --url http://127.0.0.1:8000/vqa \
    --image /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old/frame_retarget/sample_000000/000000.jpg \
    --task_type cot \
    --instruction "Place the green and white toy in the right_dark_brown_basket." \
    --question "Place the green and white toy in the right_dark_brown_basket."

SUBTASK TASK
    python scripts/vqa_client.py \
    --url http://127.0.0.1:8000/vqa \
    --image /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old/frame_retarget/sample_000000/000000.jpg \
    --task_type vqa \
    --instruction "pick all objects in to the boxes." \
    --question "What objects are to the RIGHT of the gripper?" \
    --vqa_type spatial

QA TASK
    python scripts/vqa_client.py \
    --url http://127.0.0.1:8000/vqa \
    --image /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old/frame_retarget/sample_000000/000000.jpg \
    --task_type subtask \
    --instruction "pick all objects in to the boxes."
'''

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/vqa")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--task_type", type=str, default="vqa")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--vqa_type", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    payload = {
        "image_base64": encode_image_base64(args.image),
        "task_type": args.task_type,
        "instruction": args.instruction,
        "question": args.question,
        "vqa_type": args.vqa_type,
        "generation_params": {"max_new_tokens": args.max_new_tokens},
    }

    response = post_vqa(args.url, payload)
    print(response)


if __name__ == "__main__":
    main()

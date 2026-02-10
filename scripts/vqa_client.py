import argparse
import base64
import json
import urllib.request

"""
单图也可以
python scripts/vqa_client.py \
  --url http://127.0.0.1:8000/vqa \
  --image /path/front.jpg \
  --task_type vqa \
  --question "What objects are to the RIGHT of the gripper?"
COT TASK
python scripts/vqa_client.py \
  --url http://127.0.0.1:8000/vqa \
    --images /path/front.jpg,/path/left_wrist.jpg,/path/right_wrist.jpg \
    --view_names "front view,left wrist view,right wrist view" \
  --task_type cot \
  --question "Place the green and white toy in the right_dark_brown_basket."

SUBTASK TASK
python scripts/vqa_client.py \
  --url http://127.0.0.1:8000/vqa \
    --images /path/front.jpg,/path/left_wrist.jpg,/path/right_wrist.jpg \
    --view_names "front view,left wrist view,right wrist view" \
  --task_type subtask \
  --instruction "pick all objects in to the boxes."

QA TASK
python scripts/vqa_client.py \
  --url http://127.0.0.1:8000/vqa \
    --images /path/front.jpg,/path/left_wrist.jpg,/path/right_wrist.jpg \
    --view_names "front view,left wrist view,right wrist view" \
  --task_type vqa \
  --instruction "pick all objects in to the boxes." \
  --question "What objects are to the RIGHT of the gripper?" \
  --vqa_type spatial
"""

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def encode_images_base64(image_paths: list) -> list:
    return [encode_image_base64(path) for path in image_paths]


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
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--images", type=str, default="")
    parser.add_argument("--view_names", type=str, default="")
    parser.add_argument("--task_type", type=str, default="vqa")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--vqa_type", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

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
        raise SystemExit("--image or --images is required")

    payload = {
        "image_base64_list": encode_images_base64(image_paths),
        "task_type": args.task_type,
        "instruction": args.instruction,
        "question": question,
        "vqa_type": args.vqa_type,
        "generation_params": {"max_new_tokens": args.max_new_tokens},
    }
    if args.view_names:
        view_names = [v for v in args.view_names.split(",") if v]
        payload["view_names"] = view_names

    response = post_vqa(args.url, payload)
    print(response)


if __name__ == "__main__":
    main()

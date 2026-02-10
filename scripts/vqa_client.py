import argparse
import base64
import json
import urllib.request

'''

COT TASK
    python scripts/vqa_client.py \
        --url http://127.0.0.1:8000/vqa \
        --image /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old/frame_retarget/sample_000000/000000.jpg \
        --question "Place the green and white toy in the right_dark_brown_basket. Please think step by step and answer."

SUBTASK TASK
    python scripts/vqa_client.py \
        --url http://127.0.0.1:8000/vqa \
        --image /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old/frame_retarget/sample_000000/000000.jpg \
        --question "pick all objects in to the boxes. Predict the next action in language."

QA TASK
    python scripts/vqa_client.py \
        --url http://127.0.0.1:8000/vqa \
        --image /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/data/g1_new/lerobot/Teleop_251103_Sort_Anonymous_10Hz_old/frame_retarget/sample_000000/000000.jpg \
        --question "pick all objects in to the boxes. Question (spatial): What objects are to the RIGHT of the gripper? Answer the question based on the observation."

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
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    payload = {
        "image_base64": encode_image_base64(args.image),
        "question": args.question,
        "generation_params": {"max_new_tokens": args.max_new_tokens},
    }

    response = post_vqa(args.url, payload)
    print(response)


if __name__ == "__main__":
    main()

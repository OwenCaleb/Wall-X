import argparse
import base64
import json
import urllib.request

'''
python scripts/vqa_client.py \
  --url http://127.0.0.1:8000/vqa \
  --image /path/to/your.jpg \
  --question "Please think step by step and answer."
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

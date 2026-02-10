import argparse
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

'''
python scripts/vqa_inference.py \
  --model_path /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/models/wallx/wall-oss-flow-copy \
  --config /mnt/nas_ssd/workspace/wenboli/projects/Wall-X/workspace/lerobot_example/config_qact_custom.yml \
  --host 0.0.0.0 \
  --port 8000
'''

class VQAWrapper(object):
    def __init__(self, model_path: str, train_config: dict = None):

        self.device = self._setup_device()
        if train_config is None:
            try:
                with open(os.path.join(model_path, "config.yml"), "r") as f:
                    train_config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                print(f"load train_config.yml fail: {e}")
        self.processor = self._load_processor(train_config["processor_path"])
        self.model = self._load_model(model_path, train_config)

    def _setup_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_processor(self, model_path: str) -> AutoProcessor:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def _load_model(
        self, model_path: str, train_config: dict
    ) -> Qwen2_5_VLMoEForAction:
        model = Qwen2_5_VLMoEForAction.from_pretrained(
            model_path, train_config=train_config
        )
        if self.device == "cuda":
            model = model.to(self.device, dtype=torch.bfloat16)
        else:
            model.to(self.device)
        model.eval()
        return model

    def generate(self, image: Image.Image, text_prompt: str, **kwargs) -> str:
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generation_params = {
            "max_new_tokens": 1024,  # default value, can be overridden by kwargs
            "do_sample": False,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            **kwargs,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_params)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


def build_prompt(task_type: str, instruction: str, question: str, vqa_type: str) -> str:
    role_start_symbol = "<|im_start|>"
    role_end_symbol = "<|im_end|>"
    vision_start_symbol = "<|vision_start|>"
    vision_end_symbol = "<|vision_end|>"
    image_pad_symbol = "<|image_pad|>"

    prologue = f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
    user_request = (
        f"{role_start_symbol}user\nObservation: "
        f"{vision_start_symbol}{image_pad_symbol}{vision_end_symbol}\n"
        f"Instruction:"
    )

    instruction = instruction or ""
    question = question or ""
    vqa_type = vqa_type or ""

    if task_type == "cot":
        text_prompt = "\nPlease think step by step and answer.\n"
        if not instruction:
            instruction = question
        user_message = (
            f"{user_request} {instruction}\n"
            f"Question: {question}{text_prompt}{role_end_symbol}\n"
        )
    elif task_type == "subtask":
        text_prompt = "\nPredict the next action in language.\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
    else:
        vqa_type_text = f" ({vqa_type})" if vqa_type else ""
        text_prompt = "\nAnswer the question based on the observation.\n"
        if not instruction:
            instruction = question
        user_message = (
            f"{user_request} {instruction}\n"
            f"Question{vqa_type_text}: {question}{text_prompt}{role_end_symbol}\n"
        )

    assistant_output = f"{role_start_symbol}assistant\n"
    return prologue + user_message + assistant_output


def load_train_config(model_path: str, config_path: str = None) -> dict:
    if config_path:
        with open(config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(model_path, "config.yml"), "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def decode_image(payload: dict) -> Image.Image:
    if "image_base64" in payload:
        raw = base64.b64decode(payload["image_base64"])
        return Image.open(BytesIO(raw)).convert("RGB")
    if "image_path" in payload:
        return Image.open(payload["image_path"]).convert("RGB")
    raise ValueError("image_base64 or image_path is required")


class VQAServer:
    def __init__(self, model_path: str, config_path: str = None):
        train_config = load_train_config(model_path, config_path)
        self.wrapper = VQAWrapper(model_path=model_path, train_config=train_config)

    def handle(self, payload: dict) -> dict:
        task_type = payload.get("task_type", "vqa")
        instruction = payload.get("instruction", "")
        question = payload.get("question", "")
        vqa_type = payload.get("vqa_type", "")
        if task_type in ["vqa", "cot"] and not question:
            raise ValueError("question is required for vqa/cot")
        if task_type == "subtask" and not instruction:
            raise ValueError("instruction is required for subtask")
        image = decode_image(payload)
        generation_params = payload.get("generation_params", {})
        prompt = build_prompt(task_type, instruction, question, vqa_type)
        answer = self.wrapper.generate(image, prompt, **generation_params)
        return {"answer": answer}


class VQARequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{\"status\":\"ok\"}")

    def do_POST(self):
        if self.path != "/vqa":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
            response = self.server.app.handle(payload)
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)


def serve(model_path: str, config_path: str, host: str, port: int) -> None:
    app = VQAServer(model_path=model_path, config_path=config_path)
    httpd = ThreadingHTTPServer((host, port), VQARequestHandler)
    httpd.app = app
    print(f"VQA server listening on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    serve(args.model_path, args.config, args.host, args.port)
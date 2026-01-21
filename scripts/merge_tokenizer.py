from transformers import AutoProcessor
import os
'''
把“FAST 动作 tokenizer”的离散动作词表，追加进 Qwen2.5-VL 的文本 tokenizer 里，从而让同一个 tokenizer 同时能编码文本 + 机器人动作 token。
'''
processor_path = "/home/liwenbo/projects/VLA/wall-x/Pretrained_models/Qwen2.5-3B-Instruct"
action_tokenizer_path = "/home/liwenbo/projects/VLA/wall-x/fast"
use_fast_tokenizer = True

# 加载 Qwen2.5-VL 的 processor，并用 fast tokenizer 实现（Rust 版）以加速。
processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
# processor 不是 Processor，而是纯 tokenizer 本体 Qwen2TokenizerFast
# print(type(processor)) 
processor.padding_side = "left"

action_tokenizer = AutoProcessor.from_pretrained(
    action_tokenizer_path, trust_remote_code=True
)

new_tokens = ["<|propri|>", "<|action|>"]
new_tokens += [f"<|action_token_{i}|>" for i in range(action_tokenizer.vocab_size)]
num_added_tokens = processor.add_tokens(new_tokens)

begin_idx_token = "<|action_token_0|>"
token_id = processor.convert_tokens_to_ids(begin_idx_token)
processor.init_kwargs["action_token_start_index"] = token_id
processor.init_kwargs["action_token_vocab_size"] = action_tokenizer.vocab_size

new_tokenizer_dir = "/home/liwenbo/projects/VLA/wall-x/fast_new_qwen25"
os.makedirs(new_tokenizer_dir, exist_ok=True)
processor.save_pretrained(new_tokenizer_dir)

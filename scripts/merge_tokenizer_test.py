from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/liwenbo/projects/VLA/wall-x/fast_new_qwen25", use_fast=True)
print("action_token_0 id =", tok.convert_tokens_to_ids("<|action_token_0|>"))
print("action_token_start_index =", tok.init_kwargs.get("action_token_start_index"))
print("action_token_vocab_size =", tok.init_kwargs.get("action_token_vocab_size"))
print("Total vocab size =", len(tok))
print("<action_> in special?", "<action_>" in tok.all_special_tokens)
print("all_special_tokens:", tok.all_special_tokens[:30])

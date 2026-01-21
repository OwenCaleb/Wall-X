import torch
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
import yaml

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader) #安全设置

    # Set model_type in data config if not already set 
    config["data"]["model_type"] = config.get("model_type")
    return config

# Load the model 这个官网默认代码加载模型太慢了
model_path = "/home/liwenbo/projects/VLA/wall-x/Pretrained_models/wall-oss-flow"  # or your local path
config_path = "/home/liwenbo/projects/VLA/wall-x/workspace/lerobot_example/config_qact.yml"
config = load_config(config_path)
model = Qwen2_5_VLMoEForAction.from_pretrained(model_path, train_config=config)
model.eval()

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).bfloat16()

# Your inference code here...


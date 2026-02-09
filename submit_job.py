"""
Usage:
python submit_job.py --train_job_name "test-train-job-api" --description "test train job api" --num_gpu 4

Reference:
https://apiexplorer.ksyun.com/#/api/239/CreateTrainJob/2024-06-12/1246

pip install -U kingsoftcloud-sdk-python

python3 submit_job.py \
  --train_job_name "wenboli_train_wallx_gr1_for_all_20_epoch" \
  --num_gpu 4
"""

import os
import json
from ksyun.common import credential
from ksyun.common.profile.client_profile import ClientProfile
from ksyun.common.profile.http_profile import HttpProfile
from ksyun.common.exception.ksyun_sdk_exception import KsyunSDKException
from ksyun.client.aicp.v20240612 import client as aicp_client, models as aicp_models

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_job_name", type=str, default="test-train-job-api")
parser.add_argument("--description", type=str, default="")
parser.add_argument("--num_gpu", type=int, default=0)
parser.add_argument("--num_cpu", type=int, default=-1)
parser.add_argument("--memory", type=int, default=-1)
parser.add_argument("--train_steps", type=int, default=8000)
parser.add_argument("--emergency_job", action="store_true", default=False)
args = parser.parse_args()

assert args.num_gpu in [0,1,2,3,4,8], "num_gpus must be restricted"
node_config_dict = {
    0: {"num_cpu": 1, "memory": 4},
    1: {"num_cpu": 12, "memory": 112},
    2: {"num_cpu": 25, "memory": 225},
    3: {"num_cpu": 38, "memory": 338},
    4: {"num_cpu": 51, "memory": 451},
    8: {"num_cpu": 103, "memory": 902}
}

if args.num_cpu == -1:
    args.num_cpu = node_config_dict[args.num_gpu]["num_cpu"]
if args.memory == -1:
    args.memory = node_config_dict[args.num_gpu]["memory"]

print(f"Using {args.num_gpu} GPUs, {args.num_cpu} CPUs, {args.memory}GB memory")

if args.emergency_job:
    job_priority = "kaic-high"
else:
    job_priority = "kaic-normal"

config = {
    "TrainJobName": args.train_job_name,
    "ResourcePoolId": "9210ffe0-b529-4ca9-a996-b509c9d7722d",
    "QueueName": "a800-gpu",
    "Priority": job_priority,
    "Description": args.description,
    "Command": f"""echo "[job] start"
echo "[time] $(date +'%Y-%m-%d %H:%M:%S')"

# ---- 环境变量（与你前面调试保持一致）----
export HOME=/mnt/nas_ssd/workspace/wenboli
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 如果你希望完全控制 HF cache（可选）
# export HF_HOME=$HOME/.cache/huggingface
# export HF_HUB_CACHE=$HF_HOME/hub
# export TRANSFORMERS_CACHE=$HF_HOME/transformers
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
# ---- Conda ----
source ~/.bashrc
source /opt/conda/etc/profile.d/conda.sh
conda activate wallx

echo "[diag] CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "[diag] CONDA_PREFIX=$CONDA_PREFIX"
echo "[diag] PATH(head)=$(echo $PATH | tr ':' '\n' | head -n 5)"

echo "[diag] which conda: $(which conda)"
echo "[diag] which python: $(which python)"
echo "[diag] which pip: $(which pip)"
echo "[diag] which accelerate: $(which accelerate)"

python - <<'PY'
import sys, site, os, inspect
print("[diag] sys.executable:", sys.executable)
print("[diag] sys.prefix:", sys.prefix)
print("[diag] CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("[diag] site.getsitepackages:", site.getsitepackages())
try:
    import transformers
    print("[diag] transformers version:", transformers.__version__)
    print("[diag] transformers file:", inspect.getfile(transformers))
except Exception as e:
    print("[diag] import transformers failed:", repr(e))
PY

python -c "import sys; print('python ok', sys.executable)"
python -c "import transformers; print('transformers', transformers.__version__)"

# ---- 代码目录 ----
cd /mnt/nas_ssd/workspace/wenboli/projects/Wall-X

# ---- 训练 ----
/bin/bash workspace/lerobot_example/run.sh

echo "[job] done"
echo "[time] $(date +'%Y-%m-%d %H:%M:%S')"
""",
#     "Command": """cd /mnt/nas_ssd/workspace/xinyusun/Projects/mindon_pi0
# /root/miniconda3/envs/pi05/bin/python scripts/test_tqdm.py""",
    "Framework": "pytorch",
    "ImageSource": "Personal",
    "FrameworkReplicas": {
        "Worker": 0,
        "Chief": 0,
        "Evaluator": 0,
        "PS": 0,
        "Master": 1
    },
    "RestartPolicy": "Never",
    "Envs": [{
            "Name": "PYTHONUNBUFFERED",
            "Value": "1"
        }],
    "SupportTensorboard": False,
    "ImageId": "f2318d00-145f-475c-8aad-2dc8ddaba4cb",
    "ImageRepoId": "autosave-notebook-image",
    "ImageTagId": "autosave-kaic-11db2143-8223-4336-9cab-74f7b6b0c489",
    "GPUType": "GM302",
    "GPUNumber": args.num_gpu,
    "CPUNum": args.num_cpu,
    "Memory": args.memory,
    "StorageConfigs": [{
            "StorageConfigId": "8f1a83d7-85a8-4abf-9c12-123f1fa82fbf",
            "MountPath": "/mnt/nas_ssd/data",
            "StorageConfigType": "DataSet"
        }, {
            "StorageConfigId": "6781b761-abbc-4692-80df-4e02cae92a19"
            "",
            "MountPath": "/mnt/nas_ssd/workspace",
            "StorageConfigType": "DataSet"
        }],
    "AccessType": "QueueMember",
    "MaxRuntime": 72,
    "SelfHealing": True,
    "RunOnCPU": False
}

print(json.dumps(config, indent=4))


try:
    cred = credential.Credential(
        os.environ.get("KSYUN_SECRET_ID", "AKLTM2vQvRsPQCqS18FCbCfE"),
        os.environ.get("KSYUN_SECRET_KEY", "ON1xReVUBebCZE5k7W2pgEUZTZB28JezOhiigoho")
    )

    httpProfile = HttpProfile()
    httpProfile.endpoint = "aicp.api.ksyun.com"
    httpProfile.reqMethod = "POST"
    httpProfile.reqTimeout = 60
    httpProfile.scheme = "http"
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile

    aicpClient = aicp_client.AicpClient(cred, "cn-northwest-3", profile=clientProfile)
    print(aicpClient.call_json("CreateTrainJob", config))
except KsyunSDKException as err:
    print(err)
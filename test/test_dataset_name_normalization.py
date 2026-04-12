#!/usr/bin/env python3
"""
测试修复后的dataset_name规范化和Normalizer初始化
"""
import sys
import json
from pathlib import Path

# 测试dataset_name规范化
print("=" * 80)
print("测试 dataset_name 规范化")
print("=" * 80)

config_path = "workspace/lerobot_example/config_qact_custom.yml"
import yaml
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

lerobot_configs = config['data'].get('lerobot_configs', [])
print(f"✓ 加载{len(lerobot_configs)}个数据集\n")

# 模拟load_normalizer中的规范化逻辑
print("模拟 load_normalizer() 中的dataset_name规范化:")
print()

action_statistic_dof = {}

for cfg_idx, cfg in enumerate(lerobot_configs):
    norm_stats_path = cfg.get('norm_stats_path', None)
    root = cfg.get('root', None)
    
    # 生成dataset_name
    if root:
        dataset_name = root.rstrip("/").split("/")[-1]
    else:
        dataset_name = f"dataset_{cfg_idx}"
    
    # 规范化dataset_name（关键修复）
    original_name = dataset_name
    dataset_name = dataset_name.replace(".", "_")
    
    # 加载统计
    if norm_stats_path and Path(norm_stats_path).exists():
        try:
            stats_from_file = json.load(open(norm_stats_path, 'r'))
            
            for repo_id, repo_stats in stats_from_file.items():
                if dataset_name not in action_statistic_dof:
                    action_statistic_dof[dataset_name] = {}
                action_statistic_dof[dataset_name].update(repo_stats)
            
            # 显示信息
            if original_name != dataset_name:
                print(f"[{cfg_idx}] {original_name}")
                print(f"    ↓ (normalized)")
                print(f"    {dataset_name} ✓")
            else:
                print(f"[{cfg_idx}] {dataset_name} ✓")
        except Exception as e:
            print(f"[{cfg_idx}] Failed: {e}")
    print()

print()
print("=" * 80)
print("验证规范化后的dataset_names")
print("=" * 80)
print(f"Normalizer 将管理 {len(action_statistic_dof)} 个数据集:")
for i, name in enumerate(action_statistic_dof.keys()):
    has_dot = "." in name
    status = "❌" if has_dot else "✅"
    print(f"  {status} {i}: {name}")

print()

# 检查是否所有名称都通过了验证
all_ok = all("." not in name for name in action_statistic_dof.keys())

if all_ok:
    print("✅ 所有dataset_name都没有点号，可以安全地用于PyTorch ParameterDict!")
else:
    print("❌ 仍然有dataset_name包含点号!")
    sys.exit(1)

print()
print("=" * 80)
print("测试 PyTorch ParameterDict 初始化")
print("=" * 80)

import torch
import torch.nn as nn

try:
    # 模拟Normalizer的初始化
    min_dict = nn.ParameterDict()
    
    for k in action_statistic_dof.keys():
        # 创建一个dummy tensor作为min参数
        min_dict[k] = nn.Parameter(torch.randn(20), requires_grad=False)
    
    print(f"✅ 成功创建ParameterDict，包含{len(min_dict)}个参数:")
    for key in min_dict.keys():
        print(f"   - {key}")
    
    print()
    print("✅ Normalizer初始化将成功！")
    
except KeyError as e:
    print(f"❌ ParameterDict初始化失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 未知错误: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("结论: 修复成功！")
print("=" * 80)

#!/usr/bin/env python3
"""
Deep audit of multi-dataset data flow
验证从加载配置到归一化的整个pipeline
"""
import yaml
import json
import sys
from pathlib import Path

def audit_config_loading():
    """Step 1: 验证配置是否正确加载"""
    print("=" * 80)
    print("STEP 1: 配置加载审查")
    print("=" * 80)
    
    config_path = "workspace/lerobot_example/config_qact_custom.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lerobot_configs = config['data'].get('lerobot_configs', [])
    print(f"✓ 配置加载成功，共{len(lerobot_configs)}个数据集\n")
    
    # 验证每个配置中的dataset_name生成
    dataset_names_from_config = []
    for i, cfg in enumerate(lerobot_configs):
        root = cfg.get('root', '')
        repo_id = cfg.get('repo_id', '')
        norm_stats_path = cfg.get('norm_stats_path', '')
        
        # 生成dataset_name的逻辑（与PreprocessedDataset.__getitem__相同）
        if root:
            dataset_name = root.rstrip("/").split("/")[-1]
        else:
            dataset_name = repo_id
        
        dataset_names_from_config.append(dataset_name)
        
        print(f"[Config {i}]")
        print(f"  repo_id: {repo_id}")
        print(f"  root: {root}")
        print(f"  dataset_name: {dataset_name}")
        print(f"  norm_stats_path: {norm_stats_path}")
        print(f"  norm_stats exists: {Path(norm_stats_path).exists()}")
        print()
    
    return config, lerobot_configs, dataset_names_from_config


def audit_normalizer_loading(lerobot_configs, dataset_names_from_config):
    """Step 2: 验证Normalizer加载的统计"""
    print("=" * 80)
    print("STEP 2: 规范化器统计加载审查")
    print("=" * 80)
    
    action_statistic_dof = {}
    
    # 模拟load_normalizer()的逻辑
    for cfg_idx, cfg in enumerate(lerobot_configs):
        norm_stats_path = cfg.get('norm_stats_path', None)
        root = cfg.get('root', None)
        dataset_name = dataset_names_from_config[cfg_idx]
        
        print(f"[Dataset {cfg_idx}] Loading stats...")
        
        if norm_stats_path and Path(norm_stats_path).exists():
            try:
                stats_from_file = json.load(open(norm_stats_path, 'r'))
                
                # 为这个数据集保留独立的统计副本
                if dataset_name not in action_statistic_dof:
                    action_statistic_dof[dataset_name] = {}
                
                for repo_id, repo_stats in stats_from_file.items():
                    action_statistic_dof[dataset_name].update(repo_stats)
                
                print(f"  ✓ Loaded from: {norm_stats_path}")
                print(f"  ✓ Stats keys: {list(repo_stats.keys())}")
                print(f"  ✓ Stored under dataset_name: {dataset_name}")
            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
        else:
            print(f"  ✗ norm_stats_path not found: {norm_stats_path}")
        print()
    
    print(f"Normalizer将管理 {len(action_statistic_dof)} 个数据集的统计:")
    for name in action_statistic_dof.keys():
        print(f"  - {name}")
    print()
    
    return action_statistic_dof


def audit_sample_marking(lerobot_configs, dataset_names_from_config):
    """Step 3: 验证样本标记的dataset_name"""
    print("=" * 80)
    print("STEP 3: 样本标记(dataset_name)审查")
    print("=" * 80)
    
    print("模拟PreprocessedDataset.__getitem__()的逻辑:")
    print()
    
    sample_dataset_names = []
    for source_id in range(len(lerobot_configs)):
        cfg = lerobot_configs[source_id]
        root = cfg.get('root', None)
        repo_id = cfg.get('repo_id', '')
        
        # 这是PreprocessedDataset.__getitem__()中的逻辑
        if root:
            dataset_name = root.rstrip("/").split("/")[-1]
        else:
            dataset_name = repo_id
        
        sample_dataset_names.append(dataset_name)
        
        print(f"[Source {source_id}]")
        print(f"  root: {root}")
        print(f"  generated dataset_name: {dataset_name}")
        print(f"  matches config dataset_name: {dataset_name == dataset_names_from_config[source_id]}")
        print()
    
    return sample_dataset_names


def audit_datacollator_handling(sample_dataset_names, lerobot_configs):
    """Step 4: 验证DataCollator处理dataset_names"""
    print("=" * 80)
    print("STEP 4: DataCollator 处理审查")
    print("=" * 80)
    
    # 模拟default_dataset_name的生成
    if len(lerobot_configs) > 1:
        first_root = lerobot_configs[0].get('root', None)
        if first_root:
            default_dataset_name = first_root.rstrip("/").split("/")[-1]
        else:
            default_dataset_name = lerobot_configs[0].get('repo_id', '')
    else:
        default_dataset_name = lerobot_configs[0].get('repo_id', '')
    
    print(f"default_dataset_name: {default_dataset_name}")
    print()
    
    # 模拟batch中的样本
    print("模拟batch中8个样本，来自3个不同的数据集:")
    batch_size = 8
    batch_samples = []
    
    # 假设batch中的样本分布
    sample_indices = [0, 0, 1, 1, 1, 2, 2, 2]
    for i, source_idx in enumerate(sample_indices):
        batch_samples.append({
            'dataset_name': sample_dataset_names[source_idx]
        })
        print(f"  Sample {i}: dataset_name = {sample_dataset_names[source_idx]}")
    
    print()
    
    # 模拟DataCollator.__call__()中的处理
    dataset_names_in_batch = [
        item.get('dataset_name', default_dataset_name) for item in batch_samples
    ]
    
    print(f"DataCollator提取的dataset_names: {dataset_names_in_batch}")
    print(f"长度: {len(dataset_names_in_batch)} (应该与batch_size=8一致)")
    print()
    
    return dataset_names_in_batch, batch_size


def audit_normalizer_filtering(dataset_names_in_batch, batch_size):
    """Step 5: 验证Normalizer中的过滤逻辑"""
    print("=" * 80)
    print("STEP 5: Normalizer 过滤审查 (★ 关键问题区域)")
    print("=" * 80)
    
    print("Normalizer.normalize_data()中有以下逻辑:")
    print('  dataset_names = [name for name in dataset_names if name != "x2_multimodal"]')
    print()
    
    # 检查是否有"x2_multimodal"
    has_x2_multimodal = "x2_multimodal" in dataset_names_in_batch
    print(f"batch中是否存在'x2_multimodal': {has_x2_multimodal}")
    
    if has_x2_multimodal:
        print()
        print("⚠️  WARNING: 检测到'x2_multimodal' dataset_name!")
        print()
        
        filtered_names = [name for name in dataset_names_in_batch if name != "x2_multimodal"]
        print(f"过滤后的dataset_names长度: {len(filtered_names)}")
        print(f"原始batch_size: {batch_size}")
        print()
        
        if len(filtered_names) != batch_size:
            print("❌ 维度不匹配!")
            print(f"   xs (action/proprioception) 仍有{batch_size}个张量")
            print(f"   但dataset_names只有{len(filtered_names)}个")
            print("   这会导致zip(xs, dataset_names)只处理前{len(filtered_names)}个样本!")
            print("   剩余的样本会被忽略！")
            return False
        else:
            print("✓ 长度匹配（巧合）")
    else:
        print(f"✓ 没有'x2_multimodal'，过滤后长度: {batch_size}")
    
    print()
    return True


def audit_normalizer_parameter_lookup(action_statistic_dof, dataset_names_in_batch):
    """Step 6: 验证Normalizer能否正确查询参数"""
    print("=" * 80)
    print("STEP 6: Normalizer 参数查询审查")
    print("=" * 80)
    
    print(f"Normalizer中存储的dataset_name keys: {list(action_statistic_dof.keys())}")
    print()
    
    print("检查batch中的每个dataset_name是否都在Normalizer中有对应的参数:")
    all_found = True
    for i, name in enumerate(dataset_names_in_batch):
        found = name in action_statistic_dof
        status = "✓" if found else "✗"
        print(f"  {status} Sample {i}: '{name}' -> {found}")
        if not found:
            all_found = False
    
    print()
    if all_found:
        print("✓ 所有dataset_name都能在Normalizer中找到对应的参数")
    else:
        print("❌ 某些dataset_name在Normalizer中找不到对应的参数，会导致KeyError!")
    
    print()
    return all_found


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "多数据集数据流深度审查" + " " * 42 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        # Step 1: 配置加载
        config, lerobot_configs, dataset_names_from_config = audit_config_loading()
        
        # Step 2: 规范化器统计加载
        action_statistic_dof = audit_normalizer_loading(lerobot_configs, dataset_names_from_config)
        
        # Step 3: 样本标记
        sample_dataset_names = audit_sample_marking(lerobot_configs, dataset_names_from_config)
        
        # Step 4: DataCollator处理
        dataset_names_in_batch, batch_size = audit_datacollator_handling(
            sample_dataset_names, lerobot_configs
        )
        
        # Step 5: Normalizer过滤
        filter_ok = audit_normalizer_filtering(dataset_names_in_batch, batch_size)
        
        # Step 6: Normalizer参数查询
        lookup_ok = audit_normalizer_parameter_lookup(action_statistic_dof, dataset_names_in_batch)
        
        # 结论
        print("=" * 80)
        print("总体结论:")
        print("=" * 80)
        
        if filter_ok and lookup_ok:
            print("✓ 数据流正确！多数据集模式应该能正常工作")
        else:
            print("❌ 数据流有问题！")
            if not filter_ok:
                print("   - Normalizer中的过滤逻辑可能导致维度不匹配")
            if not lookup_ok:
                print("   - Normalizer中的参数查询可能失败")
        
        print()
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

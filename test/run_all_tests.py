#!/usr/bin/env python3
"""
一键运行所有验证脚本
验证多数据集系统的完整性
"""
import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print()
    print("=" * 100)
    print(f"✓ {description}")
    print("=" * 100)
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, cwd="/mnt/nas_ssd/workspace/wenboli/projects/Wall-X")
        if result.returncode == 0:
            print()
            print(f"✅ {description} - 成功")
            return True
        else:
            print()
            print(f"❌ {description} - 失败 (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ 执行错误: {e}")
        return False

def main():
    print()
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 20 + "多数据集系统完整验证" + " " * 56 + "║")
    print("╚" + "=" * 98 + "╝")
    print()
    print("运行环境:")
    print(f"  工作目录: /mnt/nas_ssd/workspace/wenboli/projects/Wall-X")
    print(f"  Python: {sys.version.split()[0]}")
    print()
    
    tests = [
        (
            "python test/test_dataset_name_normalization.py",
            "TEST 1/3: 数据集名称规范化验证 (点号处理)"
        ),
        (
            "python test/test_multidataset_flow.py",
            "TEST 2/3: 多数据集完整数据流验证"
        ),
        (
            "python test/verify_normalizer_multidataset_logic.py 2>&1 | tail -50",
            "TEST 3/3: Normalizer多数据集逻辑深度验证"
        ),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc.split("/")[0].strip(), success))
    
    # 总结
    print()
    print()
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 35 + "验证总结" + " " * 53 + "║")
    print("╚" + "=" * 98 + "╝")
    print()
    
    all_passed = True
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
        if not success:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ 所有验证通过!")
        print()
        print("系统状态:")
        print("  ✅ 配置完整")
        print("  ✅ 统计独立")
        print("  ✅ 样本标记正确")
        print("  ✅ 批处理正确")
        print("  ✅ Normalizer逻辑正确")
        print("  ✅ 点号规范化完成")
        print()
        print("可以立即启动多数据集训练:")
        print("  python train_qact.py --config workspace/lerobot_example/config_qact_custom.yml")
        print()
        return 0
    else:
        print("❌ 有验证失败")
        print()
        print("请检查:")
        print("  1. Python依赖是否完整")
        print("  2. 配置文件是否存在")
        print("  3. 查看各个测试脚本的详细输出")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())

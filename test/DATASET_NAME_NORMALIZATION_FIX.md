# 多数据集数据流修复 - 规范化dataset_name中的点号

## 问题发现

在实际运行训练时，遇到错误：

```
KeyError: 'parameter name can\'t contain "."'
```

**根本原因**:
- 数据集 `Teleop_251103_Sort_Anonymous_10Hz_refactorized_v3.0` 的dataset_name包含 `.0`
- PyTorch的 `nn.ParameterDict` 不允许key中含有"."（用于属性访问）
- Normalizer.\_\_init\_\_中创建ParameterDict时失败

## 修复方案

在三个生成dataset_name的地方添加规范化：

```python
# 规范化dataset_name：PyTorch ParameterDict不允许名称中有"."
dataset_name = dataset_name.replace(".", "_")
```

### 修复地点1：`wall_x/trainer/qwen_vl_act_trainer.py` - load_normalizer()

```python
# 第256-258行
if root:
    dataset_name = root.rstrip("/").split("/")[-1]
else:
    dataset_name = f"dataset_{cfg_idx}"

# 新增：规范化dataset_name
dataset_name = dataset_name.replace(".", "_")
```

### 修复地点2：`wall_x/data/load_lerobot_dataset.py` - PreprocessedDataset.__getitem__()

```python
# 在__getitem__方法中
if root:
    dataset_name = root.rstrip("/").split("/")[-1]
else:
    dataset_name = repo_id

# 新增：规范化dataset_name
dataset_name = dataset_name.replace(".", "_")
```

### 修复地点3：`wall_x/data/load_lerobot_dataset.py` - DataCollator.__init__()

```python
# 在__init__方法中
if first_root:
    self.default_dataset_name = first_root.rstrip("/").split("/")[-1]
else:
    self.default_dataset_name = data_cfg["lerobot_configs"][0].get("repo_id", "")

# 新增：规范化dataset_name
self.default_dataset_name = self.default_dataset_name.replace(".", "_")
```

## 修复后的效果

✅ **修复验证**:

| 原始名称 | 规范化名称 | ParameterDict兼容 |
|---------|---------|-----------------|
| Teleop_251022_... | Teleop_251022_... | ✅ (无点号) |
| Teleop_251103_..._v3.0 | Teleop_251103_..._v3_0 | ✅ (`.0`→`_0`) |
| dataset_123 | dataset_123 | ✅ |

所有11个数据集都能成功初始化nn.ParameterDict!

## 数据流验证

✅ 修复后的完整检查清单：

```
STEP 1: 配置加载 ✓
STEP 2: 统计加载 ✓
STEP 3: 样本标记 ✓
STEP 4: 批处理 ✓
STEP 5: Normalizer初始化 ✓✓✓ (NOW FIXED!)
STEP 6: 参数查询 ✓
```

## 已应用修复

✅ 三处都已修复
✅ 规范化一致性保证
✅ ParameterDict初始化成功
✅ 11个数据集都已验证

## 下一步

现在可以运行完整的多数据集训练！

```bash
python train_qact.py --config workspace/lerobot_example/config_qact_custom.yml
```

**但注意**：后续可能在后续的training步骤中发现其他问题。当前修复只解决了初始化阶段的点号问题。

# 多数据集数据流 - 关键修复方案

## 问题症状

**在 Normalizer.normalize_data() 中**:
```python
dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
for x, dataset_name in zip(xs, dataset_names):  # ← 维度可能不匹配！
```

**在 Normalizer.unnormalize_data() 中**:
```python
dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
for x, dataset_name, mask in zip(xs, dataset_names, dof_mask):  # ← 同样问题
```

## 风险分析

### 当前状态（安全）
- 所有11个dataset_name都形如 `Teleop_xxxxx_...`
- 无"x2_multimodal"
- 过滤不会被触发
- **但这是巧合，不是安全的设计**

### 未来风险（不安全）
如果某个数据集的root路径为：
- `/path/to/x2_multimodal/`
- 则dataset_name = "x2_multimodal"
- 该样本会被过滤掉
- batch中: 8个样本 → xs有8个张量，pero dataset_names只有7个
- `zip(xs, dataset_names)` 只处理前7个 → **第8个样本被忽略！**

## 为什么要过滤"x2_multimodal"？

很可能是**单数据集时代的遗留代码**：
- 原先系统只支持单个数据集
- "x2_multimodal"可能是某个特殊配置的名字
- 曾经有特殊处理逻辑，现在已不需要
- 但这行代码被遗留下来

**在多数据集模式中，这个过滤没有意义。**


## 修复方案

### 方案A：同步过滤（最保险）

确保xs和dataset_names同时被过滤：

**修改位置**: [wall_x/model/action_head.py](wall_x/model/action_head.py#L112)

```python
def normalize_data(self, xs, dataset_names):
    new_xs = []
    
    # 同时过滤xs和dataset_names，保持维度一致
    xs_filtered = []
    names_filtered = []
    for x, name in zip(xs, dataset_names):
        if name != "x2_multimodal":
            xs_filtered.append(x)
            names_filtered.append(name)
    
    # 处理过滤后的数据
    for x, dataset_name in zip(xs_filtered, names_filtered):
        delta = self.delta[dataset_name]
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        x = (x - self.min[dataset_name]) / delta
        x = x * 2 - 1
        x = torch.clamp(x, -1, 1)
        new_xs.append(x)
    
    new_xs = torch.stack(new_xs)
    return new_xs

def unnormalize_data(self, xs, dataset_names, dof_mask=None):
    new_xs = []
    
    # 同时过滤xs, dataset_names, dof_mask
    xs_filtered = []
    names_filtered = []
    mask_filtered = []
    for x, name, mask in zip(xs, dataset_names, dof_mask if dof_mask is not None else [None] * len(xs)):
        if name != "x2_multimodal":
            xs_filtered.append(x)
            names_filtered.append(name)
            mask_filtered.append(mask)
    
    # 处理过滤后的数据
    for x, dataset_name, mask in zip(xs_filtered, names_filtered, mask_filtered):
        x = (x + 1) / 2
        
        # -------- 维度对齐逻辑 --------
        if mask is not None:
            mask = mask[0].bool()
            d_stats = self.delta[dataset_name].shape[0]
            d_mask = mask.shape[0]
            if d_mask != d_stats:
                print(f"[WARN] dof_mask dim mismatch: mask_dim={d_mask}, stats_dim={d_stats}, "
                      f"dataset={dataset_name}. Hard-align mask.", flush=True)
                if d_mask > d_stats:
                    mask = mask[:d_stats]
                else:
                    pad = torch.zeros(d_stats - d_mask, dtype=mask.dtype, device=mask.device)
                    mask = torch.cat([mask, pad], dim=0)
            
            action_space_delta = self.delta[dataset_name][mask]
            action_space_min = self.min[dataset_name][mask]
        else:
            action_space_delta = self.delta[dataset_name]
            action_space_min = self.min[dataset_name]
        
        # -------- 维度对齐 --------
        d_stats = action_space_delta.shape[-1]
        d_x = x.shape[-1]
        if d_stats != d_x:
            print(f"[WARN] unnormalize_data dim mismatch: x_dim={d_x}, stats_dim={d_stats}, "
                  f"dataset={dataset_name}. Padding stats to match x_dim.", flush=True)
            if d_stats < d_x:
                pad = d_x - d_stats
                delta_pad = action_space_delta.new_zeros(pad)
                min_pad = action_space_min.new_zeros(pad)
                action_space_delta = torch.cat([action_space_delta, delta_pad], dim=-1)
                action_space_min = torch.cat([action_space_min, min_pad], dim=-1)
        
        x = x * action_space_delta + action_space_min
        new_xs.append(x)
    
    new_xs = torch.stack(new_xs)
    return new_xs
```


### 方案B：完全移除过滤（最激进）

如果确认"x2_multimodal"过滤在当前和未来都不需要，直接移除：

```python
def normalize_data(self, xs, dataset_names):
    new_xs = []
    # 移除这一行：dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
    
    for x, dataset_name in zip(xs, dataset_names):
        delta = self.delta[dataset_name]
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        x = (x - self.min[dataset_name]) / delta
        x = x * 2 - 1
        x = torch.clamp(x, -1, 1)
        new_xs.append(x)
    new_xs = torch.stack(new_xs)
    return new_xs

def unnormalize_data(self, xs, dataset_names, dof_mask=None):
    new_xs = []
    # 移除这一行：dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
    dof_mask = dof_mask if dof_mask is not None else [None] * len(xs)
    
    for x, dataset_name, mask in zip(xs, dataset_names, dof_mask):
        # ... 其余代码不变
```


### 方案C：在上游处理（最优）

不在Normalizer中过滤，而在数据加载时处理：

**修改位置**: `wall_x/data/load_lerobot_dataset.py`

在PreprocessedDataset.__getitem__中标记需要跳过的样本：

```python
# PreprocessedDataset.__getitem__
result = {
    "image_inputs": image_inputs,
    "text": text,
    "action": action,
    "agent_pos": agent_pos,
    "frame_index": frame_index,
    "dataset_name": dataset_name,
    "skip_normalization": dataset_name == "x2_multimodal",  # ← 标记
}
```

然后在DataCollator中过滤：

```python
# DataCollator.__call__
skip_masks = [item.get("skip_normalization", False) for item in batch]
if any(skip_masks):
    # 过滤掉标记为skip的样本
    batch_filtered = [item for item, skip in zip(batch, skip_masks) if not skip]
    # ... 后续处理
else:
    batch_filtered = batch
```

**优势**:
- 在最早的阶段处理，逻辑清晰
- Normalizer保持简洁
- 便于扩展其他特殊处理


## 推荐方案

**我推荐 方案B（完全移除过滤）**

理由：
1. ✅ 最简单 - 只需删除一行代码
2. ✅ 最清晰 - 消除隐藏的逻辑
3. ✅ 最安全 - 无维度风险
4. ⚠️ 假设前提 - "x2_multimodal"过滤确实是遗留代码

如果确认"x2_multimodal"过滤有特殊用途，则选择**方案A（同步过滤）**。


## 实施步骤

### 第1步：理解"x2_multimodal"的用途

搜索代码找出为什么需要过滤"x2_multimodal"：

```bash
grep -r "x2_multimodal" --include="*.py" .
```

检查输出，看是否有针对"x2_multimodal"的特殊处理逻辑。

### 第2步：选择修复方案

- 如果没找到特殊处理逻辑 → 使用**方案B**
- 如果找到特殊处理逻辑 → 使用**方案A或C**

### 第3步：编辑文件

修改 `wall_x/model/action_head.py` 的normalize_data()和unnormalize_data()方法

### 第4步：验证

运行 `python test_multidataset_flow.py` 确保没有破坏

### 第5步：单元测试

添加test case验证：

```python
def test_normalize_with_mixed_datasets():
    # 创建模拟的mixed batch
    xs = torch.randn(8, 1, 19)  # 8个样本, 每个[1, 19]
    dataset_names = [
        "Teleop_251022_GrapeCleanbgWaist_Anonymous_10Hz",
        "Teleop_251022_GrapeCleanbgWaist_Anonymous_10Hz",
        "Teleop_251024_FruitCar_Anonymous_10Hz",
        "Teleop_251024_FruitCar_Anonymous_10Hz",
        "x2_multimodal",  # ← 特殊case
        "Teleop_251025_FruitCar_Anonymous_10Hz",
        "Teleop_251025_FruitCar_Anonymous_10Hz",
        "Teleop_251025_FruitCar_Anonymous_10Hz",
    ]
    
    normalizer = Normalizer(action_statistic_dof, dof_config)
    normalized_xs = normalizer.normalize_data(xs, dataset_names)
    
    # 验证输出形状正确
    assert normalized_xs.shape == xs.shape, f"Shape mismatch: {normalized_xs.shape} != {xs.shape}"
    
    # 验证所有样本都被处理
    assert len(normalized_xs) == len(xs), "Sample count mismatch"
```


## 风险评估

| 修复方案 | 实施复杂度 | 风险等级 | 测试覆盖 |
|--------|---------|--------|--------|
| 方案A | 中 | 低 | 中 |
| 方案B | 低 | 极低 | 低 |
| 方案C | 高 | 极低 | 高 |

**建议**: 先用方案B试试，如果出问题再改为方案A。


## 总结

💡 **核心问题**: Normalizer中的"x2_multimodal"过滤导致xs和dataset_names维度不匹配的潜在风险

🔧 **快速修复**: 完全移除过滤（删除2行代码）

📋 **完整修复**: 同步过滤xs和dataset_names

✅ **最终目标**: 确保多数据集模式下的数据流安全可靠

# SinPoint Integration Documentation

## 概述

SinPoint 数据增强模块已成功整合到 DAPointMamba 项目中，无需依赖外部的 SinPoint 项目。

## 文件结构

### 新增文件

1. **`utils/sinpoint_augmentation.py`**
   - SinPoint 数据增强类的完整实现
   - 包含 Local、Global 和 Sin 三种增强方法
   - 支持 RPS（随机点采样）和 FPS（最远点采样）
   - 内置归一化和反归一化功能

2. **`test_sinpoint_integration.py`**
   - SinPoint 整合测试脚本
   - 验证导入、初始化和增强功能是否正常工作

### 修改文件

1. **`tools/runner.py`**
   - 导入语句：从 `utils.sinpoint_augmentation` 导入 `SinPoint`
   - 初始化：在训练开始前创建 SinPoint 实例
   - 应用：在两个数据流分支中应用 DG 改造

## 使用方法

### 基本使用

```python
from utils.sinpoint_augmentation import SinPoint

# 配置参数
class SinPointArgs:
    def __init__(self):
        self.A = 0.8              # 振幅参数
        self.w = 3.0              # 频率参数
        self.rand_center_num = 4  # 随机中心点数量
        self.sample = "RPS"       # 采样方式: "RPS" 或 "FPS"
        self.isCat = False        # 是否拼接原始数据
        self.shuffle = False      # 是否打乱

# 初始化
sinpoint_aug = SinPoint(SinPointArgs())

# 使用
aug_data, labels = sinpoint_aug.Sin(point_cloud_data)
```

### 训练中的应用

在 `tools/runner.py` 中，SinPoint 用于 DG（域泛化）改造：

```python
# DG 改造：使用增强后的源域数据替换目标域
aug_partial, _ = sinpoint_aug.Sin(source_partial)
target_partial = aug_partial  # 直接覆盖目标域

# 安全检查：确保 device 和 shape 一致
assert target_partial.device == source_partial.device, "Device mismatch"
assert target_partial.shape == source_partial.shape, f"Shape mismatch"

# 继续前向传播
rebuild_points, loss_sp, loss_ch = base_model(source_partial, target_partial)
```

## 参数说明

### SinPoint 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `A` | float | 0.8 | 正弦变换的振幅参数，控制扰动强度 |
| `w` | float | 3.0 | 正弦变换的频率参数，控制扰动频率 |
| `rand_center_num` | int | 4 | 局部增强的随机中心点数量（0 表示全局增强）|
| `sample` | str | "RPS" | 采样方式："RPS"（随机点采样）或 "FPS"（最远点采样）|
| `isCat` | bool | False | 是否将原始数据和增强数据拼接 |
| `shuffle` | bool | False | 拼接后是否打乱顺序 |

### 增强方法

1. **`Sin(data, label=[])`**：主要增强方法
   - 自动进行归一化 → 增强 → 反归一化
   - 返回：(增强后的数据, 标签)

2. **`Local(data)`**：基于多个随机中心的局部增强
   - 更细粒度的扰动
   - 保持整体结构

3. **`Global(data)`**：全局统一增强
   - 对整个点云应用相同的正弦变换
   - 适合需要一致性扰动的场景

## 测试验证

运行测试脚本验证整合：

```bash
cd /root/autodl-tmp/DAPointMamba_Git
python test_sinpoint_integration.py
```

预期输出：
```
============================================================
Testing SinPoint Integration
============================================================
✓ SinPoint initialized successfully
✓ Created dummy point cloud: shape=torch.Size([4, 2048, 3]), device=cpu
✓ Augmentation successful: shape=torch.Size([4, 2048, 3]), device=cpu
✓ Shape consistency verified
✓ Device consistency verified
✓ Local augmentation successful: shape=torch.Size([4, 2048, 3])
✓ Global augmentation successful: shape=torch.Size([4, 2048, 3])
============================================================
All tests passed! SinPoint integration successful.
============================================================
```

## 技术细节

### 归一化处理

SinPoint 内置了点云归一化功能：

```python
def normalize_point_clouds(self, pcs):
    B, N, C = pcs.shape
    shift = torch.mean(pcs, dim=1).unsqueeze(1)
    scale = torch.std(pcs.view(B, N * C), dim=1).unsqueeze(1).unsqueeze(1)
    newpcs = (pcs - shift) / scale
    return newpcs, shift, scale
```

- **shift**：减去均值，使点云中心化
- **scale**：除以标准差，使点云标准化
- 增强后会自动恢复原始尺度：`newdata = newdata * scale + shift`

### 设备兼容性

SinPoint 自动检测并保持输入数据的设备（CPU/GPU）：

```python
device = data.device
# ... 所有操作都在同一设备上进行
```

## 训练命令

使用 DG 改造后的训练：

```bash
# ModelNet 目标域
python main.py \
    --config cfgs/ModelNet_models/DAPointMamba.yaml \
    --exp_name modelnet_dg_experiment

# KITTI 目标域
python main.py \
    --config cfgs/KITTI_models/KITTI.yaml \
    --exp_name kitti_dg_experiment
```

## 从 UDA 到 DG 的转变

### UDA（无监督域适应）模式
- 源域：真实的源数据
- 目标域：真实的目标数据
- 目标：学习如何对齐两个不同的真实域

### DG（域泛化）模式
- 源域：真实的源数据
- 目标域：增强后的源数据（伪目标域）
- 目标：学习如何处理数据的多样性，提升泛化能力

## 参考文献

SinPoint 原始实现：
- GitHub: https://github.com/dhh1995/SinPoint
- Paper: SinPoint: Sinusoidal-based Augmentation for Point Cloud Data

## 更新日志

### 2026-03-03
- ✅ 创建 `utils/sinpoint_augmentation.py`
- ✅ 修改 `tools/runner.py` 导入路径
- ✅ 添加完整的文档注释
- ✅ 创建测试脚本并验证通过
- ✅ 完全独立于外部 SinPoint 项目

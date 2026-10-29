# 反浮动点系统使用说明

## 概述

反浮动点系统已集成到你的配置文件 `configs/omnire_extended_cam_lidar.yaml` 中，用于消除空气中的雾状高斯点。

### 2. 深度估计集成
**文件**: `models/trainers/base.py`

- 在 `_init_losses()` 后添加了 `_init_depth_estimator()` 方法
- 在 `forward()` 中添加了 `_update_depth_points()` 调用
- 自动从图像估计深度并设置到高斯模型

**文件**: `datasets/base/pixel_source.py`

- 添加 `img_path` 到 `image_infos` 字典，用于深度估计

### 3. 配置文件更新
**文件**: `configs/omnire_extended_cam_lidar.yaml`

在 `model.Background` 下添加了三个新配置块：

#### (1) 深度估计器配置 (ctrl.depth_estimator)
```yaml
ctrl:
  depth_estimator:
    encoder: vitl           # 模型大小: vits/vitb/vitl
    max_depth: 80           # 最大深度(米)
    cache_dir: null         # 深度缓存目录(可选)
    max_cache_items: 100    # 内存缓存最大条目数
```

#### (2) 浮动点检测配置 (ctrl.floater_detection)
```yaml
ctrl:
  floater_detection:
    enabled: true           # 启用浮动点检测
    grad_thresh: 0.0001     # 梯度阈值
    opacity_thresh: 0.1     # 不透明度阈值
    isolation_thresh: 2.0   # 孤立度阈值(场景单位)
    depth_margin: 1.0       # 与深度表面的最大距离(米)
    soft_prune_until: 10000 # 软剪枝步数
    hard_prune_interval: 1000 # 硬剪枝间隔
    warmup_steps: 500       # 预热步数
    floater_score_thresh: 0.7 # 浮动点分数阈值
```

#### (3) 表面引导密度化配置 (ctrl.surface_guided_densification)
```yaml
ctrl:
  surface_guided_densification:
    enabled: true           # 启用表面引导
    pull_factor: 0.5        # 向表面拉动因子
    max_dup_distance: 1.0   # 克隆最大距离(米)
    max_split_distance: 1.2 # split源点离表面过远则不split
    max_new_points_ratio_per_refine: 0.05 # 每次refine最多新增占比
    bright_new_points_ratio_cap: 0.2 # 亮区新增点占比上限
    split_new_points_ratio: 0.6 # 新增预算中split占比
```

### 短训（20k）建议节奏

如果目标是 20k 先看清晰度，再决定是否继续长训，建议采用“先成形、后抑雾”：

```yaml
floater_detection:
  warmup_steps: 1500
  soft_prune_until: 14000
  max_soft_prune_ratio: 0.01
  soft_opacity_shrink: 0.995
  max_prune_ratio_per_hard_step: 0.004
  max_candidate_prune_ratio: 0.35
  reset_alpha_grace_steps: 1200
  protect_high_grad_percentile: 0.85
  far_surface_priority_weight: 0.15

surface_guided_densification:
  bright_region_densify_scale: 1.05
  max_new_points_ratio_per_refine: 0.08
  bright_new_points_ratio_cap: 0.3
  split_new_points_ratio: 0.65
```

这组参数的目的：前 14k 优先稳住结构清晰度，后 6k 再温和去除雾状点。

### 快速验证（10k）建议节奏

如果目标是 10k 就先看策略有效性，建议采用“**9成时间先成形，最后2k轻抑雾**”：

```yaml
trainer:
  optim:
    num_iters: 10000
  gaussian_ctrl_general_cfg:
    stop_split_at: 10000

floater_detection:
  warmup_steps: 1000
  soft_prune_until: 8000
  max_soft_prune_ratio: 0.01
  soft_opacity_shrink: 0.995
  max_prune_ratio_per_hard_step: 0.003
  max_candidate_prune_ratio: 0.30
  reset_alpha_grace_steps: 1000
  protect_high_grad_percentile: 0.80
  far_surface_priority_weight: 0.10

surface_guided_densification:
  bright_region_densify_scale: 1.05
  bright_luma_quantile: 0.98
  bright_luma_min: 0.7
  bright_emissive_quantile: 0.995
  bright_emissive_min: 1.0
  bright_surface_distance_min: 0.35
  max_new_points_ratio_per_refine: 0.10
  bright_new_points_ratio_cap: 0.35
  split_new_points_ratio: 0.65
```

这组参数的目的：前 8k 更优先保证全视角可见清晰度，8k-10k 做轻量去雾，避免后视在短训中被过早削弱。

#### (4) 正则化配置 (reg)
```yaml
reg:
  spatial_sparsity:
    enabled: true
    w: 0.01                 # 损失权重
    grid_size: 1.0          # 体素大小
    max_density: 10.0       # 最大密度
    distance_scale: 2.0     # 距离缩放
  
  depth_consistency:
    w: 0.1                  # 损失权重
    max_surface_distance: 0.5 # 最大表面距离(米)
    use_ray_consistency: true # 使用射线一致性
```

## 功能开关

如果想临时禁用反浮动点系统，修改配置：

```yaml
model:
  Background:
    ctrl:
      floater_detection:
        enabled: false  # 禁用浮动点检测
      surface_guided_densification:
        enabled: false  # 禁用表面引导
    reg:
      spatial_sparsity:
        enabled: false  # 禁用空间稀疏性
```

## 参数调优建议

### 场景类型

**白天场景**（浮动点较少）：
```yaml
floater_detection:
  floater_score_thresh: 0.8  # 更严格
  warmup_steps: 1000
spatial_sparsity:
  w: 0.005  # 降低权重
```

**夜间场景**（浮动点较多）：
```yaml
floater_detection:
  floater_score_thresh: 0.6  # 更宽松
  warmup_steps: 500
spatial_sparsity:
  w: 0.02  # 提高权重
depth_consistency:
  w: 0.15  # 提高权重
```

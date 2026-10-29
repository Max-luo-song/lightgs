# 光影重建策略总结

本文档总结了该项目中关于光影重建的核心策略和技术实现，包括PBR（Physically Based Rendering）材质系统、全局光照、环境光等方面。

---
## 一、PBR材质系统

### 1.1 核心架构

项目实现了完整的PBR材质表示，将传统3DGS扩展为支持物理材质属性的渲染系统。

**实现文件**: `models/gaussians/vanilla.py`

### 1.2 PBR材质参数

每个3D高斯点（Gaussian）都包含以下PBR属性：

#### 1.2.1 基础材质参数
- **Albedo（反照率/基础色）**: 从原有的`_features_dc`通过sigmoid激活得到

- **Roughness（粗糙度）**: `_roughness` 张量，每个高斯一个标量值
  - 通过sigmoid映射到[0,1]范围
  - 控制表面的微观粗糙程度，影响镜面反射的锐利度
  - 学习率: 0.001

- **Metallic（金属度）**: `_metallic` 张量
  - 通过sigmoid映射到[0,1]范围
  - 控制材质是金属(1)还是电介质(0)
  - 学习率: 0.001

- **Normals（法线）**: `_normals` 张量 (N, 3)
  - 显式存储的表面法线向量
  - 归一化后用于光照计算
  - 学习率: 0.001

#### 1.2.2 Anisotropic Spherical Gaussians (ASG)参数

用于表示各向异性的镜面高光，每个高斯点包含4个ASG波瓣（lobes）：

- **ASG Amplitude（振幅）**: `_asg_amplitude` (N, 4, 3)
  - 每个波瓣的RGB颜色强度
  - 通过sigmoid激活
  - 学习率: 0.001

- **ASG Sharpness（锐度）**: `_asg_sharpness` (N, 4, 2)
  - 控制镜面高光的集中程度（λ, μ参数）
  - 通过exp激活
  - 学习率: 0.001

- **ASG Direction（方向）**: `_asg_dir` (N, 4, 3)
  - 每个波瓣的光照方向
  - 归一化为单位向量
  - 学习率: 0.001

---

## 二、全局光照系统

### 2.1 GlobalLightingModule

**实现文件**: `models/modules.py`

动态估计场景级别的漫反射光照，通过球谐系数表示。

#### 关键组件
1. **时间编码器**: SinusoidalEncoder
   - 将时间戳编码为高维特征（sin/cos位置编码）
   - 频率范围: 2^0 到 2^4

2. **相机外观嵌入**: nn.Embedding
   - 为每个相机学习独立的appearance embedding
   - 维度: 16维（可配置）

3. **MLP网络**:
   - 输入: 时间编码 + 相机嵌入
   - 隐藏层: 64维 × 4层（可配置）
   - 输出: 3 × (sh_degree+1)² 个SH系数
   - 包含skip connection以提升训练稳定性

### 2.2 环境光照 (EnvLight)

**实现文件**: `models/modules.py`

使用Cubemap表示远距离环境光照（如天空）。

## 三、光照计算流程

### 3.1 训练阶段光照流水线

**主要文件**: `models/trainers/scene_graph.py`, `models/trainers/base.py`

## 四、反浮动点系统（Anti-Floater）

### 4.1 概述

**实现文件**: `models/floater_detection.py`

针对夜间场景和复杂光照条件，实现了反浮动点检测系统，用于消除空气中的"雾状"高斯点。

### 4.2 深度估计集成

使用Depth Anything V2进行单目深度估计（depth_estimator）：

### 4.3 浮动点检测策略（floater_detection）

基于4个维度的综合评分：

1. **梯度活动度**（权重0.25）
   - 检测低梯度、不活跃的高斯点

2. **孤立度**（权重0.20）
   - 检测远离其他点的孤立高斯

3. **深度违规**（权重0.35）
   - 检测远离深度表面的点

4. **低不透明度**（权重0.20）
   - 检测半透明点

### 4.4 剪枝策略

- **预热期**（0-500步）: 不进行剪枝
- **软剪枝期**（500-10000步）: 降低浮动点的学习率
- **硬剪枝期**（>10000步）: 每1000步直接删除浮动点

### 4.5 表面引导密度化（surface_guided_densification）

**效果**:
- Split操作：新高斯被拉向最近深度表面
- Clone操作：拒绝距离表面过远的克隆

### 4.6 正则化损失

#### 空间稀疏性正则化（spatial_sparsity）

#### 深度一致性正则化（depth_consistency）

---

## 五、优化器状态管理

### 5.1 PBR参数的优化

**实现文件**: `models/gaussians/basics.py`, `models/gaussians/vanilla.py`

针对PBR参数的Adam优化器状态进行了特殊处理：

#### 参数分组（get_param_groups）
```python
groups = {
    "xyz": [self._means],
    "sh_dc": [self._features_dc],
    "sh_rest": [self._features_rest],
    "opacity": [self._opacities],
    "scaling": [self._scales],
    "rotation": [self._quats],
    # PBR参数
    "roughness": [self._roughness],
    "metallic": [self._metallic],
    "normals": [self._normals],
    "asg_amplitude": [self._asg_amplitude],
    "asg_sharpness": [self._asg_sharpness],
    "asg_dir": [self._asg_dir],
}
```

#### 学习率调度
```yaml
gaussian_optim_general_cfg:
  # ... 标准3DGS参数 ...
  # PBR参数
  roughness:
    lr: 0.001
  metallic:
    lr: 0.001
  normals:
    lr: 0.001
  asg_amplitude:
    lr: 0.001
  asg_sharpness:
    lr: 0.001
  asg_dir:
    lr: 0.001
```

### 5.2 动态密度化时的状态同步

**关键修复**: 在split/clone/cull操作时正确处理optimizer state

---

## 六、训练损失函数

### 6.1 渲染损失

```yaml
losses:
  rgb:
    w: 0.8                     # RGB重建损失
  ssim:
    w: 0.2                     # 结构相似性损失
  mask:
    w: 0.05                    # 掩码损失
    opacity_loss_type: bce     # 二元交叉熵
  depth:
    w: 0.01                    # 深度监督损失
    inverse_depth: False
    normalize: False
    loss_type: l1
  affine:
    w: 0.00001                 # 仿射变换正则化
```

### 6.2 PBR相关正则化

1. **空间稀疏性**: 限制高斯点的空间密度
2. **深度一致性**: 约束高斯点接近深度表面
3. **Sharp Shape正则化**: 
   ```yaml
   sharp_shape_reg:
     w: 1.0
     step_interval: 10
     max_gauss_ratio: 10.0
   ```

---

**最后更新**: 2026-04-01
**版本**: DriveStudio/OmniRe with PBR Extension

# 当前项目光照处理总结

本文基于当前代码（`models/`、`configs/`）梳理项目对光照的实际处理路径。

## 1. 颜色/光照表示层

### 1.1 基础 SH 颜色表示（非 PBR 路径）
- 高斯颜色基础参数是 `features_dc + features_rest`，并使用球谐函数表达视角相关颜色。  
  参考：`models/gaussians/vanilla.py:1209-1215`、`models/gaussians/basics.py:76-89`
- 训练中 SH 阶数按步数逐步放开（`sh_degree_interval`）。  
  参考：`configs/omnire_extended_cam_lidar.yaml:109-122`

### 1.2 PBR 扩展（夜景主路径）
- 当 `use_pbr=True` 时，为每个高斯增加：
  - `roughness`
  - `metallic`
  - `normals`
  - `asg_amplitude / asg_sharpness / asg_dir`
  - 可选 `emissive / emissive_intensity`
  参考：`models/gaussians/vanilla.py:85-107, 206-221`
- 这些参数有独立优化组与学习率。  
  参考：`models/gaussians/vanilla.py:310-329`、`configs/omnire_extended_cam_lidar.yaml:74-104`

## 2. 渲染层光照计算

### 2.1 全局光照（GlobalLightingModule）
- 模块输入：`normalized_time` + `camera/image embedding`，输出全局 SH 系数。  
  参考：`models/modules.py:209-262`
- 在 `MultiTrainer` 前向中生成 `global_sh_coeffs` 并传给高斯渲染。  
  参考：`models/trainers/scene_graph.py:343-354`、`models/trainers/base.py:448-466`

### 2.2 PBR 着色公式（当前实现）
- `compute_pbr_color` 主要流程：
  - `albedo = sigmoid(features_dc)`
  - `normal` 归一化
  - 用 `GlobalLighting` 输出 SH 在法线方向上求 `diffuse`
  - 用 ASG lobe + 类 Phong 指数近似计算 `specular`
  - 可选 `emissive` 直接相加
  - 最后做 `L/(1+L)` tone mapping
  参考：`models/gaussians/vanilla.py:1131-1199`

> 现状说明：`metallic` 参数已建模并参与优化，但在当前 `compute_pbr_color` 中未进入着色公式。  
> 参考：`models/gaussians/vanilla.py:1177-1181` 与 `metallic` 全局检索结果

### 2.3 天空/环境光融合
- 当前配置里 `Sky` 使用 `EnvLight`（可学习立方体环境贴图），不是 `SkyModel`。  
  参考：`configs/night_pbr.yaml:111-117`、`models/modules.py:174-207`
- 渲染融合方式：
  - `rgb_sky`
  - `rgb_sky_blend = rgb_sky * (1-opacity)`
  - 最终 `rgb = rgb_gaussians + rgb_sky * (1-opacity)`（再过 affine）
  参考：`models/trainers/scene_graph.py:405-413`、`models/trainers/base.py:617-625`

## 3. 光照相关监督与稳态控制

### 3.1 天空掩码监督
- 用 `sky_masks` 监督占据概率（BCE/safe BCE），约束天空区域不被高斯错误占据。  
  参考：`models/trainers/base.py:234-243, 947-949`

### 3.2 强光区域重加权与几何抑制
- RGB loss 支持按亮度阈值对亮区重加权，并可对后视相机进一步加权。  
  参考：`models/trainers/base.py:894-919`
- `highlight_geometry_suppression` 对高亮区域 opacity 额外惩罚，防止“用雾状高斯拟合强光”。  
  参考：`models/trainers/base.py:927-945`

### 3.3 Free-space 约束（夜间强光稳定）
- 基于深度/伪深度抑制前景空域不合理 opacity，支持亮区重加权、后视相机加权、可选 sky mask 过滤。  
  参考：`models/trainers/base.py:723-861`

### 3.4 反浮点与发光稀疏
- 高斯侧有 `spatial_sparsity`、`depth_consistency`、`emissive_sparsity` 正则。  
  参考：`models/gaussians/vanilla.py:1281-1308`
- 亮区 densify 有“发光优先”策略：亮且低 opacity 的高斯优先 duplicate、限制 split，减少强光雾化。  
  参考：`models/gaussians/vanilla.py:980-1069`

## 4. 配置层实践（当前主线）

`omnire_extended_cam_lidar.yaml` / `night_pbr*.yaml` 的共同主线是：
- 开启 `use_pbr: True`、`use_emissive: True`
- 启用 `GlobalLighting`（SH degree=2）
- `Sky` 用 `EnvLight`
- 配合 `free_space + depth_consistency + spatial_sparsity + highlight suppression`

典型配置位置：
- `configs/omnire_extended_cam_lidar.yaml:44-67, 136-224, 439-444`
- `configs/night_pbr_anti_floater.yaml:44-66, 134-205`

## 5. 一句话结论

项目当前光照方案是：**“高斯 SH 表示 + 全局时变 SH 光照 + PBR(含 ASG 高光) + 可学习环境光 + 强光/空域/深度联合约束”**，核心目标是把夜间高亮现象更多交给光照与发光分支解释，而不是让几何 opacity 无限制堆积。

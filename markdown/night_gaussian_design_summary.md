# 黑夜高斯特殊设计总结

基于当前项目代码梳理，`night_pbr*` 这一套“黑夜高斯”不是单纯换了个配置，而是围绕**夜间高光、弱光、雾状浮点**做了完整增强。

## 1. 显式 PBR 夜景表达

- `models/gaussians/vanilla.py` 为背景高斯增加了 PBR 属性：
  - `roughness`
  - `metallic`
  - `normals`
  - `asg_amplitude / asg_sharpness / asg_dir`
- 支持 `use_pbr: True` 后，颜色不再只靠 SH，而是走：
  - **Diffuse + Specular(ASG) + Tone Mapping**
- 额外支持 `use_emissive: True`，把夜景中的**灯、牌、反光源**从纯几何里拆出来，用显式发光通道表达。

## 2. 全局光照建模

- `models/modules.py` 中接入了 `GlobalLightingModule`
- `models/trainers/scene_graph.py` 会根据 `normed_time` 和 `cam_id` 生成 `global_sh_coeffs`
- 这些全局 SH 系数会传给高斯渲染，用于让夜景的整体照明随时间和相机变化，而不是固定环境光

## 3. 强光 / 过曝区域专门处理

- `models/trainers/base.py` 对亮区 RGB loss 做了重权重：
  - 用亮度阈值 `luma_thresh` 找 `bright_mask`
  - 可对后视相机额外加权
  - 目标是避免夜间强光区域训练信号失衡
- 同文件还增加了 `highlight_geometry_suppression`：
  - 对高亮区域的 `opacity` 额外惩罚
  - 明确抑制“用一团雾状高斯去硬拟合过曝高光”的做法

## 4. 反浮点 / 去雾机制

- `models/floater_detection.py` 提供了多维度浮点检测：
  - 低梯度
  - 低不透明度
  - 空间孤立
  - 偏离深度表面
- `models/trainers/base.py` 会初始化深度估计器，并在训练时把深度点更新给高斯模型
- `models/gaussians/vanilla.py` 中实现了：
  - `adaptive_floater_pruning`
  - 软剪枝 / 硬剪枝
  - 高梯度、高不透明度、贴近表面的保护策略

## 5. 表面引导的密度化

- `split_gaussians` 会把新高斯向深度表面拉近，减少漂浮在空中的点
- `dup_gaussians` 会拒绝离表面太远的 clone
- 还加入了：
  - `bright_region_densify_scale`
  - `max_new_points_ratio_per_refine`
  - `bright_new_points_ratio_cap`
  - `split_new_points_ratio`

这套逻辑的核心是：**夜景里优先让结构贴近真实表面，避免光源附近生成空气浮点。**

## 6. 亮区“发光优先”策略

- `_compute_bright_gaussian_mask()` 不只看颜色亮度，还会结合：
  - emissive 强度分位数
  - 距离深度表面的门控
- `_suppress_foggy_growth_candidates()` 会把亮且低 opacity 的种子限制住，减少雾化扩散
- 如果启用了 `highlight_emission_first`，会优先让高亮表达走 emissive 通道，而不是继续裂变几何

## 7. 夜景专用配置文件

当前已经有三套直接相关配置：

| 配置文件 | 作用 |
|---|---|
| `configs/night_pbr.yaml` | 基础夜景 PBR 方案 |
| `configs/night_pbr_extended.yaml` | 更完整的夜景扩展版，带深度/浮点/表面约束 |
| `configs/night_pbr_anti_floater.yaml` | 专门强调反浮点、深度一致性、emissive 的增强版 |

## 结论

这套“黑夜高斯”的特殊设计可以概括为三句话：

1. **用 PBR + emissive 显式表达夜景光照。**
2. **用深度、浮点检测、表面引导，把雾状高斯压下去。**
3. **用亮区重权重和高光抑制，避免强光区域被几何硬拟合。**


# 2026-04-20 11:50:16 Explicit Light Source Branch Plan

本文记录当前夜间强光建模的下一步方案：先优先尝试 **显式车灯 Light Source Branch**，再根据效果逐步推进 bloom、饱和像素监督、实例级约束、道路反光等后续设计。

## 1. 当前判断

目前代码已经引入了较完整的夜间强光/反浮点策略：

- PBR 分解渲染：`Diffuse + ASG Specular + Tone Mapping`
- Emissive 发光通道
- 各向异性 emissive SH
- emissive cone prior
- emissive surface support / cluster 正则
- highlight RGB reweight
- highlight geometry suppression
- free-space loss
- rear pseudo depth
- floater detection + adaptive pruning
- surface-guided densification
- fog hard gate schedule
- depth consistency / spatial sparsity

这些策略的核心仍然是：**普通背景 Gaussian 可以学成发光点，然后再用规则和 loss 管住它**。

实践上，如果极强车灯仍然导致雾状伪点、空气密度、拖尾或后视强光不稳定，下一阶段应该把车灯从普通背景高斯中拆出来，变成一个显式、低自由度、强约束的独立光源分支。

一句话目标：

> 让车灯由专门的 Light Source Branch 承担能量和方向性，让 Background Gaussians 只负责可解释几何与材质，不再用空气密度拟合强光。

## 2. 第一阶段优先方案：Explicit Light Source Branch

### 2.1 设计目标

新增一个独立的 `LightSourceGaussians` 或 `LightNodes` 分支，用少量专用高斯表达车灯/尾灯/灯牌等强主动光源。

该分支和普通 `Background` 高斯分开：

- 独立参数
- 独立 optimizer group
- 独立 densify/prune 策略
- 独立 geometry support loss
- 独立渲染输出
- 独立 mask/regularization

这样可以从表示层上减少“背景高斯冒充车灯”的路径。

### 2.2 建议参数

每个 light Gaussian 建议包含：

- `position`: 光源中心
- `scale`: 小尺度形状，限制体积膨胀
- `orientation`: 光源方向/局部坐标
- `color`: 发光颜色，RGB
- `intensity`: HDR 发光强度，建议 log-domain 参数化
- `cone_axis`: 主发光方向
- `cone_angle` 或 `cone_sharpness`: 控制方向性
- `temporal_embedding` 或 `frame_embedding`: 支持刹车灯/转向灯等时变强度
- 可选 `instance_id`: 绑定 2D/3D 灯簇实例

初版可以先做最小可行版本：

- `position`
- `scale`
- `color`
- `intensity`
- `cone_axis`
- `cone_sharpness`

暂时不做完整 instance/temporal。

### 2.3 渲染输出

Light branch 单独输出：

- `light_rgb`
- `light_alpha`
- `light_depth`
- `light_hdr`
- 可选 `light_core_mask`

最终融合可以先简单做：

```text
rgb = background_rgb + light_rgb
```

后续再升级为：

```text
rgb = background_rgb
    + light_core_rgb
    + camera_bloom(light_core_hdr)
    + road_reflection(light_sources, road_material)
```

### 2.4 初始化方式

初版可以用图像启发式生成车灯候选 mask：

- 白色强光：高 luma，RGB 接近白色
- 红色尾灯：R 高，且 R 明显大于 G/B
- 黄色灯/转向灯：R/G 高，B 低
- 连通域过滤：面积、长宽比、亮度分位数
- 结合 `dynamic_masks` 或车辆区域 mask，优先保留车辆区域内/边缘附近的灯区
- 后视相机单独放宽阈值

然后结合深度：

- 有 LiDAR depth：在对应 ray 终止面附近 seed light Gaussian
- 无 LiDAR depth：使用 rear pseudo depth 或 monocular depth
- 深度不可靠：沿 ray 在合理车辆距离范围内初始化，并依赖 surface support / mask loss 收敛

### 2.5 几何约束

Light branch 必须比普通 emissive 更强约束：

- `light_surface_support_loss`: 高强度 light Gaussian 必须贴近深度表面
- `light_scale_clip`: 限制最大尺度，防止膨胀成雾团
- `light_opacity_cap`: light core 可以亮，但不能靠大 opacity 体积解释光晕
- `light_ray_clearance_loss`: 光源前方 ray segment 内惩罚 background opacity
- `light_sparsity_loss`: 限制 light branch 数量和总能量
- `light_non_mask_penalty`: light branch 投影到非灯区时受惩罚

### 2.6 Densify / Prune 策略

Light branch 不应沿用背景高斯的自由 densification。

建议：

- warmup 阶段只优化已有 light seeds
- 允许 duplicate，不允许或严格限制 split
- split 必须满足：
  - 在 light mask 内有稳定梯度
  - 靠近 surface
  - 不在 free-space
  - scale 足够小
- prune 优先删除：
  - 投影不落在 light mask 的点
  - 远离 surface 的点
  - 高 opacity 但低 emissive 的点
  - 对 light reconstruction 贡献低的点

### 2.7 和现有 Background 的关系

需要显式让 Background 不再承担车灯 core：

- 对 background emissive 加更强 sparsity
- 对 high-light mask 内的 background opacity 继续使用 `highlight_geometry_suppression`
- 对 light mask 前方的 background opacity 使用 `light_ray_clearance_loss`
- 对 Background densify：
  - light mask/free-space 中禁止 split
  - 只允许少量 surface duplicate

初版不要立刻删除当前 Background emissive，可以并行存在：

- `Background.emissive`: 处理小灯牌、远处弱光、未被 light branch 捕捉的发光
- `LightSourceBranch`: 专门处理极强车灯/尾灯 core

训练稳定后，再逐步收紧 Background emissive。

## 3. 第一阶段实现建议

### 3.1 最小改动版本

优先做一个低风险版本：

1. 新增 `models/gaussians/light_source.py`
2. 增加 LightSource 分支配置，例如：

```yaml
model:
  LightSource:
    type: models.gaussians.LightSourceGaussians
    init:
      from_light_masks: true
      max_lights_per_image: 64
    ctrl:
      enabled: true
      use_cone_emission: true
      max_scale: 0.15
      allow_split: false
    reg:
      surface_support:
        enabled: true
        w: 0.05
        margin: 0.15
      sparsity:
        enabled: true
        w: 0.01
      non_mask_penalty:
        enabled: true
        w: 0.02
```

3. 在 trainer / scene graph 中把 `LightSource` 和 `Background` 一起渲染
4. 在 loss 中新增 light mask 相关监督
5. 当前 Background 的 anti-floater/free-space/highlight 策略保持不变

### 3.2 推荐训练调度

建议分四阶段：

#### Stage A: 0-2k

- 稳几何
- 强 free-space / depth consistency
- Background emissive 强稀疏
- Light branch 只初始化，不大幅优化强度

#### Stage B: 2k-6k

- 启用 Light branch intensity/color/cone 优化
- 禁止 light split
- 允许少量 duplicate
- Background 在 light mask 内继续压 opacity

#### Stage C: 6k-12k

- 放开 light branch 小幅 refinement
- 开始记录 light residual
- 若 light branch 能解释 core，再降低 bright RGB 对 Background 的影响

#### Stage D: 12k+

- 小步优化 light shape/intensity
- 继续禁止 free-space split
- 根据 light residual 决定是否引入 bloom/PSF 分支

## 4. 后续设计方案

### 4.1 Core / Bloom / Reflection 三分解

真实车灯图像不只是灯具表面：

- `core`: 真实灯具表面，小尺度、贴车体
- `bloom/glare`: 镜头或传感器造成的 2D 光晕/拖影
- `reflection`: 路面、车身、湿地、牌照等反射

后续应从：

```text
rgb = background_rgb + emissive_rgb
```

升级到：

```text
rgb = background_rgb
    + light_core_rgb
    + camera_bloom(light_core_hdr)
    + road_reflection(light_sources, road_material)
```

关键点：bloom 不要用 3D Gaussian 表达，而应作为 image-space camera PSF。

### 4.2 Image-Space Bloom / PSF 分支

对 Light branch 输出的 HDR core 做图像空间扩散：

- 取 `light_hdr` 的高亮阈值
- 每个 camera 学习独立 PSF kernel
- kernel 可以是 anisotropic Gaussian、streak kernel 或少量 basis kernel 的组合
- bloom 加到最终 RGB 前后再 tone map

作用：

- 光晕由相机成像模型解释
- 减少空气中为了拟合光晕而生长的 Gaussian
- 让 light core 保持贴表面、小尺度

### 4.3 饱和像素下界监督

LDR 图像中车灯经常饱和。饱和像素的监督应该是“至少这么亮”，不是“精确等于白色”。

建议高亮饱和区域使用 lower-bound loss：

```text
if gt_luma > sat_thresh:
    loss = relu(gt_lower_bound - pred_hdr_luma)
else:
    loss = normal photometric loss
```

这可以避免饱和白块对几何、opacity、emissive 强度造成错误梯度。

### 4.4 实例级 Light Cluster / Support

当前全局 `emissive_cluster` 可能把左右车灯、多车灯源、灯牌错误拉近。

后续应改成实例级：

- 每个 2D light connected component 对应一个 `light_instance_id`
- 实例内部做 cluster
- 实例之间不互相吸引
- 同一车辆左右灯可加 pair prior：
  - 高度接近
  - 深度接近
  - 横向间距合理
  - 颜色/强度相近但不强制相同

### 4.5 Per-Light Cone Axis

当前 cone prior 如果使用全局 `axis_world`，不适合车辆灯。

后续应改成：

```text
cone_axis_i = vehicle_forward_axis or learned_axis_i
```

对于动态车辆：

- 前灯沿车辆 forward
- 尾灯沿 `-forward`
- 侧向灯沿 lateral

如果没有可靠车辆姿态，可以先用相机 ray 初始化，再用时序平滑和 surface support 修正。

### 4.6 车灯时序一致性

强车灯单帧容易误拟合，跨帧约束很重要。

建议后续加入：

- light position 随车辆节点运动
- intensity 可变但平滑
- cone axis 平滑
- 同一 light instance 在相邻帧投影应稳定落在灯区
- 刹车灯/转向灯允许短时强度突变，但需要显式 temporal gate

### 4.7 Light Ray Clearance

针对车灯区域增加更强的 ray segment free-space：

```text
camera ---- clear segment ---- light core/surface
```

在 `light_depth - margin` 之前惩罚 Background opacity，只允许 light core 附近有发光。

这比普通 free-space 更聚焦车灯前方空气伪密度。

### 4.8 Road Reflection Model

车灯对路面的贡献不应由空气高斯承担。

后续可以基于 road mask / road node 加简化反射：

```text
road_reflection = BRDF_road(normal, view_dir, light_pos, light_intensity, roughness)
```

初版只需要：

- light 到 road point 的距离衰减
- road normal 与 half vector 的 specular
- roughness 控制长条形反光
- road mask 内生效
- reflection 强度和 light core intensity 耦合

## 5. 实验优先级

建议按以下顺序推进：

1. **Light Source Branch + 基础 light mask 初始化**
   - 目标：先验证显式车灯分支是否能减少背景雾状高斯。

2. **Saturation-aware lower-bound loss**
   - 目标：减少饱和白块对几何和强度的错误监督。

3. **Image-space bloom / PSF**
   - 目标：把光晕从 3D 几何里拆出来。

4. **实例级 cluster/support**
   - 目标：避免全局发光点互相吸引。

5. **Per-light cone axis + temporal consistency**
   - 目标：让车灯方向性和车辆运动绑定。

6. **Road reflection**
   - 目标：让路面高光由光源和材质解释，而不是由空域 Gaussian 拟合。

## 6. 预期观察指标

第一阶段完成后，重点看：

- 车灯前方空气 opacity 是否下降
- free-space loss 是否更稳定
- floater ratio 是否下降
- Background 点数增长是否变慢
- light branch 是否集中在真实灯具表面
- 后视相机强光区域是否减少雾团和白色拖尾
- 非强光区域 PSNR/SSIM 是否不明显退化
- light branch 是否过拟合成贴图式白块

建议导出额外 buffer：

- `background_rgb`
- `light_rgb`
- `light_alpha`
- `light_depth`
- `light_hdr`
- `background_opacity`
- `light_mask_pred`
- `light_mask_gt`

## 7. 风险

- Light mask 初始化错误会导致 light branch 学错位置。
- 没有实例约束时，多车灯源可能仍然互相污染。
- 如果 light branch 太弱，Background 仍会继续承担强光。
- 如果 light branch 太强，可能压制真实几何或把反光误解释成灯源。
- 饱和区 lower-bound loss 如果阈值过宽，可能导致亮度失控。
- Bloom/PSF 若过早加入，可能掩盖 core 几何错误。

因此第一阶段只建议做显式 light branch 的最小闭环，不要同时加入所有后续模块。


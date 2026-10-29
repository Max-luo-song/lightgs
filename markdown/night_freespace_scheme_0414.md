# 夜间 3DGS Free-Space 约束方案

## 目标

针对夜间强光场景中的“雾气高斯 / 漂浮高斯”，在**生成之前**就限制表面前空域长出错误密度，并与现有的 PBR、emissive、亮区抑制、浮点剪枝联动。

## 核心思想

1. **表面前必须尽量为空**  
   对有可信深度的像素，约束 `[0, d* - δ]` 区间的累计贡献接近 0。

2. **后向弱监督用软约束**  
   没有真实深度的后视相机，只用伪深度或多视角一致性做弱 Free-Space。

3. **把 Free-Space 接入 densify / prune**  
   不只做 loss，还要参与 split / dup / prune 决策，避免空气里继续长点。

## 方案结构

### A. 前向真实深度 Free-Space

- 数据源：LiDAR 深度 / 融合深度
- 约束方式：
  - 射线表面前 prefix opacity penalty
  - 或表面前密度积分惩罚
  - 或透射率约束 `T(d* - δ) ≈ 1`

推荐最小实现：

```math
L_{fs} = \sum_r c_r \sum_i \mathbf{1}(z_i < d_r - \delta_r) \cdot w_i
```

其中：
- `d_r`：真实深度
- `c_r`：深度置信度
- `δ_r`：表面容忍带
- `w_i`：高斯对射线的贡献权重

### B. 后视伪深度 Free-Space

- 数据源：
  - 单目深度
  - 邻近视角重投影
  - 时序一致性
  - VO/SfM 稀疏点
- 约束方式：只做弱空域约束，不做强深度回归

```math
L_{fs,rear} = c(u,v)\sum_i \mathbf{1}(z_i < \hat d - \delta) \cdot w_i
```

### C. 高斯级空域惩罚

把 Free-Space 变成高斯自身的 pruning score：

- 距表面过远
- 长期低贡献
- 低 opacity + 大尺度
- 位于亮区附近

可形成：

```math
S_{prune} = aS_{float} + bS_{surface} + cS_{bright} + dS_{fs}
```

## 与现有模块的关系

| 模块 | 作用 |
|---|---|
| `use_pbr` / `use_emissive` | 让亮度解释走显式光照，而不是几何堆积 |
| `bright_region_reweight` | 提升亮区训练信号 |
| `highlight_geometry_suppression` | 抑制过曝区靠 opacity 拟合 |
| `surface_guided_densification` | 新点向真实表面靠拢 |
| `floater_detection` | 删除已生成的漂浮点 |
| `Free-Space` | 事前阻止空气里长出错误密度 |

## 推荐落地顺序

### Phase 1：前向真实深度 Free-Space

- 先接入 LiDAR 深度像素
- 只对静态背景、非强反光区域启用
- 先做 prefix opacity penalty

### Phase 2：后视软 Free-Space

- 给后向相机接单目/多视角伪深度
- 使用置信度调权
- 只压明显违反空域的高斯

### Phase 3：并入 prune / densify

- 把 Free-Space violation 加进 floater 分数
- 在 split/dup 前过滤空气伪密度种子
- 与 `surface_guided_densification` 联动，优先在表面附近生长

## 参数建议

- `delta`：给表面留容忍带，避免削薄真实表面
- 前向权重：强一些
- 后视权重：弱一些
- 强光区域：降低 Free-Space 权重，更多依赖 emissive / PBR 解释

## 验证指标

1. 表面前 prefix opacity 是否下降
2. 漂浮高斯数量是否下降
3. 后视强光周围 halo / 泛白是否减弱
4. densify 与 prune 是否更稳定

## 一句话结论

Free-Space 的本质是：**不是只告诉模型“表面在哪”，而是明确告诉它“空气里不该有东西”。**

## 实现状态

- 已在 trainer 侧接入 `free_space_loss`
- 已在 `omnire_extended_cam_lidar.yaml` 和 `night_pbr_anti_floater.yaml` 增加默认配置
- 与现有 depth loss 解耦，保持对夜间高亮区域的弱化权重

# 强光“浮点+雾化”分流方案（实现版）

## 问题与思路
当前后视强光区域容易被模型用“空气高斯堆积”拟合，导致视觉上雾化与漂浮感。  
本次方案目标是把高亮现象从纯几何拟合中分流出来：**后视强光更强调重建约束，同时抑制几何不透明度滥用，并启用 emissive/PBR 路径承接光能表达**。

## 已实施改动
1. **后视强光加权重建（loss 侧）**
   - 文件：`models/trainers/base.py`
   - 在 `bright_region_reweight` 中新增按相机 id 的权重放大：
     - `rear_camera_ids`
     - `rear_camera_weight`
   - 作用：后视强光区域获得更强训练信号，减少被全局平均稀释。

2. **高亮几何抑制项（loss 侧）**
   - 文件：`models/trainers/base.py`
   - 新增 `highlight_geometry_suppression`：
     - 在高亮 + 非路面区域对 `opacity` 施加额外惩罚
     - 支持后视相机更高惩罚倍率
   - 作用：直接阻止“靠高不透明度几何堆叠拟合高亮”的路径。

3. **高亮发光优先的致密化预算策略（model 侧）**
   - 文件：`models/gaussians/vanilla.py`
   - 在 `_apply_densify_budgets` 增加高亮分流控制：
     - `highlight_emission_first`
     - `highlight_emissive_intensity_min`
     - `highlight_low_opacity_max`
     - `highlight_duplicate_priority`
     - `highlight_split_budget_ratio_cap`
   - 作用：强光相关点优先走“发光+温和复制”，抑制激进 split 形成雾状膨胀。

4. **配置接线（启用 PBR + Emissive + GlobalLighting）**
   - 文件：`configs/omnire_extended_cam_lidar.yaml`
   - 新增/启用：
     - `model.Background.ctrl.use_pbr: True`
     - `model.Background.ctrl.use_emissive: True`
     - `trainer.losses.highlight_geometry_suppression`
     - `trainer.losses.rgb.bright_region_reweight.rear_camera_*`
     - `model.Background.ctrl.surface_guided_densification.highlight_*`
     - `model.Background.reg.emissive_sparsity`
     - `model.GlobalLighting` 模块

## 里程碑与验证口径
1. **M1：后视视觉变化出现**
   - cam 6/7/10/11/12 中强光邻域雾化密度明显下降（先看视觉，不以全局 PSNR 判定）。
2. **M2：训练稳定**
   - 背景高斯数曲线无“爆增后断崖式回落”。
   - hard/soft prune 不出现异常激增。
3. **M3：质量可持续**
   - 在 1w~2w 内后视改善不反弹；再决定是否推进 4w 长训。

## 下轮建议
1. 先跑 1w 对照（旧配置 vs 新配置）仅比较后视强光 ROI。
2. 若后视仍弱：继续上调 `rear_camera_weight` 与 `highlight_geometry_suppression.w`（小步）。
3. 若过抑制导致光柱发灰：下调几何抑制，放宽 `highlight_split_budget_ratio_cap`。

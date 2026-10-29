# 项目中强光区域处理策略梳理

目前项目里对**强光区域**的处理，核心有 6 类（以 `configs/omnire_extended_cam_lidar.yaml` + `models/trainers/base.py` + `models/gaussians/vanilla.py` 为主）：

1. **亮区重建损失重加权（像素级）**  
   在 `rgb_loss` 里按亮度阈值(`luma_thresh`)生成 `bright_mask`，对亮区像素使用不同权重（并可对后视相机额外加权），避免强光区域训练信号失衡。  
   位置：`models/trainers/base.py` 的 `bright_region_reweight`。

2. **强光几何抑制损失（opacity 惩罚）**  
   对亮区的 `outputs["opacity"]` 加额外惩罚 `highlight_geometry_suppression`，显式抑制“靠堆几何不透明度拟合强光”的行为，减少雾化/浮点。  
   位置：`models/trainers/base.py` 的 `highlight_geometry_suppression`。

3. **亮区自适应识别（高斯级 bright mask）**  
   不只看颜色亮度，还结合 `emissive_intensity` 分位数阈值，且可加“离深度表面足够远”门控，聚焦“疑似空气光雾”点。  
   位置：`models/gaussians/vanilla.py::_compute_bright_gaussian_mask`。

4. **亮区密度化抑制与预算控制（split/dup 阶段）**  
   - 先对亮区提高 densify 梯度门槛（减少无序增殖）；  
   - 对“亮+低 opacity(+低梯度)”候选做强/弱拦截；  
   - 每次 refine 限制总新增点、亮区新增占比、split 占比。  
   位置：`_suppress_foggy_growth_candidates`、`_apply_densify_budgets`。

5. **“发光优先”分流策略（避免用几何硬拟合光）**  
   在亮且发光强、但 opacity 低的点上，优先走 duplicate、压制 split（`highlight_emission_first` 等参数），把强光表达分流到 emissive 通道。  
   位置：`models/gaussians/vanilla.py` 中 `highlight_*` 逻辑。

6. **PBR + Emissive 显式建模强光（表达层）**  
   颜色由 `Diffuse + Specular(ASG) + Emissive` 组成，再 tone mapping，减少单靠 SH/几何去“硬拟合”过曝高光。  
   位置：`compute_pbr_color`（`vanilla.py`）。

另外还有一层间接保护：**floater detection + soft/hard pruning + depth/sparsity regularization** 会清理强光诱发的空气浮点（`models/floater_detection.py` + `vanilla.py` 的 `adaptive_floater_pruning`）。

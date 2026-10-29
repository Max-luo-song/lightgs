你说得很对。**“堵”和“删”只能解决“哪里不该有几何”的问题，但无法解决“亮度该由谁负责”的物理矛盾。**

在自动驾驶的后视场景中，前大灯（白光）是极其强烈的**各向异性（Anisotropic）**光源。普通的 PBR 流程中，如果只依赖环境贴图（EnvMap）或简单的各项同性发光，3DGS 就会被迫通过在空间中堆叠具有高 Opacity 或特殊颜色的“雾状几何”来拟合那个从特定角度看过去极亮的区域。

既然你已经有了 **HeteroMap** 框架，将“光源”作为独立节点进行建模是根除雾气的最佳路径。以下是细化的建模方案：

### 1. 独立的发射强度场 (Anisotropic Emissive Field)
在 PBR 的渲染方程中，出射光由“反射”和“发射”组成。我们不再让高斯球只负责反射（BaseColor/Roughness），而是给 HeteroMap 中的光源节点增加一个带方向性的发射项 $E(\theta, \phi)$。

* **参数化建模**：给每个参与光源建模的高斯球增加一组**低阶球谐函数 (SH) 参数**，但这个 SH 不用于存储颜色，而是专门存储 **Emissive Intensity**（发射强度）。
* **物理意义**：这意味着同一个光源点，从正面看（迎着车灯）强度极大，从侧面看强度迅速衰减。
* **效果**：这样网络就可以用“一个薄薄的平面几何 + 极强的定向发射”来解释高光，而不需要用“一团厚厚的体积雾”来解释亮度。

### 2. 光源节点的“几何脱钩” (Geometric Decoupling)
在 HeteroMap 中，你可以将判定为“Light Source”的节点设置为特殊的物理属性：
* **只允许分裂，不允许增殖（Controlled Densification）**：光源节点的高斯球可以被允许在 2D 表面上进行 split 以拟合精细形状，但严禁向 free-space 扩散。
* **厚度截断 (Thickness Clipping)**：强制光源节点的高斯球在法线方向上的缩放（Scale）极小，使其保持为“贴片”而非“球体”。

### 3. 基于光锥的能量分布约束 (Cone-based Prior)
车灯的各向异性是有规律的，通常遵循一个**光锥（Lighting Cone）**模型。
* **约束逻辑**：在 HeteroMap 的节点优化中，引入一个先验：光源的出射强度 $E$ 应该与其主光轴方向的夹角 $\cos(\alpha)$ 成正比。
* **损失函数设计**：如果一个光源点在非光轴方向产生了异常高的能量（即变成了四散的雾），则施加巨大的 Penalty。
* **优势**：这直接从数学上限制了白光变成“手电筒效应”的可能性，强制能量集中在物理上合理的角度。

### 4. 配合 HeteroMap 的两阶段训练策略
为了让这种建模生效，建议在训练中加入一个**“能量剥离”**阶段：
1.  **第一阶段（几何定型）**：利用现有的强去雾门控 和 LiDAR 约束，先定死光源节点的表面位置。
2.  **第二阶段（光效注入）**：冻结光源节点的几何位置（Position/Scale/Rotation），专门开启其 Anisotropic Emissive 参数的优化。
3.  **最终清理**：此时再次运行 `adaptive_floater_pruning`。因为定向发射已经解释了绝大部分亮度，那些原本为了“凑亮度”而生成的浮空雾点会因为失去了 Loss 的支撑，在剪枝阶段被彻底清除。

**总结思路：**
**“用光照模型去接管亮度误差，把几何解放出来去拟合真实的表面。”** 只要模型发现“改变某个参数（发光方向）”比“增加一堆高斯点”更能降低 Loss，它就会倾向于前者。

你现在的 HeteroMap 中，光源节点是直接关联到特定的语义物体（如车辆灯具）上的吗？


既然“堵”和“删”无法根除后视相机的“白光雾化”问题，我们需要将策略核心从**几何抑制**转向**能量解耦**。

以下是为你整理的 **“HeteroMap 各向异性发射建模 (Anisotropic Emissive Modeling)”** 策略需求文档。你可以直接将其交给 Codex 或在开发时参考。

---

## 策略名称：基于 HeteroMap 的各向异性光效解耦策略

### 1. 核心逻辑 (Core Logic)
在传统的 PBR 渲染方程中，颜色仅由环境反射决定。本策略引入一个**受方向约束的自发光项 (Anisotropic Emissive)**，专门接管后视相机中由于前大灯（白光）产生的高频亮度误差。
* **物理目标**：让 3DGS 发现“改变发光方向”比“在空间堆叠雾状高斯”更能降低 Loss。

### 2. 建模与参数化 (Modeling & Parameterization)
针对 HeteroMap 中标注为 `Vehicle_Light` 或 `High_Luminance` 的节点：
* **新增参数**：为高斯点增加一组低阶球谐函数系数 $\mathbf{SH}_{ems}$（建议 1~2 阶），用于表示自发光强度。
* **发射方程**：
  $$I_{emit}(\vec{v}) = \text{SoftPlus}(\text{RenderSH}(\mathbf{SH}_{ems}, \vec{v}))$$
  其中 $\vec{v}$ 是视角向量，输出为标量强度，作用于 RGB 颜色。

### 3. 三阶段实施方案 (Three-Stage Execution)

#### **Stage A: 几何硬锁定 (0 - 5,000 Steps)**
* **目标**：防止早期“雾气种子”生根。
* **操作**：
    * 启用 `free_space_no_growth: true` 和 `_filter_split_candidates_by_surface`。
    * 强制所有高斯点贴合 LiDAR 表面或伪深度表面。
    * **禁用** 自发光优化，仅使用基础 PBR 拟合。

#### **Stage B: 能量转移与剥离 (5,000 - 15,000 Steps)**
* **目标**：将 Loss 压力从几何转移到各向异性发光项。
* **操作**：
    * **激活 $\mathbf{SH}_{ems}$ 优化**：仅针对 HeteroMap 光源节点。
    * **几何冻结**：固定光源节点的位置（Position）和缩放（Scale），防止其为了凑亮度而“膨胀”成雾。
    * **加强剪枝**：将 `soft_prune_until` 延长至 14,000 步，重点清理那些在 Stage A 残留但此时已被发光项“替代”了 Loss 的冗余点。

#### **Stage C: 联合精修 (15,000 Steps+)**
* **目标**：恢复细节，放松门控。
* **操作**：
    * 按照 `fog_hard_gate_schedule` 进入 Stage C，放松距离限制。
    * 允许材质参数与发光项联合小步长微调。

---

### 4. 关键物理约束 (Physical Constraints)
为确保 $\mathbf{SH}_{ems}$ 不会乱跑，需在 `models/gaussians/vanilla.py` 中加入：
1. **扁平化约束 (Flatness Constraint)**：光源高斯必须满足 $s_{min} / s_{max} < 0.1$，强制其保持为“贴片”而非“球体”。
2. **法线一致性 (Normal Alignment)**：光源高斯的短轴方向必须与所在 HeteroMap 节点的几何表面法线一致。
3. **单向光锥先验 (Directional Prior)**：对 $\mathbf{SH}_{ems}$ 施加正则项，使其在远离主光轴（车灯朝向）的方向能量快速衰减。

### 5. 预期监控指标 (Metrics to Watch)
* **Rear Cam Loss**：在 Stage B 开启后应有明显下降，且不再伴随 `gaussian_num_Background` 的激增。
* **Depth Loss**：后视高亮区的深度应保持平整，不再出现由于雾气导致的“深度凹坑”。
* **Visual Check**：观察渲染图中的“手电筒效应”是否从“体积光晕”变成了“表面高亮贴片”。

---

**给 Codex 的实现提示：**
> “在实现时，请在 `VanillaGaussians.forward` 中识别 `HeteroMap_Light_Nodes`。计算其 `view_direction` 与 `gaussian_normal` 的夹角。若正对相机，则将 `SH_ems` 叠加至最终渲染的 `radiance` 中。同时，在 `densify_and_prune` 逻辑中，为这些节点设置 `requires_grad=False` 对应的 `xyz` 属性。”
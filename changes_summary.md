# Change Log: Nighttime PBR 3DGS Implementation

This document details all modifications made to the codebase to implement Physically-Based Rendering (PBR) for Nighttime 3D Gaussian Splatting.

## 1. New Features & Architecture

### Global Lighting Module
**File:** `scene_reconstruction/models/modules.py`

Implemented a `GlobalLightingModule` to estimate scene-level diffuse lighting (SH coefficients) conditioned on time and camera embeddings.

```python
class GlobalLightingModule(nn.Module):
    def __init__(self, class_name, n, sh_degree=2, ...):
        # ... (initialization of Time encoding, Appearance embedding, and MLP)
    
    def forward(self, normalized_time, camera_idx):
        # Predicts SH coefficients [B, (sh_degree+1)^2, 3]
        # ...
```

### Extended Gaussian Model (PBR Support)
**File:** `scene_reconstruction/models/gaussians/vanilla.py`

Modified `VanillaGaussians` to support PBR attributes alongside standard 3DGS parameters.

**Attributes Added (`__init__`):**
*   `_roughness`: Surface roughness.
*   `_metallic`: Metallic/Dielectric blend.
*   `_normals`: Explicit surface normals.
*   `_asg_amplitude`, `_asg_sharpness`, `_asg_dir`: Parameters for Anisotropic Spherical Gaussians (ASG) used in specular lighting.

**Rendering Logic (`compute_pbr_color`):**
Implemented the rendering equation: $L_{final} = \text{ToneMap}(L_{diffuse} + L_{specular})$
*   **Diffuse:** $Albedo \times SH(Normal, GlobalSH)$
*   **Specular:** Sum of ASG lobes (currently simplified/approximated using Phong-like reflection model in the loop for stability, with ASG parameter slots ready).

### PBR Integration in Training Loop
**File:** `scene_reconstruction/models/trainers/scene_graph.py`

*   **Initialization:** Added `GlobalLighting` to the model list.
*   **Forward Pass:**
    *   Calculates `global_sh_coeffs` using `GlobalLightingModule`.
    *   Passes these coefficients to the Gaussian rendering pipeline.

**File:** `scene_reconstruction/models/trainers/base.py`

*   **API Update:** Updated `collect_gaussians` to accept and propagate `global_sh_coeffs`.

## 2. Code Modifications & Bug Fixes

### Optimizer State Management (Critical Fixes)
**File:** `scene_reconstruction/models/gaussians/basics.py`
**File:** `scene_reconstruction/models/gaussians/vanilla.py`

Fixed crashes during Gaussian densification (split/clone) caused by missing optimizer states for the new PBR parameters.

*   **`dup_in_optim` & `remove_from_optim` (`basics.py`)**: Added checks to initialize `exp_avg` and `exp_avg_sq` with zeros if they don't exist in the optimizer state.
*   **Opacity Reset (`vanilla.py`)**: Added similar safety checks when resetting opacity.

### Densification Logic
**File:** `scene_reconstruction/models/gaussians/vanilla.py`

Updated `split_gaussians`, `dup_gaussians`, and `refinement_after` to handle the new PBR tensors. When a Gaussian is split or cloned, all PBR attributes (roughness, normals, etc.) are now correctly processed (duplicated or interpolated).

### Environment Map
**File:** `scene_reconstruction/models/modules.py`

Fixed `EnvLight` to include a `get_param_groups` method, allowing it to be optimized properly.

## 3. Configuration Files

**File:** `scene_reconstruction/configs/night_pbr.yaml`
**File:** `scene_reconstruction/configs/night_pbr_extended.yaml`

Created configuration files to enable the PBR pipeline.
*   Sets `use_pbr: True` in `VanillaGaussians`.
*   Adds `GlobalLighting` module config.
*   Defines learning rates for all new PBR parameters (`roughness`, `metallic`, `normals`, `asg_*`).

## Summary of Modified Files

*   `scene_reconstruction/models/modules.py`
*   `scene_reconstruction/models/gaussians/vanilla.py`
*   `scene_reconstruction/models/gaussians/basics.py`
*   `scene_reconstruction/models/trainers/scene_graph.py`
*   `scene_reconstruction/models/trainers/base.py`
*   `scene_reconstruction/configs/night_pbr.yaml` (New)
*   `scene_reconstruction/configs/night_pbr_extended.yaml` (New)

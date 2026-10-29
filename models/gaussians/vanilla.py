"""
Filename: 3dgs.py

Author: Ziyu Chen (ziyu.sjtu@gmail.com)

Description:
Unofficial implementation of 3DGS based on the work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 
This implementation is modified from the nerfstudio GaussianSplattingModel.

- Original work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
- Codebase reference: nerfstudio GaussianSplattingModel (https://github.com/nerfstudio-project/nerfstudio/blob/gaussian-splatting/nerfstudio/models/gaussian_splatting.py)

Original paper: https://arxiv.org/abs/2308.04079
"""

from typing import Dict, List, Tuple, Optional
from omegaconf import OmegaConf
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter

import torch.nn.functional as F
from models.gaussians.basics import *
from models.floater_detection import (
    FloaterDetector, 
    FloaterDetectionConfig,
    SpatialSparsityRegularizer,
    SpatialSparsityConfig,
    DepthConsistencyRegularizer,
    evaluate_floater_reduction,
)

logger = logging.getLogger()

class VanillaGaussians(nn.Module):

    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 30.,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 300,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg
        self.networks_cfg = networks
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.step = 0
        
        self.device = device
        self.ball_gaussians=self.ctrl_cfg.get("ball_gaussians", False)
        self.gaussian_2d = self.ctrl_cfg.get("gaussian_2d", False)
        self.use_pbr = self.ctrl_cfg.get("use_pbr", False)
        self.use_emissive = False
        self.use_anisotropic_emissive = False
        self.emissive_sh_degree = 0
        self.emissive_sh_dim = 0
        
        # for evaluation
        self.in_test_set = False
        
        # init models
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self._means = torch.zeros(1, 3, device=self.device)
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        else:
            if self.gaussian_2d:
                self._scales = torch.zeros(1, 2, device=self.device)
            else:
                self._scales = torch.zeros(1, 3, device=self.device)
        self._quats = torch.zeros(1, 4, device=self.device)
        self._opacities = torch.zeros(1, 1, device=self.device)
        self._features_dc = torch.zeros(1, 3, device=self.device)
        self._features_rest = torch.zeros(1, num_sh_bases(self.sh_degree) - 1, 3, device=self.device)

        if self.use_pbr:
            # PBR attributes
            self._roughness = torch.zeros(1, 1, device=self.device)
            self._metallic = torch.zeros(1, 1, device=self.device)
            self._normals = torch.zeros(1, 3, device=self.device)
            # ASG attributes (4 lobes)
            # Amplitude (3 RGB), Sharpness (2 lambda,mu), Direction (2 theta,phi or 3 xyz)
            # Paper: "Each Gaussian is assigned a small set of ASG lobes (four in our case)"
            # Simplified: 1 lobe for now to check feasibility or 4? Let's use 1 to start.
            # No, let's follow the paper "four in our case".
            self.num_asg_lobes = 4
            self._asg_amplitude = torch.zeros(1, self.num_asg_lobes, 3, device=self.device)
            self._asg_sharpness = torch.zeros(1, self.num_asg_lobes, 2, device=self.device) 
            # Directions can be local or global. Assuming local to normal?
            # Or explicit directions.
            self._asg_dir = torch.zeros(1, self.num_asg_lobes, 3, device=self.device)
            
            # Emissive attributes for surface-supported bright radiance.
            self.use_emissive = self.ctrl_cfg.get("use_emissive", False)
            if self.use_emissive:
                self._emissive = torch.zeros(1, 3, device=self.device)
                self._emissive_intensity = torch.zeros(1, 1, device=self.device)
                self.use_anisotropic_emissive = bool(self.ctrl_cfg.get("use_anisotropic_emissive", False))
                self.emissive_sh_degree = int(self.ctrl_cfg.get("emissive_sh_degree", 1))
                self.emissive_sh_degree = max(0, self.emissive_sh_degree)
                self.emissive_sh_dim = num_sh_bases(self.emissive_sh_degree) if self.use_anisotropic_emissive else 0
                if self.use_anisotropic_emissive:
                    self._emissive_sh = torch.zeros(1, self.emissive_sh_dim, 3, device=self.device)
        
        # Initialize floater detection system
        self._init_floater_detection()
        
        # Depth points cache for surface constraints
        self._depth_points: Optional[torch.Tensor] = None
        # Runtime stats for adaptive anti-fog controls
        self._last_floater_ratio: float = 0.0
        self._last_detected_floaters: int = 0
        self._initial_num_points_for_growth_cap: Optional[int] = None
        self._last_densify_budget_scale: float = 1.0

    def _init_floater_detection(self):
        """Initialize floater detection and regularization systems."""
        # Check if floater detection is enabled in config
        floater_cfg = self.ctrl_cfg.get("floater_detection", None)
        if floater_cfg is not None and floater_cfg.get("enabled", False):
            self.floater_detector = FloaterDetector(
                config=FloaterDetectionConfig(
                    enabled=True,
                    grad_thresh=floater_cfg.get("grad_thresh", 0.0001),
                    opacity_thresh=floater_cfg.get("opacity_thresh", 0.1),
                    isolation_thresh=floater_cfg.get("isolation_thresh", 2.0),
                    depth_margin=floater_cfg.get("depth_margin", 1.0),
                    soft_prune_until=floater_cfg.get("soft_prune_until", 10000),
                    hard_prune_interval=floater_cfg.get("hard_prune_interval", 1000),
                    warmup_steps=floater_cfg.get("warmup_steps", 500),
                    floater_score_thresh=floater_cfg.get("floater_score_thresh", 0.7),
                ),
                device=str(self.device),
            )
            logger.info(f"[{self.class_prefix}] FloaterDetector initialized")
        else:
            self.floater_detector = None
        
        # Initialize spatial sparsity regularizer
        sparsity_cfg = self.reg_cfg.get("spatial_sparsity", None) if self.reg_cfg else None
        if sparsity_cfg is not None and sparsity_cfg.get("enabled", True):
            self.spatial_sparsity_reg = SpatialSparsityRegularizer(
                config=SpatialSparsityConfig(
                    enabled=True,
                    grid_size=sparsity_cfg.get("grid_size", 1.0),
                    max_density=sparsity_cfg.get("max_density", 10.0),
                    weight=sparsity_cfg.get("w", 0.01),
                    distance_scale=sparsity_cfg.get("distance_scale", 2.0),
                )
            )
            logger.info(f"[{self.class_prefix}] SpatialSparsityRegularizer initialized")
        else:
            self.spatial_sparsity_reg = None
        
        # Initialize depth consistency regularizer
        depth_cfg = self.reg_cfg.get("depth_consistency", None) if self.reg_cfg else None
        if depth_cfg is not None:
            self.depth_consistency_reg = DepthConsistencyRegularizer(
                max_surface_distance=depth_cfg.get("max_surface_distance", 0.5),
                weight=depth_cfg.get("w", 0.1),
                use_ray_consistency=depth_cfg.get("use_ray_consistency", True),
            )
            logger.info(f"[{self.class_prefix}] DepthConsistencyRegularizer initialized")
        else:
            self.depth_consistency_reg = None

    def set_depth_points(self, depth_points: torch.Tensor):
        """
        Set the estimated depth points for surface constraints.
        
        Args:
            depth_points: Surface points from depth estimation (M, 3)
        """
        self._depth_points = depth_points.to(self.device)

    @property
    def sh_degree(self):
        return self.ctrl_cfg.sh_degree

    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        self._means = Parameter(init_means)
        
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        if self.ball_gaussians:
            self._scales = Parameter(torch.log(avg_dist.repeat(1, 1)))
        else:
            if self.gaussian_2d:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 2)))
            else:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)

        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))
        
        if self.use_pbr:
            # Initialize PBR params
            self._roughness = Parameter(torch.logit(0.5 * torch.ones(self.num_points, 1, device=self.device)))
            self._metallic = Parameter(torch.logit(0.5 * torch.ones(self.num_points, 1, device=self.device)))
            # Initialize normals (random or from somewhere? For now random)
            self._normals = Parameter(F.normalize(torch.randn(self.num_points, 3, device=self.device), dim=-1))
            
            # ASG params
            self._asg_amplitude = Parameter(torch.logit(0.1 * torch.ones(self.num_points, self.num_asg_lobes, 3, device=self.device)))
            self._asg_sharpness = Parameter(torch.logit(0.5 * torch.ones(self.num_points, self.num_asg_lobes, 2, device=self.device)))
            self._asg_dir = Parameter(F.normalize(torch.randn(self.num_points, self.num_asg_lobes, 3, device=self.device), dim=-1))
            
            # Emissive params (Phase 3) - initialize to near zero
            if self.use_emissive:
                self._emissive = Parameter(torch.zeros(self.num_points, 3, device=self.device))
                self._emissive_intensity = Parameter(torch.full((self.num_points, 1), -5.0, device=self.device))  # exp(-5) ≈ 0.007
                if self.use_anisotropic_emissive:
                    self._emissive_sh = Parameter(torch.zeros(self.num_points, self.emissive_sh_dim, 3, device=self.device))

        # Keep a stable reference for early-growth guardrails.
        self._initial_num_points_for_growth_cap = int(self.num_points)

        
    @property
    def colors(self):
        if self.sh_degree > 0:
            return SH2RGB(self._features_dc)
        else:
            return torch.sigmoid(self._features_dc)
    @property
    def shs_0(self):
        return self._features_dc
    @property
    def shs_rest(self):
        return self._features_rest
    @property
    def num_points(self):
        return self._means.shape[0]
    @property
    def get_scaling(self):
        if self.ball_gaussians:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales).repeat(1, 2)
                scaling = torch.cat([scaling, torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales).repeat(1, 3)
        else:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales)
                scaling = torch.cat([scaling[..., :2], torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales)
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacities)
    @property
    def get_quats(self):
        return self.quat_act(self._quats)
    
    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)
    
    def preprocess_per_train_step(self, step: int):
        self.step = step
        
    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        self.after_train(radii, xys_grad, last_size)
        if step % self.ctrl_cfg.refine_interval == 0:
            self.refinement_after(step, optimizer)

    def after_train(
        self,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (radii > 0).flatten()
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask
            
            grads = xys_grad.norm(dim=-1)
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )
        
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = {
            self.class_prefix+"xyz": [self._means],
            self.class_prefix+"sh_dc": [self._features_dc],
            self.class_prefix+"sh_rest": [self._features_rest],
            self.class_prefix+"opacity": [self._opacities],
            self.class_prefix+"scaling": [self._scales],
            self.class_prefix+"rotation": [self._quats],
        }
        if self.use_pbr:
            groups[self.class_prefix+"roughness"] = [self._roughness]
            groups[self.class_prefix+"metallic"] = [self._metallic]
            groups[self.class_prefix+"normals"] = [self._normals]
            groups[self.class_prefix+"asg_amplitude"] = [self._asg_amplitude]
            groups[self.class_prefix+"asg_sharpness"] = [self._asg_sharpness]
            groups[self.class_prefix+"asg_dir"] = [self._asg_dir]
            if self.use_emissive:
                groups[self.class_prefix+"emissive"] = [self._emissive]
                groups[self.class_prefix+"emissive_intensity"] = [self._emissive_intensity]
                if self.use_anisotropic_emissive:
                    groups[self.class_prefix+"emissive_sh"] = [self._emissive_sh]
        return groups
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()
    
    def adaptive_floater_pruning(self, step: int, optimizer: torch.optim.Optimizer) -> int:
        """
        Perform adaptive floater detection and pruning.
        
        Args:
            step: Current training step
            optimizer: Optimizer instance
            
        Returns:
            Number of pruned Gaussians
        """
        if self.floater_detector is None:
            return 0
        
        config = self.floater_detector.config
        floater_ctrl = self.ctrl_cfg.get("floater_detection", {})
        
        # Check if we should prune at this step
        if step < config.warmup_steps:
            return 0
        
        # Optional safety: after opacity reset, force soft-pruning for a short grace window.
        reset_grace_steps = int(floater_ctrl.get("reset_alpha_grace_steps", 0))
        reset_soft_only = bool(floater_ctrl.get("reset_alpha_soft_only", True))
        force_soft_prune = False
        if reset_soft_only and reset_grace_steps > 0:
            reset_interval = int(self.ctrl_cfg.get("reset_alpha_interval", 0))
            reset_offset = int(self.ctrl_cfg.get("refine_interval", 0))
            if reset_interval > 0 and step >= reset_offset:
                steps_since_reset = (step - reset_offset) % reset_interval
                force_soft_prune = steps_since_reset < reset_grace_steps

        if (not force_soft_prune) and step >= config.soft_prune_until and step % config.hard_prune_interval != 0:
            return 0
        
        # Detect floaters
        floater_mask, floater_scores = self.floater_detector.detect_floaters(
            means=self._means.detach(),
            opacities=self.get_opacity.detach(),
            xys_grad_norm=self.xys_grad_norm,
            vis_counts=self.vis_counts if hasattr(self, 'vis_counts') else None,
            depth_points=self._depth_points,
            step=step,
        )
        
        # Safety gate 1: protect high-opacity points (likely stable geometry).
        protect_opacity_thresh = float(floater_ctrl.get("protect_high_opacity_thresh", -1.0))
        if protect_opacity_thresh > 0:
            opacity_vals = self.get_opacity.detach().squeeze(-1)
            protect_mask = (opacity_vals > protect_opacity_thresh) & floater_mask
            if protect_mask.any():
                floater_mask = floater_mask & (~protect_mask)
                logger.info(
                    f"[{self.class_prefix}] Floater protection (opacity): kept {protect_mask.sum().item()} points"
                )

        # Safety gate 2: protect high-contribution points by gradient percentile.
        protect_grad_percentile = float(floater_ctrl.get("protect_high_grad_percentile", 0.0))
        if (
            0.0 < protect_grad_percentile < 1.0
            and self.xys_grad_norm is not None
            and hasattr(self, "vis_counts")
            and self.vis_counts is not None
            and self.xys_grad_norm.shape[0] == floater_mask.shape[0]
            and self.vis_counts.shape[0] == floater_mask.shape[0]
        ):
            avg_grad = self.xys_grad_norm / self.vis_counts.clamp(min=1)
            grad_thr = torch.quantile(avg_grad.detach(), protect_grad_percentile)
            protect_mask = (avg_grad >= grad_thr) & floater_mask
            if protect_mask.any():
                floater_mask = floater_mask & (~protect_mask)
                logger.info(
                    f"[{self.class_prefix}] Floater protection (grad): kept {protect_mask.sum().item()} points"
                )

        # Safety gate 3: protect points near depth surfaces.
        protect_surface_distance = float(floater_ctrl.get("protect_surface_distance", 0.0))
        if (
            protect_surface_distance > 0
            and self._depth_points is not None
            and self._depth_points.shape[0] > 0
        ):
            candidate_idx = torch.where(floater_mask)[0]
            if candidate_idx.numel() > 0:
                depth_pts = self._depth_points
                if depth_pts.shape[0] > 50000:
                    sample_idx = torch.randperm(depth_pts.shape[0], device=self.device)[:50000]
                    depth_pts = depth_pts[sample_idx]
                dists = torch.cdist(self._means[candidate_idx].detach(), depth_pts)
                min_dist, _ = dists.min(dim=1)
                protect_local = min_dist < protect_surface_distance
                if protect_local.any():
                    protect_idx = candidate_idx[protect_local]
                    floater_mask[protect_idx] = False
                    logger.info(
                        f"[{self.class_prefix}] Floater protection (surface): kept {protect_idx.numel()} points"
                    )

        num_floaters = floater_mask.sum().item()
        self._last_detected_floaters = int(num_floaters)
        self._last_floater_ratio = float(num_floaters / max(1, self.num_points))
        
        if num_floaters == 0:
            return 0
        
        # Soft pruning: gently shrink floater opacity logits so these points fade out.
        if step < config.soft_prune_until or force_soft_prune:
            soft_ratio = float(floater_ctrl.get("max_soft_prune_ratio", 0.02))
            soft_ratio = max(0.0, min(1.0, soft_ratio))
            shrink = float(floater_ctrl.get("soft_opacity_shrink", 0.98))
            shrink = max(0.90, min(0.999, shrink))
            topk = int(max(1, self.num_points * soft_ratio))
            if topk < num_floaters:
                scores = floater_scores[floater_mask]
                floater_idx = torch.where(floater_mask)[0]
                _, sel = torch.topk(scores, k=topk, largest=True, sorted=False)
                selected_idx = floater_idx[sel]
            else:
                selected_idx = torch.where(floater_mask)[0]

            with torch.no_grad():
                self._opacities[selected_idx] = self._opacities[selected_idx] + np.log(shrink)
            logger.info(
                f"[{self.class_prefix}] Soft pruning: shrunk opacity logits for {selected_idx.numel()} floaters"
            )
            return 0
        
        # Hard pruning with surface-distance priority and per-step prune cap.
        prune_cap_ratio = float(floater_ctrl.get("max_prune_ratio_per_hard_step", 0.02))
        prune_cap_ratio = max(0.0, min(1.0, prune_cap_ratio))
        candidate_cap_ratio = float(floater_ctrl.get("max_candidate_prune_ratio", 1.0))
        candidate_cap_ratio = max(0.0, min(1.0, candidate_cap_ratio))
        far_surface_weight = float(floater_ctrl.get("far_surface_priority_weight", 0.25))
        far_surface_margin = float(floater_ctrl.get("light_far_surface_margin", config.depth_margin))
        free_space_weight = float(floater_ctrl.get("free_space_priority_weight", 0.0))
        free_space_margin = float(floater_ctrl.get("free_space_priority_margin", far_surface_margin))

        candidate_idx = torch.where(floater_mask)[0]
        if candidate_idx.numel() == 0:
            return 0

        priority_scores = floater_scores[candidate_idx]
        if self._depth_points is not None and self._depth_points.shape[0] > 0 and (far_surface_weight > 0 or free_space_weight > 0):
            depth_pts = self._depth_points
            if depth_pts.shape[0] > 50000:
                sample_idx = torch.randperm(depth_pts.shape[0], device=self.device)[:50000]
                depth_pts = depth_pts[sample_idx]
            dists = torch.cdist(self._means[candidate_idx].detach(), depth_pts)
            min_dist, _ = dists.min(dim=1)
            if far_surface_weight > 0:
                far_surface_score = torch.clamp(
                    (min_dist - far_surface_margin) / max(far_surface_margin, 1e-6),
                    min=0.0,
                    max=1.0,
                )
                priority_scores = priority_scores + far_surface_weight * far_surface_score
            if free_space_weight > 0:
                free_space_score = torch.clamp(
                    (min_dist - free_space_margin) / max(free_space_margin, 1e-6),
                    min=0.0,
                    max=1.0,
                )
                priority_scores = priority_scores + free_space_weight * free_space_score

        prune_limit = int(max(1, self.num_points * prune_cap_ratio))
        candidate_limit = int(max(1, candidate_idx.numel() * candidate_cap_ratio))
        n_prune = min(candidate_idx.numel(), prune_limit, candidate_limit)
        if n_prune <= 0:
            return 0
        _, top_idx = torch.topk(priority_scores, k=n_prune, largest=True, sorted=False)
        prune_idx = candidate_idx[top_idx]
        final_mask = torch.zeros_like(floater_mask)
        final_mask[prune_idx] = True

        logger.info(
            f"[{self.class_prefix}] Hard pruning: removing {n_prune}/{num_floaters} floaters "
            f"(cap_total={prune_limit}, cap_candidate={candidate_limit})"
        )
        self.prune_points(final_mask, optimizer)
        
        return n_prune

    def refinement_after(self, step, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            effective_stop_split_at = self._get_effective_stop_split_at()
            do_densification = (
                self.step < effective_stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                bright_densify_scale = self._compute_bright_densify_scale()
                if bright_densify_scale > 1.0:
                    bright_mask = self._compute_bright_gaussian_mask()
                    bright_high_grads = (avg_grad_norm > (self.ctrl_cfg.densify_grad_thresh * bright_densify_scale)).squeeze()
                    high_grads = torch.where(bright_mask, bright_high_grads, high_grads)
                
                splits = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples

                dups = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads

                # For anti-fog stability: reject split seeds that are too far from depth surfaces.
                splits = self._filter_split_candidates_by_surface(splits)

                # For anti-fog source suppression: block growth of bright/free-space/low-opacity seeds.
                splits, dups = self._suppress_foggy_growth_candidates(splits, dups, avg_grad_norm)
                # Surface-supported emissive geometry: allow shape refinement, block duplicate diffusion.
                emissive_geo_cfg = self.ctrl_cfg.get("emissive_surface_geometry", None)
                if (
                    emissive_geo_cfg is not None
                    and bool(emissive_geo_cfg.get("enabled", False))
                    and bool(emissive_geo_cfg.get("dup_block", True))
                    and self.use_pbr
                    and self.use_emissive
                ):
                    high_emissive_mask = self._compute_high_emissive_mask()
                    dups = dups & (~high_emissive_mask)

                supported_geo_cfg = None
                surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
                if surface_guided_cfg is not None:
                    supported_geo_cfg = surface_guided_cfg.get("supported_highlight_geometry", None)
                if supported_geo_cfg is not None and bool(supported_geo_cfg.get("enabled", False)):
                    supported_bright_mask = self._compute_surface_supported_bright_mask(supported_geo_cfg)
                    if supported_bright_mask.any():
                        if bool(supported_geo_cfg.get("block_duplicate", True)):
                            dups = dups & (~supported_bright_mask)
                        if bool(supported_geo_cfg.get("surface_split_only", True)):
                            splits = self._filter_split_candidates_by_surface(splits)

                # For anti-fog stability: cap per-step densification and bright-region share.
                splits, dups = self._apply_densify_budgets(splits, dups, avg_grad_norm, nsamps)
                # Safety: make sure masks stay boolean; integer 0/1 masks would be interpreted as indices.
                splits = splits.bool()
                dups = dups.bool()
                
                if self.use_pbr:
                    (
                        split_means, split_feature_dc, split_feature_rest, split_opacities, split_scales, split_quats,
                        split_roughness, split_metallic, split_normals, split_asg_amplitude, split_asg_sharpness, split_asg_dir,
                        split_emissive, split_emissive_intensity, split_emissive_sh
                    ) = self.split_gaussians(splits, nsamps)
                else:
                    (
                        split_means, split_feature_dc, split_feature_rest, split_opacities, split_scales, split_quats,
                    ) = self.split_gaussians(splits, nsamps)

                if self.use_pbr:
                    (
                        dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats,
                        dup_roughness, dup_metallic, dup_normals, dup_asg_amplitude, dup_asg_sharpness, dup_asg_dir,
                        dup_emissive, dup_emissive_intensity, dup_emissive_sh
                    ) = self.dup_gaussians(dups)
                else:
                    (
                        dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats,
                    ) = self.dup_gaussians(dups)
                
                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                
                if self.use_pbr:
                    self._roughness = Parameter(torch.cat([self._roughness.detach(), split_roughness, dup_roughness], dim=0))
                    self._metallic = Parameter(torch.cat([self._metallic.detach(), split_metallic, dup_metallic], dim=0))
                    self._normals = Parameter(torch.cat([self._normals.detach(), split_normals, dup_normals], dim=0))
                    self._asg_amplitude = Parameter(torch.cat([self._asg_amplitude.detach(), split_asg_amplitude, dup_asg_amplitude], dim=0))
                    self._asg_sharpness = Parameter(torch.cat([self._asg_sharpness.detach(), split_asg_sharpness, dup_asg_sharpness], dim=0))
                    self._asg_dir = Parameter(torch.cat([self._asg_dir.detach(), split_asg_dir, dup_asg_dir], dim=0))
                    if self.use_emissive:
                        self._emissive = Parameter(torch.cat([self._emissive.detach(), split_emissive, dup_emissive], dim=0))
                        self._emissive_intensity = Parameter(
                            torch.cat([self._emissive_intensity.detach(), split_emissive_intensity, dup_emissive_intensity], dim=0)
                        )
                        if self.use_anisotropic_emissive:
                            self._emissive_sh = Parameter(
                                torch.cat([self._emissive_sh.detach(), split_emissive_sh, dup_emissive_sh], dim=0)
                            )
                
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                
                # Also update xys_grad_norm and vis_counts for new points
                if self.xys_grad_norm is not None:
                    n_new = split_scales.shape[0] + dup_scales.shape[0]
                    self.xys_grad_norm = torch.cat(
                        [self.xys_grad_norm, torch.zeros(n_new, device=self.device, dtype=self.xys_grad_norm.dtype)],
                        dim=0,
                    )
                if hasattr(self, 'vis_counts') and self.vis_counts is not None:
                    n_new = split_scales.shape[0] + dup_scales.shape[0]
                    self.vis_counts = torch.cat(
                        [self.vis_counts, torch.zeros(n_new, device=self.device, dtype=self.vis_counts.dtype)],
                        dim=0,
                    )
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")
                    
            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        if old_params in optimizer.state:
                            param_state = optimizer.state[old_params]
                            if "exp_avg" in param_state:
                                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                            if "exp_avg_sq" in param_state:
                                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            
            # ========== Floater Pruning (Anti-fog) ==========
            self.adaptive_floater_pruning(step, optimizer)
            self._apply_emissive_surface_thickness_clip()
            
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def _get_surface_growth_stage_cfg(self) -> Optional[dict]:
        return self._get_surface_schedule_stage_cfg("growth_control_schedule")

    def _get_surface_schedule_stage_cfg(self, schedule_key: str) -> Optional[dict]:
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is None:
            return None
        schedule = surface_guided_cfg.get(schedule_key, None)
        if schedule is None:
            return None
        for stage in schedule:
            until_step = int(stage.get("until_step", -1))
            if until_step < 0 or self.step <= until_step:
                return stage
        return None

    def _get_fog_hard_gate_stage_cfg(self) -> Optional[dict]:
        return self._get_surface_schedule_stage_cfg("fog_hard_gate_schedule")

    def _get_effective_stop_split_at(self) -> int:
        stop_split_at = int(self.ctrl_cfg.get("stop_split_at", 0))
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is None:
            return stop_split_at

        adaptive_stop_split_at = int(surface_guided_cfg.get("adaptive_stop_split_at", -1))
        if adaptive_stop_split_at > 0:
            stop_split_at = min(stop_split_at, adaptive_stop_split_at)

        stage_cfg = self._get_surface_growth_stage_cfg()
        if stage_cfg is not None:
            stage_stop_split_at = int(stage_cfg.get("stop_split_at", -1))
            if stage_stop_split_at > 0:
                stop_split_at = min(stop_split_at, stage_stop_split_at)
        return stop_split_at

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        
        if self.use_pbr:
            self._roughness = Parameter(self._roughness[~culls].detach())
            self._metallic = Parameter(self._metallic[~culls].detach())
            self._normals = Parameter(self._normals[~culls].detach())
            self._asg_amplitude = Parameter(self._asg_amplitude[~culls].detach())
            self._asg_sharpness = Parameter(self._asg_sharpness[~culls].detach())
            self._asg_dir = Parameter(self._asg_dir[~culls].detach())
            if self.use_emissive:
                self._emissive = Parameter(self._emissive[~culls].detach())
                self._emissive_intensity = Parameter(self._emissive_intensity[~culls].detach())
                if self.use_anisotropic_emissive:
                    self._emissive_sh = Parameter(self._emissive_sh[~culls].detach())
        
        # Sync statistical buffers
        if self.xys_grad_norm is not None:
            if self.xys_grad_norm.shape[0] == culls.shape[0]:
                self.xys_grad_norm = self.xys_grad_norm[~culls]
            else:
                self.xys_grad_norm = None
        if hasattr(self, 'vis_counts') and self.vis_counts is not None:
            if self.vis_counts.shape[0] == culls.shape[0]:
                self.vis_counts = self.vis_counts[~culls]
            else:
                self.vis_counts = None
        if self.max_2Dsize is not None:
            if self.max_2Dsize.shape[0] == culls.shape[0]:
                self.max_2Dsize = self.max_2Dsize[~culls]
            else:
                self.max_2Dsize = None

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        This function splits gaussians that are too large.
        With surface-guided densification, new Gaussians are pulled toward depth surfaces.
        """
        split_mask = split_mask.bool()

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        
        # ========== Surface-Guided Split (Phase 2) ==========
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is not None and surface_guided_cfg.get("enabled", False):
            if self._depth_points is not None and self._depth_points.shape[0] > 0:
                pull_factor = surface_guided_cfg.get("pull_factor", 0.5)
                
                # Sample depth points for efficiency
                depth_pts = self._depth_points
                if depth_pts.shape[0] > 50000:
                    sample_idx = torch.randperm(depth_pts.shape[0], device=self.device)[:50000]
                    depth_pts = depth_pts[sample_idx]
                
                # Find nearest surface points
                dists = torch.cdist(new_means, depth_pts)
                min_dist, nearest_idx = dists.min(dim=1)
                nearest_surface_pts = depth_pts[nearest_idx]
                
                # Pull toward surface
                direction = nearest_surface_pts - new_means
                direction = F.normalize(direction, dim=-1)
                pull_distance = min_dist.unsqueeze(-1) * pull_factor
                
                new_means = new_means + direction * pull_distance
                
                print(f"    Surface-guided: avg pull distance = {(min_dist * pull_factor).mean().item():.4f}")
        
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)

        if self.use_pbr:
            new_roughness = self._roughness[split_mask].repeat(samps, 1)
            new_metallic = self._metallic[split_mask].repeat(samps, 1)
            new_normals = self._normals[split_mask].repeat(samps, 1)
            new_asg_amplitude = self._asg_amplitude[split_mask].repeat(samps, 1, 1)
            new_asg_sharpness = self._asg_sharpness[split_mask].repeat(samps, 1, 1)
            new_asg_dir = self._asg_dir[split_mask].repeat(samps, 1, 1)
            if self.use_emissive:
                new_emissive = self._emissive[split_mask].repeat(samps, 1)
                new_emissive_intensity = self._emissive_intensity[split_mask].repeat(samps, 1)
                if self.use_anisotropic_emissive:
                    new_emissive_sh = self._emissive_sh[split_mask].repeat(samps, 1, 1)
                else:
                    new_emissive_sh = torch.empty((new_means.shape[0], 0, 3), device=self.device, dtype=new_means.dtype)
            else:
                new_emissive = torch.empty((new_means.shape[0], 3), device=self.device, dtype=new_means.dtype)
                new_emissive_intensity = torch.empty((new_means.shape[0], 1), device=self.device, dtype=new_means.dtype)
                new_emissive_sh = torch.empty((new_means.shape[0], 0, 3), device=self.device, dtype=new_means.dtype)
            return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, \
                   new_roughness, new_metallic, new_normals, new_asg_amplitude, new_asg_sharpness, new_asg_dir, \
                   new_emissive, new_emissive_intensity, new_emissive_sh
                   
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small.
        With surface-guided densification, clones far from surfaces can be rejected.
        """
        dup_mask = dup_mask.bool()
        # ========== Surface-Guided Clone Filter (Phase 2) ==========
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is not None and surface_guided_cfg.get("enabled", False):
            if self._depth_points is not None and self._depth_points.shape[0] > 0:
                max_dup_distance = surface_guided_cfg.get("max_dup_distance", 1.0)
                hard_gate_stage_cfg = self._get_fog_hard_gate_stage_cfg()
                if hard_gate_stage_cfg is not None:
                    max_dup_distance = float(hard_gate_stage_cfg.get("dup_surface_distance_max", max_dup_distance))
                
                # Sample depth points for efficiency
                depth_pts = self._depth_points
                if depth_pts.shape[0] > 50000:
                    sample_idx = torch.randperm(depth_pts.shape[0], device=self.device)[:50000]
                    depth_pts = depth_pts[sample_idx]
                
                # Compute distances to nearest surface
                dup_means_candidate = self._means[dup_mask]
                dists = torch.cdist(dup_means_candidate, depth_pts)
                min_dist, _ = dists.min(dim=1)
                
                # Only keep clones near surfaces
                valid_clone_mask = min_dist < max_dup_distance
                n_rejected = (~valid_clone_mask).sum().item()
                
                if n_rejected > 0:
                    print(f"    Surface-guided clone: rejected {n_rejected} clones far from surface")
                    # Update dup_mask to exclude far clones
                    dup_indices = torch.where(dup_mask)[0]
                    dup_mask = dup_mask.clone()
                    dup_mask[dup_indices[~valid_clone_mask]] = False
        
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]

        if self.use_pbr:
            dup_roughness = self._roughness[dup_mask]
            dup_metallic = self._metallic[dup_mask]
            dup_normals = self._normals[dup_mask]
            dup_asg_amplitude = self._asg_amplitude[dup_mask]
            dup_asg_sharpness = self._asg_sharpness[dup_mask]
            dup_asg_dir = self._asg_dir[dup_mask]
            if self.use_emissive:
                dup_emissive = self._emissive[dup_mask]
                dup_emissive_intensity = self._emissive_intensity[dup_mask]
                if self.use_anisotropic_emissive:
                    dup_emissive_sh = self._emissive_sh[dup_mask]
                else:
                    dup_emissive_sh = torch.empty((dup_means.shape[0], 0, 3), device=self.device, dtype=dup_means.dtype)
            else:
                dup_emissive = torch.empty((dup_means.shape[0], 3), device=self.device, dtype=dup_means.dtype)
                dup_emissive_intensity = torch.empty((dup_means.shape[0], 1), device=self.device, dtype=dup_means.dtype)
                dup_emissive_sh = torch.empty((dup_means.shape[0], 0, 3), device=self.device, dtype=dup_means.dtype)
            return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats, \
                   dup_roughness, dup_metallic, dup_normals, dup_asg_amplitude, dup_asg_sharpness, dup_asg_dir, \
                   dup_emissive, dup_emissive_intensity, dup_emissive_sh

        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats

    def _filter_split_candidates_by_surface(self, split_mask: torch.Tensor) -> torch.Tensor:
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is None or not surface_guided_cfg.get("enabled", False):
            return split_mask
        if self._depth_points is None or self._depth_points.shape[0] == 0:
            return split_mask

        max_split_distance = float(surface_guided_cfg.get("max_split_distance", -1.0))
        hard_gate_stage_cfg = self._get_fog_hard_gate_stage_cfg()
        if hard_gate_stage_cfg is not None:
            max_split_distance = float(hard_gate_stage_cfg.get("split_surface_distance_max", max_split_distance))
        if max_split_distance <= 0:
            return split_mask

        split_idx = torch.where(split_mask)[0]
        if split_idx.numel() == 0:
            return split_mask

        depth_pts = self._depth_points
        if depth_pts.shape[0] > 50000:
            sample_idx = torch.randperm(depth_pts.shape[0], device=self.device)[:50000]
            depth_pts = depth_pts[sample_idx]

        dists = torch.cdist(self._means[split_idx], depth_pts)
        min_dist, _ = dists.min(dim=1)
        keep = min_dist < max_split_distance
        if hard_gate_stage_cfg is not None:
            min_grad_scale = float(hard_gate_stage_cfg.get("split_min_grad_scale", 0.0))
            if min_grad_scale > 0 and self.xys_grad_norm is not None and self.vis_counts is not None:
                if self.xys_grad_norm.shape[0] == self.num_points and self.vis_counts.shape[0] == self.num_points:
                    avg_grad = self.xys_grad_norm / self.vis_counts.clamp(min=1)
                    grad_thr = self.ctrl_cfg.densify_grad_thresh * min_grad_scale
                    keep = keep & (avg_grad[split_idx].squeeze(-1) >= grad_thr)
        rejected = (~keep).sum().item()
        if rejected > 0:
            split_mask = split_mask.clone()
            split_mask[split_idx[~keep]] = False
            print(f"    Surface-guided split: rejected {rejected} split seeds far from surface")
        return split_mask

    def _compute_free_space_gaussian_mask(self) -> torch.Tensor:
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is None or not surface_guided_cfg.get("enabled", False):
            return torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
        if self._depth_points is None or self._depth_points.shape[0] == 0:
            return torch.zeros(self.num_points, dtype=torch.bool, device=self.device)

        free_space_distance_min = float(surface_guided_cfg.get("free_space_seed_distance_min", -1.0))
        free_space_opacity_max = float(surface_guided_cfg.get("free_space_seed_opacity_max", -1.0))
        if free_space_distance_min <= 0 or free_space_opacity_max <= 0:
            return torch.zeros(self.num_points, dtype=torch.bool, device=self.device)

        depth_pts = self._depth_points
        if depth_pts.shape[0] > 50000:
            sample_idx = torch.randperm(depth_pts.shape[0], device=self.device)[:50000]
            depth_pts = depth_pts[sample_idx]
        dists = torch.cdist(self._means.detach(), depth_pts)
        min_dist, _ = dists.min(dim=1)

        opacity_vals = self.get_opacity.detach().squeeze(-1)
        free_space_mask = (min_dist > free_space_distance_min) & (opacity_vals < free_space_opacity_max)

        if bool(surface_guided_cfg.get("free_space_seed_only_bright", False)):
            bright_mask = self._compute_bright_gaussian_mask()
            free_space_mask = free_space_mask & bright_mask
        return free_space_mask

    def _compute_surface_supported_bright_mask(self, cfg) -> torch.Tensor:
        if self._depth_points is None or self._depth_points.shape[0] == 0:
            return torch.zeros(self.num_points, dtype=torch.bool, device=self.device)

        albedo = torch.sigmoid(self._features_dc.detach())
        luma = 0.299 * albedo[:, 0] + 0.587 * albedo[:, 1] + 0.114 * albedo[:, 2]
        luma_min = float(cfg.get("luma_min", 0.65))
        bright_mask = luma > luma_min

        if self.use_pbr and self.use_emissive:
            emissive_intensity = torch.exp(torch.clamp(self._emissive_intensity.detach().squeeze(-1), min=-8.0, max=8.0))
            bright_mask = bright_mask | (emissive_intensity > float(cfg.get("emissive_min", 0.8)))

        if not bright_mask.any():
            return bright_mask

        candidate_idx = torch.where(bright_mask)[0]
        depth_pts = self._depth_points
        max_surface_samples = int(cfg.get("max_surface_samples", 50000))
        if max_surface_samples > 0 and depth_pts.shape[0] > max_surface_samples:
            sample_idx = torch.randperm(depth_pts.shape[0], device=self.device)[:max_surface_samples]
            depth_pts = depth_pts[sample_idx]

        dists = torch.cdist(self._means[candidate_idx], depth_pts)
        min_dist, _ = dists.min(dim=1)
        supported = min_dist <= float(cfg.get("surface_distance_max", 0.35))
        supported_mask = torch.zeros_like(bright_mask)
        supported_mask[candidate_idx[supported]] = True
        return supported_mask

    def _suppress_foggy_growth_candidates(
        self,
        splits: torch.Tensor,
        dups: torch.Tensor,
        avg_grad_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is None or not surface_guided_cfg.get("enabled", False):
            return splits, dups
        if not bool(surface_guided_cfg.get("suppress_foggy_growth", False)):
            return splits, dups

        fog_seed_opacity_max = float(surface_guided_cfg.get("fog_seed_opacity_max", -1.0))
        fog_seed_opacity_soft_max = float(
            surface_guided_cfg.get("fog_seed_opacity_soft_max", fog_seed_opacity_max)
        )
        if fog_seed_opacity_max <= 0 and fog_seed_opacity_soft_max <= 0:
            return splits, dups

        bright_mask = self._compute_bright_gaussian_mask()
        free_space_mask = self._compute_free_space_gaussian_mask()
        supported_geo_cfg = surface_guided_cfg.get("supported_highlight_geometry", None)
        if supported_geo_cfg is not None and bool(supported_geo_cfg.get("enabled", False)):
            supported_bright_mask = self._compute_surface_supported_bright_mask(supported_geo_cfg)
            if supported_bright_mask.any():
                bright_mask = bright_mask & (~supported_bright_mask)
        foggy_mask = bright_mask | free_space_mask
        if not foggy_mask.any():
            return splits, dups

        opacity = self.get_opacity.detach().squeeze(-1)
        # Strong hit: block split+dup for bright/free-space seeds with low opacity.
        strong_block_mask = torch.zeros_like(foggy_mask)
        if fog_seed_opacity_max > 0:
            strong_block_mask = foggy_mask & (opacity < fog_seed_opacity_max)

        # Weak hit: block split only for brighter free-space seeds with mid opacity but weak gradients.
        grad_vals = avg_grad_norm.squeeze(-1).detach()
        low_grad_scale = float(surface_guided_cfg.get("fog_seed_low_grad_scale", 1.0))
        low_grad_scale = max(0.0, low_grad_scale)
        low_grad_mask = grad_vals < (self.ctrl_cfg.densify_grad_thresh * low_grad_scale)
        weak_block_mask = torch.zeros_like(foggy_mask)
        if fog_seed_opacity_soft_max > 0:
            weak_block_mask = foggy_mask & (opacity < fog_seed_opacity_soft_max) & low_grad_mask & (~strong_block_mask)

        if not strong_block_mask.any() and not weak_block_mask.any():
            return splits, dups

        # Free-space specific gate: block split by default and heavily throttle duplicates.
        free_space_split_block = bool(surface_guided_cfg.get("free_space_split_block", True))
        free_space_dup_keep_ratio = float(surface_guided_cfg.get("free_space_dup_keep_ratio", 1.0))
        free_space_dup_keep_ratio = max(0.0, min(1.0, free_space_dup_keep_ratio))
        free_space_only_weak = bool(surface_guided_cfg.get("free_space_only_weak_block", False))
        hard_gate_stage_cfg = self._get_fog_hard_gate_stage_cfg()
        if hard_gate_stage_cfg is not None:
            free_space_split_block = bool(hard_gate_stage_cfg.get("free_space_split_block", free_space_split_block))
            free_space_dup_keep_ratio = float(
                hard_gate_stage_cfg.get("free_space_dup_keep_ratio", free_space_dup_keep_ratio)
            )
            free_space_dup_keep_ratio = max(0.0, min(1.0, free_space_dup_keep_ratio))
        free_space_block_mask = free_space_mask & ((~free_space_only_weak) | (opacity < fog_seed_opacity_soft_max))
        if hard_gate_stage_cfg is not None and bool(hard_gate_stage_cfg.get("free_space_no_growth", False)):
            free_space_block_mask = free_space_mask

        # Strong + weak both suppress split; only strong suppresses duplicate.
        split_block_mask = strong_block_mask | weak_block_mask | (free_space_block_mask if free_space_split_block else torch.zeros_like(free_space_block_mask))
        dup_block_mask = strong_block_mask

        if hard_gate_stage_cfg is not None:
            if bool(hard_gate_stage_cfg.get("free_space_no_growth", False)):
                split_block_mask = split_block_mask | free_space_mask
                dup_block_mask = dup_block_mask | free_space_mask
            if bool(hard_gate_stage_cfg.get("bright_low_opacity_no_growth", False)):
                bright_opacity_max = float(hard_gate_stage_cfg.get("bright_low_opacity_max", fog_seed_opacity_soft_max))
                bright_block = bright_mask & (opacity < bright_opacity_max)
                split_block_mask = split_block_mask | bright_block
                dup_block_mask = dup_block_mask | bright_block
            if bool(hard_gate_stage_cfg.get("unsupported_highlight_no_growth", False)):
                unsupported_opacity_max = float(hard_gate_stage_cfg.get("unsupported_highlight_opacity_max", 1.0))
                unsupported_bright_block = bright_mask & (opacity < unsupported_opacity_max)
                split_block_mask = split_block_mask | unsupported_bright_block
                dup_block_mask = dup_block_mask | unsupported_bright_block
            hard_min_grad_scale = float(hard_gate_stage_cfg.get("fog_min_grad_scale", 0.0))
            if hard_min_grad_scale > 0:
                hard_low_grad = grad_vals < (self.ctrl_cfg.densify_grad_thresh * hard_min_grad_scale)
                hard_grad_block = foggy_mask & hard_low_grad
                split_block_mask = split_block_mask | hard_grad_block
                dup_block_mask = dup_block_mask | hard_grad_block

        new_splits = splits & (~split_block_mask)
        new_dups = dups & (~dup_block_mask)

        # Keep only top-k duplicates inside free-space candidates.
        if free_space_dup_keep_ratio < 1.0:
            free_space_dup_candidates = torch.where(new_dups & free_space_block_mask)[0]
            if free_space_dup_candidates.numel() > 0:
                keep_k = int(free_space_dup_candidates.numel() * free_space_dup_keep_ratio)
                keep_k = max(0, min(free_space_dup_candidates.numel(), keep_k))
                if keep_k == 0:
                    new_dups[free_space_dup_candidates] = False
                elif keep_k < free_space_dup_candidates.numel():
                    grad_vals = avg_grad_norm.squeeze(-1).detach()
                    _, topk = torch.topk(
                        grad_vals[free_space_dup_candidates],
                        k=keep_k,
                        largest=True,
                        sorted=False,
                    )
                    keep_idx = free_space_dup_candidates[topk]
                    drop_mask = torch.ones_like(free_space_dup_candidates, dtype=torch.bool)
                    drop_mask[topk] = False
                    drop_idx = free_space_dup_candidates[drop_mask]
                    new_dups[drop_idx] = False

        rejected_split_strong = int((splits & strong_block_mask).sum().item())
        rejected_split_weak = int((splits & weak_block_mask).sum().item())
        rejected_dup_strong = int((dups & strong_block_mask).sum().item())
        rejected_split_free = int((splits & free_space_block_mask).sum().item())
        rejected_dup_free = int((dups & free_space_block_mask).sum().item()) - int((new_dups & free_space_block_mask).sum().item())
        if rejected_split_strong > 0 or rejected_split_weak > 0 or rejected_dup_strong > 0:
            print(
                f"    Fog-growth suppression: split strong={rejected_split_strong}, split weak={rejected_split_weak}, "
                f"dup strong={rejected_dup_strong} "
                f"(op_strong<{fog_seed_opacity_max:.3f}, op_soft<{fog_seed_opacity_soft_max:.3f})"
            )
        if rejected_split_free > 0 or rejected_dup_free > 0:
            print(
                f"    Free-space gate: split_rejected={rejected_split_free}, dup_rejected={rejected_dup_free}, "
                f"dup_keep_ratio={free_space_dup_keep_ratio:.2f}"
            )

        if bool(surface_guided_cfg.get("fog_gate_debug", False)):
            split_candidates = int(splits.sum().item())
            dup_candidates = int(dups.sum().item())
            bright_candidates = int((bright_mask & (splits | dups)).sum().item())
            free_space_candidates = int((free_space_mask & (splits | dups)).sum().item())
            print(
                f"    Fog-growth gate stats: split_cand={split_candidates}, dup_cand={dup_candidates}, "
                f"bright_cand={bright_candidates}, free_space_cand={free_space_candidates}"
            )
        return new_splits, new_dups

    def _select_densify_group(
        self,
        split_idx: torch.Tensor,
        dup_idx: torch.Tensor,
        avg_grad_norm: torch.Tensor,
        point_budget: int,
        nsamps: int,
        split_budget_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if point_budget <= 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty

        grad_vals = avg_grad_norm.squeeze(-1)
        split_budget_ratio = max(0.0, min(1.0, split_budget_ratio))
        split_point_budget = int(point_budget * split_budget_ratio)
        split_k = min(split_idx.numel(), split_point_budget // max(1, nsamps))
        selected_split_idx = torch.empty(0, dtype=torch.long, device=self.device)
        if split_k > 0:
            _, top_split = torch.topk(grad_vals[split_idx], k=split_k, largest=True, sorted=False)
            selected_split_idx = split_idx[top_split]

        used_points = selected_split_idx.numel() * max(1, nsamps)
        remaining_points = max(0, point_budget - used_points)
        dup_k = min(dup_idx.numel(), remaining_points)
        selected_dup_idx = torch.empty(0, dtype=torch.long, device=self.device)
        if dup_k > 0:
            _, top_dup = torch.topk(grad_vals[dup_idx], k=dup_k, largest=True, sorted=False)
            selected_dup_idx = dup_idx[top_dup]

        return selected_split_idx, selected_dup_idx

    def _apply_densify_budgets(
        self,
        splits: torch.Tensor,
        dups: torch.Tensor,
        avg_grad_norm: torch.Tensor,
        nsamps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is None or not surface_guided_cfg.get("enabled", False):
            return splits, dups

        max_new_ratio = float(surface_guided_cfg.get("max_new_points_ratio_per_refine", 1.0))
        split_budget_ratio = float(surface_guided_cfg.get("split_new_points_ratio", 0.6))
        bright_cap_ratio = float(surface_guided_cfg.get("bright_new_points_ratio_cap", 1.0))
        free_space_cap_ratio = float(surface_guided_cfg.get("free_space_new_points_ratio_cap", 1.0))

        stage_cfg = self._get_surface_growth_stage_cfg()
        if stage_cfg is not None:
            max_new_ratio = float(stage_cfg.get("max_new_points_ratio_per_refine", max_new_ratio))
            split_budget_ratio = float(stage_cfg.get("split_new_points_ratio", split_budget_ratio))
            bright_cap_ratio = float(stage_cfg.get("bright_new_points_ratio_cap", bright_cap_ratio))
            free_space_cap_ratio = float(stage_cfg.get("free_space_new_points_ratio_cap", free_space_cap_ratio))

        max_new_ratio = max(0.0, min(1.0, max_new_ratio))
        split_budget_ratio = max(0.0, min(1.0, split_budget_ratio))
        bright_cap_ratio = max(0.0, min(1.0, bright_cap_ratio))
        free_space_cap_ratio = max(0.0, min(1.0, free_space_cap_ratio))
        if max_new_ratio <= 0.0:
            return torch.zeros_like(splits), torch.zeros_like(dups)

        # Dynamic budget decay when floater ratio is high.
        dynamic_ratio_enable = bool(surface_guided_cfg.get("dynamic_budget_from_floater_ratio", True))
        if dynamic_ratio_enable:
            floater_ratio_trigger = float(surface_guided_cfg.get("floater_ratio_budget_trigger", 0.01))
            dynamic_ratio = float(surface_guided_cfg.get("dynamic_max_new_points_ratio_per_refine", max_new_ratio))
            if stage_cfg is not None:
                dynamic_ratio = float(stage_cfg.get("dynamic_max_new_points_ratio_per_refine", dynamic_ratio))
            dynamic_ratio = max(0.0, min(max_new_ratio, dynamic_ratio))
            if self._last_floater_ratio > floater_ratio_trigger:
                max_new_ratio = dynamic_ratio

        # Trigger-based extra suppression when growth pressure is high.
        budget_scale = 1.0
        floater_soft_trigger = float(surface_guided_cfg.get("growth_pressure_floater_ratio", -1.0))
        floater_hard_trigger = float(surface_guided_cfg.get("growth_pressure_floater_ratio_hard", -1.0))
        floater_soft_scale = float(surface_guided_cfg.get("growth_pressure_scale", 0.7))
        floater_hard_scale = float(surface_guided_cfg.get("growth_pressure_scale_hard", 0.4))
        if floater_hard_trigger > 0 and self._last_floater_ratio >= floater_hard_trigger:
            budget_scale *= max(0.0, min(1.0, floater_hard_scale))
        elif floater_soft_trigger > 0 and self._last_floater_ratio >= floater_soft_trigger:
            budget_scale *= max(0.0, min(1.0, floater_soft_scale))

        if self._initial_num_points_for_growth_cap is not None and self._initial_num_points_for_growth_cap > 0:
            growth_ratio = self.num_points / max(1, self._initial_num_points_for_growth_cap)
            growth_soft_trigger = float(surface_guided_cfg.get("growth_pressure_ratio", -1.0))
            growth_hard_trigger = float(surface_guided_cfg.get("growth_pressure_ratio_hard", -1.0))
            growth_soft_scale = float(surface_guided_cfg.get("growth_pressure_ratio_scale", 0.75))
            growth_hard_scale = float(surface_guided_cfg.get("growth_pressure_ratio_scale_hard", 0.45))
            if growth_hard_trigger > 0 and growth_ratio >= growth_hard_trigger:
                budget_scale *= max(0.0, min(1.0, growth_hard_scale))
            elif growth_soft_trigger > 0 and growth_ratio >= growth_soft_trigger:
                budget_scale *= max(0.0, min(1.0, growth_soft_scale))

        if stage_cfg is not None:
            budget_scale *= float(stage_cfg.get("budget_scale", 1.0))
        budget_scale = max(0.0, min(1.0, budget_scale))
        self._last_densify_budget_scale = budget_scale
        max_new_ratio *= budget_scale
        split_budget_ratio *= budget_scale
        split_budget_ratio = max(0.0, min(1.0, split_budget_ratio))

        total_budget = int(self.num_points * max_new_ratio)
        if total_budget <= 0:
            return torch.zeros_like(splits), torch.zeros_like(dups)

        split_idx = torch.where(splits)[0]
        dup_idx = torch.where(dups)[0]
        if split_idx.numel() == 0 and dup_idx.numel() == 0:
            return splits, dups

        free_space_mask = self._compute_free_space_gaussian_mask()
        growth_cap_ratio = float(surface_guided_cfg.get("bg_growth_cap_ratio_before_8k", -1.0))
        growth_cap_until = int(surface_guided_cfg.get("bg_growth_cap_until_step", 8000))
        if self._initial_num_points_for_growth_cap is None:
            self._initial_num_points_for_growth_cap = int(self.num_points)
        if (
            growth_cap_ratio > 0
            and self.step < growth_cap_until
            and self._initial_num_points_for_growth_cap is not None
            and self.num_points > int(self._initial_num_points_for_growth_cap * growth_cap_ratio)
        ):
            splits = splits & (~free_space_mask)
            dups = dups & (~free_space_mask)
            split_idx = torch.where(splits)[0]
            dup_idx = torch.where(dups)[0]
            if split_idx.numel() == 0 and dup_idx.numel() == 0:
                return splits, dups

        bright_mask = self._compute_bright_gaussian_mask()
        # In strong-light regions, reserve more growth budget for low-opacity duplicates
        # and suppress aggressive split bursts to avoid fog-like point clouds.
        if bool(surface_guided_cfg.get("highlight_emission_first", False)):
            opacity_vals = self.get_opacity.detach().squeeze(-1)
            emissive_thr = float(surface_guided_cfg.get("highlight_emissive_intensity_min", 1.0))
            low_opacity_gate = float(surface_guided_cfg.get("highlight_low_opacity_max", 0.35))
            if self.use_pbr and self.use_emissive:
                emissive_intensity = torch.exp(self._emissive_intensity.detach().squeeze(-1))
                bright_emit = emissive_intensity > emissive_thr
                split_block = bright_emit & (opacity_vals < low_opacity_gate)
                if split_block.any():
                    splits = splits & (~split_block)
                    split_idx = torch.where(splits)[0]
                    if split_idx.numel() == 0 and dup_idx.numel() == 0:
                        return splits, dups
                dup_prior = bool(surface_guided_cfg.get("highlight_duplicate_priority", True))
                if dup_prior:
                    split_budget_ratio = min(split_budget_ratio, float(surface_guided_cfg.get("highlight_split_budget_ratio_cap", 0.35)))

        bright_split_idx = split_idx[bright_mask[split_idx]]
        non_bright_split_idx = split_idx[~bright_mask[split_idx]]
        bright_dup_idx = dup_idx[bright_mask[dup_idx]]
        non_bright_dup_idx = dup_idx[~bright_mask[dup_idx]]
        free_split_idx = split_idx[free_space_mask[split_idx]]
        free_dup_idx = dup_idx[free_space_mask[dup_idx]]

        bright_budget = int(total_budget * bright_cap_ratio)
        bright_sel_split, bright_sel_dup = self._select_densify_group(
            bright_split_idx,
            bright_dup_idx,
            avg_grad_norm,
            bright_budget,
            nsamps,
            split_budget_ratio,
        )
        bright_used = bright_sel_split.numel() * max(1, nsamps) + bright_sel_dup.numel()
        non_bright_budget = max(0, total_budget - bright_used)
        non_bright_sel_split, non_bright_sel_dup = self._select_densify_group(
            non_bright_split_idx,
            non_bright_dup_idx,
            avg_grad_norm,
            non_bright_budget,
            nsamps,
            split_budget_ratio,
        )

        selected_split_idx = torch.cat([bright_sel_split, non_bright_sel_split], dim=0)
        selected_dup_idx = torch.cat([bright_sel_dup, non_bright_sel_dup], dim=0)

        # Free-space candidates get their own strict cap.
        if free_space_cap_ratio < 1.0 and (free_split_idx.numel() > 0 or free_dup_idx.numel() > 0):
            free_space_budget = int(total_budget * free_space_cap_ratio)
            keep_free_split, keep_free_dup = self._select_densify_group(
                free_split_idx,
                free_dup_idx,
                avg_grad_norm,
                free_space_budget,
                nsamps,
                split_budget_ratio,
            )
            free_keep_mask = torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
            if keep_free_split.numel() > 0:
                free_keep_mask[keep_free_split] = True
            if keep_free_dup.numel() > 0:
                free_keep_mask[keep_free_dup] = True

            if selected_split_idx.numel() > 0:
                keep_sel_split = (~free_space_mask[selected_split_idx]) | free_keep_mask[selected_split_idx]
                selected_split_idx = selected_split_idx[keep_sel_split]
            if selected_dup_idx.numel() > 0:
                keep_sel_dup = (~free_space_mask[selected_dup_idx]) | free_keep_mask[selected_dup_idx]
                selected_dup_idx = selected_dup_idx[keep_sel_dup]

        new_splits = torch.zeros_like(splits)
        new_dups = torch.zeros_like(dups)
        if selected_split_idx.numel() > 0:
            new_splits[selected_split_idx] = True
        if selected_dup_idx.numel() > 0:
            new_dups[selected_dup_idx] = True

        selected_points = selected_split_idx.numel() * max(1, nsamps) + selected_dup_idx.numel()
        bright_points = bright_sel_split.numel() * max(1, nsamps) + bright_sel_dup.numel()
        free_points = int((new_splits & free_space_mask).sum().item()) * max(1, nsamps) + int((new_dups & free_space_mask).sum().item())
        print(
            f"    Densify budget: selected {selected_points}/{total_budget} new points "
            f"(bright={bright_points}, bright_cap={bright_budget}, free_space={free_points}, "
            f"scale={self._last_densify_budget_scale:.3f})"
        )
        return new_splits, new_dups

    def _compute_bright_densify_scale(self) -> float:
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is None:
            return 1.0
        return float(surface_guided_cfg.get("bright_region_densify_scale", 1.0))

    def _compute_bright_gaussian_mask(self) -> torch.Tensor:
        surface_guided_cfg = self.ctrl_cfg.get("surface_guided_densification", None)
        if surface_guided_cfg is None:
            return torch.zeros(self.num_points, dtype=torch.bool, device=self.device)

        albedo = torch.sigmoid(self._features_dc)
        luma = 0.299 * albedo[:, 0] + 0.587 * albedo[:, 1] + 0.114 * albedo[:, 2]

        # Adaptive bright threshold from current Gaussian distribution.
        luma_quantile = float(surface_guided_cfg.get("bright_luma_quantile", 0.98))
        luma_quantile = max(0.0, min(1.0, luma_quantile))
        luma_min = float(surface_guided_cfg.get("bright_luma_min", 0.7))
        if luma.numel() > 0:
            luma_thr = torch.quantile(luma.detach(), luma_quantile).item()
            luma_thr = max(luma_min, luma_thr)
        else:
            luma_thr = luma_min
        bright_mask = luma > luma_thr

        if self.use_pbr and self.use_emissive:
            emissive_intensity = torch.exp(self._emissive_intensity.squeeze(-1))
            emissive_quantile = float(surface_guided_cfg.get("bright_emissive_quantile", 0.995))
            emissive_quantile = max(0.0, min(1.0, emissive_quantile))
            emissive_min = float(surface_guided_cfg.get("bright_emissive_min", 1.0))
            if emissive_intensity.numel() > 0:
                emissive_thr = torch.quantile(emissive_intensity.detach(), emissive_quantile).item()
                emissive_thr = max(emissive_min, emissive_thr)
            else:
                emissive_thr = emissive_min
            bright_mask = bright_mask | (emissive_intensity > emissive_thr)

        # Optional: keep only bright points sufficiently far from depth surfaces
        # to focus suppression on likely fog-like free-space Gaussians.
        bright_surface_distance_min = float(surface_guided_cfg.get("bright_surface_distance_min", -1.0))
        if (
            bright_surface_distance_min > 0
            and self._depth_points is not None
            and self._depth_points.shape[0] > 0
            and bright_mask.any()
        ):
            bright_idx = torch.where(bright_mask)[0]
            depth_pts = self._depth_points
            if depth_pts.shape[0] > 50000:
                sample_idx = torch.randperm(depth_pts.shape[0], device=self.device)[:50000]
                depth_pts = depth_pts[sample_idx]
            dists = torch.cdist(self._means[bright_idx], depth_pts)
            min_dist, _ = dists.min(dim=1)
            far_from_surface = min_dist > bright_surface_distance_min
            filtered_mask = torch.zeros_like(bright_mask)
            filtered_mask[bright_idx[far_from_surface]] = True
            bright_mask = filtered_mask

        return bright_mask

    def _get_emissive_cone_axis(self) -> Optional[torch.Tensor]:
        cone_cfg = self.ctrl_cfg.get("emissive_cone_prior", None)
        if cone_cfg is None or not bool(cone_cfg.get("enabled", False)):
            return None
        axis = cone_cfg.get("axis_world", [1.0, 0.0, 0.0])
        axis_tensor = torch.tensor(axis, device=self.device, dtype=self._means.dtype)
        if axis_tensor.numel() != 3:
            return None
        axis_norm = torch.norm(axis_tensor, p=2)
        if axis_norm <= 1e-8:
            return None
        return axis_tensor / axis_norm

    def _compute_high_emissive_mask(self) -> torch.Tensor:
        if not (self.use_pbr and self.use_emissive):
            return torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
        emissive_geo_cfg = self.ctrl_cfg.get("emissive_surface_geometry", None)
        if emissive_geo_cfg is None or not bool(emissive_geo_cfg.get("enabled", False)):
            return torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
        emissive_intensity = torch.exp(torch.clamp(self._emissive_intensity.detach().squeeze(-1), min=-8.0, max=8.0))
        q = float(emissive_geo_cfg.get("emissive_quantile", emissive_geo_cfg.get("light_quantile", 0.995)))
        q = max(0.0, min(1.0, q))
        min_intensity = float(emissive_geo_cfg.get("min_intensity", 1.0))
        if emissive_intensity.numel() == 0:
            return torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
        thr = torch.quantile(emissive_intensity, q).item()
        thr = max(min_intensity, thr)
        return emissive_intensity >= thr

    def _compute_emissive_surface_support_loss(self, cfg) -> Optional[torch.Tensor]:
        if self._depth_points is None or self._depth_points.shape[0] == 0:
            return None
        high_emissive_mask = self._compute_high_emissive_mask()
        if not high_emissive_mask.any():
            return None
        candidate_idx = torch.where(high_emissive_mask)[0]
        max_points = int(cfg.get("max_points", 256))
        if max_points > 0 and candidate_idx.numel() > max_points:
            emissive_vals = torch.exp(torch.clamp(self._emissive_intensity[candidate_idx].detach().squeeze(-1), min=-8.0, max=8.0))
            _, topk = torch.topk(emissive_vals, k=max_points, largest=True, sorted=False)
            candidate_idx = candidate_idx[topk]

        depth_pts = self._depth_points
        max_surface_samples = int(cfg.get("max_surface_samples", 50000))
        if max_surface_samples > 0 and depth_pts.shape[0] > max_surface_samples:
            sample_idx = torch.randperm(depth_pts.shape[0], device=self.device)[:max_surface_samples]
            depth_pts = depth_pts[sample_idx]

        dists = torch.cdist(self._means[candidate_idx], depth_pts)
        min_dist, _ = dists.min(dim=1)
        margin = float(cfg.get("margin", 0.15))
        support_loss = torch.relu(min_dist - margin)
        return support_loss.mean() * float(cfg.get("w", 0.05))

    def _compute_emissive_cluster_loss(self, cfg) -> Optional[torch.Tensor]:
        high_emissive_mask = self._compute_high_emissive_mask()
        if high_emissive_mask.sum().item() < 2:
            return None
        candidate_idx = torch.where(high_emissive_mask)[0]
        max_points = int(cfg.get("max_points", 128))
        if max_points > 0 and candidate_idx.numel() > max_points:
            emissive_vals = torch.exp(torch.clamp(self._emissive_intensity[candidate_idx].detach().squeeze(-1), min=-8.0, max=8.0))
            _, topk = torch.topk(emissive_vals, k=max_points, largest=True, sorted=False)
            candidate_idx = candidate_idx[topk]
        if candidate_idx.numel() < 2:
            return None

        pts = self._means[candidate_idx]
        dists = torch.cdist(pts, pts)
        dists.fill_diagonal_(1e6)
        nearest = dists.min(dim=1).values
        target_radius = float(cfg.get("target_radius", 0.25))
        cluster_loss = torch.relu(nearest - target_radius)
        return cluster_loss.mean() * float(cfg.get("w", 0.01))

    def _apply_emissive_surface_thickness_clip(self) -> None:
        if not (self.use_pbr and self.use_emissive):
            return
        emissive_geo_cfg = self.ctrl_cfg.get("emissive_surface_geometry", None)
        if emissive_geo_cfg is None or not bool(emissive_geo_cfg.get("enabled", False)):
            return
        max_scale = float(emissive_geo_cfg.get("max_scale", -1.0))
        if max_scale <= 0:
            return
        high_emissive_mask = self._compute_high_emissive_mask()
        if not high_emissive_mask.any():
            return
        max_log_scale = np.log(max_scale)
        with torch.no_grad():
            self._scales[high_emissive_mask] = torch.clamp(self._scales[high_emissive_mask], max=max_log_scale)

    def compute_pbr_color(self, cam, global_sh_coeffs):
        # Albedo
        albedo = torch.sigmoid(self._features_dc) # Base color
        
        # Normals
        normals = F.normalize(self._normals, dim=-1, eps=1e-6)
        normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # View directions
        viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]
        viewdirs = -F.normalize(viewdirs, dim=-1, eps=1e-6) # Point to camera
        viewdirs = torch.nan_to_num(viewdirs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Diffuse Component (Ld)
        if global_sh_coeffs is not None:
             global_sh_coeffs = torch.nan_to_num(global_sh_coeffs, nan=0.0, posinf=1e3, neginf=-1e3)
             # Evaluate SH at normals
             # global_sh_coeffs: [1, (deg+1)^2, 3] or [B, ...]
             sh_coeffs_expanded = global_sh_coeffs.unsqueeze(0).expand(self.num_points, -1, -1)
             
             num_coeffs = global_sh_coeffs.shape[-2]
             global_sh_degree = int(num_coeffs ** 0.5) - 1
             
             diffuse_light = spherical_harmonics(global_sh_degree, normals, sh_coeffs_expanded)
             diffuse_light = torch.clamp(diffuse_light, 0.0)
             L_d = albedo * diffuse_light
        else:
             L_d = albedo * 0.1 # Fallback ambient
             
        # Specular Component (Ls) - Simplified
        # Iterate over lobes
        L_s = torch.zeros_like(L_d)
        
        # ASG Params
        asg_amp = torch.sigmoid(self._asg_amplitude)
        asg_sharp = torch.exp(torch.clamp(self._asg_sharpness, min=-8.0, max=8.0)) # lambda, mu
        asg_dir = F.normalize(self._asg_dir, dim=-1, eps=1e-6)
        asg_dir = torch.nan_to_num(asg_dir, nan=0.0, posinf=0.0, neginf=0.0)
        
        for i in range(self.num_asg_lobes):
            l_i = asg_dir[:, i, :] # Light direction
            h_i = F.normalize(viewdirs + l_i, dim=-1, eps=1e-6)
            h_i = torch.nan_to_num(h_i, nan=0.0, posinf=0.0, neginf=0.0)
            ndoth = torch.clamp((normals * h_i).sum(dim=-1, keepdim=True), 0.0, 1.0)
            
            # Simple Phong approx for now
            roughness = torch.sigmoid(self._roughness)
            shininess = 1.0 / (roughness + 1e-6)
            spec = torch.pow(ndoth, shininess * 10.0) 
            
            L_s += asg_amp[:, i, :] * spec

        L_HDR = L_d + L_s
        
        # ========== Emissive Component (Phase 3) ==========
        if self.use_emissive:
            emissive_color = torch.sigmoid(self._emissive)  # RGB in [0,1]
            emissive_intensity = torch.exp(torch.clamp(self._emissive_intensity, min=-8.0, max=8.0))  # Intensity scaling
            if self.use_anisotropic_emissive:
                # Directional emissive field E(theta, phi) represented by low-order SH.
                emissive_sh = self._emissive_sh
                directional_energy = spherical_harmonics(self.emissive_sh_degree, viewdirs, emissive_sh)
                directional_energy = F.relu(torch.nan_to_num(directional_energy, nan=0.0, posinf=0.0, neginf=0.0))
                directional_energy = directional_energy.mean(dim=-1)
                directional_floor = float(self.ctrl_cfg.get("anisotropic_emissive_floor", 0.05))
                directional_floor = max(0.0, min(1.0, directional_floor))
                directional_gain = directional_floor + directional_energy

                # Cone prior: suppress excessive emission outside the dominant light axis.
                cone_cfg = self.ctrl_cfg.get("emissive_cone_prior", None)
                if cone_cfg is not None and bool(cone_cfg.get("enabled", False)):
                    cone_axis = self._get_emissive_cone_axis()
                    if cone_axis is not None:
                        cos_alpha = torch.clamp((viewdirs * cone_axis.unsqueeze(0)).sum(dim=-1), min=0.0, max=1.0)
                        cone_power = float(cone_cfg.get("power", 8.0))
                        cone_power = max(1.0, cone_power)
                        cone_floor = float(cone_cfg.get("floor", 0.05))
                        cone_floor = max(0.0, min(1.0, cone_floor))
                        cone_gain = cone_floor + (1.0 - cone_floor) * torch.pow(cos_alpha, cone_power)
                        directional_gain = directional_gain * cone_gain
                        max_gain = float(cone_cfg.get("max_gain", 4.0))
                        if max_gain > 0:
                            directional_gain = torch.clamp(directional_gain, max=max_gain)
                emissive_intensity = emissive_intensity * directional_gain.unsqueeze(-1)
            L_emissive = emissive_color * emissive_intensity
            L_HDR = L_HDR + L_emissive
        
        # Tone Mapping
        L_HDR = torch.nan_to_num(L_HDR, nan=0.0, posinf=1e3, neginf=0.0)
        L_HDR = torch.clamp(L_HDR, min=0.0)
        L_LDR = L_HDR / (1.0 + L_HDR)
        L_LDR = torch.nan_to_num(L_LDR, nan=0.0, posinf=1.0, neginf=0.0)
        L_LDR = torch.clamp(L_LDR, 0.0, 1.0)
        
        return L_LDR

    def get_gaussians(self, cam: dataclass_camera, global_sh_coeffs: torch.Tensor = None) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        # get colors of gaussians
        if self.use_pbr:
            rgbs = self.compute_pbr_color(cam, global_sh_coeffs)
        else:
            colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
            if self.sh_degree > 0:
                viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
                viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
                n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
                rgbs = spherical_harmonics(n, viewdirs, colors)
                rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
            else:
                rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=self._means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict
    
    def compute_reg_loss(self):
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            w = sharp_shape_reg_cfg.w
            max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
            step_interval = sharp_shape_reg_cfg.step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.get_scaling
                scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * w
                loss_dict["sharp_shape_reg"] = scale_reg

        flatten_reg = self.reg_cfg.get("flatten", None)
        if flatten_reg is not None:
            sclaings = self.get_scaling
            min_scale, _ = torch.min(sclaings, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss_dict["flatten"] = flatten_loss * flatten_reg.w
        
        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg:
            if (self.cur_radii > 0).sum():
                opacity = torch.sigmoid(self._opacities)
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        # compute the max of scaling
        max_s_square_reg = self.reg_cfg.get("max_s_square_reg", None)
        if max_s_square_reg is not None and not self.ball_gaussians:
            loss_dict["max_s_square"] = torch.mean((self.get_scaling.max(dim=1).values) ** 2) * max_s_square_reg.w
        
        # ========== Anti-Floater Regularization (Phase 1 & 2) ==========
        
        # Spatial sparsity regularization
        if self.spatial_sparsity_reg is not None:
            sparsity_losses = self.spatial_sparsity_reg(
                means=self._means,
                opacities=self.get_opacity,
                depth_points=self._depth_points,
            )
            for key, value in sparsity_losses.items():
                loss_dict[f"spatial_{key}"] = value
        
        # Depth consistency regularization
        if self.depth_consistency_reg is not None and self._depth_points is not None:
            depth_losses = self.depth_consistency_reg(
                means=self._means,
                opacities=self.get_opacity,
                depth_points=self._depth_points,
            )
            for key, value in depth_losses.items():
                loss_dict[f"depth_{key}"] = value
        
        # Emissive sparsity regularization (Phase 3)
        if self.use_pbr and self.use_emissive:
            emissive_reg_cfg = self.reg_cfg.get("emissive_sparsity", None) if self.reg_cfg else None
            if emissive_reg_cfg is not None:
                # Penalize too many emissive Gaussians
                emissive_intensity = torch.exp(self._emissive_intensity).squeeze()
                emissive_sparsity_loss = emissive_intensity.mean() * emissive_reg_cfg.get("w", 0.01)
                loss_dict["emissive_sparsity"] = emissive_sparsity_loss
            anisotropic_reg_cfg = self.reg_cfg.get("anisotropic_emissive", None) if self.reg_cfg else None
            if (
                self.use_anisotropic_emissive
                and anisotropic_reg_cfg is not None
                and bool(anisotropic_reg_cfg.get("enabled", True))
            ):
                cone_axis = self._get_emissive_cone_axis()
                if cone_axis is not None and self.num_points > 0:
                    axis_dirs = cone_axis.unsqueeze(0).repeat(self.num_points, 1)
                    back_dirs = -axis_dirs
                    axis_energy = spherical_harmonics(self.emissive_sh_degree, axis_dirs, self._emissive_sh).mean(dim=-1)
                    back_energy = spherical_harmonics(self.emissive_sh_degree, back_dirs, self._emissive_sh).mean(dim=-1)
                    axis_energy = F.relu(torch.nan_to_num(axis_energy, nan=0.0, posinf=0.0, neginf=0.0))
                    back_energy = F.relu(torch.nan_to_num(back_energy, nan=0.0, posinf=0.0, neginf=0.0))
                    back_ratio = float(anisotropic_reg_cfg.get("back_ratio", 0.3))
                    back_ratio = max(0.0, min(1.0, back_ratio))
                    w = float(anisotropic_reg_cfg.get("w", 0.005))
                    cone_loss = F.relu(back_energy - axis_energy * back_ratio).mean()
                    loss_dict["anisotropic_emissive_cone"] = cone_loss * w
            surface_support_cfg = self.reg_cfg.get("emissive_surface_support", None) if self.reg_cfg else None
            if surface_support_cfg is not None and bool(surface_support_cfg.get("enabled", True)):
                surface_support_loss = self._compute_emissive_surface_support_loss(surface_support_cfg)
                if surface_support_loss is not None:
                    loss_dict["emissive_surface_support"] = surface_support_loss
            cluster_cfg = self.reg_cfg.get("emissive_cluster", None) if self.reg_cfg else None
            if cluster_cfg is not None and bool(cluster_cfg.get("enabled", True)):
                cluster_loss = self._compute_emissive_cluster_loss(cluster_cfg)
                if cluster_loss is not None:
                    loss_dict["emissive_cluster"] = cluster_loss

        return loss_dict
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        N = state_dict["_means"].shape[0]
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device))
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device))
        self._features_dc = Parameter(torch.zeros((N,) + self._features_dc.shape[1:], device=self.device))
        self._features_rest = Parameter(torch.zeros((N,) + self._features_rest.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))

        if self.use_pbr:
            if "_roughness" in state_dict:
                self._roughness = Parameter(torch.zeros((N,) + self._roughness.shape[1:], device=self.device))
            if "_metallic" in state_dict:
                self._metallic = Parameter(torch.zeros((N,) + self._metallic.shape[1:], device=self.device))
            if "_normals" in state_dict:
                self._normals = Parameter(torch.zeros((N,) + self._normals.shape[1:], device=self.device))
            if "_asg_amplitude" in state_dict:
                self._asg_amplitude = Parameter(torch.zeros((N,) + self._asg_amplitude.shape[1:], device=self.device))
            if "_asg_sharpness" in state_dict:
                self._asg_sharpness = Parameter(torch.zeros((N,) + self._asg_sharpness.shape[1:], device=self.device))
            if "_asg_dir" in state_dict:
                self._asg_dir = Parameter(torch.zeros((N,) + self._asg_dir.shape[1:], device=self.device))
            if self.use_emissive:
                if "_emissive" in state_dict:
                    self._emissive = Parameter(torch.zeros((N,) + self._emissive.shape[1:], device=self.device))
                if "_emissive_intensity" in state_dict:
                    self._emissive_intensity = Parameter(torch.zeros((N,) + self._emissive_intensity.shape[1:], device=self.device))
                if self.use_anisotropic_emissive and "_emissive_sh" in state_dict:
                    self._emissive_sh = Parameter(torch.zeros((N,) + self._emissive_sh.shape[1:], device=self.device))

        if self.use_pbr and self.use_emissive and self.use_anisotropic_emissive and "_emissive_sh" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["_emissive_sh"] = torch.zeros((N, self.emissive_sh_dim, 3), device=self.device, dtype=self._means.dtype)

        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
    
    def export_gaussians_to_ply(self, alpha_thresh: float) -> Dict:
        means = self._means
        direct_color = self.colors
        
        activated_opacities = self.get_opacity
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }

    def prune_points(self, mask: torch.Tensor, optimizer: torch.optim.Optimizer = None) -> None:
            """
            从外部强制剔除指定的高斯点。
            Args:
                mask: torch.Tensor (bool), 形状为 [num_points], True 表示要剔除的点。
                optimizer: 当前的优化器。如果传入，将同步清理优化器状态。
            """
            if not mask.any():
                return

            # 确保是布尔型
                if mask.dtype != torch.bool:
                    mask = mask.to(torch.bool)

            n_bef = self.num_points
            # 这里的 ~mask 表示保留下来的点
            keep_mask = ~mask

            # 1. 更新模型参数
            self._means = Parameter(self._means[keep_mask].detach())
            self._scales = Parameter(self._scales[keep_mask].detach())
            self._quats = Parameter(self._quats[keep_mask].detach())
            self._features_dc = Parameter(self._features_dc[keep_mask].detach())
            self._features_rest = Parameter(self._features_rest[keep_mask].detach())
            self._opacities = Parameter(self._opacities[keep_mask].detach())

            if self.use_pbr:
                self._roughness = Parameter(self._roughness[keep_mask].detach())
                self._metallic = Parameter(self._metallic[keep_mask].detach())
                self._normals = Parameter(self._normals[keep_mask].detach())
                self._asg_amplitude = Parameter(self._asg_amplitude[keep_mask].detach())
                self._asg_sharpness = Parameter(self._asg_sharpness[keep_mask].detach())
                self._asg_dir = Parameter(self._asg_dir[keep_mask].detach())
                if self.use_emissive:
                    self._emissive = Parameter(self._emissive[keep_mask].detach())
                    self._emissive_intensity = Parameter(self._emissive_intensity[keep_mask].detach())
                    if self.use_anisotropic_emissive:
                        self._emissive_sh = Parameter(self._emissive_sh[keep_mask].detach())

            # 2. 同步更新内部统计量 (非常重要，否则 postprocess 会报错)
            if self.xys_grad_norm is not None:
                # Check size consistency before indexing
                if self.xys_grad_norm.shape[0] == keep_mask.shape[0]:
                    self.xys_grad_norm = self.xys_grad_norm[keep_mask]
                else:
                    # Size mismatch, reset to None (will be re-initialized)
                    print(f"    Warning: xys_grad_norm size mismatch ({self.xys_grad_norm.shape[0]} vs {keep_mask.shape[0]}), resetting")
                    self.xys_grad_norm = None
            if hasattr(self, 'vis_counts') and self.vis_counts is not None:
                # Check size consistency before indexing
                if self.vis_counts.shape[0] == keep_mask.shape[0]:
                    self.vis_counts = self.vis_counts[keep_mask]
                else:
                    # Size mismatch, reset to None (will be re-initialized)
                    print(f"    Warning: vis_counts size mismatch ({self.vis_counts.shape[0]} vs {keep_mask.shape[0]}), resetting")
                    self.vis_counts = None
            if self.max_2Dsize is not None:
                # Check size consistency before indexing
                if self.max_2Dsize.shape[0] == keep_mask.shape[0]:
                    self.max_2Dsize = self.max_2Dsize[keep_mask]
                else:
                    # Size mismatch, reset to None (will be re-initialized)
                    print(f"    Warning: max_2Dsize size mismatch ({self.max_2Dsize.shape[0]} vs {keep_mask.shape[0]}), resetting")
                    self.max_2Dsize = None

            # 3. 同步更新优化器状态
            if optimizer is not None:
                param_groups = self.get_gaussian_param_groups()
                # 使用你框架中现有的 remove_from_optim 工具函数
                # 注意：mask 必须是布尔类型且长度与剔除前一致
                remove_from_optim(optimizer, mask, param_groups)
            print(f"     [Online Pruning] Class {self.class_prefix} removed: {n_bef - self.num_points} points.")

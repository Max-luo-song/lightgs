"""
Floater Detection and Spatial Regularization Module

This module provides tools for detecting and removing floating Gaussians
(foggy artifacts) in 3D Gaussian Splatting, along with spatial regularization
to encourage physically plausible Gaussian distributions.

Author: Auto-generated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)


@dataclass
class FloaterDetectionConfig:
    """Configuration for floater detection."""
    enabled: bool = True
    grad_thresh: float = 0.0001  # Gradient threshold for low-activity detection
    opacity_thresh: float = 0.1  # Low opacity threshold
    isolation_thresh: float = 2.0  # Distance threshold for spatial isolation (scene units)
    depth_margin: float = 1.0  # Maximum distance from estimated depth surface (meters)
    min_nn_neighbors: int = 10  # Number of neighbors for isolation check
    soft_prune_until: int = 10000  # Step until which to use soft pruning
    hard_prune_interval: int = 1000  # Interval for hard pruning
    warmup_steps: int = 500  # Steps before starting detection
    floater_score_thresh: float = 0.7  # Score threshold for floater classification


@dataclass 
class SpatialSparsityConfig:
    """Configuration for spatial sparsity regularization."""
    enabled: bool = True
    grid_size: float = 1.0  # Voxel size for spatial grouping
    max_density: float = 10.0  # Maximum acceptable density ratio
    weight: float = 0.01  # Loss weight
    distance_scale: float = 2.0  # Scale for distance-based weighting


class FloaterDetector:
    """
    Multi-dimensional floater detection system.
    
    Detects floating Gaussians based on:
    1. Low gradient activity (not contributing to reconstruction)
    2. Low opacity (semi-transparent artifacts)
    3. Spatial isolation (far from other Gaussians)
    4. Depth violation (in front of real surfaces)
    """
    
    def __init__(self, config: FloaterDetectionConfig, device: str = 'cuda'):
        """
        Initialize FloaterDetector.
        
        Args:
            config: Detection configuration
            device: Compute device
        """
        self.config = config
        self.device = device
        
        # Statistics tracking
        self.detection_history: List[Dict] = []
    
    def compute_isolation_scores(
        self,
        means: torch.Tensor,
        k: int = None,
    ) -> torch.Tensor:
        """
        Compute spatial isolation scores for each Gaussian.
        
        Args:
            means: Gaussian centers (N, 3)
            k: Number of neighbors to consider
            
        Returns:
            Isolation scores (N,), higher = more isolated
        """
        if k is None:
            k = self.config.min_nn_neighbors
        
        N = means.shape[0]
        if N <= k:
            return torch.zeros(N, device=means.device)
        
        # Use sklearn for efficient KNN (on CPU)
        means_np = means.detach().cpu().numpy()
        
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn.fit(means_np)
        distances, _ = nn.kneighbors(means_np)
        
        # Average distance to k nearest neighbors (excluding self)
        avg_nn_dist = distances[:, 1:].mean(axis=1)
        
        # Normalize by isolation threshold
        isolation_scores = torch.from_numpy(avg_nn_dist).float().to(means.device)
        isolation_scores = torch.clamp(isolation_scores / self.config.isolation_thresh, 0, 1)
        
        return isolation_scores
    
    def compute_depth_violation_scores(
        self,
        means: torch.Tensor,
        depth_points: torch.Tensor,
        max_distance: float = None,
    ) -> torch.Tensor:
        """
        Compute how far Gaussians are from the nearest depth surface.
        
        Args:
            means: Gaussian centers (N, 3)
            depth_points: Surface points from depth estimation (M, 3)
            max_distance: Maximum distance to consider
            
        Returns:
            Violation scores (N,), higher = further from surfaces
        """
        if max_distance is None:
            max_distance = self.config.depth_margin
        
        if depth_points is None or depth_points.shape[0] == 0:
            return torch.zeros(means.shape[0], device=means.device)
        
        # Compute distances in chunks to avoid OOM
        N = means.shape[0]
        M = depth_points.shape[0]
        chunk_size = min(10000, N)
        
        min_distances = torch.zeros(N, device=means.device)
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk = means[i:end]
            
            # Sample depth points if too many
            if M > 50000:
                sample_idx = torch.randperm(M, device=depth_points.device)[:50000]
                depth_sample = depth_points[sample_idx]
            else:
                depth_sample = depth_points
            
            # Compute pairwise distances
            dists = torch.cdist(chunk, depth_sample)
            min_dists, _ = dists.min(dim=1)
            min_distances[i:end] = min_dists
        
        # Normalize by max distance
        violation_scores = torch.clamp(min_distances / max_distance, 0, 1)
        
        return violation_scores
    
    def compute_gradient_scores(
        self,
        xys_grad_norm: Optional[torch.Tensor],
        vis_counts: Optional[torch.Tensor],
        expected_size: int = None,
    ) -> torch.Tensor:
        """
        Compute low-gradient scores (inverse of gradient activity).
        
        Args:
            xys_grad_norm: Accumulated 2D projection gradient norms (N,)
            vis_counts: Visibility counts (N,)
            expected_size: Expected number of Gaussians
            
        Returns:
            Low-gradient scores (N,), higher = lower gradient activity
        """
        if xys_grad_norm is None or vis_counts is None:
            return None
        
        # Check size consistency
        if expected_size is not None:
            if xys_grad_norm.shape[0] != expected_size:
                return None  # Skip gradient scores if size mismatch
            if vis_counts.shape[0] != expected_size:
                return None
        
        # Compute average gradient
        avg_grad = xys_grad_norm / vis_counts.clamp(min=1)
        
        # Inverse normalization: high gradient -> low score
        low_grad_scores = 1.0 - torch.clamp(avg_grad / self.config.grad_thresh, 0, 1)
        
        return low_grad_scores
    
    def compute_opacity_scores(
        self,
        opacities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute low-opacity scores.
        
        Args:
            opacities: Gaussian opacities (N, 1) or (N,)
            
        Returns:
            Low-opacity scores (N,), higher = lower opacity
        """
        opacities = opacities.squeeze()
        low_opacity_scores = 1.0 - torch.clamp(opacities / self.config.opacity_thresh, 0, 1)
        return low_opacity_scores
    
    def detect_floaters(
        self,
        means: torch.Tensor,
        opacities: torch.Tensor,
        xys_grad_norm: Optional[torch.Tensor] = None,
        vis_counts: Optional[torch.Tensor] = None,
        depth_points: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect floating Gaussians using multi-dimensional scoring.
        
        Args:
            means: Gaussian centers (N, 3)
            opacities: Gaussian opacities (N, 1)
            xys_grad_norm: Accumulated gradient norms (N,)
            vis_counts: Visibility counts (N,)
            depth_points: Surface points from depth (M, 3)
            step: Current training step
            
        Returns:
            Tuple of (floater_mask, floater_scores)
            - floater_mask: Boolean mask (N,) True for floaters
            - floater_scores: Float scores (N,) in [0, 1]
        """
        N = means.shape[0]
        device = means.device
        
        # Check warmup
        if step < self.config.warmup_steps:
            return torch.zeros(N, dtype=torch.bool, device=device), torch.zeros(N, device=device)
        
        # Compute individual scores
        scores_dict = {}
        weights = {}
        
        # 1. Low gradient score
        grad_scores = self.compute_gradient_scores(xys_grad_norm, vis_counts, expected_size=N)
        if grad_scores is not None:
            scores_dict['gradient'] = grad_scores
            weights['gradient'] = 0.25
        
        # 2. Spatial isolation score
        isolation_scores = self.compute_isolation_scores(means)
        scores_dict['isolation'] = isolation_scores
        weights['isolation'] = 0.2
        
        # 3. Depth violation score
        if depth_points is not None:
            depth_scores = self.compute_depth_violation_scores(means, depth_points)
            scores_dict['depth'] = depth_scores
            weights['depth'] = 0.35
        
        # 4. Low opacity score
        opacity_scores = self.compute_opacity_scores(opacities)
        scores_dict['opacity'] = opacity_scores
        weights['opacity'] = 0.2
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Compute weighted average score
        floater_scores = torch.zeros(N, device=device)
        for key, score in scores_dict.items():
            floater_scores += weights.get(key, 0) * score
        
        # Create floater mask
        floater_mask = floater_scores > self.config.floater_score_thresh
        
        # Log statistics
        detection_stats = {
            'step': step,
            'total_gaussians': N,
            'floaters_detected': floater_mask.sum().item(),
            'floater_ratio': floater_mask.float().mean().item(),
            'mean_score': floater_scores.mean().item(),
            'max_score': floater_scores.max().item() if N > 0 else 0,
        }
        self.detection_history.append(detection_stats)
        
        if floater_mask.sum() > 0:
            logger.info(
                f"[FloaterDetector] Step {step}: Detected {floater_mask.sum().item()}/{N} "
                f"floaters ({100*floater_mask.float().mean().item():.2f}%)"
            )
        
        return floater_mask, floater_scores


class SpatialSparsityRegularizer(nn.Module):
    """
    Regularizer to encourage spatially efficient Gaussian distributions.
    
    Penalizes:
    1. High density of low-contribution Gaussians
    2. Inefficient spatial coverage
    3. Gaussians far from surfaces
    """
    
    def __init__(self, config: SpatialSparsityConfig):
        """
        Initialize regularizer.
        
        Args:
            config: Regularization configuration
        """
        super().__init__()
        self.config = config
    
    def voxelize_gaussians(
        self,
        means: torch.Tensor,
        grid_size: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign Gaussians to voxels.
        
        Args:
            means: Gaussian centers (N, 3)
            grid_size: Voxel size
            
        Returns:
            Tuple of (voxel_indices, unique_voxels)
        """
        if grid_size is None:
            grid_size = self.config.grid_size
        
        # Quantize to voxel indices
        voxel_coords = (means / grid_size).long()
        
        # Get unique voxels and inverse mapping
        unique_voxels, inverse_indices = torch.unique(
            voxel_coords, dim=0, return_inverse=True
        )
        
        return inverse_indices, unique_voxels
    
    def compute_density_inefficiency_loss(
        self,
        means: torch.Tensor,
        opacities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for inefficient density (many low-opacity Gaussians).
        
        Penalizes voxels with many Gaussians but low total opacity.
        
        Args:
            means: Gaussian centers (N, 3)
            opacities: Gaussian opacities (N, 1)
            
        Returns:
            Scalar loss value
        """
        opacities = opacities.squeeze()
        
        # Voxelize
        inverse_indices, unique_voxels = self.voxelize_gaussians(means)
        num_voxels = unique_voxels.shape[0]
        
        # Count Gaussians per voxel and sum opacities
        voxel_counts = torch.zeros(num_voxels, device=means.device)
        voxel_opacities = torch.zeros(num_voxels, device=means.device)
        
        voxel_counts.scatter_add_(0, inverse_indices, torch.ones_like(opacities))
        voxel_opacities.scatter_add_(0, inverse_indices, opacities)
        
        # Compute inefficiency: count / (opacity + eps)
        # High value = many Gaussians with low total opacity
        inefficiency = voxel_counts / (voxel_opacities + 1e-6)
        
        # Penalize voxels exceeding max density ratio
        violation = F.relu(inefficiency - self.config.max_density)
        
        return violation.mean() * self.config.weight
    
    def compute_entropy_sparsity_loss(
        self,
        opacities: torch.Tensor,
        surface_distances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute entropy-based sparsity loss with optional distance weighting.
        
        Encourages opacities to be either 0 or 1, not intermediate.
        
        Args:
            opacities: Gaussian opacities (N, 1)
            surface_distances: Optional distances to nearest surface (N,)
            
        Returns:
            Scalar loss value
        """
        opacities = opacities.squeeze().clamp(1e-6, 1 - 1e-6)
        
        # Binary entropy
        entropy = -(
            opacities * torch.log(opacities) + 
            (1 - opacities) * torch.log(1 - opacities)
        )
        
        # Optional distance weighting: penalize distant low-opacity Gaussians more
        if surface_distances is not None:
            distance_weights = torch.exp(surface_distances / self.config.distance_scale)
            entropy = entropy * distance_weights
        
        return entropy.mean() * self.config.weight
    
    def forward(
        self,
        means: torch.Tensor,
        opacities: torch.Tensor,
        depth_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all spatial sparsity losses.
        
        Args:
            means: Gaussian centers (N, 3)
            opacities: Gaussian opacities (N, 1)
            depth_points: Optional surface points (M, 3)
            
        Returns:
            Dict of loss components
        """
        losses = {}
        
        # Density inefficiency loss
        losses['density_inefficiency'] = self.compute_density_inefficiency_loss(
            means, opacities
        )
        
        # Entropy sparsity loss with distance weighting
        surface_distances = None
        if depth_points is not None and depth_points.shape[0] > 0:
            # Compute distances to nearest surface point
            # Use sampling for efficiency
            if depth_points.shape[0] > 50000:
                sample_idx = torch.randperm(depth_points.shape[0])[:50000]
                depth_sample = depth_points[sample_idx]
            else:
                depth_sample = depth_points
            
            # Chunk processing
            N = means.shape[0]
            chunk_size = 10000
            surface_distances = torch.zeros(N, device=means.device)
            
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)
                dists = torch.cdist(means[i:end], depth_sample)
                surface_distances[i:end] = dists.min(dim=1)[0]
        
        losses['entropy_sparsity'] = self.compute_entropy_sparsity_loss(
            opacities, surface_distances
        )
        
        return losses


class DepthConsistencyRegularizer(nn.Module):
    """
    Regularizer to enforce depth consistency between Gaussians and estimated depth.
    
    Constrains Gaussians to lie near estimated depth surfaces.
    """
    
    def __init__(
        self,
        max_surface_distance: float = 0.5,
        weight: float = 0.1,
        use_ray_consistency: bool = True,
    ):
        """
        Initialize regularizer.
        
        Args:
            max_surface_distance: Maximum allowed distance from surface (meters)
            weight: Loss weight
            use_ray_consistency: Whether to use ray-based consistency
        """
        super().__init__()
        self.max_surface_distance = max_surface_distance
        self.weight = weight
        self.use_ray_consistency = use_ray_consistency
    
    def compute_surface_distance_loss(
        self,
        means: torch.Tensor,
        depth_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for Gaussians far from depth surfaces.
        
        Args:
            means: Gaussian centers (N, 3)
            depth_points: Surface points (M, 3)
            
        Returns:
            Scalar loss value
        """
        if depth_points is None or depth_points.shape[0] == 0:
            return torch.tensor(0.0, device=means.device)
        
        N = means.shape[0]
        chunk_size = 10000
        
        # Sample depth points if too many
        if depth_points.shape[0] > 50000:
            sample_idx = torch.randperm(depth_points.shape[0])[:50000]
            depth_sample = depth_points[sample_idx]
        else:
            depth_sample = depth_points
        
        total_violation = 0.0
        total_count = 0
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk = means[i:end]
            
            dists = torch.cdist(chunk, depth_sample)
            min_dists = dists.min(dim=1)[0]
            
            # Penalize distances exceeding threshold
            violation = F.relu(min_dists - self.max_surface_distance)
            total_violation += violation.sum()
            total_count += chunk.shape[0]
        
        return (total_violation / total_count) * self.weight
    
    def compute_ray_consistency_loss(
        self,
        means: torch.Tensor,
        opacities: torch.Tensor,
        camera_position: torch.Tensor,
        estimated_depths: torch.Tensor,
        pixel_coords: torch.Tensor,
        intrinsics: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute ray consistency loss.
        
        Penalizes Gaussians with high opacity in front of the estimated depth.
        
        Args:
            means: Gaussian centers (N, 3)
            opacities: Gaussian opacities (N, 1)
            camera_position: Camera position (3,)
            estimated_depths: Per-pixel estimated depths (H, W)
            pixel_coords: Which pixel each Gaussian projects to (N, 2)
            intrinsics: Camera intrinsics (3, 3)
            margin: Depth margin for violation
            
        Returns:
            Scalar loss value
        """
        # Compute Gaussian depths (distance from camera)
        to_gaussians = means - camera_position.unsqueeze(0)
        gaussian_depths = to_gaussians.norm(dim=-1)
        
        # Get expected depth for each Gaussian based on pixel projection
        H, W = estimated_depths.shape
        u = pixel_coords[:, 0].clamp(0, W - 1).long()
        v = pixel_coords[:, 1].clamp(0, H - 1).long()
        expected_depths = estimated_depths[v, u]
        
        # Penalize Gaussians in front of expected depth
        depth_violation = F.relu((expected_depths - margin) - gaussian_depths)
        
        # Weight by opacity (only care about visible Gaussians)
        weighted_violation = depth_violation * opacities.squeeze()
        
        return weighted_violation.mean() * self.weight
    
    def forward(
        self,
        means: torch.Tensor,
        opacities: torch.Tensor,
        depth_points: Optional[torch.Tensor] = None,
        **ray_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute depth consistency losses.
        
        Args:
            means: Gaussian centers (N, 3)
            opacities: Gaussian opacities (N, 1)
            depth_points: Surface points (M, 3)
            **ray_kwargs: Arguments for ray consistency loss
            
        Returns:
            Dict of loss components
        """
        losses = {}
        
        # Surface distance loss
        if depth_points is not None:
            losses['surface_distance'] = self.compute_surface_distance_loss(
                means, depth_points
            )
        
        # Ray consistency loss
        if self.use_ray_consistency and 'camera_position' in ray_kwargs:
            losses['ray_consistency'] = self.compute_ray_consistency_loss(
                means, opacities, **ray_kwargs
            )
        
        return losses


def evaluate_floater_reduction(
    means: torch.Tensor,
    opacities: torch.Tensor,
    depth_points: torch.Tensor,
    distance_threshold: float = 1.0,
    opacity_threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Evaluate the effectiveness of floater reduction.
    
    Args:
        means: Gaussian centers (N, 3)
        opacities: Gaussian opacities (N, 1)
        depth_points: Surface points (M, 3)
        distance_threshold: Distance threshold for floater definition
        opacity_threshold: Opacity threshold for floater definition
        
    Returns:
        Dict of evaluation metrics
    """
    opacities = opacities.squeeze()
    N = means.shape[0]
    
    if depth_points is None or depth_points.shape[0] == 0:
        return {
            'total_points': N,
            'floater_count': 0,
            'floater_ratio': 0.0,
            'avg_floater_opacity': 0.0,
            'avg_surface_distance': 0.0,
        }
    
    # Compute distances to nearest surface
    chunk_size = 10000
    min_distances = torch.zeros(N, device=means.device)
    
    # Sample depth points
    if depth_points.shape[0] > 50000:
        sample_idx = torch.randperm(depth_points.shape[0])[:50000]
        depth_sample = depth_points[sample_idx]
    else:
        depth_sample = depth_points
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        dists = torch.cdist(means[i:end], depth_sample)
        min_distances[i:end] = dists.min(dim=1)[0]
    
    # Identify floaters: far from surface AND low opacity
    floater_mask = (min_distances > distance_threshold) & (opacities < opacity_threshold)
    floater_count = floater_mask.sum().item()
    
    metrics = {
        'total_points': N,
        'floater_count': floater_count,
        'floater_ratio': floater_count / N if N > 0 else 0,
        'avg_floater_opacity': opacities[floater_mask].mean().item() if floater_count > 0 else 0,
        'avg_surface_distance': min_distances.mean().item(),
        'max_surface_distance': min_distances.max().item(),
        'pct_near_surface': (min_distances < distance_threshold).float().mean().item(),
    }
    
    return metrics

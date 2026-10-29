#!/usr/bin/env python3
"""
Floater Reduction Validation Script

This script evaluates the effectiveness of anti-floater measures in the Gaussian
splatting model. It computes various metrics to quantify floater presence.

Usage:
    python tools/validate_floater_reduction.py --checkpoint path/to/checkpoint.pth
    
Author: Auto-generated
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.floater_detection import evaluate_floater_reduction


def load_gaussian_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Dict:
    """
    Load Gaussian parameters from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to
        
    Returns:
        Dict with Gaussian parameters
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract Gaussian parameters - adapt based on your checkpoint format
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    
    # Find Gaussian parameters (may be prefixed with class name)
    means = None
    opacities = None
    
    for key, value in state.items():
        if '_means' in key and means is None:
            means = value.to(device)
        if '_opacities' in key and opacities is None:
            opacities = value.to(device)
    
    if means is None or opacities is None:
        raise ValueError("Could not find Gaussian parameters in checkpoint")
    
    return {
        'means': means,
        'opacities': torch.sigmoid(opacities) if opacities.min() < 0 else opacities,
    }


def load_depth_points(
    depth_dir: Optional[str],
    intrinsics_file: Optional[str],
    max_points: int = 100000,
    device: str = 'cuda',
) -> Optional[torch.Tensor]:
    """
    Load depth points from precomputed depth maps.
    
    Args:
        depth_dir: Directory containing depth .npy files
        intrinsics_file: Path to camera intrinsics
        max_points: Maximum number of points to load
        device: Device for tensors
        
    Returns:
        Tensor of 3D points (N, 3) or None
    """
    if depth_dir is None or not os.path.exists(depth_dir):
        print("Warning: No depth directory provided, skipping depth-based metrics")
        return None
    
    depth_dir = Path(depth_dir)
    depth_files = list(depth_dir.glob("*.npy"))
    
    if not depth_files:
        print(f"Warning: No depth files found in {depth_dir}")
        return None
    
    print(f"Found {len(depth_files)} depth files")
    
    # Load a subset of depth maps and convert to points
    all_points = []
    
    # Load intrinsics if available
    if intrinsics_file and os.path.exists(intrinsics_file):
        intrinsics = np.load(intrinsics_file)
    else:
        # Default intrinsics (adjust for your camera)
        intrinsics = np.array([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ], dtype=np.float32)
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    for depth_file in depth_files[:10]:  # Sample subset
        depth = np.load(depth_file)
        H, W = depth.shape
        
        # Create meshgrid
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # Valid mask
        valid = (depth > 0.1) & (depth < 80.0)
        
        # Backproject to 3D
        u_valid = u[valid].astype(np.float32)
        v_valid = v[valid].astype(np.float32)
        d_valid = depth[valid]
        
        x = (u_valid - cx) * d_valid / fx
        y = (v_valid - cy) * d_valid / fy
        z = d_valid
        
        points = np.stack([x, y, z], axis=-1)
        
        # Subsample
        if len(points) > max_points // len(depth_files):
            idx = np.random.choice(len(points), max_points // len(depth_files), replace=False)
            points = points[idx]
        
        all_points.append(points)
    
    all_points = np.concatenate(all_points, axis=0)
    print(f"Loaded {len(all_points)} depth points")
    
    return torch.from_numpy(all_points).float().to(device)


def compute_comprehensive_metrics(
    means: torch.Tensor,
    opacities: torch.Tensor,
    depth_points: Optional[torch.Tensor],
    distance_thresholds: list = [0.5, 1.0, 2.0],
    opacity_thresholds: list = [0.1, 0.2, 0.3],
) -> Dict:
    """
    Compute comprehensive floater metrics.
    
    Args:
        means: Gaussian centers (N, 3)
        opacities: Gaussian opacities (N, 1)
        depth_points: Optional surface points (M, 3)
        distance_thresholds: Distance thresholds to evaluate
        opacity_thresholds: Opacity thresholds to evaluate
        
    Returns:
        Dict of metrics
    """
    metrics = {
        'total_gaussians': means.shape[0],
        'mean_opacity': opacities.mean().item(),
        'median_opacity': opacities.median().item(),
        'low_opacity_ratio': (opacities < 0.1).float().mean().item(),
    }
    
    # Opacity distribution
    for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]:
        metrics[f'opacity_below_{thresh}'] = (opacities < thresh).sum().item()
    
    if depth_points is not None:
        # Distance to surface metrics
        N = means.shape[0]
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
        
        metrics['mean_surface_distance'] = min_distances.mean().item()
        metrics['median_surface_distance'] = min_distances.median().item()
        metrics['max_surface_distance'] = min_distances.max().item()
        
        # Floater counts for various thresholds
        for dist_thresh in distance_thresholds:
            for op_thresh in opacity_thresholds:
                floater_mask = (min_distances > dist_thresh) & (opacities.squeeze() < op_thresh)
                key = f'floaters_dist{dist_thresh}_op{op_thresh}'
                metrics[key] = floater_mask.sum().item()
                metrics[key + '_ratio'] = floater_mask.float().mean().item()
        
        # Near-surface statistics
        for thresh in [0.5, 1.0, 2.0]:
            near_surface = (min_distances < thresh).float().mean().item()
            metrics[f'near_surface_{thresh}m'] = near_surface
        
        # Use the standard evaluation function
        standard_metrics = evaluate_floater_reduction(
            means, opacities, depth_points,
            distance_threshold=1.0,
            opacity_threshold=0.3,
        )
        metrics.update({f'standard_{k}': v for k, v in standard_metrics.items()})
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Validate Floater Reduction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--depth_dir', type=str, default=None, help='Directory with depth maps')
    parser.add_argument('--intrinsics', type=str, default=None, help='Camera intrinsics file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.checkpoint}")
    gaussians = load_gaussian_checkpoint(args.checkpoint, args.device)
    
    print(f"Loaded {gaussians['means'].shape[0]} Gaussians")
    
    # Load depth points if available
    depth_points = load_depth_points(args.depth_dir, args.intrinsics, device=args.device)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_comprehensive_metrics(
        gaussians['means'],
        gaussians['opacities'],
        depth_points,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("FLOATER REDUCTION METRICS")
    print("=" * 60)
    
    print(f"\nBasic Statistics:")
    print(f"  Total Gaussians: {metrics['total_gaussians']:,}")
    print(f"  Mean Opacity: {metrics['mean_opacity']:.4f}")
    print(f"  Median Opacity: {metrics['median_opacity']:.4f}")
    print(f"  Low Opacity (<0.1) Ratio: {metrics['low_opacity_ratio']*100:.2f}%")
    
    if depth_points is not None:
        print(f"\nSurface Distance Statistics:")
        print(f"  Mean Distance to Surface: {metrics['mean_surface_distance']:.4f}m")
        print(f"  Median Distance to Surface: {metrics['median_surface_distance']:.4f}m")
        print(f"  Max Distance to Surface: {metrics['max_surface_distance']:.4f}m")
        
        print(f"\nNear-Surface Percentages:")
        for thresh in [0.5, 1.0, 2.0]:
            print(f"  Within {thresh}m: {metrics[f'near_surface_{thresh}m']*100:.2f}%")
        
        print(f"\nFloater Counts (distance > 1.0m AND opacity < 0.3):")
        print(f"  Count: {metrics['floaters_dist1.0_op0.3']:,}")
        print(f"  Ratio: {metrics['floaters_dist1.0_op0.3_ratio']*100:.4f}%")
    
    print("=" * 60)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")
    
    return metrics


if __name__ == '__main__':
    main()

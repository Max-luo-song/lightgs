"""
Depth Estimator Module using Depth Anything V2

This module provides a wrapper for Depth Anything V2 metric depth estimation model,
specifically designed for autonomous driving scenarios.

Author: Auto-generated
"""

import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)

# Add Depth-Anything-V2 to path
DEPTH_ANYTHING_PATH = Path(__file__).parent.parent / "third_party" / "Depth-Anything-V2"
METRIC_DEPTH_PATH = DEPTH_ANYTHING_PATH / "metric_depth"

if str(DEPTH_ANYTHING_PATH) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_PATH))
if str(METRIC_DEPTH_PATH) not in sys.path:
    sys.path.insert(0, str(METRIC_DEPTH_PATH))


class DepthAnythingEstimator:
    """
    Wrapper for Depth Anything V2 metric depth estimation.
    
    Supports:
    - Single image inference
    - Batch inference
    - Depth caching to avoid recomputation
    - Both relative and metric depth modes
    """
    
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    # Pretrained model URLs
    CHECKPOINT_URLS = {
        'vits_vkitti': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth',
        'vitb_vkitti': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth',
        'vitl_vkitti': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth',
        'vits_relative': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
        'vitb_relative': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
        'vitl_relative': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
    }
    
    def __init__(
        self,
        encoder: str = 'vitl',
        mode: str = 'metric',  # 'metric' or 'relative'
        dataset: str = 'vkitti',  # 'vkitti' for outdoor, 'hypersim' for indoor
        max_depth: float = 80.0,
        checkpoint_dir: Optional[str] = None,
        device: str = 'cuda',
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize DepthAnythingEstimator.
        
        Args:
            encoder: Model encoder size ('vits', 'vitb', 'vitl')
            mode: 'metric' for metric depth in meters, 'relative' for relative depth
            dataset: Training dataset ('vkitti' for outdoor/driving, 'hypersim' for indoor)
            max_depth: Maximum depth value in meters (80 for outdoor, 20 for indoor)
            checkpoint_dir: Directory to store/load checkpoints
            device: Device to run inference on
            cache_dir: Directory to cache depth maps
            enable_cache: Whether to cache computed depth maps
        """
        self.encoder = encoder
        self.mode = mode
        self.dataset = dataset
        self.max_depth = max_depth
        self.device = device
        self.enable_cache = enable_cache
        
        # Setup directories
        if checkpoint_dir is None:
            checkpoint_dir = DEPTH_ANYTHING_PATH / "checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "depth_anything_v2"
        self.cache_dir = Path(cache_dir)
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(f"DepthAnythingEstimator initialized: encoder={encoder}, mode={mode}, device={device}")
    
    def _load_model(self):
        """Load the Depth Anything V2 model."""
        # Try metric_depth path first (for metric mode), then regular path
        DepthAnythingV2 = None
        
        if self.mode == 'metric':
            try:
                # For metric depth, import from metric_depth subdirectory
                from depth_anything_v2.dpt import DepthAnythingV2
            except ImportError:
                pass
        
        if DepthAnythingV2 is None:
            try:
                # Try regular import (relative depth mode)
                import importlib.util
                dpt_path = DEPTH_ANYTHING_PATH / "depth_anything_v2" / "dpt.py"
                if dpt_path.exists():
                    spec = importlib.util.spec_from_file_location("dpt", dpt_path)
                    dpt_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(dpt_module)
                    DepthAnythingV2 = dpt_module.DepthAnythingV2
            except Exception as e:
                logger.warning(f"Failed to load from regular path: {e}")
        
        if DepthAnythingV2 is None:
            raise ImportError(
                "Could not import Depth Anything V2. "
                "Please ensure third_party/Depth-Anything-V2 exists and is properly set up."
            )
        
        config = self.MODEL_CONFIGS[self.encoder].copy()
        
        if self.mode == 'metric':
            config['max_depth'] = self.max_depth
        
        model = DepthAnythingV2(**config)
        
        # Get checkpoint path
        checkpoint_path = self._get_checkpoint_path()
        
        if not checkpoint_path.exists():
            logger.info(f"Checkpoint not found, downloading to {checkpoint_path}")
            self._download_checkpoint(checkpoint_path)
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        return model
    
    def _get_checkpoint_path(self) -> Path:
        """Get the checkpoint file path."""
        if self.mode == 'metric':
            filename = f"depth_anything_v2_metric_{self.dataset}_{self.encoder}.pth"
        else:
            filename = f"depth_anything_v2_{self.encoder}.pth"
        return self.checkpoint_dir / filename
    
    def _download_checkpoint(self, checkpoint_path: Path):
        """Download checkpoint from HuggingFace."""
        if self.mode == 'metric':
            key = f"{self.encoder}_{self.dataset}"
        else:
            key = f"{self.encoder}_relative"
        
        url = self.CHECKPOINT_URLS.get(key)
        if url is None:
            raise ValueError(f"Unknown checkpoint key: {key}")
        
        import urllib.request
        logger.info(f"Downloading from {url}")
        urllib.request.urlretrieve(url, checkpoint_path)
        logger.info(f"Downloaded to {checkpoint_path}")
    
    def _get_cache_key(self, image_path: str) -> str:
        """Generate a cache key for an image."""
        # Use file path and modification time for cache key
        stat = os.stat(image_path)
        key_str = f"{image_path}_{stat.st_mtime}_{self.encoder}_{self.mode}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a cache key."""
        return self.cache_dir / f"{cache_key}.npy"
    
    def _load_from_cache(self, image_path: str) -> Optional[np.ndarray]:
        """Try to load depth from cache."""
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(image_path)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, image_path: str, depth: np.ndarray):
        """Save depth to cache."""
        if not self.enable_cache:
            return
        
        cache_key = self._get_cache_key(image_path)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            np.save(cache_path, depth)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    @torch.no_grad()
    def estimate_depth(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        input_size: int = 518,
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Estimate depth for a single image.
        
        Args:
            image: Image path, numpy array (HxWxC, BGR), or tensor (CxHxW)
            input_size: Input size for the model
            return_tensor: If True, return torch.Tensor instead of numpy array
            
        Returns:
            Depth map of shape (H, W) in meters (metric mode) or relative values
        """
        # Check cache
        if isinstance(image, str):
            cached = self._load_from_cache(image)
            if cached is not None:
                if return_tensor:
                    return torch.from_numpy(cached).to(self.device)
                return cached
        
        # Load and preprocess image
        if isinstance(image, str):
            raw_image = cv2.imread(image)
            if raw_image is None:
                raise ValueError(f"Failed to load image: {image}")
        elif isinstance(image, torch.Tensor):
            # Convert CxHxW tensor to HxWxC numpy (BGR)
            raw_image = image.permute(1, 2, 0).cpu().numpy()
            raw_image = (raw_image * 255).astype(np.uint8)
            if raw_image.shape[2] == 3:
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        else:
            raw_image = image
        
        # Run inference
        depth = self.model.infer_image(raw_image, input_size)
        
        # Cache result
        if isinstance(image, str):
            self._save_to_cache(image, depth)
        
        if return_tensor:
            return torch.from_numpy(depth).to(self.device)
        return depth
    
    @torch.no_grad()
    def estimate_depth_batch(
        self,
        images: List[Union[str, np.ndarray]],
        input_size: int = 518,
        batch_size: int = 4,
        return_tensor: bool = False,
    ) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        Estimate depth for a batch of images.
        
        Args:
            images: List of image paths or numpy arrays
            input_size: Input size for the model
            batch_size: Batch size for inference
            return_tensor: If True, return torch.Tensor instead of numpy array
            
        Returns:
            List of depth maps
        """
        depths = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_depths = [self.estimate_depth(img, input_size, return_tensor) for img in batch]
            depths.extend(batch_depths)
        
        return depths
    
    def depth_to_points(
        self,
        depth: Union[np.ndarray, torch.Tensor],
        intrinsics: Union[np.ndarray, torch.Tensor],
        extrinsics: Optional[Union[np.ndarray, torch.Tensor]] = None,
        min_depth: float = 0.1,
        max_depth: Optional[float] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert depth map to 3D points.
        
        Args:
            depth: Depth map (H, W)
            intrinsics: Camera intrinsic matrix (3, 3) or (4, 4)
            extrinsics: Optional camera extrinsic matrix (4, 4) for world coordinates
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth (default: self.max_depth)
            
        Returns:
            Point cloud (N, 3) in camera or world coordinates
        """
        if max_depth is None:
            max_depth = self.max_depth
        
        use_torch = isinstance(depth, torch.Tensor)
        
        if use_torch:
            device = depth.device
            H, W = depth.shape
            
            # Create pixel coordinates
            v, u = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            
            # Valid depth mask
            valid = (depth > min_depth) & (depth < max_depth)
            
            # Get valid pixels
            u_valid = u[valid].float()
            v_valid = v[valid].float()
            d_valid = depth[valid]
            
            # Intrinsics
            if not isinstance(intrinsics, torch.Tensor):
                intrinsics = torch.from_numpy(intrinsics).float().to(device)
            
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            
            # Backproject to 3D
            x = (u_valid - cx) * d_valid / fx
            y = (v_valid - cy) * d_valid / fy
            z = d_valid
            
            points = torch.stack([x, y, z], dim=-1)
            
            # Transform to world coordinates if extrinsics provided
            if extrinsics is not None:
                if not isinstance(extrinsics, torch.Tensor):
                    extrinsics = torch.from_numpy(extrinsics).float().to(device)
                # extrinsics: cam_to_world (4x4)
                R = extrinsics[:3, :3]
                t = extrinsics[:3, 3]
                points = points @ R.T + t
            
            return points
        else:
            # NumPy version
            H, W = depth.shape
            
            v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            
            valid = (depth > min_depth) & (depth < max_depth)
            
            u_valid = u[valid].astype(np.float32)
            v_valid = v[valid].astype(np.float32)
            d_valid = depth[valid]
            
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            
            x = (u_valid - cx) * d_valid / fx
            y = (v_valid - cy) * d_valid / fy
            z = d_valid
            
            points = np.stack([x, y, z], axis=-1)
            
            if extrinsics is not None:
                R = extrinsics[:3, :3]
                t = extrinsics[:3, 3]
                points = points @ R.T + t
            
            return points


class DepthCache:
    """
    Cache system for precomputed depth maps.
    
    Supports both file-based caching and in-memory caching for training.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_items: int = 100,
        device: str = 'cuda',
    ):
        """
        Initialize DepthCache.
        
        Args:
            cache_dir: Directory for file-based cache
            max_memory_items: Maximum items to keep in memory
            device: Device for cached tensors
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self.device = device
        
        self._memory_cache: Dict[str, torch.Tensor] = {}
        self._access_order: List[str] = []
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_key(self, image_path: str) -> str:
        """Generate cache key from image path."""
        return hashlib.md5(image_path.encode()).hexdigest()
    
    def get(self, image_path: str) -> Optional[torch.Tensor]:
        """Get depth from cache."""
        key = self._get_key(image_path)
        
        # Check memory cache first
        if key in self._memory_cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._memory_cache[key]
        
        # Check file cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pt"
            if cache_file.exists():
                depth = torch.load(cache_file, map_location=self.device)
                self._add_to_memory(key, depth)
                return depth
        
        return None
    
    def put(self, image_path: str, depth: torch.Tensor):
        """Store depth in cache."""
        key = self._get_key(image_path)
        
        # Add to memory cache
        self._add_to_memory(key, depth)
        
        # Save to file cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pt"
            torch.save(depth, cache_file)
    
    def _add_to_memory(self, key: str, depth: torch.Tensor):
        """Add item to memory cache with LRU eviction."""
        if key in self._memory_cache:
            self._access_order.remove(key)
        elif len(self._memory_cache) >= self.max_memory_items:
            # Evict least recently used
            oldest_key = self._access_order.pop(0)
            del self._memory_cache[oldest_key]
        
        self._memory_cache[key] = depth.to(self.device)
        self._access_order.append(key)
    
    def precompute(
        self,
        image_paths: List[str],
        estimator: DepthAnythingEstimator,
        show_progress: bool = True,
    ):
        """
        Precompute and cache depth maps for a list of images.
        
        Args:
            image_paths: List of image file paths
            estimator: DepthAnythingEstimator instance
            show_progress: Whether to show progress bar
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc="Precomputing depth maps")
            except ImportError:
                iterator = image_paths
        else:
            iterator = image_paths
        
        for image_path in iterator:
            if self.get(image_path) is None:
                depth = estimator.estimate_depth(image_path, return_tensor=True)
                self.put(image_path, depth)
    
    def clear_memory(self):
        """Clear memory cache."""
        self._memory_cache.clear()
        self._access_order.clear()
    
    def clear_all(self):
        """Clear both memory and file cache."""
        self.clear_memory()
        if self.cache_dir:
            for f in self.cache_dir.glob("*.pt"):
                f.unlink()


def create_depth_estimator(
    config: Optional[Dict] = None,
    **kwargs
) -> DepthAnythingEstimator:
    """
    Factory function to create DepthAnythingEstimator.
    
    Args:
        config: Configuration dict with keys: encoder, mode, dataset, max_depth, etc.
        **kwargs: Override config values
        
    Returns:
        DepthAnythingEstimator instance
    """
    default_config = {
        'encoder': 'vitl',
        'mode': 'metric',
        'dataset': 'vkitti',
        'max_depth': 80.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'enable_cache': True,
    }
    
    if config:
        default_config.update(config)
    default_config.update(kwargs)
    
    return DepthAnythingEstimator(**default_config)

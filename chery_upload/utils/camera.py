# Camera pose manipulation and trajectory generation.
import os
import torch
import numpy as np
from typing import Dict

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


def interpolate_poses(key_poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Interpolate between key poses to generate a smooth trajectory.

    Args:
        key_poses (torch.Tensor): Tensor of shape (N, 4, 4) containing key camera poses.
        target_frames (int): Number of frames to interpolate.

    Returns:
        torch.Tensor: Interpolated poses of shape (target_frames, 4, 4).
    """
    device = key_poses.device
    key_poses = key_poses.cpu().numpy()

    # Separate translation and rotation
    translations = key_poses[:, :3, 3]
    rotations = key_poses[:, :3, :3]

    # Create time array
    times = np.linspace(0, 1, len(key_poses))
    target_times = np.linspace(0, 1, target_frames)

    # Interpolate translations
    interp_translations = np.stack([
        np.interp(target_times, times, translations[:, i])
        for i in range(3)
    ], axis=-1)
    
    # Interpolate rotations using Slerp
    key_rots = R.from_matrix(rotations)
    slerp = Slerp(times, key_rots)
    interp_rotations = slerp(target_times).as_matrix()

    # Combine interpolated translations and rotations
    interp_poses = np.eye(4)[None].repeat(target_frames, axis=0)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations

    return torch.tensor(interp_poses, dtype=torch.float32, device=device)


def look_at_rotation(
    direction: torch.Tensor, up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])
) -> torch.Tensor:
    """Calculate rotation matrix to look at a specific direction."""
    front = torch.nn.functional.normalize(direction, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(front, up), dim=-1)
    up = torch.cross(right, front)
    rotation_matrix = torch.stack([right, up, -front], dim=-1)
    return rotation_matrix


def get_interp_novel_trajectories(
    dataset_type: str,
    scene_idx: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    traj_type: str = "front_center_interp",
    target_frames: int = 100,
) -> torch.Tensor:
    original_frames = per_cam_poses[ref_cam_id].shape[0]
    trajectory_generators = {
        "original_traj": original_traj,
        "left_shift_1m": left_shift_1m,
        "left_shift_3m": left_shift_3m,
        "left_shift_5m": left_shift_5m,
        "right_shift_1m": right_shift_1m,
        "right_shift_3m": right_shift_3m,
        "right_shift_5m": right_shift_5m,
        "front_shift_1m": front_shift_1m,
        "front_shift_3m": front_shift_3m,
        "front_shift_5m": front_shift_5m,
        "back_shift_1m": back_shift_1m,
        "back_shift_3m": back_shift_3m,
        "back_shift_5m": back_shift_5m,
        "change_lane_1m": change_lane_1m,
        "change_lane_2m": change_lane_2m,
        "change_lane_3.5m": change_lane_3_5m,
        "up_shift_5m": up_shift_5m,
        "front_center_interp": front_center_interp,
        "s_curve": s_curve,
        "three_key_poses": three_key_poses_trajectory,
    }

    if traj_type not in trajectory_generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return trajectory_generators[traj_type](
        dataset_type=dataset_type,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
        original_frames=original_frames,
        target_frames=target_frames,
    )


def original_traj(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return per_cam_poses[ref_cam_id]



def _shift_trajectory_fn(
    front: float,
    left: float,
    up: float,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
):
    """
    Adjust the camera pose:
    1. Increase the camera height by 5 meters.

    Args:
        per_cam_poses (Dict[int, torch.Tensor]): Dictionary of camera poses.

    Returns:
        torch.Tensor: Adjusted camera pose of shape (N, 4, 4).
    """
    assert (
        0 in per_cam_poses.keys() or 1 in per_cam_poses.keys()
    ), "Camera ID 0 or 1 is required for shift_trajectory_fn"

    # NOTE(syc): 这里取第一个相机
    current_pose = per_cam_poses[ref_cam_id]

    device = current_pose.device

    translation_matrix = torch.eye(4, dtype=torch.float32, device=device)
    translation_matrix[:3, 3] = torch.tensor(
        [front, left, up], dtype=torch.float32, device=device
    )

    rotation_matrix = torch.eye(4, dtype=torch.float32, device=device)

    transform_matrix = torch.mm(translation_matrix, rotation_matrix)
    adjusted_poses = torch.bmm(
        transform_matrix.unsqueeze(0).expand(current_pose.shape[0], -1, -1),
        current_pose,
    )

    return adjusted_poses


# Left


def left_shift_1m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=0,
        left=1,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def left_shift_3m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=0,
        left=3,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def left_shift_5m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=0,
        left=5,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


# Right


def right_shift_1m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=0,
        left=-1,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def right_shift_3m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=0,
        left=-3,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def right_shift_5m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=0,
        left=-5,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


# Front


def front_shift_1m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=1,
        left=0,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def front_shift_3m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=3,
        left=0,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def front_shift_5m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=5,
        left=0,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


# Back


def back_shift_1m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=-1,
        left=0,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def back_shift_3m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=-3,
        left=0,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def back_shift_5m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=-5,
        left=0,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


# Up


def up_shift_5m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return _shift_trajectory_fn(
        front=0,
        left=0,
        up=5,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


# Change lane
# NOTE(syc): 目前通过两条轨迹的插值实现，针对固定场景写死，未来可能需要更通用的实现
def change_lane_fn(
    shift_distance: float,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
) -> torch.Tensor:
    # 使用原始轨迹作为基础
    new_trajectory = per_cam_poses[ref_cam_id].clone()

    # 向前平移
    shifted_trajectory = _shift_trajectory_fn(
        front=shift_distance,
        left=0,
        up=0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )

    # 从第 100 帧开始切换车道，持续 45 帧
    change_start = 100
    change_duration = 45

    change_end = change_start + change_duration

    # 线性插值 >>>
    for i in range(change_duration):
        alpha = (i + 1) / change_duration
        new_trajectory[change_start + i] = (1 - alpha) * new_trajectory[
            change_start + i
        ] + alpha * shifted_trajectory[change_start + i]
    # <<<

    # 余弦插值 >>>
    t = np.linspace(0, 1, change_duration)
    alpha = 0.5 * (1 - np.cos(np.pi * t))  # 余弦插值，生成平滑单调曲线

    for i in range(change_duration):
        new_trajectory[change_start + i, :3, 3] = (1 - alpha[i]) * new_trajectory[
            change_start + i, :3, 3
        ] + alpha[i] * shifted_trajectory[change_start + i, :3, 3]
    # <<<

    # 更新 change_end 之后的轨迹，保持在目标车道
    new_trajectory[change_end:, :3, 3] = shifted_trajectory[change_end:, :3, 3]

    return new_trajectory


def change_lane_1m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return change_lane_fn(
        shift_distance=1.0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def change_lane_2m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return change_lane_fn(
        shift_distance=2.0,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


def change_lane_3_5m(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    return change_lane_fn(
        shift_distance=3.5,
        ref_cam_id=ref_cam_id,
        per_cam_poses=per_cam_poses,
    )


########################################################################################


def front_center_interp(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
) -> torch.Tensor:
    """Interpolate key frames from the front center camera."""
    assert (
        0 in per_cam_poses.keys()
    ), "Front center camera (ID 0) is required for front_center_interp"
    key_poses = per_cam_poses[0][
        :: original_frames // 4
    ]  # Select every 4th frame as key frame
    return interpolate_poses(key_poses, target_frames)


def s_curve(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Create an S-shaped trajectory using the front three cameras."""
    assert all(
        cam in per_cam_poses.keys() for cam in [0, 1, 2]
    ), "Front three cameras (IDs 0, 1, 2) are required for s_curve"
    key_poses = torch.cat(
        [
            per_cam_poses[0][0:1],
            per_cam_poses[1][original_frames // 4 : original_frames // 4 + 1],
            per_cam_poses[0][original_frames // 2 : original_frames // 2 + 1],
            per_cam_poses[2][3 * original_frames // 4 : 3 * original_frames // 4 + 1],
            per_cam_poses[0][-1:],
        ],
        dim=0,
    )
    return interpolate_poses(key_poses, target_frames)


def three_key_poses_trajectory(
    dataset_type: str,
    ref_cam_id: int,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """
    Create a trajectory using three key poses:
    1. First frame of front center camera
    2. Middle frame with interpolated rotation and position from camera 1 or 2
    3. Last frame of front center camera

    The rotation of the middle pose is calculated using Slerp between
    the start frame and the middle frame of camera 1 or 2.

    Args:
        dataset_type (str): Type of the dataset (e.g., "waymo", "pandaset", etc.).
        per_cam_poses (Dict[int, torch.Tensor]): Dictionary of camera poses.
        original_frames (int): Number of original frames.
        target_frames (int): Number of frames in the output trajectory.

    Returns:
        torch.Tensor: Trajectory of shape (target_frames, 4, 4).
    """
    # assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    # assert 1 in per_cam_poses.keys() or 2 in per_cam_poses.keys(), "Either camera 1 or camera 2 is required"

    # First key pose: First frame of front center camera
    start_pose = per_cam_poses[0][0]
    key_poses = [start_pose]

    # Select camera for middle frame
    middle_frame = int(original_frames // 2)
    chosen_cam = np.random.choice([1, 2])

    middle_pose = per_cam_poses[chosen_cam][middle_frame]

    # Calculate interpolated rotation for middle pose
    start_rotation = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    middle_rotation = R.from_matrix(middle_pose[:3, :3].cpu().numpy())
    slerp = Slerp(
        [0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()])
    )
    interpolated_rotation = slerp(0.5).as_matrix()

    # Create middle key pose with interpolated rotation and original translation
    middle_key_pose = torch.eye(4, device=start_pose.device)
    middle_key_pose[:3, :3] = torch.tensor(
        interpolated_rotation, device=start_pose.device
    )
    middle_key_pose[:3, 3] = middle_pose[:3, 3]  # Keep the original translation
    key_poses.append(middle_key_pose)

    # Third key pose: Last frame of front center camera
    key_poses.append(per_cam_poses[0][-1])

    # Stack the key poses and interpolate
    key_poses = torch.stack(key_poses)
    return interpolate_poses(key_poses, target_frames)

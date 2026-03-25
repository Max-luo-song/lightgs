#!/bin/bash
###############################################################################
# Waymo SMPL 人体姿态提取脚本
# 用法: ./extract_human_pose.sh <scene_id> <output_dir> [gpu_id]
###############################################################################

# 参数处理
if [ $# -lt 2 ]; then
    echo "ERROR: 用法: $0 <scene_id> <output_dir> [gpu_id]"
    echo "ERROR: 示例: $0 621 /path/to/processed 0"
    exit 1
fi

scene_id="$1"
output_dir="${2%/}"  # 移除末尾斜杠
gpu="${3:--1}"       # 如果未指定，使用-1（自动选择）

echo "[INFO] Scene ID: $scene_id"
echo "[INFO] Output Dir: $output_dir"
echo "[INFO] GPU ID: $gpu"

# ============================================================================
# STAGE 3: SMPL 人体姿态提取
# ============================================================================
echo "[INFO] ========== STAGE 3: SMPL Pose Extraction =========="

# Pick an available gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "[ERROR] 没有可用的 GPU"
        exit 1
    fi
fi
echo "[INFO] 使用 GPU: ${gpu}"

# 激活 Conda 环境
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate 4D-humans
export PYOPENGL_PLATFORM=osmesa

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
    --data_root "$output_dir/training" \
    --dataset qcraft \
    --scene_id "$scene_id"

if [ $? -ne 0 ]; then
    echo "[ERROR] SMPL 提取失败"
    exit 1
fi

echo "[SUCCESS] SMPL 提取完成"

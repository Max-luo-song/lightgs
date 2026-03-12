# 参数设置,记得将新场景列表写入data/qcraft_scenes.txt
################################################################################
gpu=0
segformer_path=/opt/mmsegmentation
raw_data_root="/nas_thoru/scenario_data/"
processed_data_root="/nas_thoru/oldbak/zyk/Project/scene_reconstruction-main/data/qcraft/processed"
work_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
################################################################################

# step 1: preprocess
source /opt/conda/etc/profile.d/conda.sh && conda activate main
export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root ${raw_data_root} \
    --target_dir ${processed_data_root} \
    --dataset qcraft \
    --split training \
    --split_file data/qcraft_scenes.txt \
    --workers 8 \
    --process_keys ego_masks images calib pose objects dynamic_masks lidar \
    --skip_front_wide_side_cameras # --scene_ids $scene_id \

# step 2: extract mask
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source ${SCRIPT_DIR}/utils.sh
source /opt/conda/etc/profile.d/conda.sh && conda activate mmseg2

# Pick an avaliable gpu
# source scripts/utils.sh
# if [ "${gpu}" = "-1" ]; then
#     gpu=$(pick_gpu)
#     if [ -z "${gpu}" ]; then
#         echo "no gpu found"
#         exit 1
#     fi
# fi
# echo "Using GPU: ${gpu}"

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks_mmseg2.py \
    --data_root ${processed_data_root}/training \
    --config=${segformer_path}/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py \
    --checkpoint=${segformer_path}/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth \
    --split_file data/qcraft_scenes.txt \
    --process_dynamic_mask

# step 3: process human pose
source /opt/conda/etc/profile.d/conda.sh && conda activate 4d-humans
export PYTHONPATH=$PYTHONPATH:${work_root}/third_party/Humans4D
CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
    --data_root ${processed_data_root}/training \
    --dataset qcraft \
    --split_file data/qcraft_scenes.txt \
    --verbose # --scene_id $scene_id
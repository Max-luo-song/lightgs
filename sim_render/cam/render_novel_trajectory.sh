# 参数设置
################################################################################
ckpt_path="output/qcraft_20251103_134932_QLC0N1000623_12337_12352/20251121_lidar+cam0_1_2_3_4_5_6_8_9_10_12/checkpoint_final.pth"

traj_types=(
    original_traj
    # left_shift_1m
    left_shift_3m
    # left_shift_5m
    # right_shift_1m
    # right_shift_3m
    # right_shift_5m
    # front_shift_1m
    # front_shift_3m
    # front_shift_5m
    # back_shift_1m
    # back_shift_3m
    # back_shift_5m
    # change_lane_1m
    # change_lane_2m
    # change_lane_3.5m
)

cam_ids=(0 1 2 3 5 6 7 9 10 11 12)
downscales=(1 1 1 1 1 1 1 1 1 1 1)

fps=10

render_rgb=true
render_depth=false
save_images=true
generate_lidar_pc=false
################################################################################
source scripts/utils.sh

export PYTHONPATH=$(pwd)

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
    echo "no gpu found"
    exit 1
fi


echo "Using checkpoint: $ckpt_path"

bool_args=""
if [ "$render_rgb" = true ]; then
    bool_args="$bool_args --render_rgb"
fi
if [ "$render_depth" = true ]; then
    bool_args="$bool_args --render_depth"
fi
if [ "$save_images" = true ]; then
    bool_args="$bool_args --save_images"
fi
if [ "$generate_lidar_pc" = true ]; then
    bool_args="$bool_args --generate_lidar_pc"
fi

CUDA_VISIBLE_DEVICES=${gpu} python sim_render/cam/render_novel_trajectory.py \
    --resume_from $ckpt_path \
    --traj_types "${traj_types[@]}" \
    --cam_ids "${cam_ids[@]}" \
    --downscales "${downscales[@]}" \
    --fps $fps \
    $bool_args
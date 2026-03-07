# 参数设置
################################################################################
gpu=-1

scene_idx="mtgs"

camera_ids=(0 1 2 3 5 6 9 10 12)  # camera IDs to use, e.g., (0), (0 2 4)
# camera_ids=(0)  # camera IDs to use, e.g., (0), (0 2 4)
lidar_type="lidar"  # lidar（运动补偿前）/visual（纯视觉）

config_file="configs/omnire_extended_cam_lidar.yaml"
dataset_config="qcraft/9cams_${lidar_type}"
extra_config_info="mtgs"  # 额外信息 不开启models/nodes/road.py中的refine_after

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
#########################################################################ß#######

# Pick an avaliable gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi
echo "Using GPU: ${gpu}"


output_root="output"
project_name="qcraft_${scene_idx}"

date_str=$(date +%Y%m%d)
run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info}"

run_dir="${output_root}/${project_name}/${run_name}"

# 检查输出目录是否存在，防止覆盖
if [ -d "$run_dir" ]; then
    echo "错误：目录 $run_dir 已存在！"
    rm -r "$run_dir"
    # exit 1
fi
mkdir -p "$run_dir"

# 备份当前脚本到输出目录
script_name=$(basename "$0")
backup_path="${run_dir}/${script_name}"
cp "$0" "$backup_path"

# 启动训练
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py \
    --config_file $config_file \
    --output_root $output_root \
    --project $project_name \
    --run_name $run_name \
    dataset=$dataset_config \
    data.scene_idx=$scene_idx \
    data.pixel_source.cameras="[$(IFS=','; echo "${camera_ids[*]}")]" \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep


# ckpt_path="${run_dir}/checkpoint_final.pth"

# traj_types=(
#     original_traj
#     left_shift_1m
#     left_shift_3m
#     # left_shift_5m
#     right_shift_1m
#     right_shift_3m
#     # right_shift_5m
#     # front_shift_1m
#     # front_shift_3m
#     # front_shift_5m
#     # back_shift_1m
#     # back_shift_3m
#     # back_shift_5m
#     # change_lane_1m
#     # change_lane_2m
#     # change_lane_3.5m
# )

# # cam_ids=(0 1 2 3 5 6 7 9 10 11 12)
# # downscales=(1 1 1 1 1 1 1 1 1 1 1)
# cam_ids=(0 1)
# downscales=(1 1)

# fps=10

# render_rgb=true
# render_depth=false
# save_images=true
# generate_lidar_pc=false
# ################################################################################
# source scripts/utils.sh

# export PYTHONPATH=$(pwd)

# echo "Using checkpoint: $ckpt_path"

# bool_args=""
# if [ "$render_rgb" = true ]; then
#     bool_args="$bool_args --render_rgb"
# fi
# if [ "$render_depth" = true ]; then
#     bool_args="$bool_args --render_depth"
# fi
# if [ "$save_images" = true ]; then
#     bool_args="$bool_args --save_images"
# fi
# if [ "$generate_lidar_pc" = true ]; then
#     bool_args="$bool_args --generate_lidar_pc"
# fi

# CUDA_VISIBLE_DEVICES=${gpu} python sim_render/cam/render_novel_trajectory.py \
#     --resume_from $ckpt_path \
#     --traj_types "${traj_types[@]}" \
#     --cam_ids "${cam_ids[@]}" \
#     --downscales "${downscales[@]}" \
#     --fps $fps \
#     $bool_args

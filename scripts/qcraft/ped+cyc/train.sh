# 参数设置,一次只能训练一个场景
################################################################################
scene_id="20250819_091233_Q3708_2010_2040_part02"
gpu=0
camera_ids=(0 1 2 3 5 6 7 9 10 11 12)
################################################################################
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

# 4: train
source /opt/conda/etc/profile.d/conda.sh && conda activate main

lidar_type="lidar"  # lidar（运动补偿前）/visual（纯视觉）

config_file="configs/omnire_extended_cam_lidar.yaml"
dataset_config="qcraft/11cams_${lidar_type}"
extra_config_info=""  # 额外信息，如 depth_loss

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
################################################################################

output_root="output"
project_name="qcraft_${scene_id}"

date_str=$(date +%Y%m%d)
run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info}"

run_dir="${output_root}/${project_name}/${run_name}"

# 检查输出目录是否存在，防止覆盖
if [ -d "$run_dir" ]; then
    echo "错误：目录 $run_dir 已存在！"
    exit 1
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
    data.scene_idx=$scene_id \
    data.pixel_source.cameras="[$(IFS=','; echo "${camera_ids[*]}")]" \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
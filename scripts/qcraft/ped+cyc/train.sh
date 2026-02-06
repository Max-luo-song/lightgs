# 参数设置,还是从data/qcraft_scenes.txt读取场景序列
# 在项目根目录输入 bash scripts/qcraft/ped+cyc/train.sh data/qcraft_scenes.txt
################################################################################
split_file=$1
gpu=0
camera_ids=(0 1 2 3 5 6 7 9 10 11 12)

log_file="$(dirname "$0")/train_results.txt"  # 与脚本同目录
echo "训练开始时间: $(date)" > "$log_file"  # 清空文件并写入开始时间
echo "场景ID, 状态, 时间" >> "$log_file"    # 添加标题行
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

output_root="output"
################################################################################
# 从 split_file 读取所有场景配置
# 跳过第一行（标题行），读取 scene_id, start_timestep, end_timestep
echo "正在读取场景配置文件: $split_file"
configs=()
while IFS=',' read -r scene_id start_timestep end_timestep; do
    # 跳过第一行（标题行）和空行
    if [[ "$scene_id" != "scene_id" ]] && [[ ! -z "$scene_id" ]]; then
        # 去除可能的空白字符
        scene_id=$(echo "$scene_id" | tr -d '[:space:]')
        start_timestep=$(echo "$start_timestep" | tr -d '[:space:]')
        end_timestep=$(echo "$end_timestep" | tr -d '[:space:]')
        configs+=("$scene_id:$start_timestep:$end_timestep")
    fi
done < "$split_file"

# 检查是否有读取到配置
if [ ${#configs[@]} -eq 0 ]; then
    echo "错误: 没有找到有效的场景配置!"
    exit 1
fi

echo "找到 ${#configs[@]} 个场景配置:"
for config in "${configs[@]}"; do
    echo "  $config"
done

# 遍历所有场景配置并进行训练
for config_str in "${configs[@]}"; do
    # 解析配置字符串
    IFS=':' read -r scene_id start_timestep end_timestep <<< "$config_str"

    echo "======================================================================"
    echo "开始训练场景: $scene_id"
    echo "时间范围: $start_timestep 到 $end_timestep"
    echo "======================================================================"

    # 设置项目名称和运行名称
    project_name="qcraft_${scene_id}"
    date_str=$(date +%Y%m%d)
    run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info}"
    run_dir="${output_root}/${project_name}/${run_name}"

    # 检查输出目录是否存在，防止覆盖
    if [ -d "$run_dir" ]; then
        echo "警告：目录 $run_dir 已存在，跳过此场景！"
        echo "（如需重新训练，请先删除或移动现有目录）"
        continue  # 跳过当前场景，继续下一个
    fi
    mkdir -p "$run_dir"

    # 备份当前脚本到输出目录
    script_name=$(basename "$0")
    backup_path="${run_dir}/${script_name}"
    cp "$0" "$backup_path"

    # 启动训练
    echo "使用 GPU: $gpu"
    echo "输出目录: $run_dir"

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

    # 检查训练是否成功完成
    if [ $? -eq 0 ]; then
        echo "场景 $scene_id 训练完成！"
        # 记录成功到日志文件
        echo "$scene_id, 成功, $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
    else
        echo "警告：场景 $scene_id 训练失败！"
        echo "继续下一个场景..."
        # 记录失败到日志文件
        echo "$scene_id, 失败, $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
    fi

    echo ""
done

echo "所有场景训练完成！"
echo "======================================================================"
echo "训练结果已保存到: $log_file"
echo "======================================================================"
echo "训练结果汇总:"
cat "$log_file"
echo "======================================================================"

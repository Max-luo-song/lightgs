#!/bin/bash

# 参数设置
################################################################################
# 从 qcraft_scenes.txt 文件读取 scene_id 列表
# 文件格式: scene_id,start_timestep,end_timestep (注释行以#开头)
scenes_file="data/qcraft_scenes.txt"

gpu=-1
segformer_path=/opt/mmsegmentation
camera_ids=(0 1 2 3 5 6 7 9 10 11 12)
raw_data_root="/nas_thoru/scenario_data/"
processed_data_root="/nas_thoru/users/guoluosong/road_version/data/qcraft/processed"
work_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 训练参数
lidar_type="lidar"  # lidar（运动补偿前）/visual（纯视觉）
config_file="configs/omnire_extended_cam_lidar.yaml"
dataset_config="qcraft/11cams_${lidar_type}"
extra_config_info="roadv1"  # 额外信息，如 depth_loss
# 默认值（如果文件中没有指定，则使用这些值）
default_start_timestep=0 # start frame index for training
default_end_timestep=-1 # end frame index, -1 for the last frame

# 批处理选项
stop_on_error=false  # 如果某个步骤失败，是否停止处理后续 scene (false=继续处理下一个scene)
################################################################################

# 日志函数
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

log_success() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS] $1"
}

# 从文件读取 scene 列表
# 格式: scene_id,start_timestep,end_timestep
# 返回: scene_ids 数组和对应的 start_timesteps、end_timesteps 数组
read_scenes_from_file() {
    local file=$1
    local -n scene_ids_ref=$2
    local -n start_timesteps_ref=$3
    local -n end_timesteps_ref=$4
    
    scene_ids_ref=()
    start_timesteps_ref=()
    end_timesteps_ref=()
    
    if [ ! -f "$file" ]; then
        log_error "文件不存在: $file"
        return 1
    fi
    
    log_info "从文件读取 scene 列表: $file"
    
    while IFS= read -r line || [ -n "$line" ]; do
        # 跳过空行和注释行
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [[ -z "$line" || "$line" =~ ^# ]]; then
            continue
        fi
        
        # 解析行: scene_id,start_timestep,end_timestep
        IFS=',' read -r scene_id start_timestep end_timestep <<< "$line"
        
        # 去除空格
        scene_id=$(echo "$scene_id" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        start_timestep=$(echo "$start_timestep" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        end_timestep=$(echo "$end_timestep" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        # 如果 start_timestep 或 end_timestep 为空，使用默认值
        if [ -z "$start_timestep" ]; then
            start_timestep=$default_start_timestep
        fi
        if [ -z "$end_timestep" ]; then
            end_timestep=$default_end_timestep
        fi
        
        if [ -n "$scene_id" ]; then
            scene_ids_ref+=("$scene_id")
            start_timesteps_ref+=("$start_timestep")
            end_timesteps_ref+=("$end_timestep")
            log_info "读取到 scene: $scene_id (start=$start_timestep, end=$end_timestep)"
        fi
    done < "$file"
    
    log_info "共读取到 ${#scene_ids_ref[@]} 个 scene"
    return 0
}

# 检查命令执行结果
check_result() {
    local step=$1
    local scene_id=$2
    local exit_code=$3
    
    if [ $exit_code -ne 0 ]; then
        log_error "Scene ${scene_id} - ${step} 失败 (退出码: ${exit_code})"
        if [ "$stop_on_error" = true ]; then
            log_error "由于 stop_on_error=true，停止处理后续 scene"
            exit $exit_code
        else
            log_info "由于 stop_on_error=false，跳过当前 scene 的后续步骤，继续处理下一个 scene"
            return 1
        fi
    else
        log_success "Scene ${scene_id} - ${step} 完成"
        return 0
    fi
}

# Step 1: Preprocess
step1_preprocess() {
    local scene_id=$1
    log_info "开始处理 Scene ${scene_id} - Step 1: Preprocess"
    
    source /opt/conda/etc/profile.d/conda.sh && conda activate main
    export PYTHONPATH=$(pwd)
    python datasets/preprocess.py \
        --data_root ${raw_data_root} \
        --target_dir ${processed_data_root} \
        --dataset qcraft \
        --split training \
        --scene_ids $scene_id \
        --workers 2 \
        --process_keys ego_masks images calib pose objects dynamic_masks lidar \
        --skip_front_wide_side_cameras
    
    check_result "Step 1: Preprocess" $scene_id $?
}

# Step 2: Extract mask
step2_extract_mask() {
    local scene_id=$1
    log_info "开始处理 Scene ${scene_id} - Step 2: Extract mask"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source ${SCRIPT_DIR}/utils.sh
    source /opt/conda/etc/profile.d/conda.sh && conda activate mmseg2
    
    CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks_mmseg2.py \
        --data_root ${processed_data_root}/training \
        --config=${segformer_path}/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py \
        --checkpoint=${segformer_path}/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth \
        --scene_ids $scene_id \
        --process_dynamic_mask
    
    check_result "Step 2: Extract mask" $scene_id $?
}

# Step 3: Process human pose
step3_human_pose() {
    local scene_id=$1
    log_info "开始处理 Scene ${scene_id} - Step 3: Process human pose"
    
    source /opt/conda/etc/profile.d/conda.sh && conda activate 4d-humans
    export PYTHONPATH=$PYTHONPATH:${work_root}/third_party/Humans4D
    CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
        --data_root ${processed_data_root}/training \
        --dataset qcraft \
        --scene_id $scene_id
    
    check_result "Step 3: Process human pose" $scene_id $?
}

# Step 4: Train
step4_train() {
    local scene_id=$1
    local start_timestep=$2
    local end_timestep=$3
    log_info "开始处理 Scene ${scene_id} - Step 4: Train (start=$start_timestep, end=$end_timestep)"
    
    source /opt/conda/etc/profile.d/conda.sh && conda activate main
    
    output_root="output"
    project_name="qcraft_${scene_id}"
    
    date_str=$(date +%Y%m%d%H%M)
    run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info}"
    
    run_dir="${output_root}/${project_name}/${run_name}"
    
    # 检查输出目录是否存在，防止覆盖
    if [ -d "$run_dir" ]; then
        log_error "目录 $run_dir 已存在！跳过训练"
        return 1
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
    
    check_result "Step 4: Train" $scene_id $?
}

# 处理单个 scene 的所有步骤
process_scene() {
    local scene_id=$1
    local start_timestep=$2
    local end_timestep=$3
    local scene_index=$4
    local total_scenes=$5
    
    log_info "=========================================="
    log_info "处理 Scene ${scene_index}/${total_scenes}: ${scene_id}"
    log_info "=========================================="
    
    # Step 1: Preprocess
    ####step1_preprocess $scene_id || return 1
    
    # Step 2: Extract mask
    #step2_extract_mask $scene_id || return 1
    
    # Step 3: Process human pose
    step3_human_pose $scene_id || return 1
    
    # Step 4: Train
    step4_train $scene_id $start_timestep $end_timestep || return 1
    
    log_success "Scene ${scene_id} 所有步骤完成！"
    return 0
}

# 主函数
main() {
    # 从文件读取 scene 列表
    local scene_ids=()
    local start_timesteps=()
    local end_timesteps=()
    
    if ! read_scenes_from_file "$scenes_file" scene_ids start_timesteps end_timesteps; then
        log_error "读取 scene 列表失败"
        exit 1
    fi
    
    if [ ${#scene_ids[@]} -eq 0 ]; then
        log_error "未找到任何 scene"
        exit 1
    fi
    
    log_info "开始批处理，共 ${#scene_ids[@]} 个 scene"
    
    # Pick an available gpu (只需要选择一次)
    source scripts/utils.sh
    if [ "${gpu}" = "-1" ]; then
        gpu=$(pick_gpu)
        if [ -z "${gpu}" ]; then
            log_error "未找到可用 GPU"
            exit 1
        fi
    fi
    log_info "使用 GPU: ${gpu}"
    
    # 统计信息
    local total_scenes=${#scene_ids[@]}
    local success_count=0
    local fail_count=0
    local failed_scenes=()
    
    # 遍历所有 scene_id
    for i in "${!scene_ids[@]}"; do
        local scene_id="${scene_ids[$i]}"
        local start_timestep="${start_timesteps[$i]}"
        local end_timestep="${end_timesteps[$i]}"
        local scene_index=$((i + 1))
        
        if process_scene "$scene_id" "$start_timestep" "$end_timestep" $scene_index $total_scenes; then
            ((success_count++))
        else
            ((fail_count++))
            failed_scenes+=("$scene_id")
        fi
        
        log_info "当前进度: 成功 ${success_count}, 失败 ${fail_count}, 总计 ${total_scenes}"
    done
    
    # 输出最终统计
    log_info "=========================================="
    log_info "批处理完成！"
    log_info "成功: ${success_count}/${total_scenes}"
    log_info "失败: ${fail_count}/${total_scenes}"
    
    if [ ${#failed_scenes[@]} -gt 0 ]; then
        log_error "失败的 scene:"
        for failed_scene in "${failed_scenes[@]}"; do
            log_error "  - ${failed_scene}"
        done
    fi
    log_info "=========================================="
    
    # 如果有失败的 scene，返回非零退出码
    if [ $fail_count -gt 0 ]; then
        exit 1
    fi
}

# 执行主函数
main "$@"

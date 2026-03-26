# 参数设置
################################################################################
gpu=-1

ckpt_path="output/qcraft_20250819_004633_Q3707_990_1010_part01/20260325_lidar+cam0_1_2_3_5_6_7_9_10_11_12lightgs/checkpoint_final.pth"
################################################################################

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


echo "Using checkpoint: $ckpt_path"

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python tools/eval.py \
    --resume_from $ckpt_path \
    # --enable_viewer
#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

#
# 保存済みチェックポイントを val（穴なし GT）で評価し直し、FID 最小の best epoch を求める。
#
# 使い方:
#   bash run_compute_best_epoch_val.sh [DATE] [EPOCH_START] [EPOCH_END] [EPOCH_STEP]
#   例: bash run_compute_best_epoch_val.sh 20260527_125919 10 800 10
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

DATE="${1:-20260527_125919}"
EPOCH_START="${2:-10}"
EPOCH_END="${3:-800}"
EPOCH_STEP="${4:-10}"

CHECKPOINTS_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/${DATE}"
NAME="moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze"

DATAROOT="${WORKSPACE_DIR}/data/org_data/moving_mnist"
DATAROOT_B="${WORKSPACE_DIR}/data/preprocessed/bspline_transformed"

GPU_IDS=0
MODEL="wpsb"
MODE="sb"
INPUT_NC=1
OUTPUT_NC=1
NGF=64
NDF=64
DATASET_MODE="moving_mnist_paired"
LOAD_SIZE=64
CROP_SIZE=64
NUM_TIMESTEPS=5

SEQ_OT_P_ENTROPY=${SEQ_OT_P_ENTROPY:-0}
SEQ_OT_P_ENTROPY_PENALTY=${SEQ_OT_P_ENTROPY_PENALTY:-0.0}
SEQ_OT_DIVERGENCE=${SEQ_OT_DIVERGENCE:-1}
SEQ_OT_DIVERGENCE_PENALTY=${SEQ_OT_DIVERGENCE_PENALTY:--0.5}

echo "======================================"
echo "Best-epoch search on val with NON-HOLED GT"
echo "  Domain A: ${DATAROOT}/mnist_test_seq.npy"
echo "  Domain B (val GT): ${DATAROOT_B}/transformed_global.npy  (穴なし)"
echo "  Checkpoints: ${CHECKPOINTS_DIR}/${NAME}"
echo "  Epochs: ${EPOCH_START}-${EPOCH_END} step ${EPOCH_STEP}"
echo "======================================"

cd "${SCRIPT_DIR}"

python3 compute_best_epoch_val.py \
    --epoch_start ${EPOCH_START} \
    --epoch_end ${EPOCH_END} \
    --epoch_step ${EPOCH_STEP} \
    --dataroot "${DATAROOT}" \
    --dataroot_B "${DATAROOT_B}" \
    --data_file_A mnist_test_seq.npy \
    --data_file_B transformed_global.npy \
    --name "${NAME}" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --dataset_mode "${DATASET_MODE}" \
    --input_nc ${INPUT_NC} \
    --output_nc ${OUTPUT_NC} \
    --ngf ${NGF} \
    --ndf ${NDF} \
    --load_size ${LOAD_SIZE} \
    --crop_size ${CROP_SIZE} \
    --num_timesteps ${NUM_TIMESTEPS} \
    --sb_mode seq_ot \
    --seq_ot_normalize "mean" \
    --seq_ot_p_entropy ${SEQ_OT_P_ENTROPY} \
    --seq_ot_p_entropy_penalty ${SEQ_OT_P_ENTROPY_PENALTY} \
    --seq_ot_divergence ${SEQ_OT_DIVERGENCE} \
    --seq_ot_divergence_penalty ${SEQ_OT_DIVERGENCE_PENALTY} \
    --seq_ot_monotone_penalty 1.0 \
    --lmda 0.03 \
    --sinkhorn_type "sinkhorn_log" \
    --gpu_ids ${GPU_IDS} \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --epoch ${EPOCH_END} \
    --preprocess none \
    --no_flip \
    --eval \
    --serial_batches

#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

#
# best epoch モデルで train データを生成し、FID と同じ特徴空間
# (InceptionV3 pool3, 2048次元) で fake_B を 2D 可視化する。
# ペアあり=青 / ペアなし=赤 で色分け。PCA と t-SNE の両方を出力。
#
# 使用方法:
#   bash run_visualize_fid_feature_space_best.sh <DATE> [EPOCH]
# 例:
#   bash run_visualize_fid_feature_space_best.sh 20260527_125919 best

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

DATE="${1:-20260604_121255}"
EPOCH="${2:-best}"

CHECKPOINTS_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/${DATE}"
if [ ! -d "${CHECKPOINTS_DIR}" ]; then
  echo "ERROR: チェックポイントディレクトリが見つかりません: ${CHECKPOINTS_DIR}"
  exit 1
fi

# 実験名・データ設定（run_test_moving_mnist_seg_paired_min_ot_div_confirm.sh と同じ）
NAME="moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze_geo"
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

SEQ_OT_P_ENTROPY=${SEQ_OT_P_ENTROPY:-0}
SEQ_OT_P_ENTROPY_PENALTY=${SEQ_OT_P_ENTROPY_PENALTY:-0.0}
SEQ_OT_DIVERGENCE=${SEQ_OT_DIVERGENCE:-1}
SEQ_OT_DIVERGENCE_PENALTY=${SEQ_OT_DIVERGENCE_PENALTY:--0.5}
NUM_TIMESTEPS=5

# train 全体を使いたい場合は大きめに（train は 1000 シーケンス）
NUM_TEST=${NUM_TEST:-200}
PHASE="train"

RESULTS_DIR="${RESULTS_DIR:-${CHECKPOINTS_DIR}/test_results}"

echo "======================================"
echo "FID feature-space visualization (train, fake_B)"
echo "Experiment: ${NAME}  Epoch: ${EPOCH}  Date: ${DATE}"
echo "Num seq: ${NUM_TEST}  Results: ${RESULTS_DIR}"
echo "======================================"

cd "${SCRIPT_DIR}"

python3 visualize_fid_feature_space_best.py \
    --dataroot "${DATAROOT}" \
    --dataroot_B "${DATAROOT_B}" \
    --data_file_A mnist_test_seq.npy \
    --data_file_B transformed_cloze_global.npy \
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
    --results_dir "${RESULTS_DIR}" \
    --phase "${PHASE}" \
    --epoch "${EPOCH}" \
    --num_test ${NUM_TEST} \
    --preprocess none \
    --no_flip \
    --eval \
    --serial_batches

echo ""
echo "======================================"
echo "Done. Outputs in: ${RESULTS_DIR}/${NAME}/fid_featspace_${PHASE}_${EPOCH}"
echo "======================================"

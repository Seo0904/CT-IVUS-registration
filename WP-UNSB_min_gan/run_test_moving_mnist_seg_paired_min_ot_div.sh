#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

#
# Moving MNIST Paired Dataset用 UNSB テストスクリプト
# (run_train_moving_mnist_seg_paired_min_ot_div.sh で訓練したモデル用)
#
# 使用方法:
#   bash run_test_moving_mnist_seg_paired_min_ot_div.sh <DATE> [EPOCH]
#
# 例:
#   bash run_test_moving_mnist_seg_paired_min_ot_div.sh 20260521_131937
#   bash run_test_moving_mnist_seg_paired_min_ot_div.sh 20260521_131937 latest
#   bash run_test_moving_mnist_seg_paired_min_ot_div.sh 20260521_131937 400

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# 引数チェック
# if [ -z "${1}" ]; then
#   echo "ERROR: DATEを指定してください"
#   echo "使用方法: bash $0 <DATE> [EPOCH]"
#   echo "例: bash $0 20260521_131937"
#   echo ""
#   echo "利用可能なチェックポイント:"
#   ls "${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/" 2>/dev/null || echo "  (なし)"
#   exit 1
# fi

DATE="${1:-20260619_150314}"
EPOCH="${2:-best}"

# チェックポイントディレクトリ（訓練時と同じパス）
CHECKPOINTS_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_min_gan/moving-mnist/${DATE}"

if [ ! -d "${CHECKPOINTS_DIR}" ]; then
  echo "ERROR: チェックポイントディレクトリが見つかりません: ${CHECKPOINTS_DIR}"
  exit 1
fi

# 実験名（訓練時と同じ）
NAME="moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze_geo_debias_nmono"

# データセットパス（訓練時と同じ）
DATAROOT="${WORKSPACE_DIR}/data/org_data/moving_mnist"
DATAROOT_B="${WORKSPACE_DIR}/data/preprocessed/bspline_transformed"

# GPU設定
GPU_IDS=0

# モデル設定（訓練時と同じ）
MODEL="wpsb"
MODE="sb"

# ネットワーク構造（訓練時と同じ）
INPUT_NC=1
OUTPUT_NC=1
NGF=64
NDF=64

# データセット設定（訓練時と同じ）
DATASET_MODE="moving_mnist_paired"
LOAD_SIZE=64
CROP_SIZE=64

# sequence OT（訓練時と同じ）
SEQ_OT_P_ENTROPY=${SEQ_OT_P_ENTROPY:-0}
SEQ_OT_P_ENTROPY_PENALTY=${SEQ_OT_P_ENTROPY_PENALTY:-0.0}
SEQ_OT_DIVERGENCE=${SEQ_OT_DIVERGENCE:-1}
SEQ_OT_DIVERGENCE_PENALTY=${SEQ_OT_DIVERGENCE_PENALTY:--0.5}

# タイムステップ（訓練時と同じ）

SEQ_OT_UNBALANCED=${SEQ_OT_UNBALANCED:-60}
LAMBDA_GAN=1.0  

NUM_TIMESTEPS=5

# テスト設定
NUM_TEST=200
PHASE="test"

# 結果保存先
RESULTS_DIR="${CHECKPOINTS_DIR}/test_results"

echo "======================================"
echo "Moving MNIST Paired WP-UNSB Testing"
echo "(seq_ot + divergence, ver2)"
echo "======================================"
echo "Domain A: ${DATAROOT}/mnist_test_seq.npy"
echo "Domain B: ${DATAROOT_B}/transformed_global.npy"
echo "Experiment: ${NAME}"
echo "Epoch: ${EPOCH}"
echo "Date: ${DATE}"
echo "Checkpoint: ${CHECKPOINTS_DIR}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "Num test: ${NUM_TEST}"
echo "GPU: ${GPU_IDS}"
echo "======================================"

cd "${SCRIPT_DIR}"

python3 test.py \
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
    --seq_ot_unbalanced ${SEQ_OT_UNBALANCED} \
    --lambda_GAN ${LAMBDA_GAN} \
    --seq_ot_normalize "mean" \
    --seq_ot_p_entropy ${SEQ_OT_P_ENTROPY} \
    --seq_ot_p_entropy_penalty ${SEQ_OT_P_ENTROPY_PENALTY} \
    --seq_ot_divergence ${SEQ_OT_DIVERGENCE} \
    --seq_ot_divergence_penalty ${SEQ_OT_DIVERGENCE_PENALTY} \
    --seq_ot_monotone_penalty 1.0 \
    --lmda 0.03 \
    --sinkhorn_type "sinkhorn_log" \
    --seq_ot_solver "geo"\
    --seq_ot_geo_scaling 0.99\
    --seq_ot_geo_p 2\
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
echo "Test completed!"
echo "Results saved to: ${RESULTS_DIR}/${NAME}/${PHASE}_${EPOCH}"
echo "======================================"

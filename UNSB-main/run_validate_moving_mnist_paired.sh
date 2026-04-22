#!/bin/bash
#
# Moving MNIST Paired Dataset用 UNSB テストスクリプト
# 
# 使用方法:
#   bash run_test_moving_mnist_paired.sh [CHECKPOINT_EPOCH]
#
# 例:
#   bash run_test_moving_mnist_paired.sh latest
#   bash run_test_moving_mnist_paired.sh 100

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# チェックポイントエポック（引数から取得、デフォルトはlatest）
EPOCH="${1:-latest}"

# 日付の取得
DATE=$(date +"%Y%m%d_%H%M%S")

# データセットパス（絶対パス）
DATAROOT="${WORKSPACE_DIR}/data/org_data/moving_mnist"
DATAROOT_B="${WORKSPACE_DIR}/data/preprocessed/bspline_transformed"

# 実験名（訓練時と同じ名前を使用）
NAME="moving_mnist_paired_sb"

# GPU設定
GPU_IDS=0

# モデル設定
MODEL="sb"
MODE="sb"

# ネットワーク構造（訓練時と同じ・グレースケール用）
INPUT_NC=1
OUTPUT_NC=1
NGF=64
NDF=64

# データセット設定
DATASET_MODE="moving_mnist_paired"
LOAD_SIZE=64
CROP_SIZE=64

# タイムステップ（訓練時と同じ）
NUM_TIMESTEPS=5

# テスト設定
NUM_TEST=100          # テストするサンプル数
PHASE="val"

# チェックポイントディレクトリ（最新の実験）
CHECKPOINTS_DIR="/workspace/data/experiment_result/UNSB/moving-mnist/20260214_184345"

# 結果保存先（チェックポイントと同じディレクトリ内）
RESULTS_DIR="${CHECKPOINTS_DIR}/val_results"

echo "======================================"
echo "Moving MNIST Paired UNSB Testing"
echo "======================================"
echo "Domain A: ${DATAROOT}/mnist_test_seq.npy"
echo "Domain B: ${DATAROOT_B}/transformed_aligned.npy"
echo "Experiment: ${NAME}"
echo "Epoch: ${EPOCH}"
echo "Date: ${DATE}"
echo "Checkpoint: ${CHECKPOINTS_DIR}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "GPU: ${GPU_IDS}"
echo "======================================"

# テスト実行
cd "${SCRIPT_DIR}"

python3 test.py \
    --dataroot "${DATAROOT}" \
    --dataroot_B "${DATAROOT_B}" \
    --data_file_A mnist_test_seq.npy \
    --data_file_B transformed_aligned.npy \
    --name "${NAME}" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --dataset_mode "${DATASET_MODE}" \
    --input_nc ${INPUT_NC} \
    --output_nc ${OUTPUT_NC} \
    --ngf ${NGF} \
    --ndf ${NDF} \
    --num_timesteps ${NUM_TIMESTEPS} \
    --load_size ${LOAD_SIZE} \
    --crop_size ${CROP_SIZE} \
    --gpu_ids ${GPU_IDS} \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --phase "${PHASE}" \
    --epoch "${EPOCH}" \
    --num_test ${NUM_TEST} \
    --eval \
    --no_flip \
    --serial_batches

echo ""
echo "======================================"
echo "validation completed!"
echo "Results saved to: ${RESULTS_DIR}/${NAME}/${PHASE}_${EPOCH}"
echo "======================================"

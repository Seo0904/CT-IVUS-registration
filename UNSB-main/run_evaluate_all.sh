#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
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
EPOCH="${1:-best}"

# 日付の取得
DATE=$(date +"%Y%m%d_%H%M%S")
DATE="20260611_003552"  # 訓練時と同じ日付を使用

# データセットパス（絶対パス）
DATAROOT="${WORKSPACE_DIR}/data/org_data/moving_mnist"
DATAROOT_B="${WORKSPACE_DIR}/data/preprocessed/bspline_transformed"

# 実験名（訓練時と同じ名前を使用）
NAME="moving_mnist_unpaired_sb"

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
NUM_TEST=20000         # 1 item = 1フレームなので 1000系列 x 20フレーム = 20000
PHASE="test"

# ---- 追加評価オプション（env で上書き可）----
FVD_BACKEND="${FVD_BACKEND:-i3d}"                                         # i3d=論文比較可能 / r3d18=相対比較(torchvision)
FVD_I3D_PATH="${FVD_I3D_PATH:-/workspace/.cache/fvd/i3d_torchscript.pt}"  # i3d 重み(無ければ自動で r3d18 にフォールバック)
EVAL_FVD_VIZ="${EVAL_FVD_VIZ:-1}"                                         # FVD 特徴空間の可視化+最近傍対応(1=出す)
EVAL_PAIRED="${EVAL_PAIRED:-0}"                                           # ペア指標 SSIM/PSNR/L1/L2/LPIPS を計算するか(0でスキップ)
EVAL_TXT_NAME="${EVAL_TXT_NAME:-evaluation_metrics_fvd.txt}"                  # 保存する可読テキスト名

# チェックポイントディレクトリ（最新の実験）
CHECKPOINTS_DIR="/workspace/data/experiment_result/UNSB/moving-mnist/20260611_003552"

# 結果保存先（チェックポイントと同じディレクトリ内）
RESULTS_DIR="${CHECKPOINTS_DIR}/test_results"

echo "======================================"
echo "Moving MNIST Paired UNSB Testing"
echo "======================================"
echo "Domain A: ${DATAROOT}/mnist_test_seq.npy"
echo "Domain B: ${DATAROOT_B}/transformed_global.npy"
echo "Experiment: ${NAME}"
echo "Epoch: ${EPOCH}"
echo "Date: ${DATE}"
echo "Checkpoint: ${CHECKPOINTS_DIR}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "GPU: ${GPU_IDS}"
echo "======================================"

# テスト実行
cd "${SCRIPT_DIR}"

python3 evaluate_all.py \
    --eval_kid_subset 900 \
    --eval_fvd_backend "${FVD_BACKEND}" \
    --eval_fvd_i3d "${FVD_I3D_PATH}" \
    --eval_fvd_viz ${EVAL_FVD_VIZ} \
    --eval_paired ${EVAL_PAIRED} \
    --eval_txt_name "${EVAL_TXT_NAME}" \
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
echo "Test completed!"
echo "Results saved to: ${RESULTS_DIR}/${NAME}/${PHASE}_${EPOCH}"
echo "======================================"

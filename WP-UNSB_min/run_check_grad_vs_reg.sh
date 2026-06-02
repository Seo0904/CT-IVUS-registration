#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

# reg を変えながら GRAD_LOG_EPOCH epoch まで回し、そのエポックだけ勾配統計を記録するスクリプト。
# 結果は GRAD_LOG_FILE にまとめて書き出す。

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
DATE=$(date +"%Y%m%d_%H%M%S")

DATAROOT="${WORKSPACE_DIR}/data/org_data/moving_mnist"
DATAROOT_B="${WORKSPACE_DIR}/data/preprocessed/bspline_transformed"
RESULT_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_min/grad_vs_reg/${DATE}"
GRAD_LOG_FILE="${RESULT_DIR}/grad_vs_reg.txt"

# 何 epoch 目の勾配を見るか（-1 = 全 epoch、100 = epoch 100 のみ）
GRAD_LOG_EPOCH=${GRAD_LOG_EPOCH:-100}

mkdir -p "${RESULT_DIR}"

# 確認したい reg (= --lmda) の値リスト
REG_VALUES=(0.1 0.05 0.03 0.01)

GPU_IDS=0

echo "======================================"
echo "Gradient vs reg sweep"
echo "Results: ${RESULT_DIR}"
echo "Log epoch: ${GRAD_LOG_EPOCH}"
echo "Log: ${GRAD_LOG_FILE}"
echo "======================================"

for REG in "${REG_VALUES[@]}"; do
    echo ""
    echo "------ reg=${REG} ------"

    # reg ごとにヘッダ行を挿入
    echo "# === reg=${REG} ===" >> "${GRAD_LOG_FILE}"

    python3 train.py \
      --dataroot        "${DATAROOT}" \
      --dataroot_B      "${DATAROOT_B}" \
      --data_file_A     mnist_test_seq.npy \
      --data_file_B     transformed_global.npy \
      --name            "grad_check_reg${REG}" \
      --model           wpsb \
      --mode            sb \
      --dataset_mode    moving_mnist_paired \
      --input_nc        1 \
      --output_nc       1 \
      --ngf             64 \
      --ndf             64 \
      --load_size       64 \
      --crop_size       64 \
      --batch_size      1 \
      --num_timesteps   5 \
      --lambda_GAN      0 \
      --lambda_SB       1.0 \
      --lambda_NCE      0 \
      --lmda            "${REG}" \
      --sb_mode         seq_ot \
      --seq_ot_normalize mean \
      --seq_ot_p_entropy 0 \
      --seq_ot_p_entropy_penalty 0.0 \
      --seq_ot_divergence 1 \
      --seq_ot_divergence_penalty -0.5 \
      --seq_ot_monotone_penalty 1.0 \
      --n_epochs        "${GRAD_LOG_EPOCH}" \
      --n_epochs_decay  0 \
      --lr              0.00001 \
      --print_freq      999999 \
      --save_epoch_freq 999999 \
      --save_latest_freq 999999 \
      --val_epoch_freq  999999 \
      --checkpoints_dir "${RESULT_DIR}" \
      --gpu_ids         "${GPU_IDS}" \
      --preprocess      none \
      --no_flip \
      --no_html \
      --grad_log_file   "${GRAD_LOG_FILE}"

    echo "  -> done reg=${REG}"
done

echo ""
echo "======================================"
echo "Sweep complete. Log saved to:"
echo "  ${GRAD_LOG_FILE}"
echo "======================================"

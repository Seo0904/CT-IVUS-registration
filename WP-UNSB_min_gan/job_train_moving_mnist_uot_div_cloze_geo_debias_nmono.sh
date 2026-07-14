#!/bin/bash
#============================================================
# Genkai(玄界) 用 pjsub ジョブ:
#   Moving MNIST UNSB 学習 (uot + div + cloze + geo + debias + nmono)
#   投入:  pjsub job_train_moving_mnist_uot_div_cloze_geo_debias_nmono.sh
#   ログ:  mm_nmono.<JOBID>.out
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -N mm_nmono

set -e

# --- conda 環境 ---
source ~/.bashrc 2>/dev/null || true
conda activate ctmri

# --- このジョブスクリプトのある場所 (= WP-UNSB_min_gan) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# --- バッチサイズについて ---
#   run_train_..._nmono.sh は BATCH_SIZE=4 がハードコードのため env では変えられない。
#   増やすなら run スクリプト側の BATCH_SIZE=4 を直接編集すること
#   (系列長20でメモリ消費が大きいので H100 94GB でも様子見ながら)。

# --- wandb (計算ノードからオンライン送信可を確認済み) ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "JOB on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 学習スクリプト本体を実行 (DATAROOT 等は本体が WORKSPACE_DIR から自動算出) ---
bash run_train_moving_mnist_uot_div_cloze_geo_debias_nmono.sh

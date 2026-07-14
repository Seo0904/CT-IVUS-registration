#!/bin/bash
#============================================================
# Genkai(玄界) 用 pjsub ジョブ:
#   1枚のGPUで Moving MNIST 学習を2つ同時に回す
#     run1: debias (monotoneあり)
#     run2: nmono  (monotoneなし)
#   batch=4 と小さく H100 94GB に余裕があるため同居可能。
#   GPUノード1枚で2実験 = 予算/ノード数を節約。
#
#   投入:  pjsub job_train_moving_mnist_2on1gpu.sh
#   ログ:  mm_2on1.<JOBID>.out (両プロセスの出力が混ざる)
#          個別ログは run1.log / run2.log にも保存
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -N mm_2on1

set -e

# --- conda 環境 ---
source ~/.bashrc 2>/dev/null || true
conda activate ctmri

# --- このジョブスクリプトのある場所 (= WP-UNSB_min_gan) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# --- 両プロセスとも同じ物理GPU(0)を使う ---
export CUDA_VISIBLE_DEVICES=0
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "JOB on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# --- run1: debias 版をバックグラウンド起動 ---
echo "[launch] run1 = debias"
bash run_train_moving_mnist_uot_div_cloze_geo_debias.sh > run1_debias.log 2>&1 &
PID1=$!

# わずかにずらして起動 (RESULT_DIR の DATE 衝突回避 & 初期化の同時集中を避ける) ---
sleep 5

# --- run2: nmono 版をバックグラウンド起動 ---
echo "[launch] run2 = nmono"
bash run_train_moving_mnist_uot_div_cloze_geo_debias_nmono.sh > run2_nmono.log 2>&1 &
PID2=$!

echo "run1 PID=$PID1 (log: run1_debias.log)"
echo "run2 PID=$PID2 (log: run2_nmono.log)"

# --- どちらか落ちても両方の終了を待つ ---
set +e
wait $PID1; RC1=$?
wait $PID2; RC2=$?

echo "run1 exit=$RC1 / run2 exit=$RC2"
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]

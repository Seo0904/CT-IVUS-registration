#!/bin/bash
#============================================================
# Genkai 用 pjsub ジョブ: synthRAD CT->MR WP-UNSB(seq-OT)
#   コスト空間 = flatten (Discriminator 特徴空間で L_ot を計算)
#   投入:  pjsub job_train_synthRAD_ct2mr_flatten.sh
#   ログ:  train_ct2mr_flatten.<JOBID>.out
#
# 注意 (elapse):
#   flatten ≈ 3.3s/iter (A6000実測, pixel比+25% を chunk16 で相殺後の目安)。
#   400 epoch × 126 seq × 3.3s ≈ 46h かかる。下の elapse=24:00:00 では
#   ~epoch 200 程度までしか進まない。完走させたい場合は
#     (a) rscgrp の上限まで elapse を伸ばす、もしくは
#     (b) 24h で打ち切り後、latest_net_* から continue_train で再投入する
#   のどちらかにすること。玄海の GPU 速度次第で iter 時間は変わる。
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -N train_ct2mr_flatten

set -e

# --- conda 環境 ---
source ~/.bashrc 2>/dev/null || true
conda activate ctmri

# --- このジョブスクリプトのある場所 (= WP-UNSB_ct_mri) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# --- データ位置 (リポジトリ直下の data/ から自動算出。WORKSPACE_DIR = WP-UNSB_ct_mri の親) ---
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
export DATAROOT="${DATAROOT:-${WORKSPACE_DIR}/data/preprocessed/synthRAD_maskfree_full}"
export REGION="${REGION:-brain}"

# --- D 特徴空間 OT (flatten) ---
export SEQ_OT_COST_SPACE="${SEQ_OT_COST_SPACE:-flatten}"
# feature 空間はコスト M~30 (pixel M~1.3e4 の ~1/430) なので reg/iters を再調整。
# 勾配ベースで選定 (実データ, fake_B への grad ノルムが pixel(reg=10,grad~70)と同オーダーに
# 収まる最小 reg=0.05 → grad~2x pixel)。その reg で P収束&勾配安定する最小 iters=30
# (mass=1.005, grad比 it300=1.00)。pixel の 10/300 から大幅減。
export LMDA="${LMDA:-0.05}"
export SEQ_OT_ITERS="${SEQ_OT_ITERS:-30}"
# flatten 使用時は update_D=auto で Discriminator も学習される (real/fake 分類)。
export UPDATE_D="${UPDATE_D:-auto}"
# adv は使わず純 SB + D特徴OT で見る (adv を足すなら LAMBDA_GAN=1 等で上書き)。
export LAMBDA_GAN="${LAMBDA_GAN:-0}"
# netG forward chunk: checkpoint 下では小さいほど速い (16 推奨)。
export NETG_CHUNK="${NETG_CHUNK:-16}"

# --- 実験名 (flatten と分かるように) ---
export NAME="${NAME:-synthRAD_${REGION}_ct2mr_sb_otdiv_geo_dfeat_flatten}"

# --- wandb (計算ノードからオンライン送信可なら 1) ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-dfeat,flatten}"

echo "JOB on $(hostname) at $(date)"
echo "SEQ_OT_COST_SPACE=${SEQ_OT_COST_SPACE}  UPDATE_D=${UPDATE_D}  LAMBDA_GAN=${LAMBDA_GAN}  NETG_CHUNK=${NETG_CHUNK}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 学習スクリプト本体を実行 ---
bash run_train_synthRAD_ot_div_geo_gan.sh

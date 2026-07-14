#!/bin/bash
#============================================================
# Genkai 用 pjsub ジョブ: synthRAD CT->MR WP-UNSB(seq-OT)
#   コスト空間 = random_patch (D特徴空間。毎step 1空間位置をランダム選択して L_ot を計算)
#   投入:  pjsub job_train_synthRAD_ct2mr_random_patch.sh
#   ログ:  train_ct2mr_rndpatch.<JOBID>.out
#
# 注意 (elapse):
#   random_patch ≈ flatten と同程度の速度 (追加コストの主因は D 学習で両モード共通、
#   記述子の cdist 差は netG に対して誤差)。400 epoch × 126 seq でおよそ数十時間。
#   完走させたい場合は elapse を rscgrp 上限まで伸ばすか、latest_net_* から
#   continue_train で再投入すること。玄海の GPU 速度次第で iter 時間は変わる。
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -N train_ct2mr_rndpatch

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

# --- D 特徴空間 OT (random_patch) ---
export SEQ_OT_COST_SPACE="${SEQ_OT_COST_SPACE:-random_patch}"
# 1空間位置の特徴なのでコスト M~0.15 と極小。勾配ベースで選定 (実データ, fake_B への
# grad が pixel(reg=10,grad~70)と同オーダーに収まる最小 reg=0.005 → grad~0.6x pixel)。
# その reg で P収束&勾配安定する最小 iters=50 (mass=1.0005, grad比 it300=1.01)。
export LMDA="${LMDA:-0.005}"
export SEQ_OT_ITERS="${SEQ_OT_ITERS:-50}"
# D特徴OT 使用時は update_D=auto で Discriminator も学習される (real/fake 分類)。
export UPDATE_D="${UPDATE_D:-auto}"
# adv は使わず純 SB + D特徴OT で見る (adv を足すなら LAMBDA_GAN=1 等で上書き)。
export LAMBDA_GAN="${LAMBDA_GAN:-0}"
# netG forward chunk: checkpoint 下では小さいほど速い (16 推奨)。
export NETG_CHUNK="${NETG_CHUNK:-16}"

# --- 実験名 (random_patch と分かるように) ---
export NAME="${NAME:-synthRAD_${REGION}_ct2mr_sb_otdiv_geo_dfeat_rndpatch}"

# --- wandb (計算ノードからオンライン送信可なら 1) ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-dfeat,random_patch}"

echo "JOB on $(hostname) at $(date)"
echo "SEQ_OT_COST_SPACE=${SEQ_OT_COST_SPACE}  UPDATE_D=${UPDATE_D}  LAMBDA_GAN=${LAMBDA_GAN}  NETG_CHUNK=${NETG_CHUNK}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 学習スクリプト本体を実行 ---
bash run_train_synthRAD_ot_div_geo_gan.sh

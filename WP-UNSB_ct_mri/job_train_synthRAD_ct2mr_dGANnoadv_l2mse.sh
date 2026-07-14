#!/bin/bash
#============================================================
# Genkai 用 pjsub ジョブ: synthRAD CT->MR WP-UNSB(seq-OT)
#   D = real/fake 分類器 (conditional patchGAN) だが adv 勾配は G に流さない
#   G = seq-OT(D特徴, flatten) + MSE ドリフト項  ※G_GAN 無し
#   投入:  pjsub job_train_synthRAD_ct2mr_dGANnoadv_l2mse.sh
#   ログ:  train_ct2mr_dGANnoadv_l2mse.<JOBID>.out
#
# ★ dGAN 版 (job_train_synthRAD_ct2mr_dGAN_l2mse.sh) の LAMBDA_GAN=0 版。
#   両者の差は LAMBDA_GAN のみ。実験の狙いは:
#   「D を real/fake 分類で学習した"特徴空間"だけを seq-OT のコストに使うと、
#     dOTmax(D=OT critic)で起きた M の平坦化/崩壊が避けられるか?」の検証。
#
#   注意: LAMBDA_GAN=0 だと D の判別信号は G に直接届かない
#   (wpsb_model の loss_G_GAN は lambda_GAN>0 のときだけ G に加わる)。
#   よって G が受け取るのは seq-OT(D特徴) + MSE のみ = dOTmax と同じ経路で、
#   G が定数へ崩壊しても引き戻す直接勾配が無い。崩壊再発リスクは残る前提。
#   D 自体は use_D_feat_ot=True (flatten) で should_train_D=True のため、
#   LAMBDA_GAN=0 でも patchGAN として学習される(特徴は seq-OT に供給される)。
#
# dGAN(LAMBDA_GAN=1) 版との差 (1 変数):
#   - LAMBDA_GAN : 1 -> 0        (G_GAN 勾配を切る。D特徴は分類で学習したものを使用)
#
# dOTmax 版からの差 (2 変数):
#   - D_LOSS_MODE: ot -> gan     (D 特徴 OT 最大化 → real/fake 分類)
#   - LR    : 0.002 -> 2e-4      (初期 netG_grad スパイク(実測~1e8)の step 被害を 1/10 に)
#   ※ LAMBDA_GAN は dOTmax と同じ 0 のまま。
#   ※ LR は dGAN 版と揃える(両実験を同 lr で比較)。
#
# 据え置き (dOTmax / dGAN と同一。比較可能にするため):
#   - N_GEN=4 : G:D=3:1 の排他交互更新
#   - SEQ_OT_COST_SPACE=flatten / METRIC=l2 / FEAT_NORM=l2 / SEQ_OT_DEBIAS=True
#   - LMDA=0.05, SEQ_OT_ITERS=50 (flatten+feat_norm=l2 較正値。gan でも不変)
#   - LAMBDA_MSE=1 : UNSB の SB ドリフト項 tau*mean(‖Xt-fake‖²)
#
# 注意 (elapse):
#   flatten ≈ 3.3s/iter (A6000実測)。400 epoch × 126 seq ≈ 46h。
#   48h に収まらなければ latest_net_* から continue_train で再投入すること。
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -N train_ct2mr_dGANnoadv_l2mse

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

# --- seq-OT は D 特徴空間 (flatten) のまま。D は real/fake 分類器 ---
export SEQ_OT_COST_SPACE="${SEQ_OT_COST_SPACE:-flatten}"
export D_LOSS_MODE="${D_LOSS_MODE:-gan}"          # D = real/fake 分類 (patchGAN)
export SEQ_OT_DEBIAS="${SEQ_OT_DEBIAS:-True}"     # G 側 seq-OT の中心化 (divergence)
export SEQ_OT_METRIC="${SEQ_OT_METRIC:-l2}"       # L2 距離 (debias 下では cosine と等価)
export SEQ_OT_FEAT_NORM="${SEQ_OT_FEAT_NORM:-l2}" # 特徴を単位球面化 (M を O(1) に固定)

# --- ★ adv 勾配を G に流さない (分類特徴だけを seq-OT に使う検証) ---
export LAMBDA_GAN="${LAMBDA_GAN:-0}"

# --- G:D=3:1 の排他交互更新 (dOTmax / dGAN と同じ) ---
export N_GEN="${N_GEN:-4}"

# --- ★ 初期 netG_grad スパイク(実測~1e8)の step 被害軽減のため lr を 1/10 に (dGAN と同値) ---
export LR="${LR:-0.0002}"
# D の学習率。空=--lr と同じ (G と同一 lr)。
export LR_D="${LR_D:-}"

# --- 較正済み reg/iters (flatten, feat_norm=l2; brain.h5)。gan でも不変 ---
export LMDA="${LMDA:-0.05}"
export SEQ_OT_ITERS="${SEQ_OT_ITERS:-50}"

# --- UNSB SB ドリフト MSE 項 ---
export LAMBDA_MSE="${LAMBDA_MSE:-1}"

# --- D 更新は auto (flatten で use_D_feat_ot=True → LAMBDA_GAN=0 でも D 学習される) ---
export UPDATE_D="${UPDATE_D:-auto}"
# netG forward chunk: checkpoint 下では小さいほど速い (16 推奨)。
export NETG_CHUNK="${NETG_CHUNK:-16}"

# --- 実験名 ---
export NAME="${NAME:-synthRAD_${REGION}_ct2mr_dGANnoadv_l2_featnorm_mse}"

# --- wandb (計算ノードからオンライン送信可なら 1) ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-dfeat,flatten,dGAN,l2,featnorm,mse,gd3to1,noadv}"

echo "JOB on $(hostname) at $(date)"
echo "D_LOSS_MODE=${D_LOSS_MODE} LAMBDA_GAN=${LAMBDA_GAN} N_GEN=${N_GEN} LR=${LR} METRIC=${SEQ_OT_METRIC} FEAT_NORM=${SEQ_OT_FEAT_NORM} LMDA=${LMDA} ITERS=${SEQ_OT_ITERS} LAMBDA_MSE=${LAMBDA_MSE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 学習スクリプト本体を実行 ---
bash run_train_synthRAD_ot_div_geo_gan.sh

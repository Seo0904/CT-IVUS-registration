#!/bin/bash
#============================================================
# Genkai 用 pjsub ジョブ: synthRAD CT->MR WP-UNSB(seq-OT)
#   D = OT 最大化 (OT-GAN 流 critic) + G:D=3:1 + MSE ドリフト項
#   コスト空間 = flatten (Discriminator 特徴空間で L_ot を計算)
#   距離 = L2 (cdist^2) / 特徴 = L2 正規化 (単位球面化)
#   投入:  pjsub job_train_synthRAD_ct2mr_dOTmax_l2mse.sh
#   ログ:  train_ct2mr_dOTmax_l2mse.<JOBID>.out
#
# 設定の意図:
#   - D_LOSS_MODE=ot : Discriminator は D 特徴空間の Sinkhorn divergence を
#     最大化する (OT-GAN の minibatch energy distance critic 相当)。
#     debias=True 固定なので、feat_norm=l2 のもとでは L2 と cosine は等価
#     (単位球面上で ‖x-y‖²=2(1-cos))。「L2 距離」を明示指定している。
#   - N_GEN=4 : G/D 排他交互更新で G:D=3:1 ((N-1):1)。
#   - SEQ_OT_FEAT_NORM=l2 : フレーム記述子を単位球面化。D が特徴ノルムを
#     膨らませて OT を稼ぐ発散を防ぐ (ot モードでは必須級)。
#   - LAMBDA_MSE=1 : UNSB の SB ドリフト項 tau*mean(‖Xt-fake‖²) を追加
#     (netE 項なし。both 内の同項と同じ寄与)。
#
# reg/iters は brain.h5 で較正した値 (feat_norm=l2, ot=debias):
#   LMDA=0.05, SEQ_OT_ITERS=50 で 1対1(P_IDENT_ERR)を保ちつつ勾配が流れ、
#   Sinkhorn は iters=20 で既に収束(50 は安全マージン)。
#   ※較正は 16 スライス間引き。全スライス(~90)だとコスト行列が大きく
#     収束がやや遅くなるので、その場合は SEQ_OT_ITERS=100 も検討。
#
# 注意 (elapse):
#   flatten ≈ 3.3s/iter (A6000実測)。400 epoch × 126 seq ≈ 46h。
#   下の elapse=48:00:00 で概ね完走見込みだが、足りなければ latest_net_* から
#   continue_train で再投入すること。
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -N train_ct2mr_dOTmax_l2mse

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

# --- D 特徴空間 OT (flatten) + D=OT最大化 ---
export SEQ_OT_COST_SPACE="${SEQ_OT_COST_SPACE:-flatten}"
export D_LOSS_MODE="${D_LOSS_MODE:-ot}"          # D は OT (Sinkhorn divergence) を最大化
export SEQ_OT_DEBIAS="${SEQ_OT_DEBIAS:-True}"    # ot 本体 (中心化された divergence)
export SEQ_OT_METRIC="${SEQ_OT_METRIC:-l2}"      # L2 距離 (debias 下では cosine と等価)
export SEQ_OT_FEAT_NORM="${SEQ_OT_FEAT_NORM:-l2}"  # 特徴を単位球面化 (正規化あり)

# --- G:D=3:1 の排他交互更新 ---
export N_GEN="${N_GEN:-4}"                        # (N-1):1 = 3:1

# --- 較正済み reg/iters (feat_norm=l2, ot=debias; brain.h5) ---
export LMDA="${LMDA:-0.05}"
export SEQ_OT_ITERS="${SEQ_OT_ITERS:-50}"

# --- UNSB SB ドリフト MSE 項 ---
export LAMBDA_MSE="${LAMBDA_MSE:-1}"

# --- D 更新は auto (flatten で use_D_feat_ot=True → D 学習される) ---
export UPDATE_D="${UPDATE_D:-auto}"
# adv (G_GAN) は使わない。D は OT 最大化のみで学習。
export LAMBDA_GAN="${LAMBDA_GAN:-0}"
# netG forward chunk: checkpoint 下では小さいほど速い (16 推奨)。
export NETG_CHUNK="${NETG_CHUNK:-16}"

# --- 実験名 ---
export NAME="${NAME:-synthRAD_${REGION}_ct2mr_dOTmax_l2_featnorm_mse}"

# --- wandb (計算ノードからオンライン送信可なら 1) ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-dfeat,flatten,dOTmax,l2,featnorm,mse,gd3to1}"

echo "JOB on $(hostname) at $(date)"
echo "D_LOSS_MODE=${D_LOSS_MODE} METRIC=${SEQ_OT_METRIC} FEAT_NORM=${SEQ_OT_FEAT_NORM} N_GEN=${N_GEN} LMDA=${LMDA} ITERS=${SEQ_OT_ITERS} LAMBDA_MSE=${LAMBDA_MSE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 学習スクリプト本体を実行 ---
bash run_train_synthRAD_ot_div_geo_gan.sh

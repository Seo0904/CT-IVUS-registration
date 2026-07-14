#!/bin/bash
#============================================================
# Genkai 用 pjsub ジョブ: synthRAD CT->MR WP-UNSB(seq-OT)
#   D = real/fake 分類器 (通常の conditional patchGAN critic)
#   G = GAN(adv) + seq-OT(D特徴, flatten) + MSE ドリフト項
#   投入:  pjsub job_train_synthRAD_ct2mr_dGAN_l2mse.sh
#   ログ:  train_ct2mr_dGAN_l2mse.<JOBID>.out
#
# ★ dOTmax 版 (job_train_synthRAD_ct2mr_dOTmax_l2mse.sh) の D-GAN 版。
#   あちらは fake_B が定数へ崩壊した (netG が tanh 飽和)。原因は
#   「G の学習信号が全部 D特徴 OT を経由 → 高次元 l2 球で M が平坦化して
#    勾配消失 → 崩壊防止項(diversity)まで一緒に死ぬ → lr=0.002 の大ステップで
#    tanh 飽和の吸収状態へドリフト」。
#   → D を real/fake 分類器に戻し LAMBDA_GAN>0 にすることで、G が定数を出しても
#     消えない直接的な realism 勾配を与えて崩壊を断つ。
#
# dOTmax 版からの変更点 (3 変数差):
#   - D_LOSS_MODE: ot -> gan     (D 特徴 OT 最大化 → real/fake 分類)
#   - LAMBDA_GAN : 0  -> 1       (G_GAN 勾配を G に流す。崩壊対策の本体)
#   - LR    : 0.002 -> 2e-4      (初期 netG_grad スパイク(実測~1e8)の step 被害を 1/10 に)
#
# 据え置き (dOTmax 版と同一。比較可能にするため):
#   - N_GEN=4 : G:D=3:1 の排他交互更新 (dOTmax と同じ)
#   - SEQ_OT_COST_SPACE=flatten : seq-OT は D 特徴空間のまま(temporal ordering 用)
#   - SEQ_OT_METRIC=l2 / SEQ_OT_FEAT_NORM=l2 : 記述子を単位球面化
#   - SEQ_OT_DEBIAS=True : G 側 seq-OT の中心化
#   - LAMBDA_MSE=1 : UNSB の SB ドリフト項 tau*mean(‖Xt-fake‖²)
#
# reg/iters は flatten+feat_norm=l2 で較正済みの値をそのまま使用:
#   LMDA=0.05, SEQ_OT_ITERS=50。feat_norm=l2 が M を単位球面上の O(1) に固定するので、
#   D を gan で学習してもこの較正は不変 (本体 run_train スクリプトの regime 表参照:
#   flatten(M~30) → lmda=0.05 / iters=50)。
#
# 注意 (elapse):
#   flatten ≈ 3.3s/iter (A6000実測)。400 epoch × 126 seq ≈ 46h。dOTmax と同じ
#   G:D=3:1 なので所要も同程度。48h に収まらなければ latest_net_* から
#   continue_train で再投入すること。
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
#PJM -N train_ct2mr_dGAN_l2mse

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
export D_LOSS_MODE="${D_LOSS_MODE:-gan}"          # ★ D = real/fake 分類 (patchGAN)
export SEQ_OT_DEBIAS="${SEQ_OT_DEBIAS:-True}"     # G 側 seq-OT の中心化 (divergence)
export SEQ_OT_METRIC="${SEQ_OT_METRIC:-l2}"       # L2 距離 (debias 下では cosine と等価)
export SEQ_OT_FEAT_NORM="${SEQ_OT_FEAT_NORM:-l2}" # 特徴を単位球面化 (M を O(1) に固定)

# --- ★ G に敵対勾配を流す (崩壊対策の本体) ---
export LAMBDA_GAN="${LAMBDA_GAN:-1}"

# --- G:D=3:1 の排他交互更新 (dOTmax と同じ) ---
export N_GEN="${N_GEN:-4}"

# --- ★ 初期 netG_grad スパイク(実測~1e8)の step 被害軽減のため lr を 1/10 に ---
export LR="${LR:-0.0002}"
# D の学習率。空=--lr と同じ (G と同一 lr)。
export LR_D="${LR_D:-}"

# --- 較正済み reg/iters (flatten, feat_norm=l2; brain.h5)。gan でも不変 ---
export LMDA="${LMDA:-0.05}"
export SEQ_OT_ITERS="${SEQ_OT_ITERS:-50}"

# --- UNSB SB ドリフト MSE 項 ---
export LAMBDA_MSE="${LAMBDA_MSE:-1}"

# --- D 更新は auto (LAMBDA_GAN>0 → D 学習される) ---
export UPDATE_D="${UPDATE_D:-auto}"
# netG forward chunk: checkpoint 下では小さいほど速い (16 推奨)。
export NETG_CHUNK="${NETG_CHUNK:-16}"

# --- 実験名 ---
export NAME="${NAME:-synthRAD_${REGION}_ct2mr_dGAN_l2_featnorm_mse}"

# --- wandb (計算ノードからオンライン送信可なら 1) ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-dfeat,flatten,dGAN,l2,featnorm,mse,gd3to1,adv}"

echo "JOB on $(hostname) at $(date)"
echo "D_LOSS_MODE=${D_LOSS_MODE} LAMBDA_GAN=${LAMBDA_GAN} N_GEN=${N_GEN} LR=${LR} METRIC=${SEQ_OT_METRIC} FEAT_NORM=${SEQ_OT_FEAT_NORM} LMDA=${LMDA} ITERS=${SEQ_OT_ITERS} LAMBDA_MSE=${LAMBDA_MSE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 学習スクリプト本体を実行 ---
bash run_train_synthRAD_ot_div_geo_gan.sh

#!/bin/bash
#============================================================
# Genkai(玄界) 用 pjsub ジョブ: synthRAD CT->MR UNSB 学習の再開
#   latest チェックポイントから n_epochs(=400) まで続きを回す。
#   既定では 361 epoch から再開 (= 360 まで完了済みの想定)。
#
#   投入:  pjsub job_resume_synthRAD_ct2mr.sh
#   ログ:  resume_ct2mr.<JOBID>.out
#
#   上書き可能な環境変数 (pjsub 時に渡す):
#     RESUME_DIR  : 再開する checkpoints_dir(=日付ディレクトリ)。
#                   未指定なら experiment_result 配下の最新ディレクトリを自動選択
#     EPOCH_COUNT : 開始 epoch (既定 361)
#     ITER_COUNT  : 開始 total_iters = W&B step 継続用 (既定 4101120 ≒ 360*11392)
#     WANDB_ID    : 同じ W&B run に追記したい場合に指定 (例: 8uknok68)
#============================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=5:00:00
#PJM -j
#PJM -N resume_ct2mr

set -e

# --- conda 環境 ---
source ~/.bashrc 2>/dev/null || true
conda activate ctmri

# --- このジョブスクリプトのある場所 (= UNSB-main) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${SCRIPT_DIR}"

# --- データ位置 (run スクリプトの DATAROOT 既定は docker パスなので上書き必須) ---
export DATAROOT="${DATAROOT:-${WORKSPACE_DIR}/data/preprocessed/synthRAD_maskfree_full}"
export REGION="${REGION:-brain}"
NAME_EXP="synthRAD_${REGION}_ct2mr_sb"

# --- 再開対象ディレクトリ ---
EXP_ROOT="${WORKSPACE_DIR}/data/experiment_result/UNSB/synthRAD_${REGION}_ct2mr"
# このジョブで続きから回す対象の日付ディレクトリ
RESUME_DATE="${RESUME_DATE:-20260617_202816}"
if [[ -z "${RESUME_DIR:-}" ]]; then
  RESUME_DIR="${EXP_ROOT}/${RESUME_DATE}"
fi
export RESUME_DIR

# --- latest チェックポイントの存在確認 ---
CKPT="${RESUME_DIR}/${NAME_EXP}/latest_net_G.pth"
echo "RESUME_DIR = ${RESUME_DIR}"
echo "checking   = ${CKPT}"
if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] latest チェックポイントが見つかりません: ${CKPT}" >&2
  echo "        RESUME_DIR=... を明示指定して投入してください。" >&2
  ls -la "${RESUME_DIR}/${NAME_EXP}/" 2>/dev/null | tail -20 >&2 || true
  exit 1
fi

# --- 再開パラメータ ---
export RESUME=1
export EPOCH_COUNT="${EPOCH_COUNT:-361}"
export ITER_COUNT="${ITER_COUNT:-4101120}"   # ≒ 360 epoch * 11392 slices (W&B step 継続用)
# 同じ W&B run に追記 (run id を指定)
export WANDB_ID="${WANDB_ID:-yady7muc}"

# --- wandb (計算ノードからオンライン送信可を確認済み) ---
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "JOB on $(hostname) at $(date)"
echo "EPOCH_COUNT=${EPOCH_COUNT}  ITER_COUNT=${ITER_COUNT}  WANDB_ID=${WANDB_ID:-<new run>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 再開学習を実行 (run スクリプト内の RESUME ブロックが continue_train を付与) ---
bash run_train_synthRAD_ct2mr.sh

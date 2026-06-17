#!/bin/bash
# ============================================================
# Genkai (九大スパコン) 上で Docker 環境を conda で再現するスクリプト
#   元: Dockerfile (nvidia/cuda:12.1.1-devel-ubuntu22.04, Python3.12)
# 使い方:
#   bash setup_genkai_conda.sh
# ============================================================
set -euo pipefail

ENV_NAME="ctmri"

# --- 1. conda を有効化 (Genkai のモジュール名は環境に合わせて変更) ---
# 例: module load miniconda / module load python など
# module load miniconda3
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda が見つかりません。先に 'module load miniconda3' 等を実行してください。" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 2. 環境作成 (Docker と同じ Python 3.12) ---
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[INFO] 既存の ${ENV_NAME} を再利用します。"
else
    conda create -y -n "${ENV_NAME}" python=3.12
fi
conda activate "${ENV_NAME}"

# --- 3. Dockerfile と同じ順序で pip インストール ---
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==69.5.1"

# visdom は build-isolation を切って入れる (Dockerfile と同じ)
python -m pip install --no-build-isolation --no-cache-dir visdom==0.2.4

# PyTorch は CUDA 12.1 ビルド (Docker の CUDA 12.1.1 に対応)
python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

# 残りの依存 (cu121 版を上書きしないよう torch / torchvision 行を除外)
grep -ivE '^(torch|torchvision)([=<>!~ ]|$)' requirements.txt > /tmp/req_genkai.txt
python -m pip install --no-cache-dir -r /tmp/req_genkai.txt

# --- 4. 動作確認 ---
echo "==== 確認 ===="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
echo "完了: conda activate ${ENV_NAME}"

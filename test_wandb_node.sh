#!/bin/bash
#============================================================
# 計算ノードから wandb オンライン送信できるか確認するジョブ (pjsub)
#   投入:  pjsub test_wandb_node.sh
#   結果:  test_wandb_node.sh.<JOBID>.out を確認
#============================================================
#PJM -L rscgrp=b-batch   # 存在しなければ pjsub がエラーを返す。その場合は名前を変える
#PJM -L gpu=1
#PJM -L elapse=0:10:00
#PJM -j
#PJM -N wandb_test

set -e

# conda 有効化 (base が自動で効かない場合は明示)
source ~/.bashrc 2>/dev/null || true
conda activate ctmri

echo "===== ノード情報 ====="
hostname
date

echo "===== 1. 外部HTTPS到達性 (api.wandb.ai) ====="
curl -sS --max-time 20 -o /dev/null -w "wandb_http=%{http_code}\n" https://api.wandb.ai || \
    echo "curl 失敗: 計算ノードは外部接続不可の可能性"

echo "===== 2. wandb オンライン送信テスト ====="
python -c "
import wandb
r = wandb.init(project='genkai-node-test')
wandb.log({'x': 1})
url = r.url
r.finish()
print('ONLINE OK ->', url)
"

echo "===== 完了 ====="

#!/bin/bash
# 実験A: 全volume版(削らない=cloze無し)の MR を target / balanced OT
#   - データ: synthRAD_maskfree_full/brain/brain.h5 (MRは全スライス健在)
#   - unbalanced 無し (balanced)
# 他は run_train_synthRAD_ot_div_geo_gan.sh の設定そのまま (lmda=10, iters=300, normalize=none, geo)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export DATAROOT="/workspace/data/preprocessed/synthRAD_maskfree_full"
unset H5_PATH                       # 既定の <dataroot>/brain/brain.h5 を使う
unset SEQ_OT_UNBALANCED             # balanced
export NAME="synthRAD_brain_ct2mr_otdiv_geo_gan_A_nocloze_bal"
# W&B タグ（既定タグに追加）。実行時にさらに足したいときは WANDB_TAGS_EXTRA を上書き。
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-nocloze,balanced,expA}"

exec "${SCRIPT_DIR}/run_train_synthRAD_ot_div_geo_gan.sh" "$@"

#!/bin/bash
# 実験C: cloze版(MRをランダムに10〜30枚黒化)の MR を target / unbalanced OT (rho=50)
#   - データ: synthRAD_maskfree_full/brain/brain_cloze.h5
#   - unbalanced rho=50 … 黒化で消えた target (T>N) の余りフレームを drop させる狙い
#     (reg=10, iters=300, normalize=none での較正スイートスポット。レンジ30〜75)
# 他は run_train_synthRAD_ot_div_geo_gan.sh の設定そのまま
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export DATAROOT="/workspace/data/preprocessed/synthRAD_maskfree_full"
export H5_PATH="/workspace/data/preprocessed/synthRAD_maskfree_full/brain/brain_cloze.h5"
export SEQ_OT_UNBALANCED=50         # unbalanced rho
export NAME="synthRAD_brain_ct2mr_otdiv_geo_gan_C_cloze_unb50"
# W&B タグ（既定タグに追加）。実行時にさらに足したいときは WANDB_TAGS_EXTRA を上書き。
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-cloze,unbalanced,rho50,expC}"

exec "${SCRIPT_DIR}/run_train_synthRAD_ot_div_geo_gan.sh" "$@"

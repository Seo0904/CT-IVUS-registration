#!/bin/bash
# 実験B: cloze版(MRをランダムに10〜30枚黒化)の MR を target / balanced OT
#   - データ: synthRAD_maskfree_full/brain/brain_cloze.h5 (split.json は同ディレクトリを共用)
#   - unbalanced 無し (balanced) … 黒スライスは get_valid_frame_idx で除外されるが
#     balanced は全質量を強制輸送するため、対照(=unbalancedの効果を見る基準)として実行
# 他は run_train_synthRAD_ot_div_geo_gan.sh の設定そのまま
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export DATAROOT="/workspace/data/preprocessed/synthRAD_maskfree_full"
export H5_PATH="/workspace/data/preprocessed/synthRAD_maskfree_full/brain/brain_cloze.h5"
unset SEQ_OT_UNBALANCED             # balanced
export NAME="synthRAD_brain_ct2mr_otdiv_geo_gan_B_cloze_bal"
# W&B タグ（既定タグに追加）。実行時にさらに足したいときは WANDB_TAGS_EXTRA を上書き。
export WANDB_TAGS_EXTRA="${WANDB_TAGS_EXTRA:-cloze,balanced,expB}"

exec "${SCRIPT_DIR}/run_train_synthRAD_ot_div_geo_gan.sh" "$@"

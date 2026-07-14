#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

#
# synthRAD brain CT->MR 用 UNSB(seq-OT) 訓練スクリプト
#
# ドメインA: CT volume (1患者=1 sequence, スライス軸=frame軸)
# ドメインB: MR volume (同上)
# dataset_mode=ct_mri, region=brain, batch_size=1 (1患者ずつ)

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# 日付の取得
DATE=$(date +"%Y%m%d_%H%M%S")

# データ領域 (brain / pelvis)
REGION=${REGION:-brain}

# 保存先ディレクトリ（絶対パス）
RESULT_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_ct_mri/synthRAD-${REGION}/${DATE}"

# ディレクトリ作成
mkdir -p "${RESULT_DIR}"

# データセットパス（絶対パス）: <DATAROOT>/<REGION>/<REGION>.h5 を読む
# full = 症例数(=seq数)を増やした版 (brain: 180症例, train126/val18/test36)。
DATAROOT="${DATAROOT:-${WORKSPACE_DIR}/data/preprocessed/synthRAD_maskfree_full}"

# h5 ファイルを直接上書き（空なら <DATAROOT>/<REGION>/<REGION>.h5）。cloze版などを指す用。
# split.json は常に <DATAROOT>/<REGION>/split.json を使う(cloze版も同じ split を共用)。
H5_PATH=${H5_PATH:-}
H5_FILE="${H5_PATH:-${DATAROOT}/${REGION}/${REGION}.h5}"

# 実験名（保存先の最後のフォルダ名として使用）
NAME="${NAME:-synthRAD_${REGION}_ct2mr_sb_w_otdiv_015_geo_gan}"

# GPU設定
GPU_IDS=0

# ===== W&B logging (optional) =====
USE_WANDB=${USE_WANDB:-1}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-WP-UNSB_ct_mri}
WANDB_GROUP=${WANDB_GROUP:-synthRAD-${REGION}}
WANDB_TAGS=${WANDB_TAGS:-synthRAD,${REGION},ct2mr,WP-UNSB_ct_mri}
# 追加タグ（カンマ区切り）。既定タグに足したいとき WANDB_TAGS_EXTRA で渡す。
WANDB_TAGS_EXTRA=${WANDB_TAGS_EXTRA:-}
if [[ -n "${WANDB_TAGS_EXTRA}" ]]; then
  WANDB_TAGS="${WANDB_TAGS},${WANDB_TAGS_EXTRA}"
fi
WANDB_IMAGE_FREQ=${WANDB_IMAGE_FREQ:-1000}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${NAME}_${DATE}}

# モデル設定
MODEL="wpsb"
MODE="sb"

# ネットワーク構造（グレースケール用）
INPUT_NC=1
OUTPUT_NC=1
NGF=64
NDF=64

# データセット設定
DATASET_MODE="ct_mri"
SLICE_AXIS=${SLICE_AXIS:-2}            # 2=axial をシーケンス軸に
MIN_MASK_RATIO=${MIN_MASK_RATIO:-0.02} # マスク前景がこの比率未満のスライスは除外
NUM_SLICES_PER_SEQ=${NUM_SLICES_PER_SEQ:-0}  # 0=全スライス(~200)
NETG_CHUNK=${NETG_CHUNK:-16}           # netG forward を frame 軸でこの枚数ずつ実行(32bit overflow回避)。
                                       # checkpoint=True 下では小chunkほど再計算が軽く高速(クリーン実測 pixel: 16=3.74s < 32=4.85s < 64=5.44s, 結果不変)。
NETG_CHECKPOINT=${NETG_CHECKPOINT:-True} # チャンク毎に gradient checkpointing(全frame保持でのOOM回避)
AMP_NETG=${AMP_NETG:-False}            # off: fp16 overflow による nan/inf 勾配を回避(巨大SBスケールのため)
LOAD_SIZE=256                          # preprocess=none のため実リサイズには未使用
CROP_SIZE=256
BATCH_SIZE=1                           # 1患者=1 sequence。volume毎にS/H/Wが違うので1固定

# 損失の重み
# debias で SB が中心化され桁が落ちる前提で、adv 比率を持ち上げるため lambda_GAN を引き上げる。
# 環境変数で上書き可: LAMBDA_GAN=1 にすれば debias 単独の効果を切り分けられる。
LAMBDA_GAN=${LAMBDA_GAN:-0}   # 0 = GAN無効(純SBベースライン)。debias=True/mono_penalty=0.1 のままで純SBの素の挙動を見る
LAMBDA_GAN_SEQ=0
LAMBDA_SB=${LAMBDA_SB:-1.0}
LAMBDA_NCE=0
# UNSB の SB ドリフト項 tau*||Xt - fake_B||^2 の重み(0=無効/従来)。
# seq_ot モードに UNSB 同様のドリフト正則化を入れたいとき。both では二重加算注意。
LAMBDA_MSE=${LAMBDA_MSE:-0}

# sequence OT (P entropy)
SEQ_OT_P_ENTROPY=${SEQ_OT_P_ENTROPY:-0}
SEQ_OT_P_ENTROPY_PENALTY=${SEQ_OT_P_ENTROPY_PENALTY:-0.0}

# sequence OT (divergence)
# debias=True のときは ot_cost 自体が Sinkhorn divergence になり、この手作り項は
# コード側で自動スキップされる（seq_ot_divergence の値は無視）。
SEQ_OT_DIVERGENCE=${SEQ_OT_DIVERGENCE:-1}
SEQ_OT_DIVERGENCE_PENALTY=${SEQ_OT_DIVERGENCE_PENALTY:--0.5}

# sequence OT (debias): True で solve_sample(debias=True) の Sinkhorn divergence を
# ot_cost に使う。biased OT のゲタ（自己コスト offset）が消えて SB が中心化され、
# adv 比率が回復する（MovingMNIST の動く設定に合わせる）。
SEQ_OT_DEBIAS=${SEQ_OT_DEBIAS:-True}

# sequence OT (monotone): mono は勾配あり(min_gan準拠)。penalty は netG を直接動かすので
# 1本目のログ(SB_P=debiased ot_cost と SB_U=penalty*mono_loss)を見て較正する。
# mono は値は小さいが勾配が OT より大きい(実測 grad比 ~1.8@1.0)ので、勾配で OT の ~2割に抑える。
MONO_PENALTY=${MONO_PENALTY:-0.1}

# sequence OT (unbalanced): rho を渡すと unbalanced OT になる。空にすると balanced
SEQ_OT_UNBALANCED=${SEQ_OT_UNBALANCED:-}

# sequence OT のコスト空間: pixel(画像L2) / flatten(D特徴) / random_patch(D特徴1点)
SEQ_OT_COST_SPACE=${SEQ_OT_COST_SPACE:-pixel}
# D 更新制御: auto(lambda_GAN>0 or D特徴OTで学習) / always / never
UPDATE_D=${UPDATE_D:-auto}

# ===== OT-GAN (OT-GAN-main) 由来の学習法オプション（既定値=従来挙動） =====
# G/D 排他交互更新: 0=無効(毎step両方)。N>0 で N step に 1 回 D のみ、他は G のみ。
# 比率 G:D=(N-1):1。参考コード意図なら 3(=2:1)、論文の 3:1 なら 4。
N_GEN=${N_GEN:-0}
# D の学習率。空=--lr と同じ。OT-GAN 準拠は G/D とも 3e-4 (LR=3e-4 LR_D=3e-4)
LR_D=${LR_D:-}
# D の loss: gan(patchGAN, 従来) / ot(D特徴空間の Sinkhorn divergence 最大化, OT-GAN critic 相当)
# ot は SEQ_OT_COST_SPACE=flatten/random_patch が必須、SEQ_OT_FEAT_NORM=l2 を強く推奨
D_LOSS_MODE=${D_LOSS_MODE:-gan}
# seq-OT コストの距離: l2(cdist^2, 従来) / cosine(1 - x̂ŷᵀ, OT-GAN 準拠, 値域[0,2])
SEQ_OT_METRIC=${SEQ_OT_METRIC:-l2}
# フレーム記述子の L2 正規化(単位球面化): none(従来) / l2
# 正規化すると M が O(1) になるので LMDA(reg) や MONO_PENALTY は要再較正 (OT-GAN の reg は 0.1)
SEQ_OT_FEAT_NORM=${SEQ_OT_FEAT_NORM:-none}
# OT-GAN 準拠設定の例:
#   LR=3e-4 LR_D=3e-4 N_GEN=3 D_LOSS_MODE=ot SEQ_OT_COST_SPACE=flatten \
#   SEQ_OT_METRIC=cosine SEQ_OT_FEAT_NORM=l2 SEQ_OT_DEBIAS=True LMDA=0.1

# Sinkhorn の reg(=lmda) と反復数。コスト空間でスケールが変わるので要調整:
#   pixel(M~1.3e4): lmda=10 / iters=300
#   flatten(M~30) : lmda=0.05 / iters=50   (D特徴。M小さくすぐ収束)
#   random_patch(M~0.15): lmda=0.005 / iters=50
LMDA=${LMDA:-10}
SEQ_OT_ITERS=${SEQ_OT_ITERS:-300}

# シーケンス長は volume の有効スライス数で可変（データ側で決まる）

# タイムステップ
NUM_TIMESTEPS=5

# 訓練設定
N_EPOCHS=400
N_EPOCHS_DECAY=0
LR=${LR:-0.002}
SAVE_EPOCH_FREQ=10
PRINT_FREQ=${PRINT_FREQ:-1}
DISPLAY_FREQ=${DISPLAY_FREQ:-1000}

echo "======================================"
echo "synthRAD ${REGION} CT->MR UNSB(seq-OT) Training"
echo "======================================"
echo "Data: ${H5_FILE} (A=CT, B=MR)"
echo "Experiment: ${NAME}"
echo "Date: ${DATE}"
echo "Results will be saved to: ${RESULT_DIR}"
echo "GPU: ${GPU_IDS}"
echo "======================================"

# 設定をログに保存
cat > "${RESULT_DIR}/config.txt" << EOF
Experiment Configuration
========================
Date: ${DATE}
Name: ${NAME}

Data:
  h5: ${H5_FILE}
  Region: ${REGION}  (A=CT, B=MR)
  Slice axis: ${SLICE_AXIS}  Min mask ratio: ${MIN_MASK_RATIO}
  Num slices per seq: ${NUM_SLICES_PER_SEQ} (0=all)

Model:
  Model: ${MODEL}
  Mode: ${MODE}
  Input NC: ${INPUT_NC}
  Output NC: ${OUTPUT_NC}
  NGF: ${NGF}
  NDF: ${NDF}
  Num Timesteps: ${NUM_TIMESTEPS}

Training:
  Batch Size: ${BATCH_SIZE}
  Load Size: ${LOAD_SIZE}
  Crop Size: ${CROP_SIZE}
  Print Freq: ${PRINT_FREQ}
  Learning Rate: ${LR}
  Epochs: ${N_EPOCHS}
  Decay Epochs: ${N_EPOCHS_DECAY}

Loss Weights:
  Lambda GAN: ${LAMBDA_GAN}
  Lambda SB: ${LAMBDA_SB}
  Lambda NCE: ${LAMBDA_NCE}
  Seq OT P Entropy: ${SEQ_OT_P_ENTROPY}
  Seq OT P Entropy Penalty: ${SEQ_OT_P_ENTROPY_PENALTY}
  Seq OT Divergence: ${SEQ_OT_DIVERGENCE}
  Seq OT Divergence Penalty: ${SEQ_OT_DIVERGENCE_PENALTY}
  Seq OT Unbalanced (rho): ${SEQ_OT_UNBALANCED:-none(balanced)}

OT-GAN style options:
  N Gen (G/D alternation): ${N_GEN} (0=off)
  LR D: ${LR_D:-same as LR}
  D Loss Mode: ${D_LOSS_MODE}
  Seq OT Metric: ${SEQ_OT_METRIC}
  Seq OT Feat Norm: ${SEQ_OT_FEAT_NORM}

Wandb:
  Use: ${USE_WANDB}
  Project: ${WANDB_PROJECT}
  Entity: ${WANDB_ENTITY}
  Mode: ${WANDB_MODE}
  Group: ${WANDB_GROUP}
  Tags: ${WANDB_TAGS}
  Run Name: ${WANDB_RUN_NAME}
  Image Freq: ${WANDB_IMAGE_FREQ}
EOF

# UNSB-mainディレクトリに移動して実行
cd "${SCRIPT_DIR}"

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PWD=$(pwd)"
echo "train.py path=$(realpath train.py)"

cmd=(python3 train.py
  --dataroot ${DATAROOT}
  --region ${REGION}
  --slice_axis ${SLICE_AXIS}
  --min_mask_ratio ${MIN_MASK_RATIO}
  --num_slices_per_seq ${NUM_SLICES_PER_SEQ}
  --netg_chunk ${NETG_CHUNK}
  --netg_checkpoint ${NETG_CHECKPOINT}
  --amp_netg ${AMP_NETG}
  --name ${NAME}
  --model ${MODEL}
  --mode ${MODE}
  --dataset_mode ${DATASET_MODE}
  --input_nc ${INPUT_NC}
  --output_nc ${OUTPUT_NC}
  --ngf ${NGF}
  --ndf ${NDF}
  --load_size ${LOAD_SIZE}
  --crop_size ${CROP_SIZE}
  --batch_size ${BATCH_SIZE}
  --num_timesteps ${NUM_TIMESTEPS}
  --lambda_GAN ${LAMBDA_GAN}
  --lambda_SB ${LAMBDA_SB}
  --lambda_NCE ${LAMBDA_NCE}
  --lambda_MSE ${LAMBDA_MSE}
  --save_ot_details
  --ot_details_max_samples 10
  --save_feature_tsne
  --lmda ${LMDA}
  --seq_ot_iters ${SEQ_OT_ITERS}
  --sinkhorn_type "sinkhorn_log"
  --sb_mode seq_ot
  --seq_ot_normalize "none"
  --seq_ot_solver "geo"
  --seq_ot_geo_scaling 0.99
  --seq_ot_geo_p 2
  --seq_ot_p_entropy ${SEQ_OT_P_ENTROPY}
  --seq_ot_p_entropy_penalty ${SEQ_OT_P_ENTROPY_PENALTY}
  --seq_ot_divergence ${SEQ_OT_DIVERGENCE}
  --seq_ot_divergence_penalty ${SEQ_OT_DIVERGENCE_PENALTY}
  --seq_ot_debias ${SEQ_OT_DEBIAS}
  --seq_ot_cost_space ${SEQ_OT_COST_SPACE}
  --update_D ${UPDATE_D}
  --n_gen ${N_GEN}
  --D_loss_mode ${D_LOSS_MODE}
  --seq_ot_metric ${SEQ_OT_METRIC}
  --seq_ot_feat_norm ${SEQ_OT_FEAT_NORM}
  --n_epochs ${N_EPOCHS}
  --n_epochs_decay ${N_EPOCHS_DECAY}
  --lr ${LR}
  --print_freq ${PRINT_FREQ}
  --display_freq ${DISPLAY_FREQ}
  --seq_ot_monotone_penalty ${MONO_PENALTY}
  --save_epoch_freq ${SAVE_EPOCH_FREQ}
  --checkpoints_dir ${RESULT_DIR}
  --gpu_ids ${GPU_IDS}
  --preprocess none
  --no_flip
  --display_id 0
  --num_threads 0
)

# h5 を直接指定（cloze版など）。空なら dataset 既定の <dataroot>/<region>/<region>.h5
if [[ -n "${H5_PATH}" ]]; then
  cmd+=(--h5_path ${H5_PATH})
fi

# unbalanced OT: SEQ_OT_UNBALANCED が空でなければ rho を渡す（空なら未指定=balanced）
if [[ -n "${SEQ_OT_UNBALANCED}" ]]; then
  cmd+=(--seq_ot_unbalanced ${SEQ_OT_UNBALANCED})
fi

# D の学習率: LR_D が空でなければ渡す（空なら --lr と同じ）
if [[ -n "${LR_D}" ]]; then
  cmd+=(--lr_D ${LR_D})
fi

if [[ "${USE_WANDB}" == "1" ]]; then
  cmd+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" --wandb_run_name "${WANDB_RUN_NAME}" --wandb_group "${WANDB_GROUP}" --wandb_tags "${WANDB_TAGS}" --wandb_image_freq "${WANDB_IMAGE_FREQ}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    cmd+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

"${cmd[@]}"


echo "======================================"
echo "Training completed!"
echo "Results saved to: ${RESULT_DIR}"
echo "======================================"

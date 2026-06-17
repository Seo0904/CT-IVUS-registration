#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

#
# test_visualize_seq.py を test_best_0 ... test_best_9 に対して実行するラッパー
#
# 各 run の images ディレクトリ:
#   ${CHECKPOINTS_DIR}/test_best_${i}/${NAME}/test_best/images
# 出力先（seq_vis）:
#   ${CHECKPOINTS_DIR}/test_best_${i}/${NAME}/test_best/seq_vis
#
# 使用方法:
#   bash run_visualize_seq_best.sh [DATE]
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

DATE="${1:-20260527_125919}"
NAME="moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze"
CHECKPOINTS_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/${DATE}"
VIS_PY="${SCRIPT_DIR}/test_visualize_seq.py"

cd "${SCRIPT_DIR}"

for i in $(seq 0 9); do
  IMAGES_DIR="${CHECKPOINTS_DIR}/test_best_${i}/${NAME}/test_best/images"
  echo "##### visualize test_best_${i} #####"
  echo "  images_dir: ${IMAGES_DIR}"

  if [ ! -d "${IMAGES_DIR}" ]; then
    echo "  WARNING: images ディレクトリが見つかりません。スキップします。"
    continue
  fi

  python3 "${VIS_PY}" "${IMAGES_DIR}"
done

echo ""
echo "======================================"
echo "All visualizations completed!"
echo "  各 seq_vis は test_best_0 ... test_best_9 の test_best/seq_vis 配下に保存されました"
echo "======================================"

#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

#
# run_test_moving_mnist_seg_paired_min_ot_div_confirm.sh を 11 回実行するラッパー
#
#  1 回目 : epoch 320  -> 保存先 test_320
#  2-11   : epoch best -> 保存先 test_best_0 ... test_best_9
#
# 使用方法:
#   bash run_test_11times.sh [DATE]
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

DATE="${1:-20260527_125919}"
CHECKPOINTS_DIR="${WORKSPACE_DIR}/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/${DATE}"
TEST_SH="${SCRIPT_DIR}/run_test_moving_mnist_seg_paired_min_ot_div_confirm.sh"

# 1 回目: epoch 320
echo "##### Run 1/11 : epoch=320 -> test_320 #####"
RESULTS_DIR="${CHECKPOINTS_DIR}/test_320" bash "${TEST_SH}" "${DATE}" 320

# 2-11 回目: epoch best を 10 回
for i in $(seq 0 9); do
  echo "##### Run $((i + 2))/11 : epoch=best -> test_best_${i} #####"
  RESULTS_DIR="${CHECKPOINTS_DIR}/test_best_${i}" bash "${TEST_SH}" "${DATE}" best
done

echo ""
echo "======================================"
echo "All 11 runs completed!"
echo "  ${CHECKPOINTS_DIR}/test_320"
echo "  ${CHECKPOINTS_DIR}/test_best_0 ... test_best_9"
echo "======================================"

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image


FOLDER_NAMES = ["real_A", "real_B", "fake_B"]
FILENAME_PATTERN = re.compile(r"seq(\d+)_frame(\d+)\.png")


def collect_sequences(images_root: str) -> Dict[str, Dict[int, List[int]]]:
	"""images_root 以下の各フォルダごとに {folder: {seq_id: [frame_ids]}} を集計する。"""

	result: Dict[str, Dict[int, List[int]]] = {}

	for folder in FOLDER_NAMES:
		folder_path = os.path.join(images_root, folder)
		if not os.path.isdir(folder_path):
			raise FileNotFoundError(f"フォルダが見つかりません: {folder_path}")

		seq_to_frames: Dict[int, List[int]] = {}
		for fname in sorted(os.listdir(folder_path)):
			m = FILENAME_PATTERN.match(fname)
			if not m:
				continue
			seq_id = int(m.group(1))
			frame_id = int(m.group(2))
			seq_to_frames.setdefault(seq_id, []).append(frame_id)

		# frame_id をソートしておく
		for seq_id in list(seq_to_frames.keys()):
			seq_to_frames[seq_id] = sorted(seq_to_frames[seq_id])

		result[folder] = seq_to_frames

	return result


def choose_sequence(
	seq_info: Dict[str, Dict[int, List[int]]], target_seq: Optional[int]
) -> int:
	"""3フォルダすべてに存在する seq を選ぶ。target_seq が指定されていればそれを優先。"""

	common: Optional[set[int]] = None
	for folder in FOLDER_NAMES:
		seq_set = set(seq_info[folder].keys())
		common = seq_set if common is None else common & seq_set

	if not common:
		raise RuntimeError("3フォルダすべてに共通する seq が見つかりませんでした。")

	if target_seq is not None:
		if target_seq not in common:
			raise ValueError(
				f"指定した seq={target_seq} は3フォルダで共通ではありません。共通している seq: {sorted(common)}"
			)
		return target_seq

	# 指定がなければ最小の seq を使う
	return min(common)


def get_frames_to_show(
	seq_info: Dict[str, Dict[int, List[int]]], seq_id: int, max_frames: int
) -> List[int]:
	"""3フォルダすべてに存在する frame の共通部分から表示する frame を決める。"""

	common_frames: Optional[set[int]] = None
	for folder in FOLDER_NAMES:
		frames = set(seq_info[folder].get(seq_id, []))
		if not frames:
			raise RuntimeError(f"フォルダ {folder} に seq{seq_id} のフレームがありません。")
		common_frames = frames if common_frames is None else common_frames & frames

	if not common_frames:
		raise RuntimeError(f"3フォルダすべてに共通する frame がありません (seq{seq_id})。")

	frames_sorted = sorted(common_frames)
	return frames_sorted[:max_frames]


def load_image(path: str):
	if not os.path.isfile(path):
		raise FileNotFoundError(f"画像ファイルが見つかりません: {path}")
	return Image.open(path)


def plot_triplet_sequence(
	images_root: str,
	seq_id: Optional[int] = None,
	max_frames: int = 10,
	fig_title: str = "Moving MNIST comparison",
	save_dir: Optional[str] = None,
	out_name: Optional[str] = None,
) -> Tuple[int, List[int]]:
	"""realA / realB / fake_B の3行で、同じ seq のフレームを並べて表示する。

	Returns
	-------
	(使用した seq_id, 使用した frame_id のリスト)
	"""

	if max_frames <= 0:
		raise ValueError("max_frames は 1 以上にしてください。")

	seq_info = collect_sequences(images_root)
	chosen_seq = choose_sequence(seq_info, seq_id)
	frames = get_frames_to_show(seq_info, chosen_seq, max_frames)

	n_rows = len(FOLDER_NAMES)
	n_cols = len(frames)

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
	if n_rows == 1:
		axes = [axes]

	fig.suptitle(fig_title)

	for row_idx, folder in enumerate(FOLDER_NAMES):
		for col_idx, frame_id in enumerate(frames):
			ax = axes[row_idx][col_idx] if n_cols > 1 else axes[row_idx]
			img_path = os.path.join(
				images_root, folder, f"seq{chosen_seq}_frame{frame_id}.png"
			)
			img = load_image(img_path)
			ax.imshow(img, cmap="gray")
			ax.axis("off")

			if col_idx == 0:
				ax.set_ylabel(folder, rotation=0, labelpad=30, fontsize=10, va="center")

			if row_idx == n_rows - 1:
				ax.set_xlabel(f"frame {frame_id}", fontsize=8)

	plt.tight_layout(rect=[0, 0, 1, 0.95])

	# 保存先が指定されていれば保存する（デフォルトは images/plt/seqX_nY.png）
	if save_dir is not None:
		os.makedirs(save_dir, exist_ok=True)
		if out_name is None or out_name == "":
			out_name = f"seq{chosen_seq}_n{len(frames)}.png"
		save_path = os.path.join(save_dir, out_name)
		fig.savefig(save_path, dpi=150, bbox_inches="tight")
		print(f"Figure saved to: {save_path}")
	plt.show()

	return chosen_seq, frames


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"realA / realB / fake_B の3フォルダから、" "同じ seq の画像を3行で並べて表示します。"
		)
	)

	parser.add_argument(
		"--base-dir",
		type=str,
		default="/workspace/data/experiment_result/WP-UNSB/moving-mnist",
		help="実験結果のベースディレクトリ (run_id の一つ上)",
	)
	parser.add_argument(
		"--run-id",
		type=str,
		default="20260310_030937",
		help="例: 20260310_030937 のような実験ID",
	)
	parser.add_argument(
		"--exp-name",
		type=str,
		default="moving_mnist_seg_paired_sb",
		help="例: moving_mnist_seg_paired_sb など test_results/ 以下のディレクトリ名",
	)
	parser.add_argument(
		"--result-tag",
		type=str,
		default="test_best",
		help="test_results/exp-name/ 以下の結果ディレクトリ名 (既定: test_best)",
	)
	parser.add_argument(
		"--seq",
		type=int,
		default=100,
		help="表示したい seq 番号。省略時は3フォルダ共通の最小の seq を自動選択",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=10,
		help="表示する最大フレーム数 (1〜20 程度)。共通部分から先頭 max-frames 枚を表示",
	)
	parser.add_argument(
		"--title",
		type=str,
		default="Moving MNIST comparison",
		help="figure のタイトル",
	)
	parser.add_argument(
		"--out-name",
		type=str,
		default=None,
		help=(
			"保存するファイル名。省略時は seq{seq}_n{枚数}.png という形式で自動決定"
		),
	)

	return parser.parse_args()


def main() -> None:
	args = parse_args()

	images_root = os.path.join(
		args.base_dir,
		args.run_id,
		"test_results",
		args.exp_name,
		args.result_tag,
		"images",
	)

	print(f"images_root: {images_root}")

	save_dir = os.path.join(images_root, "plt")

	seq_id, frames = plot_triplet_sequence(
		images_root=images_root,
		seq_id=args.seq,
		max_frames=args.max_frames,
		fig_title=args.title,
		save_dir=save_dir,
		out_name=args.out_name,
	)

	print(f"使用した seq: {seq_id}")
	print(f"使用した frames: {frames}")


if __name__ == "__main__":
	main()


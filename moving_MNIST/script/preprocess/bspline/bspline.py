import numpy as np
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')  # non-GUI backend for server
import matplotlib.pyplot as plt
import gryds
from tqdm import tqdm
import urllib.request
import os


class MovingMNISTBSplineTransformer:
	"""全シーケンスで 1 つの B-spline 変換を共有して適用するクラス"""

	def __init__(self, trans_mrange=0.05, trans_grid=3, seed=None):
		self.trans_mrange = trans_mrange
		self.trans_grid = trans_grid
		if seed is not None:
			np.random.seed(seed)

	def generate_bspline_transform(self):
		"""ランダムな B-spline 変換を 1 つ生成"""
		random_grid = np.random.uniform(
			-self.trans_mrange,
			self.trans_mrange,
			size=(2, self.trans_grid, self.trans_grid),
		)
		return gryds.BSplineTransformation(random_grid)

	def transform_frame(self, frame, bspline):
		"""単一フレームに B-spline 変換を適用"""
		frame_normalized = frame.astype(np.float64)
		if frame_normalized.max() > 1:
			frame_normalized = frame_normalized / 255.0

		interpolator = gryds.Interpolator(frame_normalized)
		trans_frame = interpolator.transform(bspline)

		trans_frame = np.clip(trans_frame, 0, 1)
		return trans_frame

	def transform_batch(self, data):
		"""バッチ全体に **同一** B-spline 変換を適用

		Args:
			data: (N, T, H, W)

		Returns:
			(N, T, H, W)
		"""

		num_sequences = data.shape[0]
		transformed_data = np.zeros_like(data, dtype=np.float64)

		# ★ ここで 1 つだけ B-spline 変換を作成し，全シーケンスで共有 ★
		bspline = self.generate_bspline_transform()

		print(f"Processing {num_sequences} sequences with ONE shared B-spline...")
		for i in tqdm(range(num_sequences), desc="B-spline Transform (global)"):
			sequence = data[i]
			num_frames = sequence.shape[0]
			for t in range(num_frames):
				transformed_data[i, t] = self.transform_frame(sequence[t], bspline)

		return transformed_data


def download_moving_mnist(save_path):
	url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"

	if not os.path.exists(save_path):
		print(f"Downloading Moving MNIST to {save_path}...")
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		urllib.request.urlretrieve(url, save_path)
		print("Download complete!")
	else:
		print(f"Moving MNIST already exists at {save_path}")


def load_moving_mnist(data_path):
	"""(T, N, H, W) -> (N, T, H, W) に転置して返す"""
	data = np.load(data_path)
	data = np.transpose(data, (1, 0, 2, 3))
	print(f"Loaded data shape: {data.shape}")
	print(f"  - Sequences: {data.shape[0]}")
	print(f"  - Frames per sequence: {data.shape[1]}")
	print(f"  - Frame size: {data.shape[2]}x{data.shape[3]}")
	return data


def visualize_comparison(original, transformed, sequence_idx=0, save_path=None):
	num_frames = min(20, original.shape[0])

	fig, axes = plt.subplots(2, num_frames, figsize=(2 * num_frames, 4))

	for t in range(num_frames):
		axes[0, t].imshow(original[t], cmap="gray")
		axes[0, t].axis("off")
		if t == 0:
			axes[0, t].set_title(f"Original\nFrame {t}", fontsize=8)
		else:
			axes[0, t].set_title(f"Frame {t}", fontsize=8)

		axes[1, t].imshow(transformed[t], cmap="gray")
		axes[1, t].axis("off")
		if t == 0:
			axes[1, t].set_title(f"B-spline (global)\nFrame {t}", fontsize=8)
		else:
			axes[1, t].set_title(f"Frame {t}", fontsize=8)

	plt.suptitle(
		f"Sequence {sequence_idx}: ONE T applied to ALL sequences", fontsize=10
	)
	plt.tight_layout()

	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, dpi=150, bbox_inches="tight")
		print(f"Saved visualization to {save_path}")
	else:
		plt.show()
	plt.close()


if __name__ == "__main__":
	base_dir = Path("./")
	data_dir = base_dir / "data"
	output_dir = data_dir / "preprocessed/bspline_transformed"

	data_path = data_dir / "org_data/moving_mnist/mnist_test_seq.npy"

	data_dir.mkdir(parents=True, exist_ok=True)
	output_dir.mkdir(parents=True, exist_ok=True)

	download_moving_mnist(str(data_path))
	data = load_moving_mnist(str(data_path))

	trans_grid = 3
	trans_mrange = 0.1

	transformer = MovingMNISTBSplineTransformer(
		trans_mrange=trans_mrange,
		trans_grid=trans_grid,
		seed=42,
	)

	num_sequences = data.shape[0]

	print(f"\n{'=' * 50}")
	print("B-spline Transformation (GLOBAL) Demo")
	print(f"  - trans_mrange: {trans_mrange}")
	print(f"  - trans_grid: {trans_grid}")
	print(f"  - Processing {num_sequences} sequences with ONE T")
	print(f"{'=' * 50}\n")

	transformed_data = transformer.transform_batch(data)

	np.save(output_dir / "transformed_global_0.1_3.npy", transformed_data)
	print(f"\nSaved transformed data to {output_dir / 'transformed_global_0.1_3.npy'}")

	for i in range(min(3, num_sequences)):
		visualize_comparison(
			data[i],
			transformed_data[i],
			sequence_idx=i,
			save_path=str(
				data_dir
				/ f"dust_box/bspline_transformed_global/comparison_seq{i}.png"
			),
		)

	print("\nDone!")


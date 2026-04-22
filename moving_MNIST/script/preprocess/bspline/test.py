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
    """
    Moving MNISTデータセットにB-spline変換を適用するクラス
    
    各シーケンス X_i (64フレーム) に対して同じ変換 T_i を適用
    異なるシーケンス X_j には異なる変換 T_j を適用
    """
    
    def __init__(self, trans_mrange=0.05, trans_grid=3, seed=None):
        """
        Args:
            trans_mrange: B-spline変形の強度 (default: 0.05)
            trans_grid: B-splineグリッドのサイズ (default: 3)
            seed: 乱数シード（再現性のため）
        """
        self.trans_mrange = trans_mrange
        self.trans_grid = trans_grid
        if seed is not None:
            np.random.seed(seed)
    
    def generate_bspline_transform(self):
        """
        ランダムなB-spline変換を生成
        
        Returns:
            gryds.BSplineTransformation: B-spline変換オブジェクト
        """
        random_grid = np.random.uniform(
            -self.trans_mrange, 
            self.trans_mrange,
            size=(2, self.trans_grid, self.trans_grid)
        )
        return gryds.BSplineTransformation(random_grid)
    
    def transform_frame(self, frame, bspline):
        """
        単一フレームにB-spline変換を適用
        
        Args:
            frame: 入力画像 (H, W)
            bspline: B-spline変換オブジェクト
            
        Returns:
            変換された画像
        """
        # 正規化
        frame_normalized = frame.astype(np.float64)
        if frame_normalized.max() > 1:
            frame_normalized = frame_normalized / 255.0
        
        interpolator = gryds.Interpolator(frame_normalized)
        trans_frame = interpolator.transform(bspline)
        
        # クリッピングと正規化
        trans_frame = np.clip(trans_frame, 0, 1)
        
        return trans_frame
    
    def transform_sequence(self, sequence):
        """
        シーケンス全体に同じB-spline変換を適用
        
        Args:
            sequence: 入力シーケンス (T, H, W) - T frames
            
        Returns:
            変換されたシーケンス (T, H, W)
        """
        # このシーケンス用の変換を1つ生成
        bspline = self.generate_bspline_transform()
        
        num_frames = sequence.shape[0]
        transformed_sequence = np.zeros_like(sequence, dtype=np.float64)
        
        # 同じ変換を全フレームに適用
        for t in range(num_frames):
            transformed_sequence[t] = self.transform_frame(sequence[t], bspline)
        
        return transformed_sequence
    
    def transform_batch(self, data):
        """
        バッチ全体を処理（各シーケンスに異なる変換を適用）
        
        Args:
            data: Moving MNISTデータ (N, T, H, W)
                  N: シーケンス数
                  T: フレーム数 (通常20)
                  H, W: 画像サイズ (64x64)
                  
        Returns:
            変換されたデータ (N, T, H, W)
        """
        num_sequences = data.shape[0]
        transformed_data = np.zeros_like(data, dtype=np.float64)
        
        print(f"Processing {num_sequences} sequences...")
        for i in tqdm(range(num_sequences), desc="B-spline Transform"):
            # 各シーケンス X_i に異なる変換 T_i を適用
            transformed_data[i] = self.transform_sequence(data[i])
        
        return transformed_data


def download_moving_mnist(save_path):
    """
    Moving MNISTデータセットをダウンロード
    
    Args:
        save_path: 保存先パス
    """
    url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    
    if not os.path.exists(save_path):
        print(f"Downloading Moving MNIST to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        urllib.request.urlretrieve(url, save_path)
        print("Download complete!")
    else:
        print(f"Moving MNIST already exists at {save_path}")


def load_moving_mnist(data_path):
    """
    Moving MNISTデータを読み込み
    
    Args:
        data_path: データファイルのパス
        
    Returns:
        data: (N, T, H, W) 形式のデータ
              元のデータは (T, N, H, W) なので転置
    """
    data = np.load(data_path)
    # (T, N, H, W) -> (N, T, H, W) に転置
    data = np.transpose(data, (1, 0, 2, 3))
    print(f"Loaded data shape: {data.shape}")
    print(f"  - Sequences: {data.shape[0]}")
    print(f"  - Frames per sequence: {data.shape[1]}")
    print(f"  - Frame size: {data.shape[2]}x{data.shape[3]}")
    return data


def visualize_comparison(original, transformed, sequence_idx=0, save_path=None):
    """
    オリジナルと変換後のシーケンスを比較可視化
    
    Args:
        original: オリジナルシーケンス (T, H, W)
        transformed: 変換後シーケンス (T, H, W)
        sequence_idx: シーケンスのインデックス（タイトル用）
        save_path: 保存先パス（Noneの場合は表示）
    """
    num_frames = min(100, original.shape[0])  # 最大10フレーム表示
    
    fig, axes = plt.subplots(2, num_frames, figsize=(2*num_frames, 4))
    
    for t in range(num_frames):
        # Original
        axes[0, t].imshow(original[t], cmap='gray')
        axes[0, t].axis('off')
        if t == 0:
            axes[0, t].set_title(f'Original\nFrame {t}', fontsize=8)
        else:
            axes[0, t].set_title(f'Frame {t}', fontsize=8)
        
        # Transformed
        axes[1, t].imshow(transformed[t], cmap='gray')
        axes[1, t].axis('off')
        if t == 0:
            axes[1, t].set_title(f'B-spline\nFrame {t}', fontsize=8)
        else:
            axes[1, t].set_title(f'Frame {t}', fontsize=8)
    
    plt.suptitle(f'Sequence {sequence_idx}: Same T_i applied to all frames', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # パス設定
    base_dir = Path("./")
    data_dir = base_dir / "data"
    output_dir = data_dir / "preprocessed/bspline_transformed"
    
    data_path = data_dir / "org_data/moving_mnist/mnist_test_seq.npy"
    
    # ディレクトリ作成
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データダウンロード & 読み込み
    download_moving_mnist(str(data_path))
    data = load_moving_mnist(str(data_path))
    
    # B-spline変換パラメータ
    trans_grid = 3       # グリッドサイズ
    trans_mrange = 0.15  # 変形の強度
    
    # 変換器を初期化
    transformer = MovingMNISTBSplineTransformer(
        trans_mrange=trans_mrange,
        trans_grid=trans_grid,
        seed=42  # 再現性のため
    )
    
    # デモ: 最初の10シーケンスを処理
    num_demo_sequences = data.shape[0]

    
    print(f"\n{'='*50}")
    print("B-spline Transformation Demo")
    print(f"  - trans_mrange: {trans_mrange}")
    print(f"  - trans_grid: {trans_grid}")
    print(f"  - Processing {num_demo_sequences} sequences")
    print(f"{'='*50}\n")
    
    # バッチ変換
    transformed_data = transformer.transform_batch(data)
    
    # 結果の保存
    
    np.save(output_dir / "transformed.npy", transformed_data)
    print(f"\nSaved transformed data to {output_dir}")
    
    # 可視化（最初の3シーケンス）
    for i in range(min(3, num_demo_sequences)):
        visualize_comparison(
            data[i], 
            transformed_data[i],
            sequence_idx=i,
            save_path=str(data_dir / f"dust_box/bspline_transformed/comparison_seq{i}.png")
        )
    
    print("\nDone!")
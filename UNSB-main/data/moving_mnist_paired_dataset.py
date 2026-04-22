"""
Moving MNIST Paired Dataset for UNSB
2つの異なるMoving MNISTファイルをドメインA/Bとして扱うデータセット

ドメインA: オリジナルのMoving MNIST
ドメインB: B-spline変換後のMoving MNIST
"""
import os
import numpy as np
from PIL import Image
import torch
from data.base_dataset import BaseDataset, get_transform
import random


class MovingMnistPairedDataset(BaseDataset):
    """
    2つのMoving MNISTファイルをペアとして扱うデータセット
    
    - ドメインA: dataroot/mnist_test_seq.npy
    - ドメインB: dataroot_B/transformed.npy (または指定パス)
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add dataset-specific options"""
        parser.add_argument('--dataroot_B', type=str, default='',
                          help='Path to domain B data (default: uses dataroot)')
        parser.add_argument('--data_file_A', type=str, default='mnist_test_seq.npy',
                          help='Filename for domain A data')
        parser.add_argument('--data_file_B', type=str, default='transformed.npy',
                          help='Filename for domain B data')
        parser.add_argument('--num_frames_per_seq', type=int, default=20,
                          help='Number of frames to use per sequence')
        parser.add_argument('--use_random_frame', action='store_true',
                          help='If True, randomly select frame from sequence')
        parser.add_argument('--train_ratio', type=float, default=0.7,
                          help='Ratio of data to use for training (default: 0.7 = 70%%)')
        parser.add_argument('--val_ratio', type=float, default=0.1,
                          help='Ratio of data to use for validation (default: 0.1 = 10%%)')
        # Note: --phase is already defined in train_options.py / test_options.py
        return parser

    def __init__(self, opt):
        """
        Initialize the paired Moving MNIST dataset.
        
        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)
        
        # ファイルパスの構築
        self.data_file_A = os.path.join(opt.dataroot, opt.data_file_A)
        
        if hasattr(opt, 'dataroot_B') and opt.dataroot_B and opt.dataroot_B != '':
            self.data_file_B = os.path.join(opt.dataroot_B, opt.data_file_B)
        else:
            self.data_file_B = os.path.join(opt.dataroot, opt.data_file_B)
        
        # ファイル存在確認
        if not os.path.exists(self.data_file_A):
            raise FileNotFoundError(f"Domain A file not found: {self.data_file_A}")
        if not os.path.exists(self.data_file_B):
            raise FileNotFoundError(f"Domain B file not found: {self.data_file_B}")
        
        print(f"Loading Domain A from: {self.data_file_A}")
        print(f"Loading Domain B from: {self.data_file_B}")
        
        # データ読み込み
        # 形式: (T, N, H, W) または (N, T, H, W)
        data_A = np.load(self.data_file_A)
        data_B = np.load(self.data_file_B)
        
        print(f"Raw Domain A shape: {data_A.shape}")
        print(f"Raw Domain B shape: {data_B.shape}")
        
        # 形状の確認と調整 (N, T, H, W) に統一
        # Moving MNISTは通常 (T, N, H, W) = (20, 10000, 64, 64)
        if data_A.shape[0] == 20:  # (T, N, H, W) 形式の場合
            data_A = np.transpose(data_A, (1, 0, 2, 3))
        if data_B.shape[0] == 20:  # (T, N, H, W) 形式の場合
            data_B = np.transpose(data_B, (1, 0, 2, 3))
        
        # Train/Val/Test分割
        train_ratio = opt.train_ratio if hasattr(opt, 'train_ratio') else 0.7
        val_ratio = opt.val_ratio if hasattr(opt, 'val_ratio') else 0.1
        # test_ratio = 1 - train_ratio - val_ratio (default: 0.2)
        
        num_sequences = data_A.shape[0]
        train_end = int(num_sequences * train_ratio)
        val_end = int(num_sequences * (train_ratio + val_ratio))
        
        self.phase = opt.phase if hasattr(opt, 'phase') else ('train' if opt.isTrain else 'test')
        
        if self.phase == 'train':
            # 学習時: 最初の70%を使用 (0-6999)
            self.data_A = data_A[:train_end]
            self.data_B = data_B[:train_end]
            print(f"Train mode: using sequences 0-{train_end-1} ({train_end} sequences)")
        elif self.phase == 'val':
            # 検証時: 次の10%を使用 (7000-7999)
            self.data_A = data_A[train_end:val_end]
            self.data_B = data_B[train_end:val_end]
            print(f"Validation mode: using sequences {train_end}-{val_end-1} ({val_end - train_end} sequences)")
            print("  -> Paired mode enabled for GT comparison (SSIM, L2)")
        else:
            # テスト時: 残りの20%を使用 (8000-9999)
            self.data_A = data_A[val_end:]
            self.data_B = data_B[val_end:]
            print(f"Test mode: using sequences {val_end}-{num_sequences-1} ({num_sequences - val_end} sequences)")
            print("  -> Paired mode enabled for GT comparison (SSIM, L2)")
        
        print(f"Domain A shape (after transpose): {self.data_A.shape}")
        print(f"Domain B shape (after transpose): {self.data_B.shape}")
        
        self.num_sequences_A = self.data_A.shape[0]
        self.num_sequences_B = self.data_B.shape[0]
        self.num_frames = self.data_A.shape[1]
        
        self.num_frames_per_seq = min(opt.num_frames_per_seq, self.num_frames) if hasattr(opt, 'num_frames_per_seq') else self.num_frames
        self.use_random_frame = opt.use_random_frame if hasattr(opt, 'use_random_frame') else False
        
        # Transform設定（グレースケールデータ用）
        self.transform = get_transform(opt, grayscale=True)
        
        print(f"Dataset initialized:")
        print(f"  - Domain A sequences: {self.num_sequences_A}")
        print(f"  - Domain B sequences: {self.num_sequences_B}")
        print(f"  - Frames per sequence: {self.num_frames}")
        print(f"  - Using random frame: {self.use_random_frame}")

    def __getitem__(self, index):
        """
        Return a data point and its metadata.
        
        Parameters:
            index -- a random integer for data indexing
        
        Returns:
            a dictionary containing A, B, A_paths, B_paths
        """
        # シーケンスとフレームのインデックスを計算
        if self.use_random_frame:
            seq_idx_A = index % self.num_sequences_A
            frame_idx = random.randint(0, self.num_frames - 1)
        else:
            seq_idx_A = index // self.num_frames_per_seq
            frame_idx = index % self.num_frames_per_seq
        
        # 学習時: Domain Bからはランダムにサンプリング（unpaired設定）
        # Val/Test時: 同じシーケンス・フレームを使用（paired設定でGT比較可能）
        if self.phase == 'train':
            seq_idx_B = random.randint(0, self.num_sequences_B - 1)
            frame_idx_B = random.randint(0, self.num_frames - 1)
        else:
            # Paired: AとBで同じシーケンス・フレームを使用
            seq_idx_B = seq_idx_A
            frame_idx_B = frame_idx
        
        # フレームを取得
        frame_A = self.data_A[seq_idx_A, frame_idx]  # (H, W)
        frame_B = self.data_B[seq_idx_B, frame_idx_B]  # (H, W) - val/testではGT
        
        # 正規化 [0, 255] -> [0, 1]
        if frame_A.max() > 1:
            frame_A = frame_A / 255.0
        if frame_B.max() > 1:
            frame_B = frame_B / 255.0
        
        # PIL Imageに変換（グレースケールのまま）
        frame_A_uint8 = (frame_A * 255).astype(np.uint8)
        frame_B_uint8 = (frame_B * 255).astype(np.uint8)
        
        img_A = Image.fromarray(frame_A_uint8, mode='L')
        img_B = Image.fromarray(frame_B_uint8, mode='L')
        
        # Transform適用
        A = self.transform(img_A)
        B = self.transform(img_B)
        
        return {
            'A': A,
            'B': B,
            'A_paths': f"seq{seq_idx_A}_frame{frame_idx}",
            'B_paths': f"seq{seq_idx_B}_frame{frame_idx_B}"
        }

    def __len__(self):
        """Return the total number of samples."""
        if self.use_random_frame:
            return self.num_sequences_A
        else:
            return self.num_sequences_A * self.num_frames_per_seq

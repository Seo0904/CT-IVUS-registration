"""
Moving MNIST Paired Dataset for UNSB (sequence-return version)
1サンプル = 1シーケンス (T, 1, H, W) を返す

train:
  - A: indexで指定されたシーケンス
  - B: ランダムな別シーケンス (unpaired)
val/test:
  - A,B: 同一シーケンス (paired)
"""
import os
import numpy as np
import torch
from data.base_dataset import BaseDataset
import random


class MovingMnistPairedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dataroot_B', type=str, default='',
                            help='Path to domain B data (default: uses dataroot)')
        parser.add_argument('--data_file_A', type=str, default='mnist_test_seq.npy',
                            help='Filename for domain A data')
        parser.add_argument('--data_file_B', type=str, default='transformed.npy',
                            help='Filename for domain B data')
        parser.add_argument('--num_frames_per_seq', type=int, default=20,
                            help='Number of frames to use per sequence (default: 20)')
        parser.add_argument('--train_ratio', type=float, default=0.7,
                            help='Ratio of data to use for training (default: 0.7 = 70%%)')
        parser.add_argument('--val_ratio', type=float, default=0.1,
                            help='Ratio of data to use for validation (default: 0.1 = 10%%)')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.data_file_A = os.path.join(opt.dataroot, opt.data_file_A)
        if hasattr(opt, 'dataroot_B') and opt.dataroot_B and opt.dataroot_B != '':
            self.data_file_B = os.path.join(opt.dataroot_B, opt.data_file_B)
        else:
            self.data_file_B = os.path.join(opt.dataroot, opt.data_file_B)

        if not os.path.exists(self.data_file_A):
            raise FileNotFoundError(f"Domain A file not found: {self.data_file_A}")
        if not os.path.exists(self.data_file_B):
            raise FileNotFoundError(f"Domain B file not found: {self.data_file_B}")

        print(f"Loading Domain A from: {self.data_file_A}")
        print(f"Loading Domain B from: {self.data_file_B}")

        data_A = np.load(self.data_file_A)
        data_B = np.load(self.data_file_B)

        print(f"Raw Domain A shape: {data_A.shape}")
        print(f"Raw Domain B shape: {data_B.shape}")

        # (N,T,H,W) に統一
        if data_A.ndim != 4:
            raise ValueError(f"Expected Domain A to be 4D (T,N,H,W or N,T,H,W), got {data_A.ndim}D")
        if data_B.ndim != 4:
            raise ValueError(f"Expected Domain B to be 4D (T,N,H,W or N,T,H,W), got {data_B.ndim}D")

        # MovingMNIST標準は (T,N,H,W)=(20,10000,64,64)
        # 先頭がTっぽいなら転置
        if data_A.shape[0] <= 50 and data_A.shape[1] > 50:  # 保守的判定
            data_A = np.transpose(data_A, (1, 0, 2, 3))
        if data_B.shape[0] <= 50 and data_B.shape[1] > 50:
            data_B = np.transpose(data_B, (1, 0, 2, 3))

        if data_A.shape[:2] != data_B.shape[:2]:
            print(f"[WARN] A and B (N,T) differ: A={data_A.shape[:2]} B={data_B.shape[:2]} (still proceed)")

        num_sequences = data_A.shape[0]
        num_frames_total = data_A.shape[1]

        train_ratio = opt.train_ratio if hasattr(opt, 'train_ratio') else 0.7
        val_ratio = opt.val_ratio if hasattr(opt, 'val_ratio') else 0.1
        train_end = int(num_sequences * train_ratio)
        val_end = int(num_sequences * (train_ratio + val_ratio))

        self.phase = opt.phase if hasattr(opt, 'phase') else ('train' if opt.isTrain else 'test')

        if self.phase == 'train':
            self.data_A = data_A[:train_end]
            self.data_B = data_B[:train_end]
            print(f"Train mode: using sequences 0-{train_end-1} ({train_end} sequences)")
        elif self.phase == 'val':
            self.data_A = data_A[train_end:val_end]
            self.data_B = data_B[train_end:val_end]
            print(f"Validation mode: using sequences {train_end}-{val_end-1} ({val_end-train_end} sequences)")
        else:
            self.data_A = data_A[val_end:]
            self.data_B = data_B[val_end:]
            print(f"Test mode: using sequences {val_end}-{num_sequences-1} ({num_sequences-val_end} sequences)")

        self.num_sequences_A = self.data_A.shape[0]
        self.num_sequences_B = self.data_B.shape[0]
        self.num_frames_total = self.data_A.shape[1]

        # 使うフレーム数（基本20）
        self.num_frames_per_seq = min(opt.num_frames_per_seq, self.num_frames_total) if hasattr(opt, 'num_frames_per_seq') else self.num_frames_total

        print("Dataset initialized (sequence-return):")
        print(f"  - Domain A sequences: {self.num_sequences_A}")
        print(f"  - Domain B sequences: {self.num_sequences_B}")
        print(f"  - Total frames per sequence: {self.num_frames_total}")
        print(f"  - Using frames per sequence: {self.num_frames_per_seq}")

    def _seq_to_tensor(self, seq_hw: np.ndarray) -> torch.Tensor:
        """
        seq_hw: (T,H,W) in [0,255] or [0,1] float
        returns: (T,1,H,W) float tensor normalized to [-1,1]
        """
        seq = seq_hw.astype(np.float32)
        if seq.max() > 1.0:
            seq = seq / 255.0
        # (T,H,W) -> (T,1,H,W), normalize [0,1] -> [-1,1]
        t = torch.from_numpy(seq).unsqueeze(1)  # (T,1,H,W)
        t = t * 2.0 - 1.0
        return t

    def __getitem__(self, index):
        # 1サンプル = 1シーケンス
        seq_idx_A = index % self.num_sequences_A

        if self.phase == 'train':
            # unpaired: Bは別シーケンスをランダム
            seq_idx_B = seq_idx_A
        else:
            # paired: 同一シーケンス
            seq_idx_B = seq_idx_A

        # シーケンスを取り出し (T,H,W)
        seq_A = self.data_A[seq_idx_A, :self.num_frames_per_seq]
        seq_B = self.data_B[seq_idx_B, :self.num_frames_per_seq]

        # [0,255]→[0,1] の正規化は _seq_to_tensor 内で対応
        A = self._seq_to_tensor(seq_A)  # (T,1,H,W)
        B = self._seq_to_tensor(seq_B)  # (T,1,H,W)

        return {
            'A': A,
            'B': B,
            'A_paths': f"seq{seq_idx_A}",
            'B_paths': f"seq{seq_idx_B}"
        }

    def __len__(self):
        # シーケンス数
        return self.num_sequences_A
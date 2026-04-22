"""
Moving MNIST Paired Dataset for UNSB
sequence_OT 用に sequence 単位で返す版
"""
import os
import numpy as np
from PIL import Image
import torch
import random
from data.base_dataset import BaseDataset, get_transform


class MovingMnistPairedDataset(BaseDataset):
    """
    2つのMoving MNISTファイルを sequence 単位で扱うデータセット

    - Domain A: dataroot/mnist_test_seq.npy
    - Domain B: dataroot_B/transformed.npy (または指定パス)

    返り値:
        A: (T, C, H, W)
        B: (T, C, H, W) もしくは元データ長に応じた (T_B, C, H, W)
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dataroot_B', type=str, default='',
                            help='Path to domain B data (default: uses dataroot)')
        parser.add_argument('--data_file_A', type=str, default='mnist_test_seq.npy',
                            help='Filename for domain A data')
        parser.add_argument('--data_file_B', type=str, default='transformed.npy',
                            help='Filename for domain B data')
        parser.add_argument('--num_frames_per_seq', type=int, default=20,
                            help='Number of frames to use per sequence')
        parser.add_argument('--train_ratio', type=float, default=0.7,
                            help='Ratio of data to use for training')
        parser.add_argument('--val_ratio', type=float, default=0.1,
                            help='Ratio of data to use for validation')
        parser.add_argument('--unsb', action='store_true',
                          help='If True, use UNSB mode')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.data_file_A = os.path.join(opt.dataroot, opt.data_file_A)
        if hasattr(opt, 'dataroot_B') and opt.dataroot_B:
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

        # (N, T, H, W) に統一
        if data_A.shape[0] == 20:
            data_A = np.transpose(data_A, (1, 0, 2, 3))
        if data_B.shape[0] == 20:
            data_B = np.transpose(data_B, (1, 0, 2, 3))

        train_ratio = opt.train_ratio if hasattr(opt, 'train_ratio') else 0.7
        val_ratio = opt.val_ratio if hasattr(opt, 'val_ratio') else 0.1

        num_sequences = data_A.shape[0]
        train_end = 1
        val_end = 2

        self.phase = opt.phase if hasattr(opt, 'phase') else ('train' if opt.isTrain else 'test')

        if self.phase == 'train':
            self.data_A = data_A[:train_end]
            self.data_B = data_B[:train_end]
            print(f"Train mode: using sequences 0-{train_end-1} ({train_end} sequences)")
        elif self.phase == 'val':
            self.data_A = data_A[train_end:val_end]
            self.data_B = data_B[train_end:val_end]
            print(f"Validation mode: using sequences {train_end}-{val_end-1} ({val_end - train_end} sequences)")
        else:
            self.data_A = data_A[val_end:]
            self.data_B = data_B[val_end:]
            print(f"Test mode: using sequences {val_end}-{num_sequences-1} ({num_sequences - val_end} sequences)")

        print(f"Domain A shape (after transpose): {self.data_A.shape}")
        print(f"Domain B shape (after transpose): {self.data_B.shape}")

        self.num_sequences_A = self.data_A.shape[0]
        self.num_sequences_B = self.data_B.shape[0]
        self.num_frames_A = self.data_A.shape[1]
        self.num_frames_B = self.data_B.shape[1]

        self.num_frames_per_seq = min(
            opt.num_frames_per_seq if hasattr(opt, 'num_frames_per_seq') else self.num_frames_A,
            self.num_frames_A
        )
 

        self.transform = get_transform(opt, grayscale=True)
        self.unsb = opt.unsb if hasattr(opt, 'unsb') else False

        print("Dataset initialized:")
        print(f"  - Domain A sequences: {self.num_sequences_A}")
        print(f"  - Domain B sequences: {self.num_sequences_B}")
        print(f"  - Frames per sequence (A): {self.num_frames_A}")
        print(f"  - Frames per sequence (B): {self.num_frames_B}")

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """(H, W) -> (C, H, W)"""
        if frame.max() > 1:
            frame = frame / 255.0
        frame_uint8 = (frame * 255).astype(np.uint8)
        img = Image.fromarray(frame_uint8, mode='L')
        return self.transform(img)

    def __getitem__(self, index):
        """
        sequence 単位で返す

        Returns:
            {
                'A': Tensor (T_A, C, H, W),
                'B': Tensor (T_B, C, H, W),
                'A_paths': str,
                'B_paths': str,
                'seq_idx_A': int,
                'seq_idx_B': int
            }
        """
        seq_idx_A = index

        if self.phase == 'train':
            if self.unsb:
                seq_idx_B = random.randint(0, self.num_sequences_B - 1)
                if self.num_sequences_B > 1 and seq_idx_B == seq_idx_A:
                    while seq_idx_B == seq_idx_A:
                        seq_idx_B = random.randint(0, self.num_sequences_B - 1)
            else:
                seq_idx_B = seq_idx_A
        else:
            seq_idx_B = seq_idx_A

        seq_A = self.data_A[seq_idx_A]  # (T_A, H, W)
        seq_B = self.data_B[seq_idx_B]  # (T_B, H, W)

        # A は必要なら先頭 num_frames_per_seq に制限
        seq_A = seq_A[:self.num_frames_per_seq]

        # B は長さ違いを残したいなら切らない
        # もし揃えたいなら次を有効化:
        # seq_B = seq_B[:self.num_frames_per_seq]

        A_list = [self._frame_to_tensor(frame) for frame in seq_A]
        B_list = [self._frame_to_tensor(frame) for frame in seq_B]

        A = torch.stack(A_list, dim=0)  # (T_A, C, H, W)
        B = torch.stack(B_list, dim=0)  # (T_B, C, H, W)

        return {
            'A': A,
            'B': B,
            'A_paths': f"seq{seq_idx_A}",
            'B_paths': f"seq{seq_idx_B}",
            'seq_idx_A': seq_idx_A,
            'seq_idx_B': seq_idx_B,
        }

    def __len__(self):
        return self.num_sequences_A
# UNSB-main/models/sequence_ot.py
import torch
import ot  # POT
from typing import cast

def _get_valid_frame_idx(seq: torch.Tensor) -> torch.Tensor:
    """
    seq: (T, C, H, W) in [-1, 1]
    ゼロ画素フレーム（正規化後 max == -1.0）を除いた有効フレームのインデックスを返す。
    全フレームがゼロの場合は全インデックスを返す。
    """
    frame_max = seq.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]  # (T,)
    valid = (frame_max > -1.0 + 1e-3).nonzero(as_tuple=False).squeeze(1)  # 1D
    if valid.numel() == 0:
        valid = torch.arange(seq.shape[0], device=seq.device)
    return valid

def sequence_ot_loss_torch(
    fake_seq: torch.Tensor,
    tgt_seq: torch.Tensor,
    reg: float = 0.05,
    iters: int = 50,
    monotone: bool = True,
    monotone_penalty: float = 50.0,
    normalize: str = "mean",  # "mean" or "median" or "max" or None
) -> torch.Tensor:
    """
    POT Sinkhorn2 で sequence OT loss を計算する。

    fake_seq, tgt_seq: (T, C, H, W)  ※ [-1, 1] 正規化済みを想定

    ゼロ画素フレーム対策:
    - tgt_seq の有効フレームインデックス（非ゼロフレーム）を取得
    - fake_seq, tgt_seq 共にそのインデックスでサブセットしてから OT を解く
    - これにより monotone_penalty により K=0 → Ka=0 → NaN となる問題を回避
    """
    # tgt の有効フレームインデックスで両方をサブセット
    valid_idx = _get_valid_frame_idx(tgt_seq)  # tgt_seq の非ゼロフレーム
    N = valid_idx.numel()

    fake_sub = fake_seq[valid_idx]   # (N, C, H, W)
    tgt_sub  = tgt_seq[valid_idx]    # (N, C, H, W)

    fake_flat = fake_sub.reshape(N, -1)
    tgt_flat  = tgt_sub.reshape(N, -1)

    M = torch.cdist(fake_flat, tgt_flat, p=2) ** 2  # (N, N)

    # スケール正規化
    if normalize == "mean":
        s = M.detach().mean()
        M = M / (s + 1e-8)
    elif normalize == "median":
        s = M.detach().median()
        if s < 1e-6:
            s = M.detach().mean()
        M = M / (s + 1e-8)
    elif normalize == "max":
        s = M.detach().max()
        M = M / (s + 1e-8)

    # monotone soft mask: 相対インデックスで適用（サブ行列サイズ N×N）
    if monotone:
        i = torch.arange(N, device=fake_seq.device).view(N, 1)
        j = torch.arange(N, device=fake_seq.device).view(1, N)
        M = M + (i > j).float() * monotone_penalty

    a = torch.ones(N, device=fake_seq.device, dtype=fake_seq.dtype) / N
    b = torch.ones(N, device=fake_seq.device, dtype=fake_seq.dtype) / N

    cost = ot.sinkhorn2(a, b, M, reg=reg, numItermax=iters)
    if not torch.is_tensor(cost):
        cost = torch.tensor(cost, device=fake_seq.device, dtype=fake_seq.dtype)
    cost = cast(torch.Tensor, cost)
    return cost
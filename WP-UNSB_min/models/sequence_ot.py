# UNSB-main/models/sequence_ot.py
import torch
import ot  # POT
from typing import cast, Optional, Dict, Any


def get_valid_frame_idx(seq: torch.Tensor) -> torch.Tensor:
    """
    seq: (T, C, H, W) in [-1, 1]
    ゼロ画素フレーム（正規化後 max == -1.0）を除いた有効フレームのインデックスを返す。
    全フレームがゼロの場合は全インデックスを返す。
    """
    frame_max = seq.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]  # (T,)
    valid = (frame_max > -1.0 + 1e-3).nonzero(as_tuple=False).squeeze(1)
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
    P_entropy: bool = True,
    P_entropy_penalty: float = 1.0,
    ot_divergence: bool = False,
    ot_divergence_penalty: float = -0.5,
    normalize: str = "mean",   # "mean" or "median" or "max" or None
    return_plan: bool = False,
    return_details: bool = False,
    verbose: bool = False,
):
    """
    fake_seq, tgt_seq: (T, C, H, W), values in [-1, 1]

    fake_seq は全フレーム使用
    tgt_seq は valid_idx のみ使用

    total = <P, M> + lambda * L_mono(P)
    を返す。

    return_plan=True:
        (total, P)
    return_details=True:
        (total, details_dict)
    両方 False:
        (total, terms_dict)
        terms_dict には少なくとも {ot_cost, reg_cost, mono_loss, mono_penalty, total} を含む。
    """
    # tgt の有効フレームだけ使う
    valid_idx = get_valid_frame_idx(tgt_seq)   # (N,)
    N = valid_idx.numel()
    T = fake_seq.shape[0]

    fake_sub = fake_seq              # (T, C, H, W)
    tgt_sub  = tgt_seq[valid_idx]    # (N, C, H, W)

    fake_flat = fake_sub.reshape(T, -1)
    tgt_flat  = tgt_sub.reshape(N, -1)

    # cost matrix: (T, N)
    M = torch.cdist(fake_flat, tgt_flat, p=2) ** 2

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
    elif normalize is None:
        pass
    else:
        raise ValueError(f"Unsupported normalize: {normalize}")

    # 一様分布
    a = torch.ones(T, device=fake_seq.device, dtype=fake_seq.dtype) / T
    b = torch.ones(N, device=fake_seq.device, dtype=fake_seq.dtype) / N

    # 輸送計画 P を取得
    P = ot.sinkhorn(a, b, M, reg=reg, numItermax=iters)
    P = cast(torch.Tensor, P)
    OT_entropy =  - torch.sum(P * torch.log(P + 1e-8))



    # OT distance term = <P, M>
    ot_cost = torch.sum(P * M) - reg * OT_entropy
    if verbose:
        print("M min/max/mean:", M.min().item(), M.max().item(), M.mean().item())
        print("P min/max/mean:", P.min().item(), P.max().item(), P.mean().item())
        K = torch.exp(-M / reg)
        print("K min/max/mean:", K.min().item(), K.max().item(), K.mean().item())

    mono_loss = torch.tensor(0.0, device=fake_seq.device, dtype=fake_seq.dtype)
    U: Optional[torch.Tensor] = None

    if monotone:
        # 各 fake frame i が target 側のどの index に対応しているかの重心
        j_idx = torch.arange(N, device=fake_seq.device, dtype=fake_seq.dtype).view(1, N)  # (1, N)
        row_mass = P.sum(dim=1) + 1e-8                                                     # (T,)
        U = torch.sum(P * j_idx, dim=1) / row_mass                                         # (T,)
        mono_loss = torch.relu(U[:-1] - U[1:]).sum()

    P_entropy_loss = torch.tensor(0.0, device=fake_seq.device, dtype=fake_seq.dtype)
    if P_entropy:
        row_mass = P.sum(dim=1, keepdim=True) + 1e-8
        Q = P / row_mass
        P_entropy_loss = -torch.sum(Q * torch.log(Q + 1e-8), dim=1)
        P_entropy_loss = P_entropy_loss.mean()

    
    ot_divergence_loss = torch.tensor(0.0, device=fake_seq.device, dtype=fake_seq.dtype)
    if ot_divergence:
        M_d = torch.cdist(fake_flat, fake_flat, p=2) ** 2
        b_d = torch.ones(T, device=fake_seq.device, dtype=fake_seq.dtype) / T
        P_d = ot.sinkhorn(a, b_d, M_d, reg=reg, numItermax=iters)
        OT_entropy_d =  - torch.sum(P_d * torch.log(P_d + 1e-8))
        ot_divergence_loss = torch.sum(P_d * M_d) - reg * OT_entropy_d
    

        
    mono_cost = monotone_penalty * mono_loss
    entorpy_cost = P_entropy_penalty * P_entropy_loss
    ot_divergence_cost = ot_divergence_penalty * ot_divergence_loss

    total = ot_cost + mono_cost + entorpy_cost + ot_divergence_cost

    if verbose:
        print(
            "sequence_ot terms:",
            "distance=", ot_cost.item(),
            "mono=", mono_cost.item(),
            "entropy=", entorpy_cost.item(),
            "ot_divergence=", ot_divergence_cost.item(),
            "total=", total.item(),
        )

    if return_details:
        details: Dict[str, Any] = {
            "P": P,
            "M": M,
            "U": U,
            "P_d": P_d if ot_divergence else None,
            "M_d": M_d if ot_divergence else None,
            "ot_cost": ot_cost,
            "mono_cost": mono_cost,
            "entropy_cost": entorpy_cost,
            "ot_divergence_cost": ot_divergence_cost,
            "mono_loss": mono_loss,
            "mono_penalty": monotone_penalty,
            "P_entropy_loss": P_entropy_loss,
            "P_entropy_penalty": P_entropy_penalty,
            "ot_divergence_loss": ot_divergence_loss,
            "ot_divergence_penalty": ot_divergence_penalty,
            "valid_idx": valid_idx,
            "a": a,
            "b": b,
            "total": total,
        }
        return total, details

    if return_plan:
        return total, P

    terms: Dict[str, Any] = {
        "ot_cost": ot_cost,
        "mono_cost": mono_cost,
        "entropy_cost": entorpy_cost,
        "ot_divergence_cost": ot_divergence_cost,
        "mono_loss": mono_loss,
        "mono_penalty": monotone_penalty,
        "total": total,
    }
    return total, terms
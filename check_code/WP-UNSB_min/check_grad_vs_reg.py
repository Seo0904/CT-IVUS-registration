"""
reg ごとに OT loss の fake_seq への勾配 (min/max/norm) を確認するスクリプト。
「勾配が 0 以上になるのは reg がいくつから？」を調べる。
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sequence_ot.py を参照できるようにパスを追加
sys.path.insert(0, "/workspace/WP-UNSB_min")
from models.sequence_ot import sequence_ot_loss_torch

DATA_PATH = "/workspace/data/org_data/moving_mnist/mnist_test_seq.npy"
SAVE_DIR  = "/workspace/data/dust_box/grad_vs_reg"
os.makedirs(SAVE_DIR, exist_ok=True)

# Moving MNIST から 1 シーケンス取得: (T, 1, H, W) in [-1, 1]
raw = np.load(DATA_PATH)            # (T, N, H, W)
seq = raw[:, 0, :, :]              # (T, H, W)
seq_t = torch.from_numpy(seq).float().unsqueeze(1) / 127.5 - 1.0  # (T, 1, H, W)
tgt_seq = seq_t.clone()            # ターゲットは固定

# fake_seq は tgt_seq に小さなノイズを乗せたもの（要 requires_grad）
torch.manual_seed(42)
fake_seq_base = tgt_seq + 0.1 * torch.randn_like(tgt_seq)

REG_VALUES = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005]

print(f"{'reg':>8} | {'grad_min':>12} {'grad_max':>12} {'grad_norm':>12} {'grad>=0':>8} | {'ot_cost':>10} {'total':>10}")
print("-" * 85)

results = []
for reg in REG_VALUES:
    fake_seq = fake_seq_base.clone().detach().requires_grad_(True)

    total, terms = sequence_ot_loss_torch(
        fake_seq,
        tgt_seq,
        reg=reg,
        iters=100,
        monotone=True,
        monotone_penalty=1.0,
        P_entropy=False,
        P_entropy_penalty=0.0,
        ot_divergence=False,
        ot_divergence_penalty=0.0,
        normalize=None,
        return_details=False,
        sinkhorn_type = 'sinkhorn_log',
    )

    grad = torch.autograd.grad(total, fake_seq, allow_unused=True)[0]

    if grad is None:
        print(f"{reg:>8.4f} | grad is None — OT loss has no dependency on fake_seq at this reg")
        results.append((reg, None, None, None, None, None, None))
        continue

    g_min  = grad.min().item()
    g_max  = grad.max().item()
    g_norm = grad.norm().item()
    all_nonneg = bool((grad >= 0).all().item())
    ot_cost = terms['ot_cost'].item() if terms is not None else float('nan')

    print(
        f"{reg:>8.4f} | {g_min:>12.4e} {g_max:>12.4e} {g_norm:>12.4e} {str(all_nonneg):>8}"
        f" | {ot_cost:>10.4f} {total.item():>10.4f}"
    )
    results.append((reg, g_min, g_max, g_norm, all_nonneg, ot_cost, total.item()))

# ---- プロット ----
valid = [(r, mn, mx, nm) for r, mn, mx, nm, *_ in results if mn is not None]
if valid:
    regs  = [v[0] for v in valid]
    mins  = [v[1] for v in valid]
    maxs  = [v[2] for v in valid]
    norms = [v[3] for v in valid]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Gradient of OT loss w.r.t. fake_seq vs reg", fontsize=12)

    axes[0].plot(regs, mins, "o-", color="tab:blue")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("reg"); axes[0].set_ylabel("grad min"); axes[0].set_title("Gradient min")
    axes[0].invert_xaxis()

    axes[1].plot(regs, maxs, "o-", color="tab:orange")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("reg"); axes[1].set_ylabel("grad max"); axes[1].set_title("Gradient max")
    axes[1].invert_xaxis()

    axes[2].plot(regs, norms, "o-", color="tab:green")
    axes[2].set_xlabel("reg"); axes[2].set_ylabel("grad norm"); axes[2].set_title("Gradient norm")
    axes[2].invert_xaxis()

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "grad_vs_reg.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nsaved -> {out}")

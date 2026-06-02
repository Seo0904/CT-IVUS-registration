import os
import numpy as np
import torch
import ot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "/workspace/data/org_data/moving_mnist/mnist_test_seq.npy"
SAVE_DIR  = "/workspace/data/dust_box/sinkhorn"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load first sequence: (20, 10000, 64, 64) -> (20, 1, 64, 64) in [-1, 1]
raw   = np.load(DATA_PATH)
seq   = raw[:, 0, :, :]                                    # (T, H, W)
seq_t = torch.from_numpy(seq).float().unsqueeze(1) / 127.5 - 1.0  # (T, 1, H, W)

T = seq_t.shape[0]
flat = seq_t.reshape(T, -1)  # (T, D)

# Cost matrix M: (T, T)  L2^2 distance
M = torch.cdist(flat, flat, p=2) ** 2
# s = M.detach().mean()
# M = M / (s + 1e-8)   # normalize by mean

a = torch.ones(T) / T
b = torch.ones(T) / T

REG_VALUES = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01]
METHODS    = ["sinkhorn", "sinkhorn_log"]

for method in METHODS:
    lines = [f"method: {method}", "-" * 40]
    print(f"\n=== {method} ===")

    n_reg = len(REG_VALUES)
    fig, axes = plt.subplots(1, n_reg, figsize=(3 * n_reg, 3.5))
    fig.suptitle(method, fontsize=13)

    for ax, reg in zip(axes, REG_VALUES):
        P = ot.sinkhorn(a, b, M, reg=reg, numItermax=100, method=method)
        P = torch.as_tensor(P, dtype=torch.float32)
        OT_entropy = -torch.sum(P * torch.log(P + 1e-8))
        ot_cost    = (torch.sum(P * M) - reg * OT_entropy).item()

        line = f"reg={reg:.3f}  ot_cost={ot_cost:.6f}"
        lines.append(line)
        print(line)

        P_np = P.numpy()
        im = ax.imshow(P_np, cmap="viridis", aspect="auto", origin="lower")
        ax.set_title(f"reg={reg}\ncost={ot_cost:.4f}", fontsize=8)
        ax.set_xlabel("target frame")
        ax.set_ylabel("fake frame")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # regごとの個別保存
        fig_s, ax_s = plt.subplots(figsize=(4, 4))
        im_s = ax_s.imshow(P_np, cmap="viridis", aspect="auto", origin="lower")
        ax_s.set_title(f"{method}  reg={reg}  cost={ot_cost:.4f}", fontsize=9)
        ax_s.set_xlabel("target frame")
        ax_s.set_ylabel("fake frame")
        fig_s.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
        plt.tight_layout()
        reg_str = f"{reg:.3f}".replace(".", "p")
        single_path = os.path.join(SAVE_DIR, f"{method}_reg{reg_str}.png")
        fig_s.savefig(single_path, dpi=120)
        plt.close(fig_s)

    plt.tight_layout()
    img_path = os.path.join(SAVE_DIR, f"{method}_P.png")
    fig.savefig(img_path, dpi=120)
    plt.close(fig)
    print(f"saved -> {img_path}")

    out_path = os.path.join(SAVE_DIR, f"{method}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved -> {out_path}")

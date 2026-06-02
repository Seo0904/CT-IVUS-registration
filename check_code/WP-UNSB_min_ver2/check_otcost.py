import os
import numpy as np
import torch
import ot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


DATA_PATH  = "/workspace/data/preprocessed/bspline_transformed/transformed_global.npy"
DATA_PATH2 = "/workspace/data/preprocessed/bspline_transformed/transformed_cloze_global.npy"
SAVE_DIR   = "/workspace/data/dust_box/sinkhorn_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

# (10000, 20, 64, 64): 10000 sequences, each 20 frames
raw  = np.load(DATA_PATH)   # (N, T, H, W)
raw2 = np.load(DATA_PATH2)  # (N, T, H, W)
N_total = raw.shape[0]
print(f"DATA_PATH  shape: {raw.shape}")
print(f"DATA_PATH2 shape: {raw2.shape}")

REG_VALUES = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01]
METHODS    = ["sinkhorn_log"]
N_SEQS     = 1000  # 平均を取るシーケンス数

# method x reg ごとの ot_cost を蓄積して平均
accum  = {m: {r: [] for r in REG_VALUES} for m in METHODS}
last_P = {m: {r: None for r in REG_VALUES} for m in METHODS}  # 可視化用に最後のPを保持

for seq_i in range(N_SEQS):
    # シーケンス i の全 T フレーム: (T, H, W) -> (T, 1, H, W)
    src = raw[seq_i]   # (20, 64, 64)
    tgt = raw2[seq_i]  # (20, 64, 64)

    src_t = torch.from_numpy(src).float().unsqueeze(1) * 2.0 - 1.0  # (T, 1, H, W)  float[0,1]->[-1,1]
    tgt_t = torch.from_numpy(tgt).float().unsqueeze(1) * 2.0 - 1.0  # (T, 1, H, W)

    # DATA_PATH2 側の黒フレーム除去 (学習時と同じ)
    valid_idx = get_valid_frame_idx(tgt_t)
    N_valid = valid_idx.numel()
    tgt_sub = tgt_t[valid_idx]  # (N_valid, 1, H, W)

    T = src_t.shape[0]
    flat     = src_t.reshape(T, -1)         # (T, D)
    tgt_flat = tgt_sub.reshape(N_valid, -1)  # (N_valid, D)

    M = torch.cdist(flat, tgt_flat, p=2) ** 2  # (T, N_valid)

    a = torch.ones(T)       / T
    b = torch.ones(N_valid) / N_valid

    if seq_i % 100 == 0:
        print(f"seq_i={seq_i}/{N_SEQS}  T={T}  N_valid={N_valid}  M mean={M.mean():.4f}")

    last_M = M.numpy()  # 可視化用に最後のMを保持

    for method in METHODS:
        for reg in REG_VALUES:
            P = ot.sinkhorn(a, b, M, reg=reg, numItermax=150, method=method)
            P = torch.as_tensor(P, dtype=torch.float32)
            OT_entropy = -torch.sum(P * torch.log(P + 1e-8))
            ot_cost    = (torch.sum(P * M) - reg * OT_entropy).item()
            accum[method][reg].append(ot_cost)
            last_P[method][reg] = P.numpy()  # 可視化用に上書き保持

# --- M を保存 (最後のシーケンス) ---
M_path = os.path.join(SAVE_DIR, "last_M.npy")
np.save(M_path, last_M)
print(f"\nsaved M -> {M_path}  shape={last_M.shape}  mean={last_M.mean():.4f}  min={last_M.min():.4f}  max={last_M.max():.4f}")

fig_m, ax_m = plt.subplots(figsize=(5, 4))
im_m = ax_m.imshow(last_M, cmap="viridis", aspect="auto", origin="lower")
ax_m.set_title(f"cost matrix M (last seq)  mean={last_M.mean():.2f}", fontsize=9)
ax_m.set_xlabel("target frame (valid)")
ax_m.set_ylabel("source frame")
fig_m.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)
plt.tight_layout()
fig_m.savefig(os.path.join(SAVE_DIR, "last_M.png"), dpi=120)
plt.close(fig_m)

# --- 平均 OT cost を報告・保存 (学習時の loss_SB_P = seq_ot_dist / b に対応) ---
print("\n\n========== AVERAGE OT COST ==========")
for method in METHODS:
    lines = [f"method: {method}", f"N_SEQS={N_SEQS}", "-" * 40]
    print(f"\n=== {method} ===")

    n_reg = len(REG_VALUES)
    fig, axes = plt.subplots(1, n_reg, figsize=(3 * n_reg, 3.5))
    fig.suptitle(f"{method}  (avg over {N_SEQS} seqs)", fontsize=11)

    for ax, reg in zip(axes, REG_VALUES):
        costs = accum[method][reg]
        mean_cost = float(np.mean(costs))
        std_cost  = float(np.std(costs))

        line = f"reg={reg:.3f}  ot_cost mean={mean_cost:.6f}  std={std_cost:.6f}"
        lines.append(line)
        print(line)

        P_np = last_P[method][reg]  # ループ中に保存した最後のシーケンスのP

        im = ax.imshow(P_np, cmap="viridis", aspect="auto", origin="lower")
        ax.set_title(f"reg={reg}\nmean={mean_cost:.4f}", fontsize=8)
        ax.set_xlabel("target frame (valid)")
        ax.set_ylabel("source frame")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig_s, ax_s = plt.subplots(figsize=(4, 4))
        im_s = ax_s.imshow(P_np, cmap="viridis", aspect="auto", origin="lower")
        ax_s.set_title(f"{method}  reg={reg}  mean={mean_cost:.4f}", fontsize=9)
        ax_s.set_xlabel("target frame (valid)")
        ax_s.set_ylabel("source frame")
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

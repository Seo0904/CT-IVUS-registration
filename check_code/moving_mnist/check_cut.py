import numpy as np
import os
import matplotlib.pyplot as plt

def save_sample_grid_NT(npy_path, out_path, n_samples=3, n_frames=20, seed=42):
    rng = np.random.default_rng(seed)
    data = np.load(npy_path)  # (N, T, H, W)
    N, T = data.shape[0], data.shape[1]
    idxs = rng.choice(N, size=n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, n_frames, figsize=(n_frames, n_samples))
    for r, n in enumerate(idxs):
        seq = data[n]  # (T, H, W)
        for t in range(min(n_frames, T)):
            frame = seq[t]
            if frame.ndim == 3 and frame.shape[-1] == 1:
                frame = frame[..., 0]
            ax = axes[r, t] if n_samples > 1 else axes[t]
            ax.imshow(frame, cmap="gray")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("saved grid:", out_path, "picked seq idx:", idxs, "data.shape:", data.shape)

def check_zero_positions_NT(npy_path, n_check=5, seed=42, eps=0):
    """
    eps=0: 完全に0だけを0判定
    eps>0: ほぼ0も0判定（ノイズ混じり用）
    """
    rng = np.random.default_rng(seed)
    data = np.load(npy_path)  # (N, T, H, W)
    N, T = data.shape[0], data.shape[1]
    idxs = rng.choice(N, size=n_check, replace=False)

    print("data.shape:", data.shape)
    for n in idxs:
        seq = data[n]  # (T, H, W)
        # 各フレームが「全部ゼロか」を判定
        if eps == 0:
            is_zero = np.array([np.all(seq[t] == 0) for t in range(T)])
        else:
            is_zero = np.array([np.max(np.abs(seq[t])) <= eps for t in range(T)])

        zero_ts = np.where(is_zero)[0].tolist()
        print(f"seq {n}: zero frames at t={zero_ts}  (count={len(zero_ts)}/{T})")

if __name__ == "__main__":
    NPY_PATH = "/workspace/data/preprocessed/bspline_transformed/transformed_cloze_global.npy"

    # grid
    save_sample_grid_NT(
        NPY_PATH,
        "/workspace/data/dust_box/bspline_cut/cloze_global_grid.png",
        n_samples=5, n_frames=20, seed=42
    )

    # numeric check
    check_zero_positions_NT(
        NPY_PATH,
        n_check=10, seed=42
    )
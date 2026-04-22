import numpy as np
import argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default="/workspace/data/preprocessed/bspline_transformed/transformed_global.npy")
    p.add_argument('--output', type=str, default="/workspace/data/preprocessed/bspline_transformed/transformed_cut_global.npy")
    p.add_argument('--min_remove', type=int, default=1)
    p.add_argument('--max_remove', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)

    data = np.load(args.input)  # (T, N, H, W)
    T, N = data.shape[0], data.shape[1]
    out = data.copy()
    zero_frame = np.zeros_like(out[0, 0])

    for n in range(N):
        n_remove = rng.integers(args.min_remove, args.max_remove + 1)
        n_remove = min(n_remove, T - 1)  # 1枚は残す
        remove_t = rng.choice(T, size=n_remove, replace=False)
        out[remove_t, n] = zero_frame

    np.save(args.output, out)
    print("saved:", args.output, "shape:", out.shape)

if __name__ == "__main__":
    main()
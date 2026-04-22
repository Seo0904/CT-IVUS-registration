import numpy as np
import argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default="/workspace/data/preprocessed/bspline_transformed/transformed_global.npy")
    p.add_argument('--output', type=str, default="/workspace/data/preprocessed/bspline_transformed/transformed_cut_head_tail_global.npy")
    p.add_argument('--min_head', type=int, default=0)
    p.add_argument('--max_head', type=int, default=5)
    p.add_argument('--min_tail', type=int, default=0)
    p.add_argument('--max_tail', type=int, default=5)
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
        head = rng.integers(args.min_head, args.max_head + 1)
        tail = rng.integers(args.min_tail, args.max_tail + 1)
        if head + tail >= T:
            head = 0
            tail = 0

        # 先頭 head 枚をゼロ
        if head > 0:
            out[:head, n] = zero_frame
        # 末尾 tail 枚をゼロ
        if tail > 0:
            out[T-tail:, n] = zero_frame

    np.save(args.output, out)
    print("saved:", args.output, "shape:", out.shape)

if __name__ == "__main__":
    main()
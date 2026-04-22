import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


DEFAULT_SRC = "/workspace/data/org_data/moving_mnist/mnist_test_seq.npy"
DEFAULT_TGT = "/workspace/data/preprocessed/bspline_transformed/transformed_cut.npy"
DEFAULT_OUTPUT = "/workspace/data/dust_box/SSIM/metrics_random.txt"


def load_arrays(src_path: str, tgt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    src = np.load(src_path)
    tgt = np.load(tgt_path)

    if src.shape != tgt.shape:
        raise ValueError(f"Shape mismatch between src {src.shape} and tgt {tgt.shape}")

    # float化
    src = src.astype(np.float32)
    tgt = tgt.astype(np.float32)

    # 0〜255 -> 0〜1 に正規化
    src = src / 255.0
    tgt = tgt / 255.0

    # 念のためclip
    src = np.clip(src, 0.0, 1.0)
    tgt = np.clip(tgt, 0.0, 1.0)

    return src, tgt


def get_seq_range(
    total_seq: int,
    seq_start: Optional[int] = None,
    num_seqs: Optional[int] = None,
    use_last_n_seq: Optional[int] = None,
) -> Tuple[int, int]:
    if use_last_n_seq is not None:
        if use_last_n_seq <= 0:
            raise ValueError("use_last_n_seq must be > 0")
        start = max(0, total_seq - use_last_n_seq)
        end = total_seq
        return start, end

    if seq_start is None:
        seq_start = 0
    if num_seqs is None:
        end = total_seq
    else:
        end = min(total_seq, seq_start + num_seqs)

    if not (0 <= seq_start < total_seq or (seq_start == 0 and total_seq == 0)):
        raise ValueError(f"Invalid seq_start={seq_start} for total_seq={total_seq}")
    if end < seq_start:
        raise ValueError(f"Invalid range: start={seq_start}, end={end}")

    return seq_start, end


def should_skip_black(
    src_img: np.ndarray,
    tgt_img: np.ndarray,
    skip_if_tgt_black: bool = True,
    skip_if_src_black: bool = False,
    skip_if_both_black: bool = False,
) -> bool:
    # 0〜1 前提なので「全黒」は 0
    src_black = np.all(src_img == 0)
    tgt_black = np.all(tgt_img == 0)

    if skip_if_both_black and src_black and tgt_black:
        return True
    if skip_if_tgt_black and tgt_black:
        return True
    if skip_if_src_black and src_black:
        return True

    return False


def compute_metrics_for_subset(
    src: np.ndarray,
    tgt: np.ndarray,
    data_range: float = 1.0,
    seq_start: Optional[int] = None,
    num_seqs: Optional[int] = None,
    use_last_n_seq: Optional[int] = None,
    skip_if_tgt_black: bool = True,
    skip_if_src_black: bool = False,
    skip_if_both_black: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    """
    想定:
      - 4次元: (T, N, H, W)
      - 3次元: (N, H, W)
    """

    ssim_values: List[float] = []
    l2_values: List[float] = []
    l1_values: List[float] = []
    psnr_values: List[float] = []
    per_sample: List[Dict[str, float]] = []

    if src.ndim == 4:
        T, N, H, W = src.shape
        start_seq, end_seq = get_seq_range(
            total_seq=N,
            seq_start=seq_start,
            num_seqs=num_seqs,
            use_last_n_seq=use_last_n_seq,
        )

        flat_idx = 0
        used_idx = 0

        for seq in range(start_seq, end_seq):
            for frame in range(T):
                src_img = src[frame, seq]
                tgt_img = tgt[frame, seq]

                if should_skip_black(
                    src_img,
                    tgt_img,
                    skip_if_tgt_black=skip_if_tgt_black,
                    skip_if_src_black=skip_if_src_black,
                    skip_if_both_black=skip_if_both_black,
                ):
                    flat_idx += 1
                    continue

                diff = src_img - tgt_img
                l2 = float(np.mean(diff ** 2))    # MSE on [0,1]
                l1 = float(np.mean(np.abs(diff))) # MAE on [0,1]
                psnr = float(
                    peak_signal_noise_ratio(tgt_img, src_img, data_range=data_range)
                )
                ssim = float(
                    structural_similarity(
                        tgt_img,
                        src_img,
                        data_range=data_range,
                        gaussian_weights=True,
                        use_sample_covariance=False,
                    )
                )

                path = f"seq{seq}_frame{frame}"

                ssim_values.append(ssim)
                l2_values.append(l2)
                l1_values.append(l1)
                psnr_values.append(psnr)

                per_sample.append(
                    {
                        "index": int(used_idx),
                        "original_flat_index": int(flat_idx),
                        "seq": int(seq),
                        "frame": int(frame),
                        "path": path,
                        "SSIM": ssim,
                        "L2": l2,
                        "L1": l1,
                        "PSNR": psnr,
                    }
                )
                used_idx += 1
                flat_idx += 1

    elif src.ndim == 3:
        N, H, W = src.shape
        start_seq, end_seq = get_seq_range(
            total_seq=N,
            seq_start=seq_start,
            num_seqs=num_seqs,
            use_last_n_seq=use_last_n_seq,
        )

        used_idx = 0

        for i in range(start_seq, end_seq):
            src_img = src[i]
            tgt_img = tgt[i]

            if should_skip_black(
                src_img,
                tgt_img,
                skip_if_tgt_black=skip_if_tgt_black,
                skip_if_src_black=skip_if_src_black,
                skip_if_both_black=skip_if_both_black,
            ):
                continue

            diff = src_img - tgt_img
            l2 = float(np.mean(diff ** 2))
            l1 = float(np.mean(np.abs(diff)))
            psnr = float(
                peak_signal_noise_ratio(tgt_img, src_img, data_range=data_range)
            )
            ssim = float(
                structural_similarity(
                    tgt_img,
                    src_img,
                    data_range=data_range,
                    gaussian_weights=True,
                    use_sample_covariance=False,
                )
            )

            path = f"index{i}"

            ssim_values.append(ssim)
            l2_values.append(l2)
            l1_values.append(l1)
            psnr_values.append(psnr)

            per_sample.append(
                {
                    "index": int(used_idx),
                    "original_flat_index": int(i),
                    "path": path,
                    "SSIM": ssim,
                    "L2": l2,
                    "L1": l1,
                    "PSNR": psnr,
                }
            )
            used_idx += 1

    else:
        raise ValueError(
            f"Unsupported ndim={src.ndim}. Expected 3D (N,H,W) or 4D (T,N,H,W)."
        )

    if len(per_sample) == 0:
        raise ValueError("No valid samples remained after subset selection / black-frame filtering.")

    def stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    metrics_summary = {
        "SSIM": stats(ssim_values),
        "L2": stats(l2_values),
        "L1": stats(l1_values),
        "PSNR": stats(psnr_values),
        "num_samples": len(per_sample),
    }

    return metrics_summary, per_sample


def save_metrics_txt(
    output_path: str,
    metrics_summary: Dict[str, Dict[str, float]],
    per_sample: List[Dict[str, float]],
    epoch: str = "latest",
    subset_desc: str = "",
    black_filter_desc: str = "",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"Test Results - Epoch {epoch}\n")
        f.write("=" * 60 + "\n\n")

        if subset_desc:
            f.write(f"Subset: {subset_desc}\n")
        if black_filter_desc:
            f.write(f"Black filter: {black_filter_desc}\n")
        if subset_desc or black_filter_desc:
            f.write("\n")

        f.write("All metrics were computed on images rescaled to [0,1].\n\n")
        f.write(f"Samples evaluated: {len(per_sample)}\n\n")

        for metric_name in ["SSIM", "L2", "L1", "PSNR"]:
            stats = metrics_summary[metric_name]
            f.write(f"{metric_name}:\n")
            f.write(f"  Mean: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
            f.write(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n\n")

        f.write("Per-sample metrics:\n")
        f.write("=" * 60 + "\n")
        for m in per_sample:
            f.write(f"Image {m['index']} - Path: {m['path']}\n")
            if "seq" in m and "frame" in m:
                f.write(f"  Seq: {m['seq']}, Frame: {m['frame']}\n")
            f.write(f"  OriginalFlatIndex: {m['original_flat_index']}\n")
            f.write(f"  SSIM: {m['SSIM']:.6f}\n")
            f.write(f"  L2: {m['L2']:.6f}\n")
            f.write(f"  L1: {m['L1']:.6f}\n")
            f.write(f"  PSNR: {m['PSNR']:.6f}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "srcとtgtのnpyファイルから、指定subset上で "
            "SSIM / L1 / L2(MSE) / PSNR を [0,1] スケールで計算して保存するスクリプト"
        )
    )
    parser.add_argument("--src", type=str, default=DEFAULT_SRC)
    parser.add_argument("--tgt", type=str, default=DEFAULT_TGT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--epoch", type=str, default="latest")

    parser.add_argument("--seq_start", type=int, default=8000)
    parser.add_argument("--num_seqs", type=int, default=2000)
    parser.add_argument("--use_last_n_seq", type=int, default=2000)

    parser.add_argument("--skip_if_tgt_black", action="store_true")
    parser.add_argument("--skip_if_src_black", action="store_true")
    parser.add_argument("--skip_if_both_black", action="store_true")

    parser.add_argument(
        "--data_range",
        type=float,
        default=1.0,
        help="PSNR/SSIMに使うdata_range（[0,1]画像なら1.0）",
    )

    args = parser.parse_args()

    src, tgt = load_arrays(args.src, args.tgt)

    metrics_summary, per_sample = compute_metrics_for_subset(
        src=src,
        tgt=tgt,
        data_range=args.data_range,
        seq_start=args.seq_start,
        num_seqs=args.num_seqs,
        use_last_n_seq=args.use_last_n_seq,
        skip_if_tgt_black=args.skip_if_tgt_black,
        skip_if_src_black=args.skip_if_src_black,
        skip_if_both_black=args.skip_if_both_black,
    )

    subset_desc = (
        f"seq_start={args.seq_start}, num_seqs={args.num_seqs}, use_last_n_seq={args.use_last_n_seq}"
    )
    black_filter_desc = (
        f"skip_if_tgt_black={args.skip_if_tgt_black}, "
        f"skip_if_src_black={args.skip_if_src_black}, "
        f"skip_if_both_black={args.skip_if_both_black}"
    )

    save_metrics_txt(
        output_path=args.output,
        metrics_summary=metrics_summary,
        per_sample=per_sample,
        epoch=args.epoch,
        subset_desc=subset_desc,
        black_filter_desc=black_filter_desc,
    )

    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "metrics": metrics_summary,
                "subset": {
                    "seq_start": args.seq_start,
                    "num_seqs": args.num_seqs,
                    "use_last_n_seq": args.use_last_n_seq,
                },
                "black_filter": {
                    "skip_if_tgt_black": args.skip_if_tgt_black,
                    "skip_if_src_black": args.skip_if_src_black,
                    "skip_if_both_black": args.skip_if_both_black,
                },
                "per_sample": per_sample,
            },
            f,
            indent=2,
        )

    print(f"Metrics saved to: {args.output}")
    print(f"JSON metrics saved to: {json_path}")
    print(f"Samples evaluated: {len(per_sample)}")


if __name__ == "__main__":
    main()
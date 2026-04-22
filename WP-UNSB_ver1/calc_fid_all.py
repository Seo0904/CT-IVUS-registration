#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

from scipy import linalg


# --------------------------
# 画像ローダ
# --------------------------
def load_npy_as_images(npy_path: str) -> np.ndarray:
    """
    npyを読み込み、(N, H, W) もしくは (N, H, W, C) 相当の画像配列に整形して返す。
    MovingMNISTは (T, N, H, W) or (N, T, H, W) などが多いので、全部フラット化して N枚 にする。
    """
    arr = np.load(npy_path)

    if arr.ndim == 4:
        if arr.shape[-1] in (1, 3) and arr.shape[-3] > 8 and arr.shape[-2] > 8:
            # (N, H, W, C)
            return arr
        else:
            # (T, N, H, W) or (N, T, H, W)
            a, b, h, w = arr.shape
            return arr.reshape(a * b, h, w)

    if arr.ndim == 3:
        # (N, H, W)
        return arr

    if arr.ndim == 5:
        # (T, N, H, W, C) -> (T*N, H, W, C)
        t, n, h, w, c = arr.shape
        return arr.reshape(t * n, h, w, c)

    raise ValueError(f"Unsupported npy shape: {arr.shape}")


def list_png_images(folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = []
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(str(p))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under: {folder}")
    return sorted(paths)


# --------------------------
# 黒画像除外
# --------------------------
black_image_detected = False

def is_black_image(img: np.ndarray, black_threshold: int = 0) -> bool:
    global black_image_detected

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]

    is_black = np.max(img) <= black_threshold

    if is_black and not black_image_detected:
        print("[INFO] Black image detected in dataset")
        black_image_detected = True

    return is_black


def filter_black_images(arr: np.ndarray, black_threshold: int = 0) -> np.ndarray:
    keep = []
    for i in range(arr.shape[0]):
        if not is_black_image(arr[i], black_threshold=black_threshold):
            keep.append(i)

    if len(keep) == 0:
        raise ValueError("All images were filtered out as black images.")

    return arr[keep]


# --------------------------
# サンプリング
# --------------------------
def sample_from_npy(arr: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.choice(n, size=min(k, n), replace=False)
    return [arr[i] for i in idx]


def sample_from_png(paths: List[str], k: int, seed: int, black_threshold: int = None) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(paths)
    idx = rng.choice(n, size=min(k, n), replace=False)

    imgs = []
    for i in idx:
        img = Image.open(paths[i]).convert("L")
        arr = np.array(img)
        if black_threshold is not None and is_black_image(arr, black_threshold):
            continue
        imgs.append(arr)

    return imgs


# --------------------------
# Inception特徴（FID）
# --------------------------
class InceptionFeature(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Identity()
        self.model = model.to(device).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out


def to_inception_input(img: np.ndarray, transform: T.Compose) -> torch.Tensor:
    if img.ndim == 2:
        pil = Image.fromarray(img.astype(np.uint8))
    elif img.ndim == 3:
        if img.shape[-1] == 1:
            pil = Image.fromarray(img[..., 0].astype(np.uint8))
        else:
            pil = Image.fromarray(img.astype(np.uint8))
    else:
        raise ValueError(f"Unsupported image ndim: {img.ndim}")

    pil = pil.convert("RGB")
    return transform(pil)


def get_activations(
    imgs: List[np.ndarray],
    feature_model: InceptionFeature,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    transform = T.Compose(
        [
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(
                mean=Inception_V3_Weights.DEFAULT.transforms().mean,
                std=Inception_V3_Weights.DEFAULT.transforms().std,
            ),
        ]
    )

    feats = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i + batch_size]
        x = torch.stack([to_inception_input(im, transform) for im in batch], dim=0).to(device)
        f = feature_model(x).detach().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def compute_fid(
    imgs1: List[np.ndarray],
    imgs2: List[np.ndarray],
    feature_model: InceptionFeature,
    device: torch.device,
    batch_size: int,
) -> float:
    act1 = get_activations(imgs1, feature_model, device, batch_size=batch_size)
    act2 = get_activations(imgs2, feature_model, device, batch_size=batch_size)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def repeat_fid_npy_vs_npy(arr1, arr2, feature_model, device, batch_size, sample_size, rep, base_seed):
    vals = []
    for r in range(rep):
        imgs1 = sample_from_npy(arr1, sample_size, seed=base_seed + 1000 + r)
        imgs2 = sample_from_npy(arr2, sample_size, seed=base_seed + 2000 + r)
        vals.append(compute_fid(imgs1, imgs2, feature_model, device, batch_size))
    return float(np.mean(vals)), float(np.std(vals)), vals


def repeat_fid_npy_vs_png(arr1, png_paths, feature_model, device, batch_size, sample_size, rep, base_seed, black_threshold_gen=None):
    vals = []
    for r in range(rep):
        imgs1 = sample_from_npy(arr1, sample_size, seed=base_seed + 3000 + r)
        imgs2 = sample_from_png(png_paths, sample_size, seed=base_seed + 4000 + r, black_threshold=black_threshold_gen)
        vals.append(compute_fid(imgs1, imgs2, feature_model, device, batch_size))
    return float(np.mean(vals)), float(np.std(vals)), vals


# --------------------------
# メイン
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_npy", type=str, default="/workspace/data/org_data/moving_mnist/mnist_test_seq.npy")
    parser.add_argument("--tgt_npy", type=str, default="/workspace/data/preprocessed/bspline_transformed/transformed_aligned.npy")
    parser.add_argument("--gen_png_dir", type=str, default="/workspace/data/experiment_result/UNSB/moving-mnist/20260214_184345/test_results/moving_mnist_paired_sb/test_latest/images/fake_5")

    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--rep", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ignore_black_tgt", action="store_true", help="tgt内の黒画像を除外してFIDを計算する")
    parser.add_argument("--black_threshold", type=int, default=0, help="黒画像判定の閾値。0なら完全黒のみ除外")
    parser.add_argument("--ignore_black_gen", action="store_true", help="gen内の黒画像も除外する")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_model = InceptionFeature(device)

    print("[Load] src:", args.src_npy)
    print("[Load] tgt:", args.tgt_npy)
    src_arr = load_npy_as_images(args.src_npy)
    tgt_arr = load_npy_as_images(args.tgt_npy)

    if args.ignore_black_tgt:
        before = len(tgt_arr)
        tgt_arr = filter_black_images(tgt_arr, black_threshold=args.black_threshold)
        after = len(tgt_arr)
        print(f"[Filter] tgt black removal: {before} -> {after}")

    print("[Load] gen:", args.gen_png_dir)
    gen_paths = list_png_images(args.gen_png_dir)

    black_threshold_gen = args.black_threshold if args.ignore_black_gen else None

    # ① src vs src
    fid_1_mean, fid_1_std, _ = repeat_fid_npy_vs_npy(
        src_arr, src_arr, feature_model, device,
        args.batch_size, args.sample_size, args.rep, args.seed + 0
    )

    # ② tgt vs tgt
    fid_2_mean, fid_2_std, _ = repeat_fid_npy_vs_npy(
        tgt_arr, tgt_arr, feature_model, device,
        args.batch_size, args.sample_size, args.rep, args.seed + 10000
    )

    # ③ src vs tgt
    fid_3_mean, fid_3_std, _ = repeat_fid_npy_vs_npy(
        src_arr, tgt_arr, feature_model, device,
        args.batch_size, args.sample_size, args.rep, args.seed + 20000
    )

    # ④ src vs gen
    fid_4_mean, fid_4_std, _ = repeat_fid_npy_vs_png(
        src_arr, gen_paths, feature_model, device,
        args.batch_size, args.sample_size, args.rep, args.seed + 30000,
        black_threshold_gen=black_threshold_gen
    )

    # ⑤ tgt vs gen
    fid_5_mean, fid_5_std, _ = repeat_fid_npy_vs_png(
        tgt_arr, gen_paths, feature_model, device,
        args.batch_size, args.sample_size, args.rep, args.seed + 40000,
        black_threshold_gen=black_threshold_gen
    )

    print("\n===== FID Results =====")
    print(f"1 src vs src (rand split) : {fid_1_mean:.4f} ± {fid_1_std:.4f}  (rep={args.rep})")
    print(f"2 tgt vs tgt (rand split) : {fid_2_mean:.4f} ± {fid_2_std:.4f}  (rep={args.rep})")
    print(f"3 src vs tgt              : {fid_3_mean:.4f} ± {fid_3_std:.4f}  (rep={args.rep})")
    print(f"4 src vs gen              : {fid_4_mean:.4f} ± {fid_4_std:.4f}  (rep={args.rep})")
    print(f"5 tgt vs gen              : {fid_5_mean:.4f} ± {fid_5_std:.4f}  (rep={args.rep})")

    save_dir = args.gen_png_dir.split("/test_results/")[0]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "fid_best_results.txt")

    with open(save_path, "w") as f:
        f.write("===== FID Results =====\n")
        f.write(f"ignore_black_tgt: {args.ignore_black_tgt}\n")
        f.write(f"ignore_black_gen: {args.ignore_black_gen}\n")
        f.write(f"black_threshold : {args.black_threshold}\n")
        f.write(f"1 src vs src (rand split) : {fid_1_mean:.6f} ± {fid_1_std:.6f}  (rep={args.rep})\n")
        f.write(f"2 tgt vs tgt (rand split) : {fid_2_mean:.6f} ± {fid_2_std:.6f}  (rep={args.rep})\n")
        f.write(f"3 src vs tgt              : {fid_3_mean:.6f} ± {fid_3_std:.6f}  (rep={args.rep})\n")
        f.write(f"4 src vs gen              : {fid_4_mean:.6f} ± {fid_4_std:.6f}  (rep={args.rep})\n")
        f.write(f"5 tgt vs gen              : {fid_5_mean:.6f} ± {fid_5_std:.6f}  (rep={args.rep})\n")

    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()
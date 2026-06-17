#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FID計算（npy + png混在対応）
① src内ランダム2群FID
② tgt内ランダム2群FID
③ src vs tgt
④ src vs gen
⑤ tgt vs gen

注意:
- FIDはサンプル数に依存してブレます。可能なら sample_size を増やす＆rep 回繰り返して平均を見るのがおすすめ。
- MNIST系はグレースケールなので Inception 入力用に3ch化＆299x299へリサイズします。
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

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
    # 典型例:
    # (T, N, H, W) or (N, T, H, W) or (N, H, W) or (N, H, W, C)
    if arr.ndim == 4:
        # (A, B, H, W) を (A*B, H, W) に
        # ただし (N, H, W, C) の可能性もあるので最後がCっぽいか判定
        if arr.shape[-1] in (1, 3) and arr.shape[-3] > 8 and arr.shape[-2] > 8:
            # (N, H, W, C)
            return arr
        else:
            # (T, N, H, W) or (N, T, H, W)
            a, b, h, w = arr.shape
            return arr.reshape(a * b, h, w)

    if arr.ndim == 3:
        # (N,H,W)
        return arr

    if arr.ndim == 5:
        # (T,N,H,W,C) など → (T*N,H,W,C)
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


def sample_from_npy(arr: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.choice(n, size=min(k, n), replace=False)
    imgs = []
    for i in idx:
        imgs.append(arr[i])
    return imgs


def sample_from_png(paths: List[str], k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(paths)
    idx = rng.choice(n, size=min(k, n), replace=False)
    imgs = []
    for i in idx:
        img = Image.open(paths[i]).convert("L")  # 生成がグレースケール想定。RGBでもOK
        imgs.append(np.array(img))
    return imgs


# --------------------------
# Inception特徴（FID）
# --------------------------


class InceptionFeature(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)  # ← Trueにする
        model.fc = nn.Identity()  # 2048次元特徴
        self.model = model.to(device).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # aux_logits=True だと学習時は (logits, aux) になる版もあるので保険
        if isinstance(out, tuple):
            out = out[0]
        return out

def to_inception_input(img: np.ndarray, transform: T.Compose) -> torch.Tensor:
    """
    img: (H,W) or (H,W,C) numpy
    -> torch (3,299,299) float
    """
    if img.ndim == 2:
        pil = Image.fromarray(img.astype(np.uint8))
    elif img.ndim == 3:
        # (H,W,C)
        if img.shape[-1] == 1:
            pil = Image.fromarray(img[..., 0].astype(np.uint8))
        else:
            pil = Image.fromarray(img.astype(np.uint8))
    else:
        raise ValueError(f"Unsupported image ndim: {img.ndim}")

    # transformはRGB前提にする
    pil = pil.convert("RGB")
    return transform(pil)


def get_activations(
    imgs: List[np.ndarray],
    feature_model: InceptionFeature,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    # Inception用前処理
    transform = T.Compose(
        [
            T.Resize((299, 299)),
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=Inception_V3_Weights.DEFAULT.transforms().mean,
                        std=Inception_V3_Weights.DEFAULT.transforms().std),
        ]
    )

    feats = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        x = torch.stack([to_inception_input(im, transform) for im in batch], dim=0).to(device)
        f = feature_model(x).detach().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # 数値安定化
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # sqrtmの虚部が出ることがあるので落とす
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


# --------------------------
# メイン
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_npy", type=str, default="/workspace/data/org_data/moving_mnist/mnist_test_seq.npy")
    parser.add_argument("--tgt_npy", type=str, default="/workspace/data/preprocessed/bspline_transformed/transformed_aligned.npy")
    parser.add_argument("--gen_png_dir", type=str, default="/workspace/data/experiment_result/UNSB/moving-mnist/20260214_184345/test_results/moving_mnist_paired_sb/test_latest/images/fake_5")
    parser.add_argument("--sample_size", type=int, default=2000, help="各FID計算に使うサンプル数（多いほど安定だが重い）")
    parser.add_argument("--rep", type=int, default=3, help="①②の内部FIDを何回繰り返して平均するか")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_model = InceptionFeature(device)

    print("[Load] src:", args.src_npy)
    print("[Load] tgt:", args.tgt_npy)
    src_arr = load_npy_as_images(args.src_npy)
    tgt_arr = load_npy_as_images(args.tgt_npy)

    print("[Load] gen:", args.gen_png_dir)
    gen_paths = list_png_images(args.gen_png_dir)

    # ① src内ランダム2群FID（rep回平均）
    fids_1 = []
    for r in range(args.rep):
        s1 = sample_from_npy(src_arr, args.sample_size, seed=args.seed + 1000 + r)
        s2 = sample_from_npy(src_arr, args.sample_size, seed=args.seed + 2000 + r)
        fid = compute_fid(s1, s2, feature_model, device, args.batch_size)
        fids_1.append(fid)
    fid_1_mean = float(np.mean(fids_1))
    fid_1_std = float(np.std(fids_1))

    # ② tgt内ランダム2群FID（rep回平均）
    fids_2 = []
    for r in range(args.rep):
        t1 = sample_from_npy(tgt_arr, args.sample_size, seed=args.seed + 3000 + r)
        t2 = sample_from_npy(tgt_arr, args.sample_size, seed=args.seed + 4000 + r)
        fid = compute_fid(t1, t2, feature_model, device, args.batch_size)
        fids_2.append(fid)
    fid_2_mean = float(np.mean(fids_2))
    fid_2_std = float(np.std(fids_2))

    # ③ src vs tgt
    src_s = sample_from_npy(src_arr, args.sample_size, seed=args.seed + 5000)
    tgt_s = sample_from_npy(tgt_arr, args.sample_size, seed=args.seed + 6000)
    fid_3 = compute_fid(src_s, tgt_s, feature_model, device, args.batch_size)

    # ④ src vs gen
    gen_s = sample_from_png(gen_paths, args.sample_size, seed=args.seed + 7000)
    fid_4 = compute_fid(src_s, gen_s, feature_model, device, args.batch_size)

    # ⑤ tgt vs gen
    fid_5 = compute_fid(tgt_s, gen_s, feature_model, device, args.batch_size)

    print("\n===== FID Results =====")
    print(f"1 src vs src (rand split) : {fid_1_mean:.4f} ± {fid_1_std:.4f}  (rep={args.rep})")
    print(f"2 tgt vs tgt (rand split) : {fid_2_mean:.4f} ± {fid_2_std:.4f}  (rep={args.rep})")
    print(f"3 src vs tgt              : {fid_3:.4f}")
    print(f"4 src vs gen              : {fid_4:.4f}")
    print(f"5 tgt vs gen              : {fid_5:.4f}")
    print("\n(期待関係) 2=5 かつ 3≈4 なら「genがtgt側に寄れている」解釈がしやすいです。")

    save_dir = "/workspace/data/experiment_result/UNSB/moving-mnist/20260214_184345"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "fid_latest_results.txt")

    with open(save_path, "w") as f:
        f.write("===== FID Results =====\n")
        f.write(f"1 src vs src (rand split) : {fid_1_mean:.6f} ± {fid_1_std:.6f}  (rep={args.rep})\n")
        f.write(f"2 tgt vs tgt (rand split) : {fid_2_mean:.6f} ± {fid_2_std:.6f}  (rep={args.rep})\n")
        f.write(f"3 src vs tgt              : {fid_3:.6f}\n")
        f.write(f"4 src vs gen              : {fid_4:.6f}\n")
        f.write(f"5 tgt vs gen              : {fid_5:.6f}\n")
    print(f"\nSaved to: {save_path}")
if __name__ == "__main__":
    main()

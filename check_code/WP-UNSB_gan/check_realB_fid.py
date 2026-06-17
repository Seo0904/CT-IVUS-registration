#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transformed_global.npy を「自分自身」とFID（ランダム2群に分けて比較）して
スコアをprintする簡易チェックスクリプト。

- データ: (N, T, H, W) float [0,1] を想定。全フレームをフラット化して画像群にする。
- 200件で比較（各群200枚）。
"""

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

from scipy import linalg


NPY_PATH = "/workspace/data/preprocessed/bspline_transformed/transformed_global.npy"
SAMPLE_SIZE = 200
BATCH_SIZE = 64
SEED = 0


def to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0 + 1e-6:  # [0,1] スケール
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_sequences(npy_path: str) -> np.ndarray:
    """npyを読み込み (N, T, H, W) の系列配列で返す。float[0,1]想定でuint8に変換。"""
    arr = np.load(npy_path)
    if arr.ndim != 4:
        raise ValueError(f"Expected (N, T, H, W), got: {arr.shape}")
    return to_uint8(arr)


def flatten_sequences(seqs: np.ndarray):
    """(K, T, H, W) -> 画像リスト [(H, W), ...]"""
    k, t, h, w = seqs.shape
    flat = seqs.reshape(k * t, h, w)
    return [flat[i] for i in range(flat.shape[0])]


class InceptionFeature(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        model.fc = nn.Identity()  # 2048次元特徴
        self.model = model.to(device).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out


def to_inception_input(img: np.ndarray, transform: T.Compose) -> torch.Tensor:
    pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    return transform(pil)


def get_activations(imgs, feature_model, device, batch_size=64) -> np.ndarray:
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=Inception_V3_Weights.DEFAULT.transforms().mean,
                    std=Inception_V3_Weights.DEFAULT.transforms().std),
    ])
    feats = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i + batch_size]
        x = torch.stack([to_inception_input(im, transform) for im in batch], dim=0).to(device)
        feats.append(feature_model(x).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def compute_fid(imgs1, imgs2, feature_model, device, batch_size) -> float:
    act1 = get_activations(imgs1, feature_model, device, batch_size)
    act2 = get_activations(imgs2, feature_model, device, batch_size)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Load] {NPY_PATH}")
    seqs = load_sequences(NPY_PATH)
    print(f"[Data] sequences={seqs.shape}, dtype={seqs.dtype}")

    feature_model = InceptionFeature(device)

    # 同じ系列200件同士でFID（同一サンプル -> 理論上 ~0）
    sel = seqs[:SAMPLE_SIZE]
    imgs = flatten_sequences(sel)
    print(f"[Use] {sel.shape[0]} sequences -> {len(imgs)} frames (same set on both sides)")
    fid = compute_fid(imgs, imgs, feature_model, device, BATCH_SIZE)

    print("\n===== FID Result =====")
    print(f"transformed_global self-FID (same {SAMPLE_SIZE} sequences) : {fid:.4f}")


if __name__ == "__main__":
    main()

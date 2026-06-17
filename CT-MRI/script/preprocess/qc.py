"""QC 可視化: 中央3断面の CT/MR オーバレイ と 前後ヒストグラム。"""
from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _center_slices(vol):
    x, y, z = vol.shape
    return vol[:, :, z // 2], vol[:, y // 2, :], vol[x // 2, :, :]


def save_overlay(out_dir, cid, ct_unit, mr_unit, mask):
    """
    ct_unit/mr_unit: [-1,1] の (H,W,D)。2ch CT の場合は広窓ch[0]を使う。
    各断面で CT(gray) と MR(gray) を並べ、MR に mask 輪郭を重ねる。
    """
    os.makedirs(out_dir, exist_ok=True)
    if ct_unit.ndim == 4:
        ct_unit = ct_unit[0]
    views = ["axial", "coronal", "sagittal"]
    ct_s = _center_slices(ct_unit)
    mr_s = _center_slices(mr_unit)
    mk_s = _center_slices(mask.astype(np.float32))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for c, vname in enumerate(views):
        a = axes[0][c]
        a.imshow(np.rot90(ct_s[c]), cmap="gray", vmin=-1, vmax=1)
        a.set_title(f"CT / {vname}", fontsize=9); a.axis("off")
        b = axes[1][c]
        b.imshow(np.rot90(mr_s[c]), cmap="gray", vmin=-1, vmax=1)
        b.contour(np.rot90(mk_s[c]), levels=[0.5], colors="r", linewidths=0.6)
        b.set_title(f"MR (mask outline) / {vname}", fontsize=9); b.axis("off")
    fig.suptitle(f"{cid}  CT(input) vs MR(target), normalized [-1,1]", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{cid}_overlay.png"), dpi=110)
    plt.close(fig)


def save_hist(out_dir, cid, ct_hu_raw, mr_raw, mr_proc, mask):
    """処理前後の分布: CT生HU / MR生 / MR(N4+Nyul+scale後) をマスク内で。"""
    os.makedirs(out_dir, exist_ok=True)
    m = mask > 0.5
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(ct_hu_raw[m], bins=150, range=(-1100, 2200), color="steelblue")
    axes[0].set_title(f"{cid}: CT raw HU (in-mask)"); axes[0].set_xlabel("HU")
    axes[1].hist(mr_raw[m], bins=150, color="indianred")
    axes[1].set_title("MR raw (in-mask)"); axes[1].set_xlabel("intensity")
    axes[2].hist(mr_proc[m], bins=150, range=(-1.05, 1.05), color="seagreen")
    axes[2].set_title("MR after N4+Nyul+scale (in-mask)"); axes[2].set_xlabel("[-1,1]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{cid}_hist.png"), dpi=110)
    plt.close(fig)

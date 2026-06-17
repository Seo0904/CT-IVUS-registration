"""スライド用：B-spline 自由形状変形(FFD)の概念図を生成するスクリプト

生成する図:
  Panel A: 3x3 制御点と変位ベクトル(quiver) + 基準格子(点線)
  Panel B: 変位場に従って歪んだ密な格子(warped mesh)
  Panel C: 元画像 -> 変形後画像 (上に格子を重ねて歪みを可視化)

実データ(mnist_test_seq.npy)があれば1フレームを使い、
無ければ合成画像(数字風 + 規則格子)でデモする。
"""

import os
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import gryds

# bspline.py と同じパラメータ
TRANS_GRID = 3
TRANS_MRANGE = 0.15
SEED = 42
H = W = 64  # 表示用解像度


def make_sample_image():
    """実データがあれば1フレーム、無ければ合成画像を返す (H, W) in [0,1]"""
    candidates = list(Path("/workspace").rglob("mnist_test_seq.npy"))
    if candidates:
        data = np.load(candidates[0])  # (T, N, H, W)
        frame = data[0, 0].astype(np.float64)
        frame = frame / 255.0 if frame.max() > 1 else frame
        # 64x64 にリサイズ(単純な最近傍)
        from numpy import linspace

        ys = linspace(0, frame.shape[0] - 1, H).astype(int)
        xs = linspace(0, frame.shape[1] - 1, W).astype(int)
        return frame[np.ix_(ys, xs)], f"real frame: {candidates[0].name}"

    # 合成: 中央の塊 + 規則格子で歪みを見やすく
    img = np.zeros((H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-(((xx - 32) / 12) ** 2 + ((yy - 28) / 16) ** 2))
    img += blob
    img[(yy % 8 == 0) | (xx % 8 == 0)] = np.maximum(
        img[(yy % 8 == 0) | (xx % 8 == 0)], 0.35
    )
    return np.clip(img, 0, 1), "synthetic demo image"


def deformation_field(bspline, n=64):
    """密な格子に変換を適用し、基準格子と歪んだ格子を返す"""
    grid = gryds.Grid(shape=(n, n))
    warped = grid.transform(bspline)
    # gryds: grid[0]=行(y, 正規化), grid[1]=列(x, 正規化)
    base_y, base_x = grid.grid[0] * (H - 1), grid.grid[1] * (W - 1)
    warp_y, warp_x = warped.grid[0] * (H - 1), warped.grid[1] * (W - 1)
    return (base_x, base_y), (warp_x, warp_y)


def main():
    np.random.seed(SEED)
    random_grid = np.random.uniform(
        -TRANS_MRANGE, TRANS_MRANGE, size=(2, TRANS_GRID, TRANS_GRID)
    )
    bspline = gryds.BSplineTransformation(random_grid)

    img, src_desc = make_sample_image()
    print(f"sample image: {src_desc}")

    # 変形後画像
    interp = gryds.Interpolator(img)
    img_t = np.clip(interp.transform(bspline), 0, 1)

    # 変位場(粗い格子: warped mesh 用)
    (bx, by), (wx, wy) = deformation_field(bspline, n=16)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # ---- Panel A: 制御点 + 変位矢印 ----
    ax = axes[0]
    cp = np.linspace(0, W - 1, TRANS_GRID)
    cy, cx = np.meshgrid(cp, cp, indexing="ij")
    # gryds の変位スケール(正規化 -> ピクセル)。grid[0]=y, grid[1]=x
    dispy = random_grid[0] * (H - 1)
    dispx = random_grid[1] * (W - 1)
    # 基準格子(点線)
    for i in range(TRANS_GRID):
        ax.plot(cx[i, :], cy[i, :], "k--", lw=0.6, alpha=0.4)
        ax.plot(cx[:, i], cy[:, i], "k--", lw=0.6, alpha=0.4)
    ax.scatter(cx, cy, c="tab:blue", s=80, zorder=3, label="control points")
    ax.quiver(
        cx, cy, dispx, dispy,
        angles="xy", scale_units="xy", scale=1,
        color="tab:red", width=0.008, zorder=4, label="displacement",
    )
    ax.set_title("A. 3×3 control points + displacement\n(mrange=0.15)", fontsize=11)
    ax.set_xlim(-8, W + 7)
    ax.set_ylim(H + 7, -8)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    # ---- Panel B: 歪んだ格子(deformation field) ----
    ax = axes[1]
    n = wx.shape[0]
    for i in range(n):
        ax.plot(wx[i, :], wy[i, :], color="tab:green", lw=0.7)
        ax.plot(wx[:, i], wy[:, i], color="tab:green", lw=0.7)
    ax.set_title("B. Warped grid (B-spline deformation field)", fontsize=11)
    ax.set_xlim(-8, W + 7)
    ax.set_ylim(H + 7, -8)
    ax.set_aspect("equal")

    # ---- Panel C: Original -> Transformed ----
    ax = axes[2]
    combo = np.concatenate([img, np.ones((H, 4)), img_t], axis=1)
    ax.imshow(combo, cmap="gray")
    ax.set_title("C. Original  →  Transformed", fontsize=11)
    ax.axis("off")
    ax.text(W * 0.25, -3, "Original", ha="center", fontsize=10)
    ax.text(W * 1.5 + 4, -3, "Transformed", ha="center", fontsize=10)

    plt.suptitle(
        "B-spline Free-Form Deformation (one shared T for all sequences)",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path(__file__).parent / "bspline_concept.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

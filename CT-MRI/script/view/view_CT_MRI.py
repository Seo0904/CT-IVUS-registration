"""
CT-MRI (brain) データの可視化・shape確認スクリプト

/workspace/data/org_data/CT-MRI/brain 以下の各被験者ディレクトリには
  - ct.nii.gz   : CT ボリューム
  - mr.nii.gz   : MR ボリューム
  - mask.nii.gz : マスク
が入っている。

いくつかの被験者について
  - 各 NIfTI の shape / dtype / voxel spacing / 値域を表示
  - CT / MR / mask の代表スライス(axial/coronal/sagittal)を1枚の図にまとめて保存
する。

出力先: /workspace/data/dust_box/CT-MRI
"""

import os
import argparse

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")  # GUI なし環境向け
import matplotlib.pyplot as plt


DATA_ROOT = "/workspace/data/org_data/CT-MRI/brain"
OUT_DIR = "/workspace/data/dust_box/CT-MRI"

# 各被験者で読み込むモダリティ（ファイル名）
MODALITIES = ["ct", "mr", "mask"]


def load_nii(path):
    """NIfTI を読み込んで (numpy配列, nibabelオブジェクト) を返す。"""
    img = nib.load(path)
    data = img.get_fdata()
    return data, img


def print_info(name, data, img):
    """shape や spacing などの基本情報を表示する。"""
    zooms = img.header.get_zooms()
    print(f"  [{name}]")
    print(f"    shape   : {data.shape}")
    print(f"    dtype   : {data.dtype}")
    print(f"    spacing : {tuple(round(float(z), 3) for z in zooms)} (mm)")
    print(f"    min/max : {float(np.nanmin(data)):.3f} / {float(np.nanmax(data)):.3f}")
    print(f"    mean/std: {float(np.nanmean(data)):.3f} / {float(np.nanstd(data)):.3f}")


def get_slices(vol):
    """ボリュームの中央スライス (axial, coronal, sagittal) を返す。"""
    x, y, z = vol.shape
    axial = vol[:, :, z // 2]
    coronal = vol[:, y // 2, :]
    sagittal = vol[x // 2, :, :]
    return axial, coronal, sagittal


def visualize_subject(subject_id, out_dir):
    """1被験者分を可視化して PNG を保存する。"""
    subj_dir = os.path.join(DATA_ROOT, subject_id)
    print(f"=== {subject_id} ===")

    vols = {}
    for mod in MODALITIES:
        path = os.path.join(subj_dir, f"{mod}.nii.gz")
        if not os.path.exists(path):
            print(f"  [{mod}] not found: {path}")
            continue
        data, img = load_nii(path)
        print_info(mod, data, img)
        vols[mod] = data

    if not vols:
        print("  no volume found, skip.")
        return

    mods = list(vols.keys())
    view_names = ["axial", "coronal", "sagittal"]

    fig, axes = plt.subplots(
        len(mods), 3, figsize=(9, 3 * len(mods)), squeeze=False
    )
    for r, mod in enumerate(mods):
        cmap = "gray" if mod != "mask" else "viridis"
        slices = get_slices(vols[mod])
        for c, (sl, vname) in enumerate(zip(slices, view_names)):
            ax = axes[r][c]
            # 表示向きを整える(回転)
            ax.imshow(np.rot90(sl), cmap=cmap)
            ax.set_title(f"{mod} / {vname}", fontsize=9)
            ax.axis("off")

    fig.suptitle(subject_id, fontsize=12)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject_id}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="CT-MRI brain data viewer")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="可視化する被験者ID。省略時は先頭から --num 件を自動選択",
    )
    parser.add_argument(
        "--num", type=int, default=5, help="自動選択時の件数 (default: 5)"
    )
    parser.add_argument("--out", default=OUT_DIR, help="出力先ディレクトリ")
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        all_subjects = sorted(
            d
            for d in os.listdir(DATA_ROOT)
            if os.path.isdir(os.path.join(DATA_ROOT, d)) and d != "overview"
        )
        subjects = all_subjects[: args.num]

    print(f"data root : {DATA_ROOT}")
    print(f"out dir   : {args.out}")
    print(f"subjects  : {subjects}\n")

    for sid in subjects:
        visualize_subject(sid, args.out)


if __name__ == "__main__":
    main()

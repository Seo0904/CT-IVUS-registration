"""
SynthRAD2023 CT->MRI 前処理 (マスク不使用版)。

通常版 run.py はマスク(mask.nii.gz)に依存して
  - CT 背景を空気HUに固定
  - MR の N4 領域 / Nyul 前景 / [-1,1] スケール統計
を決めていた。本スクリプトはマスクを一切読まず、強度ベース(Otsu)の前景推定と
CT の絶対HU窓のみで正規化する。対応の無い生データ(マスク無し)へ適用できる手法に
そろえるための変換。詳細は MASKFREE_CONVERSION_REPORT.md を参照。

本スクリプト固有の追加仕様:
  - 各スライスを 128x128 にリサイズ(アスペクト比保持・レターボックス)してから保存
  - スライスを 1 枚おきに間引く(frame 数を半分: vol[:, :, ::2])
  - 先頭 N volume のみ使用(既定 90)

2パス構成(Nyul の train限定fit を守る):
  Pass1 : align(CT->MR grid) + Otsu前景 + N4(MR) をキャッシュ
  Fit   : train のみで Nyul fit
  Pass2 : CT窓化 / MR(Nyul+前景scale) を確定 -> 128 resize -> frame[::2] -> h5

使い方:
  python run_maskfree.py --region brain --num_volumes 90 --out_size 128
"""
from __future__ import annotations

import os
import argparse
import random
from typing import List, Dict, Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from config import load_config, Config
import pipeline as P


# =============================================================================
# マスク不使用: Otsu 強度前景
# =============================================================================
def otsu_foreground_arr(arr_zyx: np.ndarray) -> np.ndarray:
    """MR 配列(z,y,x)から強度ベースで体(前景)を推定して 0/1 を返す。

    マスク(GT)の代わり。背景(空気/defacingのゼロ詰め)より組織が明るいことを利用し、
    非ゼロ画素に Otsu 閾値を当て、穴埋め + 最大連結成分で体領域を得る。
    """
    nz = arr_zyx[arr_zyx > 0]
    if nz.size < 50:
        return (arr_zyx > 0).astype(np.uint8)
    thr = float(threshold_otsu(nz))
    fg = arr_zyx > thr
    fg = ndi.binary_fill_holes(fg)
    lbl, n = ndi.label(fg)
    if n > 1:
        sizes = ndi.sum(np.ones_like(lbl, dtype=np.float32), lbl, index=range(1, n + 1))
        fg = (lbl == (int(np.argmax(sizes)) + 1))
    return fg.astype(np.uint8)


def otsu_mask_sitk(mr: sitk.Image) -> sitk.Image:
    """N4 用に Otsu 前景を sitk マスク(uint8, mr と同一グリッド)で返す。"""
    arr = sitk.GetArrayFromImage(mr)              # (z,y,x)
    fg = otsu_foreground_arr(arr)
    m = sitk.GetImageFromArray(fg)
    m.CopyInformation(mr)
    return sitk.Cast(m, sitk.sitkUInt8)


# =============================================================================
# case 列挙 / split
# =============================================================================
def list_cases(cfg: Config, num_volumes: int) -> List[str]:
    rdir = cfg.region_dir
    cases = sorted(
        d for d in os.listdir(rdir)
        if os.path.isdir(os.path.join(rdir, d)) and d != "overview"
    )
    return cases[:num_volumes]


def make_split(cases: List[str], cfg: Config) -> Dict[str, str]:
    sp = cfg.split
    rng = random.Random(sp.seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_tr = int(round(n * sp.ratios[0]))
    n_va = int(round(n * sp.ratios[1]))
    assign = {}
    for i, c in enumerate(shuffled):
        assign[c] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return assign


def sitk_affine(img: sitk.Image) -> np.ndarray:
    d = np.array(img.GetDirection()).reshape(3, 3)
    s = np.array(img.GetSpacing())
    o = np.array(img.GetOrigin())
    A = np.eye(4)
    A[:3, :3] = d @ np.diag(s)
    A[:3, 3] = o
    return A


# =============================================================================
# Pass1 : align(CT->MR) + Otsu前景 + N4   (マスク不使用)
# =============================================================================
def pass1(cfg: Config, cases: List[str], cache_dir: str) -> Dict[str, Any]:
    print(f"\n[Pass1] align(CT->MR) + Otsu-fg + N4   ({len(cases)} cases)  [mask-free]")
    meta: Dict[str, Any] = {}
    for i, cid in enumerate(cases):
        cdir = os.path.join(cache_dir, cid)
        # キャッシュ済み(3点とも存在)はN4を再計算しない。pass1出力はsplit非依存なので
        # 既存キャッシュ(別num_volumes実行の流用含む)をそのまま再利用してよい。
        cached = all(os.path.exists(os.path.join(cdir, fn))
                     for fn in ("ct_hu.nii.gz", "mr_n4.nii.gz", "otsu_fg.nii.gz"))
        if cached:
            meta[cid] = {"resampled": None, "size": None, "cached": True}
            print(f"  ({i+1}/{len(cases)}) {cid}  [cache hit, skip N4]")
            continue

        base = os.path.join(cfg.region_dir, cid)
        ct = P.read_image(os.path.join(base, "ct.nii.gz"))
        mr = P.read_image(os.path.join(base, "mr.nii.gz"))

        # CT を MR グリッドへ(マスクは使わない)。背景固定もしない: CTのHUは絶対量で
        # 体外は元から空気(~-1000)。窓化のみで足りる。
        if not P.same_grid(ct, mr, cfg.grid_atol):
            ct = P.resample_to(ct, mr, sitk.sitkLinear, default_value=cfg.bg_ct_hu)
            resampled = True
        else:
            resampled = False

        otsu_mask = otsu_mask_sitk(mr)             # N4 用の強度前景
        mr_n4 = P.n4_correct(mr, otsu_mask, cfg)

        cdir = os.path.join(cache_dir, cid)
        os.makedirs(cdir, exist_ok=True)
        sitk.WriteImage(ct, os.path.join(cdir, "ct_hu.nii.gz"))
        sitk.WriteImage(mr_n4, os.path.join(cdir, "mr_n4.nii.gz"))
        sitk.WriteImage(otsu_mask, os.path.join(cdir, "otsu_fg.nii.gz"))
        meta[cid] = {"resampled": resampled, "size": mr.GetSize()}
        print(f"  ({i+1}/{len(cases)}) {cid}  resampled={resampled} size={mr.GetSize()}")
    return meta


def fit_nyul(cfg: Config, cases: List[str], assign: Dict[str, str],
             cache_dir: str, hist_path: str):
    train_cases = [c for c in cases if assign[c] == "train"]
    rng = random.Random(cfg.split.seed)
    if len(train_cases) > cfg.nyul.max_fit_images:
        train_cases = rng.sample(train_cases, cfg.nyul.max_fit_images)
    print(f"\n[Fit] Nyul on {len(train_cases)} train cases (train-only, Otsu-fg)")
    mr_arrs, masks = [], []
    for cid in train_cases:
        cdir = os.path.join(cache_dir, cid)
        mr_arrs.append(sitk.GetArrayFromImage(P.read_image(os.path.join(cdir, "mr_n4.nii.gz"))))
        masks.append(sitk.GetArrayFromImage(P.read_image(os.path.join(cdir, "otsu_fg.nii.gz"))))
    P.nyul_fit(mr_arrs, masks, cfg, hist_path)
    print(f"  saved Nyul standard histogram -> {hist_path}")


# =============================================================================
# Pass2 : 確定 + 128 resize + frame[::2] + h5(CT/MR のみ)
# =============================================================================
def save_h5(h5_path: str, cid: str, split: str, ct: np.ndarray, mr: np.ndarray,
            affine: np.ndarray):
    import h5py
    with h5py.File(h5_path, "a") as f:
        for grp, data in (("CT", ct), ("MR", mr)):
            if grp not in f:
                f.create_group(grp)
        for grp, data in (("CT", ct), ("MR", mr)):
            g = f[grp]
            if cid in g:
                del g[cid]
            d = g.create_dataset(cid, data=data.astype(np.float32), compression="gzip")
            d.attrs["split"] = split
            d.attrs["affine"] = affine


def pass2(cfg: Config, cases: List[str], assign: Dict[str, str],
          cache_dir: str, hist_path: str, out_dir: str,
          out_size: int, frame_step: int):
    print(f"\n[Pass2] finalize -> resize{out_size} -> frame[::{frame_step}] -> h5(CT/MR)")
    nyul = P.nyul_load(cfg, hist_path)
    h5_path = os.path.join(out_dir, f"{cfg.region}.h5")
    if os.path.exists(h5_path):
        os.remove(h5_path)

    for i, cid in enumerate(cases):
        cdir = os.path.join(cache_dir, cid)
        ct_img = P.read_image(os.path.join(cdir, "ct_hu.nii.gz"))
        mr_img = P.read_image(os.path.join(cdir, "mr_n4.nii.gz"))
        fg_img = P.read_image(os.path.join(cdir, "otsu_fg.nii.gz"))
        ref = mr_img

        ct_hu = np.transpose(sitk.GetArrayFromImage(ct_img), (2, 1, 0))   # (x,y,z)
        mr_n4 = np.transpose(sitk.GetArrayFromImage(mr_img), (2, 1, 0))
        otsu_fg = np.transpose(sitk.GetArrayFromImage(fg_img), (2, 1, 0))

        # --- CT: 絶対HU窓 -> [-1,1] (マスク背景固定なし) ---
        lo, hi = cfg.hu_window
        ct_unit = P.window_to_unit(ct_hu, lo, hi)

        # --- MR: Nyul適用 -> Otsu前景内 [-1,1] (背景=-1) ---
        fg = P.foreground_mask(mr_n4, otsu_fg)           # (fg>0.5)&(mr>0)
        mr_nyul = P.nyul_apply(nyul, mr_n4, fg)
        mr_unit, _ = P.scale_mr_inmask(mr_nyul, fg, cfg)

        # --- 128x128 リサイズ(アスペクト比保持) ---
        ct_unit = P.resize_keep_aspect(ct_unit, (out_size, out_size), order=1, bg=-1.0)
        mr_unit = P.resize_keep_aspect(mr_unit, (out_size, out_size), order=1, bg=-1.0)

        # --- 真っ黒スライス除去: CT/MRどちらかが前景比率<=しきい値 の z をペアごと捨てる ---
        n_z0 = ct_unit.shape[-1]
        n_drop = 0
        if cfg.drop_black_frames:
            keep = P.nonblack_keep_mask(ct_unit, mr_unit, min_ratio=cfg.black_fg_ratio)
            n_drop = int((~keep).sum())
            ct_unit = ct_unit[:, :, keep]
            mr_unit = mr_unit[:, :, keep]

        # --- frame 半引き(1枚使ったら1枚使わない) ---
        ct_unit = ct_unit[:, :, ::frame_step]
        mr_unit = mr_unit[:, :, ::frame_step]

        save_h5(h5_path, cid, assign[cid], ct_unit, mr_unit, sitk_affine(ref))
        print(f"  ({i+1}/{len(cases)}) {cid}  split={assign[cid]} "
              f"ct={ct_unit.shape} mr={mr_unit.shape}  "
              f"frames {n_z0}->drop{n_drop}->[::{frame_step}]={ct_unit.shape[-1]}")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="SynthRAD CT->MRI preprocessing (mask-free)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--region", default="brain", choices=["brain", "pelvis"])
    ap.add_argument("--out_root", default="/workspace/data/preprocessed/synthRAD_maskfree")
    ap.add_argument("--num_volumes", type=int, default=90, help="先頭 N volume のみ使用")
    ap.add_argument("--out_size", type=int, default=128, help="in-plane を NxN にリサイズ")
    ap.add_argument("--frame_step", type=int, default=2, help="スライス間引き間隔(2=半分)")
    ap.add_argument("--skip_pass1", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, {"region": args.region, "out_root": args.out_root})

    out_dir = cfg.out_dir
    cache_dir = os.path.join(out_dir, "_cache")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    cases = list_cases(cfg, args.num_volumes)
    assign = make_split(cases, cfg)
    n_tr = sum(v == "train" for v in assign.values())
    n_va = sum(v == "val" for v in assign.values())
    n_te = sum(v == "test" for v in assign.values())
    print(f"[mask-free] region={cfg.region} volumes={len(cases)} "
          f"split: train={n_tr} val={n_va} test={n_te}")
    print(f"out_dir={out_dir}  out_size={args.out_size}  frame_step={args.frame_step}")

    P.save_json(os.path.join(out_dir, "split.json"), assign)
    P.save_json(os.path.join(out_dir, "config_maskfree.json"), {
        "region": cfg.region, "num_volumes": args.num_volumes,
        "out_size": args.out_size, "frame_step": args.frame_step,
        "mask_free": True, "ct_norm": "abs HU window [-1,1] (no bg fill)",
        "mr_norm": "N4(Otsu) -> Nyul(train-fit, Otsu-fg) -> Otsu-fg minmax[-1,1]",
        "hu_window": list(cfg.hu_window), "nyul": cfg.nyul.__dict__,
        "drop_black_frames": cfg.drop_black_frames,
        "black_fg_ratio": cfg.black_fg_ratio,
    })

    hist_path = os.path.join(out_dir, "nyul_standard_histogram.npy")
    if not args.skip_pass1:
        ginfo = pass1(cfg, cases, cache_dir)
        P.save_json(os.path.join(out_dir, "grid_info.json"), ginfo)
    fit_nyul(cfg, cases, assign, cache_dir, hist_path)
    pass2(cfg, cases, assign, cache_dir, hist_path, out_dir,
          args.out_size, args.frame_step)
    print("\nDone (mask-free).")


if __name__ == "__main__":
    main()

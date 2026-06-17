"""
SynthRAD2023 CT->MRI 前処理パイプライン CLI。

2パス構成(Nyul の train限定fitを正しく分けるため):
  Pass1  : 全症例を 空間整合 + 背景固定 + N4(MR) し、中間結果をキャッシュ。
  Fit    : train 症例のみで Nyul landmark を fit(リーク防止)。
  Pass2  : 全症例で CT窓化[-1,1] / MR(Nyul適用+マスク内[-1,1]) を確定し、
           NIfTI / HDF5 出力、正規化パラメータ(逆変換用)・QC を保存。

使い方例:
  python run.py --config config.yaml --region brain
  python run.py --config config.yaml --region pelvis --two_channel --limit 8
"""
from __future__ import annotations

import os
import argparse
import random
from typing import List, Dict, Any, Tuple

import numpy as np
import SimpleITK as sitk

from config import load_config, Config
import pipeline as P
import qc as QC


# -----------------------------------------------------------------------------
def list_cases(cfg: Config) -> List[str]:
    rdir = cfg.region_dir
    cases = sorted(
        d for d in os.listdir(rdir)
        if os.path.isdir(os.path.join(rdir, d)) and d != "overview"
    )
    if cfg.limit:
        cases = cases[: cfg.limit]
    return cases


def make_split(cases: List[str], cfg: Config) -> Dict[str, str]:
    """case -> 'train'/'val'/'test' の割当を返す。"""
    sp = cfg.split
    assign: Dict[str, str] = {}
    if sp.mode == "explicit":
        s = set(cases)
        for c in sp.train:
            if c in s: assign[c] = "train"
        for c in sp.val:
            if c in s: assign[c] = "val"
        for c in sp.test:
            if c in s: assign[c] = "test"
        for c in cases:
            assign.setdefault(c, "train")
    else:  # ratio
        rng = random.Random(sp.seed)
        shuffled = cases[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_tr = int(round(n * sp.ratios[0]))
        n_va = int(round(n * sp.ratios[1]))
        for i, c in enumerate(shuffled):
            assign[c] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return assign


def sitk_affine(img: sitk.Image) -> np.ndarray:
    """ITK(LPS) の index->physical 4x4 affine。h5 メタデータ用。"""
    d = np.array(img.GetDirection()).reshape(3, 3)
    s = np.array(img.GetSpacing())
    o = np.array(img.GetOrigin())
    A = np.eye(4)
    A[:3, :3] = d @ np.diag(s)
    A[:3, 3] = o
    return A


# -----------------------------------------------------------------------------
def pass1_align_n4(cfg: Config, cases: List[str], cache_dir: str) -> Dict[str, Any]:
    """空間整合 + 背景固定 + N4。キャッシュに ct_hu/mr_n4/mask を保存。"""
    print(f"\n[Pass1] align + bg-fix + N4   ({len(cases)} cases)")
    meta: Dict[str, Any] = {}
    for i, cid in enumerate(cases):
        base = os.path.join(cfg.region_dir, cid)
        ct = P.read_image(os.path.join(base, "ct.nii.gz"))
        mr = P.read_image(os.path.join(base, "mr.nii.gz"))
        mask = P.read_image(os.path.join(base, "mask.nii.gz"))
        mask = P.binarize_mask(mask)

        ct, mr, mask, ginfo = P.align_pair(ct, mr, mask, cfg)

        # 背景固定(CT=bg_ct_hu)。MRは0のまま(統計はマスク内のみなので影響なし)。
        ct_arr = sitk.GetArrayFromImage(ct)          # (z,y,x)
        mask_arr = sitk.GetArrayFromImage(mask)
        ct_arr[mask_arr <= 0] = cfg.bg_ct_hu
        ct = sitk.GetImageFromArray(ct_arr); ct.CopyInformation(mr)

        mr_n4 = P.n4_correct(mr, mask, cfg)

        cdir = os.path.join(cache_dir, cid)
        os.makedirs(cdir, exist_ok=True)
        sitk.WriteImage(ct, os.path.join(cdir, "ct_hu.nii.gz"))
        sitk.WriteImage(mr_n4, os.path.join(cdir, "mr_n4.nii.gz"))
        sitk.WriteImage(mask, os.path.join(cdir, "mask.nii.gz"))
        meta[cid] = ginfo
        print(f"  ({i+1}/{len(cases)}) {cid}  grid={ginfo['grid']} resampled={ginfo['resampled']}")
    return meta


def fit_nyul(cfg: Config, cases: List[str], assign: Dict[str, str],
             cache_dir: str, hist_path: str):
    """train症例のみ(上限 max_fit_images)で Nyul を fit。"""
    train_cases = [c for c in cases if assign[c] == "train"]
    rng = random.Random(cfg.split.seed)
    if len(train_cases) > cfg.nyul.max_fit_images:
        train_cases = rng.sample(train_cases, cfg.nyul.max_fit_images)
    print(f"\n[Fit] Nyul on {len(train_cases)} train cases (train-only, leak-safe)")

    mr_arrs, masks = [], []
    for cid in train_cases:
        cdir = os.path.join(cache_dir, cid)
        mr_arrs.append(sitk.GetArrayFromImage(P.read_image(os.path.join(cdir, "mr_n4.nii.gz"))))
        masks.append(sitk.GetArrayFromImage(P.read_image(os.path.join(cdir, "mask.nii.gz"))))
    P.nyul_fit(mr_arrs, masks, cfg, hist_path)
    print(f"  saved Nyul standard histogram -> {hist_path}")


def pass2_finalize(cfg: Config, cases: List[str], assign: Dict[str, str],
                   cache_dir: str, hist_path: str, out_dir: str):
    """CT窓化 + MR(Nyul+scale) を確定し、出力・パラメータ・QCを保存。"""
    print(f"\n[Pass2] finalize + write   ({len(cases)} cases)")
    nyul = P.nyul_load(cfg, hist_path)
    h5_path = os.path.join(out_dir, f"{cfg.region}.h5")
    if "h5" in cfg.out_formats and os.path.exists(h5_path):
        os.remove(h5_path)

    for i, cid in enumerate(cases):
        cdir = os.path.join(cache_dir, cid)
        ct_img = P.read_image(os.path.join(cdir, "ct_hu.nii.gz"))
        mr_img = P.read_image(os.path.join(cdir, "mr_n4.nii.gz"))
        mask_img = P.read_image(os.path.join(cdir, "mask.nii.gz"))
        ref = mr_img

        # sitk配列(z,y,x) -> 我々の(x,y,z)
        ct_hu = np.transpose(sitk.GetArrayFromImage(ct_img), (2, 1, 0))
        mr_n4 = np.transpose(sitk.GetArrayFromImage(mr_img), (2, 1, 0))
        mask = np.transpose(sitk.GetArrayFromImage(mask_img), (2, 1, 0))

        # --- CT: 固定HU窓 -> [-1,1] (+2ch) ---
        ct_unit, ct_params = P.process_ct(ct_hu, mask, cfg)

        # --- MR: Nyul適用 -> マスク内[-1,1] ---
        # 前景=患者輪郭 AND 正信号(defacingのゼロ詰めを除外)。ゼロ領域は最終的に-1。
        fg = P.foreground_mask(mr_n4, mask)
        mr_nyul = P.nyul_apply(nyul, mr_n4, fg)
        mr_unit, mr_params = P.scale_mr_inmask(mr_nyul, fg, cfg)

        # --- 任意リサイズ(アスペクト比保持) ---
        if cfg.out_size is not None:
            ct_unit = P.resize_keep_aspect(ct_unit, cfg.out_size, order=1, bg=-1.0)
            mr_unit = P.resize_keep_aspect(mr_unit, cfg.out_size, order=1, bg=-1.0)
            mask = P.resize_keep_aspect(mask.astype(np.float32), cfg.out_size, order=0, bg=0.0)

        # --- 真っ黒スライス除去: CT/MRどちらかが全面背景(-1)の z をペアごと捨てる ---
        n_z0 = ct_unit.shape[-1]
        n_drop = 0
        if cfg.drop_black_frames:
            keep = P.nonblack_keep_mask(ct_unit, mr_unit, min_ratio=cfg.black_fg_ratio)
            n_drop = int((~keep).sum())
            ct_unit = ct_unit[..., keep]
            mr_unit = mr_unit[..., keep]
            mask = mask[..., keep]

        # --- 出力 ---
        if "nifti" in cfg.out_formats:
            P.save_case_nifti(out_dir, cid, ct_unit, mr_unit, mask, ref)
        if "h5" in cfg.out_formats:
            P.save_case_h5(h5_path, cid, assign[cid], ct_unit, mr_unit, mask, sitk_affine(ref))

        # --- 逆変換用パラメータ ---
        params = {
            "case": cid, "split": assign[cid], "region": cfg.region,
            "ct": ct_params, "mr": mr_params,
            "nyul_histogram": os.path.basename(hist_path),
            "mr_norm_note": "MRは N4 -> Nyul(train-fit) -> mask内scale の順。逆変換はscale->Nyul逆は非可逆(参考)。",
            "affine_LPS": sitk_affine(ref).tolist(),
            "out_size": cfg.out_size,
            "frames": {
                "orig": int(n_z0), "kept": int(ct_unit.shape[-1]),
                "dropped_black": int(n_drop),
                "drop_black_frames": cfg.drop_black_frames,
                "black_fg_ratio": cfg.black_fg_ratio,
            },
        }
        pdir = os.path.join(out_dir, "params")
        os.makedirs(pdir, exist_ok=True)
        P.save_json(os.path.join(pdir, f"{cid}.json"), params)

        # --- QC ---
        if cfg.qc:
            qdir = os.path.join(out_dir, "qc")
            QC.save_overlay(qdir, cid, ct_unit, mr_unit, mask)
            QC.save_hist(qdir, cid, ct_hu, mr_n4, mr_unit, mask)

        print(f"  ({i+1}/{len(cases)}) {cid}  split={assign[cid]} "
              f"ct={ct_unit.shape} mr={mr_unit.shape}  "
              f"frames {n_z0}->{ct_unit.shape[-1]} (drop {n_drop})")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SynthRAD CT->MRI preprocessing")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--region", default=None, choices=["brain", "pelvis"])
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--two_channel", action="store_true", default=None)
    ap.add_argument("--mr_norm", default=None, choices=["minmax", "zscore"])
    ap.add_argument("--skip_pass1", action="store_true",
                    help="キャッシュ済みなら Pass1(N4)をスキップ")
    args = ap.parse_args()

    overrides = {
        "region": args.region, "out_root": args.out_root, "limit": args.limit,
        "two_channel": True if args.two_channel else None,
        "mr_norm": args.mr_norm,
    }
    cfg = load_config(args.config, overrides)

    out_dir = cfg.out_dir
    cache_dir = os.path.join(out_dir, "_cache")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    cases = list_cases(cfg)
    assign = make_split(cases, cfg)
    n_tr = sum(v == "train" for v in assign.values())
    n_va = sum(v == "val" for v in assign.values())
    n_te = sum(v == "test" for v in assign.values())
    print(f"region={cfg.region}  cases={len(cases)}  split: train={n_tr} val={n_va} test={n_te}")
    print(f"out_dir={out_dir}")

    # メタ保存(split / config)
    P.save_json(os.path.join(out_dir, "split.json"), assign)
    P.save_json(os.path.join(out_dir, "config_snapshot.json"), cfg.to_dict())

    hist_path = os.path.join(out_dir, "nyul_standard_histogram.npy")

    if not args.skip_pass1:
        ginfo = pass1_align_n4(cfg, cases, cache_dir)
        P.save_json(os.path.join(out_dir, "grid_info.json"), ginfo)
    fit_nyul(cfg, cases, assign, cache_dir, hist_path)
    pass2_finalize(cfg, cases, assign, cache_dir, hist_path, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

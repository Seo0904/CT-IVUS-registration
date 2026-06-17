"""
SynthRAD2023 CT->MRI 前処理のコア関数群。

設計の要点(なぜそうするか):
  - CT と MR は画素値の性質が真逆なので正規化思想を分ける。
      CT : 物理量(HU)の絶対値・症例間一貫性が命 → 固定HU窓で線形[-1,1]。
            症例ごとのpercentile/zscoreは禁止(症例間の絶対値整合を壊す)。
      MR : 任意単位でスキャナ/シーケンス依存 → N4(バイアス除去) + Nyul(症例間
            ヒストグラム整合, train限定fitでリーク防止) + マスク内[-1,1]。
  - ペアのボクセル対応を壊さない: MR/CTで別リサンプル・別クロップを絶対にしない。
    空間操作は常に MR グリッドを基準に CT/mask を合わせる。
  - 統計(min/max/mean/std/percentile)は必ずマスク内のみ。背景空気を混ぜない。
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from config import Config


# =============================================================================
# 空間系 (CT・MR 共通)
# =============================================================================
def read_image(path: str, dtype=sitk.sitkFloat32) -> sitk.Image:
    return sitk.ReadImage(path, dtype)


def same_grid(a: sitk.Image, b: sitk.Image, atol: float) -> bool:
    """size/spacing/origin/direction が一致するか(=同一グリッドか)。"""
    if a.GetSize() != b.GetSize():
        return False
    return (
        np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def resample_to(moving: sitk.Image, ref: sitk.Image, interp: int,
                default_value: float) -> sitk.Image:
    """moving を ref グリッドへ resample。interp は線形/最近傍。"""
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(interp)
    r.SetDefaultPixelValue(default_value)
    r.SetTransform(sitk.Transform())  # identity(既に同一空間にある前提)
    return r.Execute(moving)


def align_pair(ct: sitk.Image, mr: sitk.Image, mask: sitk.Image,
               cfg: Config) -> Tuple[sitk.Image, sitk.Image, sitk.Image, Dict[str, Any]]:
    """
    MR を基準グリッドとして CT/mask を合わせる。
    同一グリッドなら assert を通すだけ。ズレていれば(resample_if_mismatch=True時)
    CT=線形 / mask=最近傍 で MR グリッドへ resample する。
    """
    info: Dict[str, Any] = {
        "ct_size_in": ct.GetSize(), "mr_size_in": mr.GetSize(),
        "resampled": False,
    }
    ct_ok = same_grid(ct, mr, cfg.grid_atol)
    mask_ok = same_grid(mask, mr, cfg.grid_atol)

    if ct_ok and mask_ok:
        info["grid"] = "identical"
        return ct, mr, mask, info

    if not cfg.resample_if_mismatch:
        raise AssertionError(
            f"Grid mismatch but resample_if_mismatch=False. "
            f"ct_ok={ct_ok} mask_ok={mask_ok}"
        )

    if not ct_ok:
        ct = resample_to(ct, mr, sitk.sitkLinear, default_value=cfg.bg_ct_hu)
    if not mask_ok:
        mask = resample_to(mask, mr, sitk.sitkNearestNeighbor, default_value=0)
    info["resampled"] = True
    info["grid"] = "resampled_to_mr"
    return ct, mr, mask, info


def binarize_mask(mask: sitk.Image) -> sitk.Image:
    return sitk.Cast(mask > 0, sitk.sitkUInt8)


# =============================================================================
# CT(入力) 強度処理 : 固定HU窓 → 線形[-1,1]
# =============================================================================
def window_to_unit(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """clip(lo,hi) → [-1,1] 線形。背景がloなら-1に張り付く。"""
    x = np.clip(arr, lo, hi)
    return (2.0 * (x - lo) / (hi - lo) - 1.0).astype(np.float32)


def process_ct(ct_arr: np.ndarray, mask_arr: np.ndarray, cfg: Config
               ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    CT を [-1,1] 化して返す。背景(マスク外)は bg_ct_hu に固定してから窓掛け。
    two_channel=True なら (2,H,W,D)、False なら (H,W,D)。
    """
    m = mask_arr > 0.5
    ct_bg = ct_arr.copy()
    ct_bg[~m] = cfg.bg_ct_hu

    lo, hi = cfg.hu_window
    wide = window_to_unit(ct_bg, lo, hi)

    params: Dict[str, Any] = {"hu_window": [lo, hi], "two_channel": cfg.two_channel}
    if cfg.two_channel:
        slo, shi = cfg.soft_window
        soft = window_to_unit(ct_bg, slo, shi)
        out = np.stack([wide, soft], axis=0)  # (2,H,W,D)
        params["soft_window"] = [slo, shi]
    else:
        out = wide  # (H,W,D)
    return out, params


# =============================================================================
# MR(ターゲット) 強度処理 : N4 → Nyul → マスク内[-1,1]/zscore
# =============================================================================
def n4_correct(mr: sitk.Image, mask: sitk.Image, cfg: Config) -> sitk.Image:
    """N4 bias field correction。shrink で高速化し、full-res にバイアス場を適用。"""
    shrink = cfg.n4.shrink_factor
    mr_s = sitk.Shrink(mr, [shrink] * mr.GetDimension())
    mask_s = sitk.Shrink(mask, [shrink] * mask.GetDimension())

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(list(cfg.n4.n_iterations))
    _ = n4.Execute(mr_s, mask_s)

    # full解像度のバイアス場を取り出して原画像へ適用(縮小由来の解像度低下を回避)
    log_bias = n4.GetLogBiasFieldAsImage(mr)
    corrected = mr / sitk.Exp(log_bias)
    return corrected


# ---- Nyul (intensity-normalization v3) --------------------------------------
def _make_nyul(cfg: Config):
    from intensity_normalization.normalizers.population.nyul import NyulNormalizer
    return NyulNormalizer(
        output_min_value=cfg.nyul.output_min,
        output_max_value=cfg.nyul.output_max,
        min_percentile=cfg.nyul.min_percentile,
        max_percentile=cfg.nyul.max_percentile,
        percentile_step=cfg.nyul.percentile_step,
    )


def _np_adapter(arr: np.ndarray):
    from intensity_normalization.adapters.images import NumpyImageAdapter
    return NumpyImageAdapter(arr.astype(np.float32))


def foreground_mask(mr_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    """
    Nyul/scale 用の前景マスク = 患者輪郭マスク AND 正の信号。
    脳の defacing 等でマスク内に「正確に0」のボクセルが多数あり、これを含めると
    low percentile が 0 に重複して Nyul の区分線形補間がゼロ割れ→NaN になる。
    ゼロ信号領域は組織ではないので前景から除外する(最終的に背景=-1へ落とす)。
    """
    return ((mask_arr > 0.5) & (mr_arr > 0.0)).astype(np.float32)


def nyul_fit(train_mr_arrs: List[np.ndarray], train_masks: List[np.ndarray],
             cfg: Config, save_path: str):
    """
    train のみで Nyul landmark を fit し、標準ヒストグラムを保存(逆変換/再現用)。
    リーク防止のため val/test は渡さない。前景はゼロ除外マスクを使う。
    """
    nyul = _make_nyul(cfg)
    images = [_np_adapter(a) for a in train_mr_arrs]
    masks = [_np_adapter(foreground_mask(a, m)) for a, m in zip(train_mr_arrs, train_masks)]
    nyul.fit_population(images, masks)
    nyul.save_standard_histogram(save_path)
    return nyul


def nyul_load(cfg: Config, hist_path: str):
    nyul = _make_nyul(cfg)
    nyul.load_standard_histogram(hist_path)
    return nyul


def nyul_apply(nyul, mr_arr: np.ndarray, fg_arr: np.ndarray) -> np.ndarray:
    """fg_arr は foreground_mask() で作った前景(ゼロ除外)。"""
    out = nyul(_np_adapter(mr_arr), _np_adapter((fg_arr > 0.5).astype(np.float32)))
    return np.asarray(out.get_data(), dtype=np.float32)


def scale_mr_inmask(mr_arr: np.ndarray, mask_arr: np.ndarray, cfg: Config
                    ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    マスク内統計で MR を [-1,1] へ。
      minmax : マスク内 p0.5..p99.5 を [-1,1] に線形(外れ値ロバスト)、背景=-1。
      zscore : マスク内 mean/std で z化 → ±clip_zscore を [-1,1] に線形、背景=-1。
    統計は必ずマスク内のみ。
    """
    m = mask_arr > 0.5
    vals = mr_arr[m]
    out = np.full_like(mr_arr, -1.0, dtype=np.float32)

    if cfg.mr_norm == "minmax":
        lo = float(np.percentile(vals, 0.5))
        hi = float(np.percentile(vals, 99.5))
        if hi <= lo:
            hi = lo + 1e-6
        x = (mr_arr[m] - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0)
        out[m] = (2.0 * x - 1.0).astype(np.float32)
        params = {"mr_norm": "minmax", "lo": lo, "hi": hi,
                  "percentiles": [0.5, 99.5]}
    elif cfg.mr_norm == "zscore":
        mean = float(vals.mean())
        std = float(vals.std() + 1e-8)
        c = cfg.clip_zscore
        z = (mr_arr[m] - mean) / std
        z = np.clip(z, -c, c)
        out[m] = (z / c).astype(np.float32)
        params = {"mr_norm": "zscore", "mean": mean, "std": std, "clip_zscore": c}
    else:
        raise ValueError(f"Unknown mr_norm: {cfg.mr_norm}")
    return out, params


# =============================================================================
# 仕上げ : アスペクト比保持リサイズ(任意, in-plane のみ)
# =============================================================================
def resize_keep_aspect(arr: np.ndarray, target_hw: Tuple[int, int],
                       order: int, bg: float) -> np.ndarray:
    """
    arr: (H,W,D) もしくは (C,H,W,D)。in-plane(H,W)をアスペクト比保持で
    target_hw に収め、不足分を bg でパディング(レターボックス)。z(D)は不変。
    order: 1=線形(画像) / 0=最近傍(ラベル)。
    """
    from scipy.ndimage import zoom

    def _one(vol3d: np.ndarray) -> np.ndarray:
        H, W, D = vol3d.shape
        th, tw = target_hw
        s = min(th / H, tw / W)
        zoomed = zoom(vol3d, (s, s, 1.0), order=order)
        zh, zw, _ = zoomed.shape
        out = np.full((th, tw, D), bg, dtype=vol3d.dtype)
        oh, ow = (th - zh) // 2, (tw - zw) // 2
        out[oh:oh + zh, ow:ow + zw, :] = zoomed
        return out

    if arr.ndim == 3:
        return _one(arr)
    elif arr.ndim == 4:
        return np.stack([_one(arr[c]) for c in range(arr.shape[0])], axis=0)
    raise ValueError(f"Unexpected ndim {arr.ndim}")


# =============================================================================
# 真っ黒スライス除去 (z軸=最終軸)
# =============================================================================
def slice_foreground_counts(arr: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    最終軸(z)スライスごとの「前景画素数」を返す。arr:(H,W,D) または (C,H,W,D)。
    前処理後の配列は背景が厳密に -1 なので、前景 = 値が -1 より大きい画素。
    2ch CT は ch 方向に最大を取ってから数える(どちらかの窓に信号があれば前景)。
    返り値: 形 (D,) の int 配列。
    """
    a = arr if arr.ndim == 3 else arr.max(axis=0)        # (H,W,D)
    fg = a > (-1.0 + eps)
    return fg.reshape(-1, a.shape[-1]).sum(axis=0).astype(np.int64)


def slice_foreground_ratio(arr: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    最終軸(z)スライスごとの「前景比率」= 前景画素数 / 面内画素数(H*W)。
    面内サイズが volume ごとに異なっても比較できるよう、絶対画素数でなく比率を使う。
    """
    inplane = arr.shape[-3] * arr.shape[-2]   # (…,H,W,D) の H*W
    return slice_foreground_counts(arr, eps) / float(inplane)


def nonblack_keep_mask(ct: np.ndarray, mr: np.ndarray,
                       min_ratio: float = 0.001, eps: float = 1e-4) -> np.ndarray:
    """
    z軸で「残す」スライスの bool マスク(長さ D)を返す。
    CT・MR の一方でも前景比率 <= min_ratio(=ほぼ真っ黒)なら、その z を除外する(union)。
    ct/mr は最終軸(z)が一致している前提。min_ratio はサイズ非依存(面内比率)。
    """
    ct_fg = slice_foreground_ratio(ct, eps)
    mr_fg = slice_foreground_ratio(mr, eps)
    return (ct_fg > min_ratio) & (mr_fg > min_ratio)


# =============================================================================
# 入出力
# =============================================================================
def array_to_nifti(arr: np.ndarray, ref: sitk.Image, path: str):
    """
    (H,W,D) numpy(=x,y,z順) を ref の geometry(affine)で nii.gz 保存。
    SimpleITK 配列は (z,y,x) 順なので転置して戻す。
    """
    vol = np.transpose(arr, (2, 1, 0))  # (D,H,W)=(z,y,x)
    img = sitk.GetImageFromArray(vol)
    img.CopyInformation(ref)
    sitk.WriteImage(img, path)


def save_case_nifti(out_dir: str, cid: str, ct: np.ndarray, mr: np.ndarray,
                    mask: np.ndarray, ref: sitk.Image):
    cdir = os.path.join(out_dir, "nifti", cid)
    os.makedirs(cdir, exist_ok=True)
    # 1ch<->2ch を切替えて再実行したときの stale な ct*.nii.gz を掃除
    import glob
    for old in glob.glob(os.path.join(cdir, "ct*.nii.gz")):
        os.remove(old)
    if ct.ndim == 4:  # 2ch
        for c in range(ct.shape[0]):
            array_to_nifti(ct[c], ref, os.path.join(cdir, f"ct_ch{c}.nii.gz"))
    else:
        array_to_nifti(ct, ref, os.path.join(cdir, "ct.nii.gz"))
    array_to_nifti(mr, ref, os.path.join(cdir, "mr.nii.gz"))
    array_to_nifti(mask.astype(np.float32), ref, os.path.join(cdir, "mask.nii.gz"))


def save_case_h5(h5_path: str, cid: str, split: str, ct: np.ndarray,
                 mr: np.ndarray, mask: np.ndarray, affine: np.ndarray):
    import h5py
    with h5py.File(h5_path, "a") as f:
        for grp in ("CT", "MR", "MASK"):
            if grp not in f:
                f.create_group(grp)
        for grp, data in (("CT", ct), ("MR", mr), ("MASK", mask.astype(np.float32))):
            g = f[grp]
            if cid in g:
                del g[cid]
            d = g.create_dataset(cid, data=data, compression="gzip")
            d.attrs["split"] = split
            d.attrs["affine"] = affine


def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

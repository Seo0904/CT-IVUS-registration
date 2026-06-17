"""
[Step 1] SynthRAD2023 前処理パイプライン設計のための実データ確認スクリプト

目的:
  - 固定HU窓を決め打ちしないため、実データのCT HU分布を確認する
  - MR/CT/mask の spacing・配列サイズ・affine が同一グリッドかを検証する
  - マスク内のみで統計を取り、背景空気の影響を排除した分布を見る

出力:
  - 各症例の spacing/size/affine一致チェックを表形式で標準出力＋CSV
  - region ごとに CT HU ヒストグラム(マスク内) を重ね描き
  - region ごとに MR 強度ヒストグラム(マスク内) を重ね描き
  - マスク内 HU percentile サマリ(固定窓の根拠)

これは「確認専用」であり、前処理本体（N4 / Nyul / 正規化）はまだ行わない。
"""

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = "/workspace/data/org_data/synthRAD"
OUT_DIR = "/workspace/data/dust_box/CT-MRI/analysis"


def list_cases(region):
    rdir = os.path.join(DATA_ROOT, region)
    cases = sorted(
        d for d in os.listdir(rdir)
        if os.path.isdir(os.path.join(rdir, d)) and d != "overview"
    )
    return cases


def load_case(region, cid):
    """ct, mr, mask の nibabel オブジェクトを返す。"""
    base = os.path.join(DATA_ROOT, region, cid)
    ct = nib.load(os.path.join(base, "ct.nii.gz"))
    mr = nib.load(os.path.join(base, "mr.nii.gz"))
    mask = nib.load(os.path.join(base, "mask.nii.gz"))
    return ct, mr, mask


def grid_report(cid, ct, mr, mask):
    """同一グリッド検証用の情報を dict で返す。"""
    def aff_close(a, b):
        return bool(np.allclose(a.affine, b.affine, atol=1e-3))

    ct_z = tuple(round(float(z), 3) for z in ct.header.get_zooms()[:3])
    mr_z = tuple(round(float(z), 3) for z in mr.header.get_zooms()[:3])
    return {
        "case": cid,
        "ct_shape": ct.shape,
        "mr_shape": mr.shape,
        "mask_shape": mask.shape,
        "shape_match": (ct.shape == mr.shape == mask.shape),
        "ct_spacing": ct_z,
        "mr_spacing": mr_z,
        "spacing_match": (ct_z == mr_z),
        "affine_ct_mr": aff_close(ct, mr),
        "affine_ct_mask": aff_close(ct, mask),
    }


def masked_values(vol, mask, thr=0.5):
    m = mask > thr
    return vol[m], m


def analyze_region(region, n, out_dir, bins=200):
    cases = list_cases(region)
    sel = cases[:n]
    print(f"\n{'='*70}\nregion = {region}  (sampling {len(sel)} / {len(cases)} cases)\n{'='*70}")

    reports = []
    ct_hist_data = []   # (cid, masked_ct_values)
    mr_hist_data = []
    ct_pcts = []        # percentile rows for window decision

    for cid in sel:
        ct, mr, mask = load_case(region, cid)
        rep = grid_report(cid, ct, mr, mask)
        reports.append(rep)

        ct_arr = ct.get_fdata()
        mr_arr = mr.get_fdata()
        mask_arr = mask.get_fdata()

        ct_in, _ = masked_values(ct_arr, mask_arr)
        mr_in, _ = masked_values(mr_arr, mask_arr)

        ct_hist_data.append((cid, ct_in))
        mr_hist_data.append((cid, mr_in))

        # マスク内 HU percentile（固定窓の妥当性根拠）
        pcs = np.percentile(ct_in, [0.1, 0.5, 1, 5, 50, 95, 99, 99.5, 99.9])
        ct_pcts.append((cid, ct_in.min(), ct_in.max(), pcs))

        print(
            f"[{cid}] shape={rep['ct_shape']} spacing={rep['ct_spacing']} "
            f"shape_match={rep['shape_match']} spacing_match={rep['spacing_match']} "
            f"aff(ct=mr)={rep['affine_ct_mr']} aff(ct=mask)={rep['affine_ct_mask']}"
        )
        print(
            f"     CT in-mask HU: min={ct_in.min():.0f} max={ct_in.max():.0f} "
            f"p1={pcs[2]:.0f} p5={pcs[3]:.0f} p50={pcs[4]:.0f} "
            f"p95={pcs[5]:.0f} p99={pcs[6]:.0f} p99.9={pcs[8]:.0f}  "
            f"| full-CT min/max={ct_arr.min():.0f}/{ct_arr.max():.0f}"
        )
        print(
            f"     MR in-mask   : min={mr_in.min():.1f} max={mr_in.max():.1f} "
            f"p50={np.percentile(mr_in,50):.1f} p99={np.percentile(mr_in,99):.1f}"
        )

    os.makedirs(out_dir, exist_ok=True)

    # ---- CT HU ヒストグラム(マスク内) 重ね描き ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for cid, vals in ct_hist_data:
        axes[0].hist(vals, bins=bins, range=(-1100, 2200), histtype="step",
                     density=True, label=cid, linewidth=1)
    axes[0].axvline(-1000, color="k", ls="--", lw=0.8)
    axes[0].axvline(2000, color="k", ls="--", lw=0.8)
    axes[0].set_title(f"{region}: CT HU (in-mask), dashed=[-1000,2000]")
    axes[0].set_xlabel("HU"); axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)

    # 軟部窓レンジを見るためのズーム
    for cid, vals in ct_hist_data:
        axes[1].hist(vals, bins=bins, range=(-300, 400), histtype="step",
                     density=True, label=cid, linewidth=1)
    axes[1].axvline(-160, color="r", ls="--", lw=0.8)
    axes[1].axvline(240, color="r", ls="--", lw=0.8)
    axes[1].set_title(f"{region}: CT HU zoom (soft-tissue), dashed=[-160,240]")
    axes[1].set_xlabel("HU"); axes[1].set_ylabel("density")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{region}_ct_hu_hist.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  saved -> {p}")

    # ---- MR 強度ヒストグラム(マスク内) ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cid, vals in mr_hist_data:
        hi = np.percentile(vals, 99.5)
        ax.hist(vals, bins=bins, range=(0, hi), histtype="step",
                density=True, label=cid, linewidth=1)
    ax.set_title(f"{region}: MR intensity (in-mask, raw, before N4/Nyul)")
    ax.set_xlabel("intensity"); ax.set_ylabel("density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{region}_mr_hist.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  saved -> {p}")

    # ---- percentile 集計(窓決定の根拠) ----
    arr = np.stack([row[3] for row in ct_pcts])  # (n, 9)
    labels = ["p0.1", "p0.5", "p1", "p5", "p50", "p95", "p99", "p99.5", "p99.9"]
    agg = {lab: (float(arr[:, i].min()), float(arr[:, i].mean()), float(arr[:, i].max()))
           for i, lab in enumerate(labels)}
    print(f"\n  --- {region}: in-mask HU percentile across cases (min / mean / max) ---")
    print(f"  full-CT min over cases: {min(r[1] for r in ct_pcts):.0f}, "
          f"max over cases: {max(r[2] for r in ct_pcts):.0f}")
    for lab in labels:
        mn, me, mx = agg[lab]
        print(f"    {lab:6s}: {mn:8.0f} / {me:8.0f} / {mx:8.0f}")

    return reports, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["brain", "pelvis"])
    ap.add_argument("--n", type=int, default=6, help="症例数/region")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"OUT_DIR   = {args.out}")

    all_reports = []
    for region in args.regions:
        reports, _ = analyze_region(region, args.n, args.out)
        for r in reports:
            r["region"] = region
        all_reports.extend(reports)

    # CSV 出力
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "grid_report.csv")
    cols = ["region", "case", "ct_shape", "mr_shape", "mask_shape", "shape_match",
            "ct_spacing", "mr_spacing", "spacing_match", "affine_ct_mr", "affine_ct_mask"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_reports:
            f.write(",".join('"%s"' % str(r[c]) for c in cols) + "\n")
    print(f"\nsaved grid report -> {csv_path}")


if __name__ == "__main__":
    main()

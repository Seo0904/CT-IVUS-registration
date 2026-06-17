"""
未正規化 M (normalize=none) + balanced OT 前提で
  (1) 勾配がしっかり返る最小の reg(=lmda) を探す
  (2) その reg で輸送行列 P が収束する最小の iter(=max_iter) を探す
実データ(npz)から sequence_ot_loss_geo と同じ手順で M を再構成して実測する。

ソルバ呼び出しは models/sequence_ot.py の geo 版に厳密に合わせる:
  geo_solve(M, reg=reg, a=a, b=b, max_iter=iters, unbalanced=None)
  M = cdist(fake_flat, tgt_flat, p=2) ** 2,  normalize=none,  a,b 一様, balanced
"""
import numpy as np
import torch
from geomloss.ot import solve as geo_solve

NPZ = "/workspace/data/experiment_result/WP-UNSB_ct_mri/synthRAD-brain/20260611_171121/synthRAD_brain_ct2mr_sb_w_otdiv_015_geo_gan/ot_details/train_epoch0040/ot_0024.npz"
dev = "cuda" if torch.cuda.is_available() else "cpu"

d = np.load(NPZ)
fake = torch.tensor(d["fake_B"], device=dev, dtype=torch.float32)   # (T,1,128,128)
realB = torch.tensor(d["real_B"], device=dev, dtype=torch.float32)  # (T,1,128,128)
valid_idx = torch.tensor(d["valid_idx"], device=dev, dtype=torch.long)

T = fake.shape[0]
fake_flat = fake.reshape(T, -1)
tgt_flat = realB[valid_idx].reshape(valid_idx.numel(), -1)
N = tgt_flat.shape[0]
a = torch.ones(T, device=dev) / T
b = torch.ones(N, device=dev) / N

with torch.no_grad():
    M0 = torch.cdist(fake_flat, tgt_flat, p=2) ** 2
print(f"T={T} N={N}  (balanced, normalize=none)")
print(f"M: min={M0.min():.3g} median={M0.median():.6g} mean={M0.mean():.6g} max={M0.max():.6g}\n")


def solve_plan(reg, max_iter, need_grad=False):
    ff = fake_flat.clone().requires_grad_(need_grad)
    M = torch.cdist(ff, tgt_flat, p=2) ** 2
    res = geo_solve(M, reg=reg, a=a, b=b, max_iter=max_iter, unbalanced=None)
    val = res.value
    P = res.plan
    out = {}
    if need_grad:
        g = torch.autograd.grad(val, ff, retain_graph=False)[0]
        out["grad_norm"] = float(g.norm())
        out["grad_nan"] = bool(torch.isnan(g).any() or torch.isinf(g).any())
        row_g = g.norm(dim=1)
        out["frames_with_grad"] = int((row_g > 1e-8 * (row_g.max() + 1e-30)).sum())
    P = P.detach()
    out["value"] = float(val)
    out["mass"] = float(P.sum())
    out["row_viol"] = float((P.sum(1) - a).abs().sum())   # balanced: ->0
    out["col_viol"] = float((P.sum(0) - b).abs().sum())
    out["entropy"] = float(-(P * (P + 1e-30).log()).sum())
    out["Pmax"] = float(P.max())
    out["P"] = P
    return out


print("=" * 96)
print("(1) reg スイープ  (max_iter=5000 固定で収束を担保し、勾配の健全性を見る)")
print(f"{'reg':>8} {'M/reg(med)':>11} {'value':>11} {'grad_norm':>11} {'nan':>4} "
      f"{'frm_grad':>8} {'mass':>7} {'row_viol':>9} {'entropy':>9} {'Pmax':>9}")
med = float(M0.median())
regs = [0.1, 1, 5, 10, 30, 50, 100, 200, 300, 500, 1000]
for reg in regs:
    r = solve_plan(reg, 5000, need_grad=True)
    print(f"{reg:>8.1f} {med/reg:>11.1f} {r['value']:>11.4g} {r['grad_norm']:>11.4g} "
          f"{str(r['grad_nan']):>4} {r['frames_with_grad']:>8d} {r['mass']:>7.4f} "
          f"{r['row_viol']:>9.4f} {r['entropy']:>9.4f} {r['Pmax']:>9.3g}")

print("\n  目安: grad_nan=False かつ grad_norm が有限・frm_grad≈T(=102) かつ entropy>0 が「勾配が返る」状態")


print("\n" + "=" * 96)
print("(2) iter スイープ: reg ごとに P が収束する最小 max_iter を探す")
print("    収束判定: 周辺制約違反 (row_viol+col_viol) < 1e-3  かつ")
print("              直前の iter 設定からの P 相対変化 ||dP||/||P|| < 1e-2")

iters_grid = [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]
reg_cands = [0.1, 1, 5, 10, 30, 50, 100]

print(f"\n{'reg':>6} | " + " ".join(f"{it:>9d}" for it in iters_grid))
print(f"{'':>6} | " + " ".join(f"{'viol+dP':>9}" for _ in iters_grid))
print("-" * 96)

summary = {}
for reg in reg_cands:
    plans = {}
    row = []
    converged_at = None
    prevP = None
    for it in iters_grid:
        r = solve_plan(reg, it, need_grad=False)
        viol = r["row_viol"] + r["col_viol"]
        if prevP is not None:
            dP = float((r["P"] - prevP).norm() / (prevP.norm() + 1e-30))
        else:
            dP = float("inf")
        row.append(f"{viol:>4.1e}/{dP:>3.0e}")
        if converged_at is None and viol < 1e-3 and dP < 1e-2:
            converged_at = it
        prevP = r["P"]
    summary[reg] = converged_at
    print(f"{reg:>6.1f} | " + " ".join(f"{c:>9}" for c in row))

print("\n--- reg ごとの収束最小 iter ---")
for reg in reg_cands:
    c = summary[reg]
    print(f"  reg={reg:>6.1f} -> 最小 iter = {c if c else '>10000 (未収束)'}")

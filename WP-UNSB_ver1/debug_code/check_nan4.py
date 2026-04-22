import numpy as np
import torch
import ot

def _frame_weights(seq, eps=1e-8, thr=1e-4):
    score = seq.abs().mean(dim=(1, 2, 3))
    w = (score > thr).float()
    if w.sum() < 1:
        w[:] = 1.0
    w = w / (w.sum() + eps)
    return w

def _apply_monotone_soft_mask(M, penalty):
    T = M.shape[0]
    i = torch.arange(T).view(T, 1)
    j = torch.arange(T).view(1, T)
    return M + (i > j).float() * penalty

b = np.load('/workspace/data/preprocessed/bspline_transformed/transformed_cut_head_tail.npy')
b2 = b.transpose(1,0,2,3).astype(np.float32)

# head_tail の正規化後: ゼロ画素(0) → [-1, 1] で -1.0
# fake_seqが -1.0 均一 の場合、tgt のゼロフレーム(全画素-1.0)との L2^2 = 0
# → M の「ゼロフレームに対応する列」が全て0

# ゼロフレームが10個のシーケンスで確認
zero_counts = (b2.max(axis=(2,3)) == 0).sum(axis=1)
worst_idx = zero_counts.argmax()
print(f"worst seq index: {worst_idx}, zero_frames: {zero_counts[worst_idx]}")

seq = torch.from_numpy(b2[worst_idx]).unsqueeze(1)
seq_norm = seq / 255.0 * 2.0 - 1.0  # ゼロ画素は -1.0 になる

# 「学習崩壊」ケース: fake が全て -1.0
fake_collapsed = torch.full_like(seq_norm, -1.0)

T = 20
fake_flat = fake_collapsed.reshape(T, -1)
tgt_flat  = seq_norm.reshape(T, -1)
M = torch.cdist(fake_flat, tgt_flat, p=2) ** 2

print(f"\nM統計 (fake=-1.0, tgt=head_tail):")
print(f"  M.median() = {M.median().item():.8f}")
print(f"  M.mean()   = {M.mean().item():.4f}")
print(f"  Mの0要素数 = {(M==0).sum().item()} / {T*T}")
print(f"  Mの0行数   = {(M.sum(dim=1)==0).sum().item()}")
print(f"  Mの0列数   = {(M.sum(dim=0)==0).sum().item()}")

# → tgt のゼロフレーム列はどれも fake=-1.0 なので距離=0
# → 0列数 = ゼロフレーム数 = 10
# → 400要素のうち 20*10=200 が 0
# → medianは400要素の中央値 → 200/400=50% が0 → median=0 or ゼロ近傍

if M.median().item() < 1e-6:
    print(f"\n★ M.median() ≈ 0 → M / (1e-8) で爆発！")
    M_exploded = M / (M.median().item() + 1e-8)
    print(f"  M_exploded.max() = {M_exploded.max().item():.2e}")
    print(f"  → sinkhorn に渡るコスト行列が異常 → NaN")
else:
    print(f"\nM.median() > 0, 正規化は正常")

# monotone_penalty 適用後
M_norm = M / (M.median() + 1e-8)
M_pen = _apply_monotone_soft_mask(M_norm, penalty=50.0)
a = _frame_weights(fake_collapsed)
bw = _frame_weights(seq_norm)
print(f"\na weights (fake=-1.0): nonzero={( a>0).sum()}, all same? {a.max()==a.min()}")
print(f"b weights: nonzero={(bw>0).sum()}")

print("\n--- sinkhorn2 実行 ---")
try:
    cost = ot.sinkhorn2(a, bw, M_pen, reg=0.2, numItermax=50)
    val = float(cost) if hasattr(cost, '__float__') else cost.item()
    print(f"cost = {val}, isnan = {np.isnan(val)}")
except Exception as e:
    print(f"ERROR: {e}")

# ゼロフレームが半分以上のとき必ずmedian=0
print(f"\n--- 結論 ---")
nz = zero_counts[worst_idx]
pct = (T * nz) / (T * T) * 100
print(f"ゼロフレーム {nz}個 → Mの {T*nz}/{T*T} = {pct:.0f}% が 0")
print(f"{'median=0 → NaN確定' if pct >= 50 else 'median>0 → 正常'}")

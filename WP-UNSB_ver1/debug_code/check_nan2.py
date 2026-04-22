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

zero_counts = (b2.max(axis=(2,3)) == 0).sum(axis=1)
print("ゼロフレーム数分布:")
for n in range(11):
    count = (zero_counts == n).sum()
    print(f"  {n}個: {count}シーケンス")

worst_idx = zero_counts.argmax()
seq = torch.from_numpy(b2[worst_idx]).unsqueeze(1)
seq_norm = seq / 255.0 * 2.0 - 1.0

# fake_seqを均一な値（学習初期を模倣）
fake_seq = torch.zeros_like(seq_norm)

T = 20
fake_flat = fake_seq.reshape(T, -1)
tgt_flat  = seq_norm.reshape(T, -1)
M = torch.cdist(fake_flat, tgt_flat, p=2) ** 2

print(f"\nM.median() = {M.median().item():.6f}")
print(f"M ゼロ要素数: {(M==0).sum().item()} / {T*T}")

M_norm = M / (M.median() + 1e-8)
print(f"M_norm.max() = {M_norm.max().item():.2f}")

M_with_penalty = _apply_monotone_soft_mask(M_norm, penalty=50.0)
a = _frame_weights(fake_seq)
bw = _frame_weights(seq_norm)

print(f"a nonzero: {(a>0).sum().item()}, bw nonzero: {(bw>0).sum().item()}")

try:
    cost = ot.sinkhorn2(a, bw, M_with_penalty, reg=0.2, numItermax=50)
    print(f"cost={cost}, isnan={np.isnan(float(cost))}")
except Exception as e:
    print(f"ERROR: {e}")

# medianがゼロになる条件を調べる
# fake_seqが全部同じ値かつtgt_seqに多数ゼロフレームがある場合
print("\n--- M.median()=0 になるケース ---")
# ゼロフレームが半分以上ある → Mの列がゼロになる → median=0
# fake_seqが全部同じなら行も同じ → 行と列で多数0 → median=0

# worst_idxのゼロフレームの位置
seq_maxes = b2[worst_idx].max(axis=(1,2))
print(f"各フレームのmax値: {seq_maxes.tolist()}")
zero_frame_indices = np.where(seq_maxes == 0)[0]
print(f"ゼロフレームのインデックス: {zero_frame_indices.tolist()}")

# ゼロフレームが10個（=半分）あるとき、M行列でゼロ列は10列
# 20x20行列で10列がゼロ → 各行の要素の半分がゼロ → median = 0
# → M / (0 + 1e-8) → M * 1e8 → 爆発!
print(f"\nM.median()={M.median().item():.8f}")
print(f"M / (median + 1e-8) で M が 0 の場合 → max値 = {M.max().item() / 1e-8:.2e}")
print(f"→ cost行列が爆発 → sinkhorn が NaN を返す")

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

# worst_idx のゼロフレームが先頭5+末尾5=10個のシーケンス
zero_counts = (b2.max(axis=(2,3)) == 0).sum(axis=1)
worst_idx = zero_counts.argmax()
seq = torch.from_numpy(b2[worst_idx]).unsqueeze(1)
seq_norm = seq / 255.0 * 2.0 - 1.0

# ★ fake_seqが「ゼロ画素 (-1.0)」の場合
# fake_seq がゼロフレームを大量に含む tgt と一致している
# fake_flat の全フレームが同じ値 (-1.0) → fake_flat rows are identical
# tgt のゼロフレーム(=-1.0) との距離 = 0
fake_seq_minus1 = torch.full_like(seq_norm, -1.0)  # 学習初期の均一出力に近い

T = 20
fake_flat = fake_seq_minus1.reshape(T, -1)
tgt_flat  = seq_norm.reshape(T, -1)
M = torch.cdist(fake_flat, tgt_flat, p=2) ** 2

zero_elements = (M < 1e-6).sum().item()
print(f"M ゼロ要素数 (fake=-1.0, tgt=head_tail): {zero_elements} / {T*T}")
print(f"M.median() = {M.median().item():.6f}")

# ゼロフレームの列を確認
tgt_frame_max = seq_norm.abs().mean(dim=(1,2,3))
print(f"\ntgt フレームのabs mean: {[round(x.item(),4) for x in tgt_frame_max]}")
zero_tgt_frames = (tgt_frame_max < 1e-4).sum().item()
print(f"tgt ゼロフレーム数 (abs.mean < 1e-4): {zero_tgt_frames}")

# fake=-1.0 の場合、ゼロ画素フレーム(=-1.0)との距離=0
# つまりM の列(ゼロフレームに対応)が全て0になる
# → median も 0 に向かう

# -1.0 とゼロフレーム(-1.0)の距離を確認
fake_row = fake_flat[0]   # 全-1.0のベクトル
tgt_col0 = tgt_flat[0]   # ゼロフレーム(=全-1.0)
dist = ((fake_row - tgt_col0)**2).sum()
print(f"\nfake[0] vs tgt[0](ゼロフレーム) L2^2 = {dist.item():.6f}")

tgt_col5 = tgt_flat[5]   # 非ゼロフレーム
dist2 = ((fake_row - tgt_col5)**2).sum()
print(f"fake[0] vs tgt[5](非ゼロフレーム) L2^2 = {dist2.item():.2f}")

print(f"\n結論:")
print(f"  ゼロフレームが{zero_tgt_frames}個あり、fake=-1.0の場合、")
print(f"  M の {zero_tgt_frames}列が全て0")
print(f"  20x20=400要素のうち {T*zero_tgt_frames}個が0")
print(f"  → median が 0 になるには全体の50%以上が0になればよい")
print(f"  → {T*zero_tgt_frames}/{T*T} = {T*zero_tgt_frames/(T*T)*100:.1f}%")
if T*zero_tgt_frames > T*T//2:
    print(f"  → median = 0 になる！ → M/1e-8 で爆発 → NaN！")
else:
    print(f"  → まだmedian>0 だが、学習が進むにつれて危険")

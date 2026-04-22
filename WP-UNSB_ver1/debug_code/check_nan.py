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
    i = torch.arange(T, device=M.device).view(T, 1)
    j = torch.arange(T, device=M.device).view(1, T)
    return M + (i > j).float() * penalty

def sequence_ot_loss_torch(fake_seq, tgt_seq, reg=0.2, iters=50,
                            monotone=True, monotone_penalty=2.0,
                            weight_thr=1e-4, normalize="median"):
    T = fake_seq.shape[0]
    fake_flat = fake_seq.reshape(T, -1)
    tgt_flat  = tgt_seq.reshape(T, -1)
    M = torch.cdist(fake_flat, tgt_flat, p=2) ** 2
    if normalize == "median":
        s = M.detach().median()
        M = M / (s + 1e-8)
    elif normalize == "mean":
        s = M.detach().mean()
        M = M / (s + 1e-8)
    if monotone:
        M = _apply_monotone_soft_mask(M, penalty=monotone_penalty)
    a = _frame_weights(fake_seq, thr=weight_thr)
    b = _frame_weights(tgt_seq, thr=weight_thr)
    print(f"  a nonzero={( a>0).sum().item()}, b nonzero={(b>0).sum().item()}")
    print(f"  M median={M.median().item():.4f}, max={M.max().item():.4f}")
    cost = ot.sinkhorn2(a, b, M, reg=reg, numItermax=iters)
    if not torch.is_tensor(cost):
        cost = torch.tensor(cost, dtype=fake_seq.dtype)
    return cost

b = np.load('/workspace/data/preprocessed/bspline_transformed/transformed_cut_head_tail.npy')
b2 = b.transpose(1,0,2,3).astype(np.float32)

zero_counts = (b2.max(axis=(2,3)) == 0).sum(axis=1)
print(f"ゼロフレーム数の分布: min={zero_counts.min()}, max={zero_counts.max()}, mean={zero_counts.mean():.2f}")
print(f"ゼロフレーム>=10のシーケンス数: {(zero_counts>=10).sum()}")
print(f"ゼロフレーム>=15のシーケンス数: {(zero_counts>=15).sum()}")

# worst case テスト
worst_idx = zero_counts.argmax()
print(f"\n--- worst case: seq[{worst_idx}], zero_frames={zero_counts[worst_idx]} ---")
seq = torch.from_numpy(b2[worst_idx]).unsqueeze(1)
seq_norm = seq / 255.0 * 2.0 - 1.0
w = _frame_weights(seq_norm)
print(f"frame_weights nonzero: {(w>0).sum().item()}/20")
print(f"frame_weights: {[round(x,4) for x in w.tolist()]}")

# fake_seqをゼロ初期化（学習初期を模倣）
fake_seq = torch.zeros_like(seq_norm)
print("\n--- OT loss test (fake=zeros, tgt=worst_case) ---")
try:
    cost = sequence_ot_loss_torch(fake_seq, seq_norm)
    print(f"cost = {cost}")
except Exception as e:
    print(f"ERROR: {e}")

# tgt のみゼロフレームが多い場合
print("\n--- median=0のケース ---")
# 全フレームがゼロの場合
all_zero = torch.full((20,1,64,64), -1.0)  # 正規化後は-1.0 (ゼロ画素)
w2 = _frame_weights(all_zero)
print(f"全ゼロシーケンスのweights nonzero: {(w2>0).sum().item()}, sum={w2.sum().item():.6f}")

# M.median()=0 になるケース
fake2 = torch.zeros(20, 1, 64, 64)
tgt2  = torch.zeros(20, 1, 64, 64)
M_test = torch.cdist(fake2.reshape(20,-1), tgt2.reshape(20,-1), p=2) ** 2
print(f"全ゼロ同士のM: median={M_test.median().item()}, max={M_test.max().item()}")
M_normalized = M_test / (M_test.median() + 1e-8)
print(f"正規化後のM: median={M_normalized.median().item()}, max={M_normalized.max().item()}")

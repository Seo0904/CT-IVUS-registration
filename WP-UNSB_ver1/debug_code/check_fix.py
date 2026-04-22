import numpy as np
import torch
import ot
from typing import cast

# 修正後の関数
def _frame_weights(seq, eps=1e-8, thr=1e-4):
    frame_max = seq.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]
    w = (frame_max > -1.0 + 1e-3).float()
    if w.sum() < 1:
        w[:] = 1.0
    w = w / (w.sum() + eps)
    return w

def _apply_monotone_soft_mask(M, penalty):
    T = M.shape[0]
    i = torch.arange(T).view(T, 1)
    j = torch.arange(T).view(1, T)
    return M + (i > j).float() * penalty

def sequence_ot_loss_torch(fake_seq, tgt_seq, reg=0.2, iters=50,
                            monotone=True, monotone_penalty=50.0,
                            weight_thr=1e-4, normalize="mean"):
    T = fake_seq.shape[0]
    fake_flat = fake_seq.reshape(T, -1)
    tgt_flat  = tgt_seq.reshape(T, -1)
    M = torch.cdist(fake_flat, tgt_flat, p=2) ** 2
    if normalize == "mean":
        s = M.detach().mean()
        M = M / (s + 1e-8)
    elif normalize == "median":
        s = M.detach().median()
        if s < 1e-6:
            s = M.detach().mean()
        M = M / (s + 1e-8)
    elif normalize == "max":
        s = M.detach().max()
        M = M / (s + 1e-8)
    if monotone:
        M = _apply_monotone_soft_mask(M, penalty=monotone_penalty)
    a = _frame_weights(fake_seq, thr=weight_thr)
    b = _frame_weights(tgt_seq, thr=weight_thr)
    cost = ot.sinkhorn2(a, b, M, reg=reg, numItermax=iters)
    if not torch.is_tensor(cost):
        cost = torch.tensor(float(cost), dtype=fake_seq.dtype)
    return cast(torch.Tensor, cost)


b = np.load('/workspace/data/preprocessed/bspline_transformed/transformed_cut_head_tail.npy')
b2 = b.transpose(1,0,2,3).astype(np.float32)
zero_counts = (b2.max(axis=(2,3)) == 0).sum(axis=1)
worst_idx = zero_counts.argmax()
seq = torch.from_numpy(b2[worst_idx]).unsqueeze(1)
seq_norm = seq / 255.0 * 2.0 - 1.0

print("=== 修正後の動作確認 ===\n")
print(f"worst case: seq[{worst_idx}], ゼロフレーム={zero_counts[worst_idx]}個")

# _frame_weights の修正確認
aw = _frame_weights(seq_norm)
print(f"\n[修正後 _frame_weights]")
print(f"  nonzero = {(aw>0).sum().item()}/20 (ゼロフレームを除外できているか)")
print(f"  weights = {[round(x,4) for x in aw.tolist()]}")
# 期待値: t=0〜4, t=15〜19 が 0 (ゼロフレーム), t=5〜14 が 0.1

# OT loss の確認 (fake=-1.0, tgt=head_tail worst case)
fake_collapsed = torch.full_like(seq_norm, -1.0).requires_grad_(True)
print(f"\n[修正後 OT loss] fake=-1.0, normalize='mean'")
try:
    cost = sequence_ot_loss_torch(fake_collapsed, seq_norm.detach(),
                                   normalize="mean", monotone_penalty=50.0)
    print(f"  cost = {cost.item():.6f}, isnan = {cost.isnan().item()}")
    cost.backward()
    g = fake_collapsed.grad
    print(f"  grad isnan = {g.isnan().any().item()}, grad norm = {g.norm().item():.6f}")
except Exception as e:
    print(f"  ERROR: {e}")

# バッチ全体でのテスト (batch_size=8)
print(f"\n[バッチテスト] B=8シーケンス, T=20フレーム")
indices = np.random.choice(len(b2), 8, replace=False)
batch = torch.from_numpy(b2[indices]).unsqueeze(2).float()
batch_norm = batch / 255.0 * 2.0 - 1.0  # (8,20,1,64,64)
fake_batch = torch.randn_like(batch_norm).requires_grad_(True)

total_loss = 0.0
nan_count = 0
for bi in range(8):
    c = sequence_ot_loss_torch(fake_batch[bi].detach(), batch_norm[bi],
                                normalize="mean", monotone_penalty=50.0)
    total_loss += c.item()
    if np.isnan(c.item()):
        nan_count += 1

print(f"  NaN loss count: {nan_count}/8")
print(f"  平均loss: {total_loss/8:.4f}")
print(f"\n✓ NaNなし" if nan_count == 0 else f"\n✗ まだNaN発生")

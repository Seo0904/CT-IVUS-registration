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
worst_idx = zero_counts.argmax()
seq = torch.from_numpy(b2[worst_idx]).unsqueeze(1)
seq_norm = seq / 255.0 * 2.0 - 1.0
fake_collapsed = torch.full_like(seq_norm, -1.0)
T = 20
tgt_flat = seq_norm.reshape(T, -1)

print("=== 各normalize方式の比較 ===\n")

for normalize in ["median", "mean", "max"]:
    fake_t = fake_collapsed.detach().requires_grad_(True)
    fake_flat2 = fake_t.reshape(T, -1)
    M2 = torch.cdist(fake_flat2, tgt_flat, p=2) ** 2

    if normalize == "median":
        s = M2.detach().median()
    elif normalize == "mean":
        s = M2.detach().mean()
    elif normalize == "max":
        s = M2.detach().max()

    M2_norm = M2 / (s + 1e-8)
    M2_pen = _apply_monotone_soft_mask(M2_norm, penalty=50.0)
    a = _frame_weights(fake_collapsed)
    bw = _frame_weights(seq_norm)

    cost2 = ot.sinkhorn2(a, bw, M2_pen, reg=0.2, numItermax=50)
    if not torch.is_tensor(cost2):
        cost2 = torch.tensor(float(cost2), requires_grad=True)

    print(f"[{normalize}] s={s.item():.4f}, cost={cost2.item():.6f}, isnan={np.isnan(cost2.item())}")
    try:
        cost2.backward()
        g = fake_t.grad
        if g is not None:
            print(f"  grad isnan={g.isnan().any().item()}, grad norm={g.norm().item():.4f}")
        else:
            print(f"  grad is None")
    except Exception as e:
        print(f"  backward ERROR: {e}")
    print()

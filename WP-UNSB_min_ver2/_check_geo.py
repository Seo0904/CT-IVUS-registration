import time
import torch
from models.sequence_ot import sequence_ot_loss_geo, sequence_ot_loss_torch

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}")

# Moving MNIST 風: T フレーム, C=1, H=W=64
T, N, C, H, W = 20, 20, 1, 64, 64
reg = 0.1
p = 2
blur = reg ** (1.0 / p)
print(f"reg={reg} -> blur={blur:.6f}  (eps=blur**p={blur**p:.4f})")

def make_data():
    # [-1,1] の系列。tgt は徐々に変化する系列っぽく
    base = torch.randn(N, C, H, W, device=dev)
    tgt = torch.tanh(base + 0.05 * torch.arange(N, device=dev).view(N,1,1,1))
    fake = torch.tanh(torch.randn(T, C, H, W, device=dev))
    return fake, tgt

# ---------- 1) 勾配が流れるか ----------
print("\n=== 1) gradient flow ===")
fake, tgt = make_data()
fake.requires_grad_(True)
total, terms = sequence_ot_loss_geo(
    fake, tgt, reg=reg, p=p, scaling=0.9,
    monotone=True, ot_divergence=True, return_details=False, verbose=False,
)
total.backward()
g = fake.grad
print("total =", total.item())
print("terms:", {k: (v.item() if torch.is_tensor(v) else v) for k,v in terms.items()})
print("grad is None? ", g is None)
print("grad finite?  ", torch.isfinite(g).all().item())
print("grad nonzero? ", (g.abs().sum().item() > 0), " |grad|_mean =", g.abs().mean().item())

# ---------- 2) scaling で収束しているか (marginal 制約 & POT 一致) ----------
print("\n=== 2) convergence vs scaling ===")
fake2, tgt2 = make_data()
a = torch.ones(T, device=dev)/T
b = torch.ones(N, device=dev)/N
for sc in [0.3, 0.5, 0.7, 0.9, 0.95]:
    _, det = sequence_ot_loss_geo(
        fake2, tgt2, reg=reg, p=p, scaling=sc,
        monotone=False, ot_divergence=False, return_details=True,
    )
    P = det["P"]
    row_err = (P.sum(1) - a).abs().max().item()   # source marginal
    col_err = (P.sum(0) - b).abs().max().item()   # target marginal
    print(f"scaling={sc:<4}: ot_cost={det['ot_cost'].item():.6f}  "
          f"row_marg_err={row_err:.2e}  col_marg_err={col_err:.2e}")

# POT 参照値
_, detP = sequence_ot_loss_torch(
    fake2, tgt2, reg=reg, iters=200, sinkhorn_type='sinkhorn',
    monotone=False, ot_divergence=False, return_details=True,
)
print(f"POT(ref)    : ot_cost={detP['ot_cost'].item():.6f}")

# ---------- 3) 1 seq あたりの時間 ----------
print("\n=== 3) time per sequence (fwd+bwd) ===")
def bench(fn, label, iters=30):
    # warmup
    for _ in range(3):
        f, t = make_data(); f.requires_grad_(True)
        out, _ = fn(f, t); out.backward()
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        f, t = make_data(); f.requires_grad_(True)
        out, _ = fn(f, t); out.backward()
    if dev == "cuda": torch.cuda.synchronize()
    dt = (time.time()-t0)/iters*1000
    print(f"{label:<28}: {dt:7.2f} ms/seq")

bench(lambda f,t: sequence_ot_loss_geo(f, t, reg=reg, p=p, scaling=0.9,
        monotone=True, ot_divergence=True), "geo(scaling=0.9,div=on)")
bench(lambda f,t: sequence_ot_loss_geo(f, t, reg=reg, p=p, scaling=0.5,
        monotone=True, ot_divergence=True), "geo(scaling=0.5,div=on)")
bench(lambda f,t: sequence_ot_loss_geo(f, t, reg=reg, p=p, scaling=0.9,
        monotone=True, ot_divergence=False), "geo(scaling=0.9,div=off)")
bench(lambda f,t: sequence_ot_loss_torch(f, t, reg=reg, iters=200,
        monotone=True, ot_divergence=True), "POT(iters=200,div=on)")

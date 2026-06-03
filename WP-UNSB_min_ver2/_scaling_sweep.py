import math, time
import torch
from models.sequence_ot import sequence_ot_loss_geo, sequence_ot_loss_torch

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
T, N, C, H, W = 20, 20, 1, 64, 64
reg = 0.1
p = 2

base = torch.randn(N, C, H, W, device=dev)
tgt  = torch.tanh(base + 0.05 * torch.arange(N, device=dev).view(N,1,1,1))
fake = torch.tanh(torch.randn(T, C, H, W, device=dev))
a = torch.ones(T, device=dev)/T
b = torch.ones(N, device=dev)/N

def col_err(P):   # target 側が受け取る量 P.sum(0) と b=1/N のズレ
    return (P.sum(0) - b).abs().max().item()
def row_err(P):
    return (P.sum(1) - a).abs().max().item()

# ---- POT 基準: numItermax を変えて target 周辺誤差 ----
print("=== POT reference (method='sinkhorn') ===")
pot_targets = {}
for it in [200, 1000, 5000, 10000]:
    _, det = sequence_ot_loss_torch(fake, tgt, reg=reg, iters=it,
                                    sinkhorn_type='sinkhorn',
                                    monotone=False, ot_divergence=False,
                                    return_details=True)
    pot_targets[it] = col_err(det["P"])
    print(f"  numItermax={it:<6}: col_marg_err={pot_targets[it]:.3e}  ot_cost={det['ot_cost'].item():.6f}")
ref = pot_targets[10000]
print(f"  -> 基準 (numItermax=10000): col_marg_err = {ref:.3e}")

# ---- geomloss: scaling を細かくスイープ ----
print("\n=== geomloss scaling sweep ===")
# おおよその段数 n ~ log(blur/diameter)/log(scaling) を併記（diameter は粗い推定）
M = (torch.cdist(fake.reshape(T,-1), tgt.reshape(N,-1))**2)
M = M / (M.mean()+1e-8)
diam = math.sqrt(M.max().item())   # 正規化後コストの直径スケール(σ単位の目安)
blur = reg**(1.0/p)

def bench_one(sc, iters=30):
    for _ in range(3):
        _, d = sequence_ot_loss_geo(fake, tgt, reg=reg, p=p, scaling=sc,
                                    monotone=False, ot_divergence=False, return_details=True)
    if dev=="cuda": torch.cuda.synchronize()
    t0=time.time()
    for _ in range(iters):
        _, d = sequence_ot_loss_geo(fake, tgt, reg=reg, p=p, scaling=sc,
                                    monotone=False, ot_divergence=False, return_details=True)
    if dev=="cuda": torch.cuda.synchronize()
    return d, (time.time()-t0)/iters*1000

print(f"  (diameter目安={diam:.3f}, blur={blur:.3f})")
print(f"  {'scaling':>8} {'~n_steps':>9} {'col_err':>11} {'row_err':>11} {'ms':>7}")
for sc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    d, ms = bench_one(sc)
    nstep = max(1, math.ceil(math.log(blur/max(diam,blur+1e-9))/math.log(sc))) if sc < 1 else float('inf')
    ce, re = col_err(d["P"]), row_err(d["P"])
    flag = " <= POT10000" if ce <= ref else ""
    print(f"  {sc:>8} {nstep:>9} {ce:>11.3e} {re:>11.3e} {ms:>7.2f}{flag}")

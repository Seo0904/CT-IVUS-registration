import time
import torch
from models.sequence_ot import sequence_ot_loss_geo, sequence_ot_loss_torch

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
T, N, C, H, W = 20, 20, 1, 64, 64
p = 2
base = torch.randn(N, C, H, W, device=dev)
tgt  = torch.tanh(base + 0.05 * torch.arange(N, device=dev).view(N,1,1,1))
fake = torch.tanh(torch.randn(T, C, H, W, device=dev))
b = torch.ones(N, device=dev)/N
def col_err(P): return (P.sum(0) - b).abs().max().item()

for reg in [0.02, 0.01, 0.005]:
    blur = reg**(1.0/p)
    print(f"\n########## reg={reg} (blur={blur:.4f}) ##########")
    print(" -- POT method='sinkhorn_log' --")
    refs={}
    for it in [50, 200, 1000, 5000, 10000]:
        _, det = sequence_ot_loss_torch(fake, tgt, reg=reg, iters=it,
                                        sinkhorn_type='sinkhorn_log',
                                        monotone=False, ot_divergence=False, return_details=True)
        refs[it]=col_err(det["P"]); print(f"   iters={it:<6}: col_err={refs[it]:.3e}")
    ref=refs[10000]
    print(f" -- geomloss (target=POT_log@10000={ref:.3e}) --")
    found=None
    for sc in [0.3,0.5,0.7,0.8,0.9,0.95,0.99]:
        for _ in range(2):
            _, d = sequence_ot_loss_geo(fake, tgt, reg=reg, p=p, scaling=sc,
                                        monotone=False, ot_divergence=False, return_details=True)
        if dev=="cuda": torch.cuda.synchronize()
        t0=time.time()
        for _ in range(20):
            _, d = sequence_ot_loss_geo(fake, tgt, reg=reg, p=p, scaling=sc,
                                        monotone=False, ot_divergence=False, return_details=True)
        if dev=="cuda": torch.cuda.synchronize()
        ms=(time.time()-t0)/20*1000; ce=col_err(d["P"]); ok=ce<=ref
        if ok and found is None: found=sc
        print(f"   scaling={sc:<5}: col_err={ce:.3e} {ms:6.2f}ms {'<= ref' if ok else ''}")
    print(f"   => 最小 scaling: {found}")

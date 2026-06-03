import time
import torch
from models.sequence_ot import sequence_ot_loss_geo, sequence_ot_loss_torch

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
T, C, H, W = 20, 1, 64, 64       # fake は常に 20 フレーム
reg = 0.1
p = 2

def col_err(P, N):
    b = torch.ones(N, device=P.device)/N
    return (P.sum(0) - b).abs().max().item()
def row_err(P, T):
    a = torch.ones(T, device=P.device)/T
    return (P.sum(1) - a).abs().max().item()

for N in [15, 12, 8, 5]:   # tgt 有効フレーム数（fake=20 と不一致）
    print(f"\n########## reg={reg}  T(fake)={T}  N(tgt)={N} ##########")
    fake = torch.tanh(torch.randn(T, C, H, W, device=dev))
    tgt  = torch.tanh(torch.randn(N, C, H, W, device=dev))  # 全フレーム valid

    # POT 参照（plain と log 両方）
    for method in ['sinkhorn', 'sinkhorn_log']:
        for it in [200, 10000]:
            _, det = sequence_ot_loss_torch(fake, tgt, reg=reg, iters=it,
                                            sinkhorn_type=method,
                                            monotone=False, ot_divergence=False, return_details=True)
            if it == 10000 and method == 'sinkhorn':
                ref = col_err(det["P"], N)
            print(f"  POT[{method:<12} it={it:<6}]: col_err={col_err(det['P'],N):.3e} row_err={row_err(det['P'],T):.3e}")
    print(f"  -- geomloss (target = POT(plain)@10000 col_err={ref:.3e}) --")
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
        ms=(time.time()-t0)/20*1000
        ce=col_err(d["P"],N); re=row_err(d["P"],T); ok=ce<=ref
        if ok and found is None: found=sc
        print(f"   scaling={sc:<5}: col_err={ce:.3e} row_err={re:.3e} {ms:6.2f}ms {'<= ref' if ok else ''}")
    print(f"   => 最小 scaling: {found}")

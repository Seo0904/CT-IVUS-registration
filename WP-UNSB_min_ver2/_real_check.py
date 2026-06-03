import time, numpy as np, torch
import ot
from models.sequence_ot import sequence_ot_loss_geo, sequence_ot_loss_torch

f='/workspace/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/20260527_125919/moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze/ot_details/train_epoch0060/ot_0037.npz'
d=np.load(f, allow_pickle=True)
dev="cuda" if torch.cuda.is_available() else "cpu"
reg=0.1; p=2

P_saved=torch.tensor(d['P']); M_saved=torch.tensor(d['M'])
T,N=P_saved.shape
a=torch.ones(T)/T; b=torch.ones(N)/N
print(f"T={T} N={N}  reg={reg}")
print("--- saved P の周辺 ---")
print(" row sums (fake側, 理想=1/T=%.4f):"%(1/T), np.round(P_saved.sum(1).numpy(),4))
print(" col sums (tgt側 , 理想=1/N=%.4f):"%(1/N), np.round(P_saved.sum(0).numpy(),4))
print(" col_err(max|col-1/N|)=%.3e  row_err=%.3e"%(
    (P_saved.sum(0)-b).abs().max(), (P_saved.sum(1)-a).abs().max()))
print(" M stats: min=%.3e max=%.3e mean=%.3e"%(M_saved.min(),M_saved.max(),M_saved.mean()))

# 実データの fake_B / real_B からやり直す（保存時と同じ経路）
fake=torch.tensor(d['fake_B']).to(dev)
tgt =torch.tensor(d['real_B']).to(dev)

def col_err(Pt,N):
    bb=torch.ones(N,device=Pt.device)/N; return (Pt.sum(0)-bb).abs().max().item()
def row_err(Pt,T):
    aa=torch.ones(T,device=Pt.device)/T; return (Pt.sum(1)-aa).abs().max().item()

print("\n=== POT: 反復で target 周辺がどう下がるか（実データ）===")
for method in ['sinkhorn','sinkhorn_log']:
  for it in [200,1000,5000,10000,20000]:
    _,det=sequence_ot_loss_torch(fake,tgt,reg=reg,iters=it,sinkhorn_type=method,
                                 monotone=False,ot_divergence=False,return_details=True)
    print(f"  [{method:<12} it={it:<6}] col_err={col_err(det['P'],det['P'].shape[1]):.3e} "
          f"row_err={row_err(det['P'],T):.3e} ot_cost={det['ot_cost'].item():.5f}")
    if method=='sinkhorn_log' and it==10000: ref=col_err(det['P'],det['P'].shape[1])

print(f"\n=== geomloss scaling sweep (target=POT_log@10000={ref:.3e}) ===")
found=None
for sc in [0.5,0.7,0.8,0.9,0.95,0.97,0.99]:
    for _ in range(2):
        _,dg=sequence_ot_loss_geo(fake,tgt,reg=reg,p=p,scaling=sc,
                                  monotone=False,ot_divergence=False,return_details=True)
    if dev=="cuda": torch.cuda.synchronize()
    t0=time.time()
    for _ in range(20):
        _,dg=sequence_ot_loss_geo(fake,tgt,reg=reg,p=p,scaling=sc,
                                  monotone=False,ot_divergence=False,return_details=True)
    if dev=="cuda": torch.cuda.synchronize()
    ms=(time.time()-t0)/20*1000
    ce=col_err(dg['P'],dg['P'].shape[1]); re=row_err(dg['P'],T); ok=ce<=ref
    if ok and found is None: found=sc
    print(f"  scaling={sc:<5}: col_err={ce:.3e} row_err={re:.3e} {ms:6.2f}ms {'<= ref' if ok else ''}")
print(f"  => 最小 scaling: {found}")

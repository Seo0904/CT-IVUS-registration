import numpy as np, torch
from geomloss.ot import solve as geo_solve
NPZ="/workspace/data/experiment_result/WP-UNSB_ct_mri/synthRAD-brain/20260611_171121/synthRAD_brain_ct2mr_sb_w_otdiv_015_geo_gan/ot_details/train_epoch0040/ot_0024.npz"
dev="cuda" if torch.cuda.is_available() else "cpu"
d=np.load(NPZ)
fake=torch.tensor(d["fake_B"],device=dev).reshape(102,-1)
realB_full=torch.tensor(d["real_B"],device=dev).reshape(102,-1)
vi=torch.tensor(d["valid_idx"],device=dev,dtype=torch.long)
tgt=realB_full[vi]              # (99, D)  valid MR frames
realB_v=realB_full[vi]          # same, for self-test

def analyze(src, dst, reg, it=2000, tag=""):
    a=torch.ones(src.shape[0],device=dev)/src.shape[0]
    b=torch.ones(dst.shape[0],device=dev)/dst.shape[0]
    M=torch.cdist(src,dst,p=2)**2
    P=geo_solve(M,reg=reg,a=a,b=b,max_iter=it,unbalanced=None).plan.detach()
    rs=P.sum(1,keepdim=True)+1e-30
    Q=P/rs                                   # conditional rows
    perpl=torch.exp(-(Q*(Q+1e-30).log()).sum(1))   # 有効target数/行: 1=一対一, N=拡散
    top1=Q.max(1).values                            # 行の最大質量割合
    am=Q.argmax(1)
    # 準対角/単調性: argmax が src index にどれだけ沿うか
    ideal=torch.linspace(0,dst.shape[0]-1,src.shape[0],device=dev).round().long()
    diag_hit=(am==ideal).float().mean()
    mono=(am[1:]>=am[:-1]).float().mean()
    print(f"  {tag:28s} reg={reg:>5.1f}: perpl(行有効target数) mean={perpl.mean():5.2f} "
          f"median={perpl.median():5.2f} | top1 mean={top1.mean():.2f} | "
          f"argmax==対角 {diag_hit*100:4.0f}% | 単調 {mono*100:4.0f}%")

print("(A) 現在の潰れた fake_B -> 有効MR  (M中央値≈1543)")
for reg in [5,10,30,50,100]:
    analyze(fake,tgt,reg,tag="collapsed fake->MR")

print("\n(B) 理想: real_B -> real_B (完全一対一を支持。対角=0コスト)")
for reg in [5,10,30,50,100,300]:
    analyze(realB_v,realB_v,reg,tag="MR->MR (self)")
print(f"\n  N={tgt.shape[0]} : perpl=1 が完全一対一, perpl=N({tgt.shape[0]}) が完全拡散")

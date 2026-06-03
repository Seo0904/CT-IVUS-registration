import time, numpy as np, torch
from models.sequence_ot import sequence_ot_loss_geo, sequence_ot_loss_torch

f='/workspace/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/20260527_125919/moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze/ot_details/train_epoch0060/ot_0037.npz'
d=np.load(f, allow_pickle=True)
dev="cuda" if torch.cuda.is_available() else "cpu"
reg=0.1
fake=torch.tensor(d['fake_B']).to(dev); tgt=torch.tensor(d['real_B']).to(dev)
def col_err(P):
    N=P.shape[1]; b=torch.ones(N,device=P.device)/N; return (P.sum(0)-b).abs().max().item()

for norm in ['none', 'mean']:
    nz = None if norm=='none' else norm
    print(f"\n############ normalize={norm} (訓練={norm=='none'}) ############")
    # POT log, 訓練と同じ 10000
    _,det=sequence_ot_loss_torch(fake,tgt,reg=reg,iters=10000,sinkhorn_type='sinkhorn_log',
                                 normalize=nz,monotone=False,ot_divergence=False,return_details=True)
    print(f"  POT log @10000 : col_err={col_err(det['P']):.3e}")
    # geomloss いくつかの scaling
    for sc in [0.5,0.9,0.99]:
        _,dg=sequence_ot_loss_geo(fake,tgt,reg=reg,p=2,scaling=sc,
                                  normalize=nz,monotone=False,ot_divergence=False,return_details=True)
        print(f"  geo scaling={sc:<5}: col_err={col_err(dg['P']):.3e}")

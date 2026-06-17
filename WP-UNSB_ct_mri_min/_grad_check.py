import numpy as np, torch
from models.sequence_ot import sequence_ot_loss_torch, sequence_ot_loss_geo

f='/workspace/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/20260527_125919/moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze/ot_details/train_epoch0060/ot_0037.npz'
d=np.load(f,allow_pickle=True); dev='cuda'
tgt=torch.tensor(d['real_B']).to(dev)
kw=dict(reg=0.1, normalize=None, monotone=True, monotone_penalty=1.0,
        P_entropy=False, ot_divergence=True, ot_divergence_penalty=-0.5)

def grad_of(fn):
    fake=torch.tensor(d['fake_B']).to(dev).requires_grad_(True)
    tot,_=fn(fake)
    g,=torch.autograd.grad(tot, fake)
    return tot.item(), g.detach()

# POT (基準)
lp,gp = grad_of(lambda fk: sequence_ot_loss_torch(fk,tgt,iters=10000,sinkhorn_type='sinkhorn_log',**kw))
# geo 現状
lg,gg = grad_of(lambda fk: sequence_ot_loss_geo(fk,tgt,scaling=0.99,p=2,**kw))

def cmp(name,ga,gb):
    a=ga.flatten().double(); b=gb.flatten().double()
    cos=(a@b)/(a.norm()*b.norm()+1e-30)
    print(f"{name:<22} |g|={gb.norm().item():.4e}  ratio|g|(geo/pot)={(gb.norm()/ga.norm()).item():.3f}  cos(pot,geo)={cos.item():+.4f}")

print(f"POT  loss={lp:.4f}  |g_pot|={gp.norm().item():.4e}")
print(f"geo  loss={lg:.4f}")
cmp("geo(current)", gp, gg)

# 参考: ot_cost だけ（monotone/divergence を切って純粋なOT項の勾配比較）
kw2=dict(reg=0.1, normalize=None, monotone=False, P_entropy=False, ot_divergence=False)
_,gp2 = grad_of(lambda fk: sequence_ot_loss_torch(fk,tgt,iters=10000,sinkhorn_type='sinkhorn_log',**kw2))
_,gg2 = grad_of(lambda fk: sequence_ot_loss_geo(fk,tgt,scaling=0.99,p=2,**kw2))
print("\n-- ot_cost のみ（monotone/div off）--")
cmp("geo ot_cost only", gp2, gg2)

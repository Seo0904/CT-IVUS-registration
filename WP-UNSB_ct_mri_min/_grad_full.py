import numpy as np, torch
from geomloss import SamplesLoss
from models.sequence_ot import sequence_ot_loss_torch, get_valid_frame_idx
f='/workspace/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/20260527_125919/moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze/ot_details/train_epoch0060/ot_0037.npz'
d=np.load(f,allow_pickle=True); dev='cuda'
tgt=torch.tensor(d['real_B']).to(dev); reg=0.1; p=2; blur=reg**(1/p)
kw=dict(monotone=True, monotone_penalty=1.0, ot_divergence=True, ot_divergence_penalty=-0.5)
def squared_cost(x,y): return ((x[:,:,None,:]-y[:,None,:,:])**2).sum(-1)

def potg():
    fake=torch.tensor(d['fake_B']).to(dev).requires_grad_(True)
    tot,_=sequence_ot_loss_torch(fake,tgt,reg=reg,normalize=None,iters=10000,sinkhorn_type='sinkhorn_log',P_entropy=False,**kw)
    g,=torch.autograd.grad(tot,fake); return g.detach()
gp=potg()

def geo_fixed():
    fake=torch.tensor(d['fake_B']).to(dev).requires_grad_(True)
    vi=get_valid_frame_idx(tgt); T=fake.shape[0]; N=vi.numel()
    ff=fake.reshape(T,-1); tt=tgt[vi].reshape(N,-1)
    M=torch.cdist(ff,tt,p=2)**2
    a=torch.ones(T,device=dev)/T; b=torch.ones(N,device=dev)/N
    loss_fn=SamplesLoss(loss='sinkhorn',p=p,blur=blur,scaling=0.99,cost=squared_cost,debias=False,backend='tensorized')
    pot_fn =SamplesLoss(loss='sinkhorn',p=p,blur=blur,scaling=0.99,cost=squared_cost,debias=False,potentials=True,backend='tensorized')
    # OT 項: GeomLoss の損失を直接（正しい勾配）
    ot_cost = loss_fn(a, ff, b, tt)
    # 監視/penalty 用 P は detach したポテンシャルから再構成（M 依存のみ残す）
    with torch.no_grad():
        F,G = pot_fn(a, ff, b, tt); F=F.reshape(-1); G=G.reshape(-1)
    P = a[:,None]*b[None,:]*torch.exp((F[:,None]+G[None,:]-M.detach())/reg)  # 完全 detach (診断用)
    # monotone: detach した P を使う（勾配は載らない＝純penaltyにはならない点に注意）
    mono=torch.tensor(0.,device=dev)
    if kw['monotone']:
        j=torch.arange(N,device=dev,dtype=P.dtype).view(1,N); rm=P.sum(1)+1e-8
        U=(P*j).sum(1)/rm; mono=torch.relu(U[:-1]-U[1:]).sum()
    # divergence: GeomLoss 損失（fake-fake）も直接
    Md=torch.cdist(ff,ff,p=2)**2; bd=torch.ones(T,device=dev)/T
    div = loss_fn(a, ff, bd, ff)
    total = ot_cost + kw['monotone_penalty']*mono + kw['ot_divergence_penalty']*div
    g,=torch.autograd.grad(total,fake); return g.detach()

g=geo_fixed()
a=gp.flatten().double(); bb=g.flatten().double(); cos=(a@bb)/(a.norm()*bb.norm()+1e-30)
print(f"POT full grad      |g|={gp.norm():.4e}")
print(f"geo FIXED full grad|g|={g.norm():.4e} ratio={(g.norm()/gp.norm()):.3f} cos(pot)={cos:+.4f}")
print(f"finite? {torch.isfinite(g).all().item()}  nonzero? {g.abs().sum().item()>0}")

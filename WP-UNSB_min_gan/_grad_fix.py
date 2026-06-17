import numpy as np, torch
from geomloss import SamplesLoss
from models.sequence_ot import sequence_ot_loss_torch, get_valid_frame_idx

f='/workspace/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/20260527_125919/moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze/ot_details/train_epoch0060/ot_0037.npz'
d=np.load(f,allow_pickle=True); dev='cuda'
tgt=torch.tensor(d['real_B']).to(dev)
reg=0.1; p=2; blur=reg**(1/p)

# POT ot_cost-only 勾配（基準）
def pot_grad():
    fake=torch.tensor(d['fake_B']).to(dev).requires_grad_(True)
    tot,_=sequence_ot_loss_torch(fake,tgt,reg=reg,normalize=None,iters=10000,
                                 sinkhorn_type='sinkhorn_log',monotone=False,P_entropy=False,ot_divergence=False)
    g,=torch.autograd.grad(tot,fake); return g.detach()
gp=pot_grad()

def squared_cost(x,y): return ((x[:,:,None,:]-y[:,None,:,:])**2).sum(-1)

def geo_variant(detach_pot):
    fake=torch.tensor(d['fake_B']).to(dev).requires_grad_(True)
    vi=get_valid_frame_idx(tgt); T=fake.shape[0]; N=vi.numel()
    ff=fake.reshape(T,-1); tt=tgt[vi].reshape(N,-1)
    M=torch.cdist(ff,tt,p=2)**2
    a=torch.ones(T,device=dev)/T; b=torch.ones(N,device=dev)/N
    sink=SamplesLoss(loss="sinkhorn",p=p,blur=blur,scaling=0.99,cost=squared_cost,
                     debias=False,potentials=True,backend="tensorized")
    F,G=sink(a,ff,b,tt); F=F.reshape(-1); G=G.reshape(-1)
    if detach_pot:
        F=F.detach(); G=G.detach()
    P=a[:,None]*b[None,:]*torch.exp((F[:,None]+G[None,:]-M)/reg)
    ent=-torch.sum(P*torch.log(P+1e-8))
    ot=torch.sum(P*M)-reg*ent
    g,=torch.autograd.grad(ot,fake); return g.detach()

def envelope_grad():
    # 純包絡線: P を完全 detach し M のみ微分
    fake=torch.tensor(d['fake_B']).to(dev).requires_grad_(True)
    vi=get_valid_frame_idx(tgt); T=fake.shape[0]; N=vi.numel()
    ff=fake.reshape(T,-1); tt=tgt[vi].reshape(N,-1)
    M=torch.cdist(ff,tt,p=2)**2
    a=torch.ones(T,device=dev)/T; b=torch.ones(N,device=dev)/N
    sink=SamplesLoss(loss="sinkhorn",p=p,blur=blur,scaling=0.99,cost=squared_cost,
                     debias=False,potentials=True,backend="tensorized")
    with torch.no_grad():
        F,G=sink(a,ff,b,tt); F=F.reshape(-1); G=G.reshape(-1)
        P=a[:,None]*b[None,:]*torch.exp((F[:,None]+G[None,:]-M.detach())/reg)
    ot=torch.sum(P*M)   # P 定数, M のみ微分
    g,=torch.autograd.grad(ot,fake); return g.detach()

def cmp(name,gb):
    a=gp.flatten().double(); b=gb.flatten().double()
    cos=(a@b)/(a.norm()*b.norm()+1e-30)
    print(f"{name:<28} |g|={gb.norm():.4e}  ratio={ (gb.norm()/gp.norm()):.3f}  cos(pot,·)={cos:+.4f}")

print(f"POT ot_cost grad           |g|={gp.norm():.4e}")
cmp("geo: F,G grad付き(現状)", geo_variant(False))
cmp("geo: F,G detach", geo_variant(True))
cmp("geo: envelope(P detach,Mのみ)", envelope_grad())

"""
Minimal test: just load G weights directly and run forward.
No DDI, no DataParallel, no optimizer - just weights and forward pass.
"""
import sys, os, torch, numpy as np
sys.path.insert(0, '/workspace/WP-UNSB')

from models.networks import define_G
from options.base_options import BaseOptions

# Create a minimal opt
class DummyOpt:
    input_nc = 1; output_nc = 1; ngf = 64; netG = 'resnet_9blocks_cond'
    normG = 'instance'; no_dropout = True; init_type = 'xavier'; init_gain = 0.02
    no_antialias = False; no_antialias_up = False; gpu_ids = [0]
    embedding_dim = 512; embedding_type = 'positional'
    num_timesteps = 5; stylegan2_G_num_downsampling = 1
    n_mlp = 3; style_dim = 512

opt = DummyOpt()

# Create netG
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                not opt.no_dropout, opt.init_type, opt.init_gain,
                opt.no_antialias, opt.no_antialias_up, opt.gpu_ids, opt)

# Load weights directly
ckpt_path = '/workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347/moving_mnist_paired_sb/latest_net_G.pth'
state_dict = torch.load(ckpt_path, map_location='cuda:0')
netG.load_state_dict(state_dict)
netG.eval()

print(f"Loaded weights from {ckpt_path}")
print(f"netG type: {type(netG)}")

# Load some training data
data_A = np.load('/workspace/data/org_data/moving_mnist/mnist_test_seq.npy')  # (20,10000,64,64)
data_A = data_A.transpose(1, 0, 2, 3)  # (10000,20,64,64)

# Take first few sequences
device = torch.device('cuda:0')
results = []

for seq_idx in range(5):
    seq = data_A[seq_idx]  # (20,64,64)
    inp = torch.from_numpy(seq).float().unsqueeze(1) / 255.0  # (20,1,64,64)
    inp = inp * 2 - 1  # normalize to [-1,1]
    inp = inp.to(device)
    
    T = inp.size(0)
    time_idx = torch.zeros(T, dtype=torch.long, device=device)
    z = torch.randn(T, 4 * opt.ngf, device=device)
    
    with torch.no_grad():
        fake = netG(inp, time_idx, z)
    
    fake_255 = ((fake + 1) / 2 * 255).clamp(0, 255)
    inp_255 = ((inp + 1) / 2 * 255).clamp(0, 255)
    
    print(f"Seq {seq_idx}: input_mean={inp_255.mean().item():.1f}, "
          f"fake_mean={fake_255.mean().item():.1f}, "
          f"ratio={fake_255.mean().item()/max(inp_255.mean().item(),1e-6):.3f}")
    results.append(fake_255.mean().item())

print(f"\nOverall fake mean: {np.mean(results):.1f}")

# Also try with multi-step inference like original UNSB
print("\n=== Multi-step inference (like original UNSB) ===")
tau = 0.01
T_steps = opt.num_timesteps  # 5
incs = np.array([0] + [1/(i+1) for i in range(T_steps-1)])
times = np.cumsum(incs)
times = times / times[-1]
times = 0.5 * times[-1] + 0.5 * times
times = np.concatenate([np.zeros(1), times])
times = torch.tensor(times).float().to(device)
print(f"Time schedule: {times.cpu().numpy()}")

for seq_idx in range(5):
    seq = data_A[seq_idx]
    inp = torch.from_numpy(seq).float().unsqueeze(1) / 255.0
    inp = inp * 2 - 1
    inp = inp.to(device)
    
    N = inp.size(0)  # 20 frames
    
    with torch.no_grad():
        for t in range(T_steps):
            if t > 0:
                delta = times[t] - times[t-1]
                denom = times[-1] - times[t-1]
                inter = (delta / denom).reshape(-1,1,1,1)
                scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
            
            Xt = inp if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt)
            
            t_idx = (t * torch.ones(N, device=device)).long()
            z = torch.randn(N, 4 * opt.ngf, device=device)
            Xt_1 = netG(Xt, t_idx, z)
    
    fake_255 = ((Xt_1 + 1) / 2 * 255).clamp(0, 255)
    inp_255 = ((inp + 1) / 2 * 255).clamp(0, 255)
    print(f"Seq {seq_idx}: input_mean={inp_255.mean().item():.1f}, "
          f"fake_mean={fake_255.mean().item():.1f}, "
          f"ratio={fake_255.mean().item()/max(inp_255.mean().item(),1e-6):.3f}")

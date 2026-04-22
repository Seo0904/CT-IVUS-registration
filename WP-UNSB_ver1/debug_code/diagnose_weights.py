"""
Check if training produces weight updates, and verify the checkpoint integrity.
"""
import sys, os, torch, numpy as np, copy
sys.path.insert(0, '/workspace/WP-UNSB')

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

sys.argv = [
    'train.py',
    '--dataroot', '/workspace/data/org_data/moving_mnist',
    '--dataroot_B', '/workspace/data/preprocessed/bspline_transformed',
    '--data_file_B', 'transformed_aligned.npy',
    '--name', 'moving_mnist_paired_sb',
    '--model', 'wpsb',
    '--dataset_mode', 'moving_mnist_paired',
    '--input_nc', '1', '--output_nc', '1',
    '--batch_size', '8',
    '--load_size', '64', '--crop_size', '64',
    '--num_frames_per_seq', '20',
    '--train_ratio', '0.7', '--val_ratio', '0.1',
    '--checkpoints_dir', '/workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347',
    '--display_id', '-1',
    '--lambda_SB', '1.0',
    '--lr', '1e-05',
    '--nce_idt', 'True',
    '--continue_train',
    '--epoch', 'latest',
    '--gpu_ids', '0',
    '--no_html',
]

opt = TrainOptions().parse()
dataset = create_dataset(opt)
model = create_model(opt)

data_iter = iter(dataset)
data = next(data_iter)

model.data_dependent_initialize(data, None)
model.setup(opt)
model.parallelize()

# Capture weights BEFORE any optimization
weights_before = {}
for name, p in model.netG.named_parameters():
    weights_before[name] = p.detach().cpu().clone()

# Also load weights directly from checkpoint for comparison
ckpt_path = '/workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347/moving_mnist_paired_sb/latest_net_G.pth'
ckpt_state = torch.load(ckpt_path, map_location='cpu')

print("=== Weight comparison: loaded model vs checkpoint file ===")
total_diff = 0
total_elements = 0
for name in ckpt_state:
    # Model is wrapped in DataParallel, so we need to add "module." prefix
    model_key = f"module.{name}"
    if model_key in dict(model.netG.named_parameters()):
        model_param = dict(model.netG.named_parameters())[model_key].detach().cpu()
        diff = (model_param - ckpt_state[name]).abs().sum().item()
        total_diff += diff
        total_elements += ckpt_state[name].numel()

print(f"  Total abs diff: {total_diff:.6f}")
print(f"  Total elements: {total_elements}")
print(f"  Mean abs diff: {total_diff/total_elements:.10f}")

# Forward pass - check output
model.set_input(data)
with torch.no_grad():
    model.forward()
fake_before = model.fake_B.detach().cpu().clone()
mean_before = ((fake_before + 1) / 2 * 255).mean().item()
print(f"\nBefore any optimization:")
print(f"  fake_B mean (0-255): {mean_before:.1f}")

# Run ONE optimization step
model.set_input(data)
model.optimize_parameters()
fake_after_1 = model.fake_B.detach().cpu().clone()
mean_after_1 = ((fake_after_1 + 1) / 2 * 255).mean().item()

# Check weight change
print(f"\nAfter 1 optimization step:")
print(f"  fake_B mean (0-255): {mean_after_1:.1f}")
total_weight_change = 0
max_change = 0
max_change_name = ""
for name, p in model.netG.named_parameters():
    diff = (p.detach().cpu() - weights_before[name]).abs()
    s = diff.sum().item()
    m = diff.max().item()
    total_weight_change += s
    if m > max_change:
        max_change = m
        max_change_name = name
print(f"  Total weight change: {total_weight_change:.6f}")
print(f"  Max single param change: {max_change:.6f} ({max_change_name})")

# Forward with the updated model
model.set_input(data)
with torch.no_grad():
    model.forward()
fake_new_fwd = model.fake_B.detach().cpu()
mean_new_fwd = ((fake_new_fwd + 1) / 2 * 255).mean().item()
print(f"\nForward with updated weights (same data):")
print(f"  fake_B mean (0-255): {mean_new_fwd:.1f}")

# Another step
data2 = next(data_iter)
model.set_input(data2)
model.optimize_parameters()
fake_after_2 = model.fake_B.detach().cpu()
mean_after_2 = ((fake_after_2 + 1) / 2 * 255).mean().item()
print(f"\nAfter 2nd optimization step (new data):")
print(f"  fake_B mean (0-255): {mean_after_2:.1f}")

# Check total weight change from checkpoint
total_weight_change_2 = 0
for name, p in model.netG.named_parameters():
    diff = (p.detach().cpu() - weights_before[name]).abs()
    total_weight_change_2 += diff.sum().item()
print(f"  Total weight change from checkpoint: {total_weight_change_2:.6f}")

# Now SAVE these updated weights temporarily and reload
tmp_save = '/tmp/test_net_G.pth'
net = model.netG
torch.save(net.module.cpu().state_dict(), tmp_save)
net.cuda(0)

# Create a fresh model and load the saved weights
from models.networks import define_G

class DummyOpt:
    input_nc = 1; output_nc = 1; ngf = 64; netG = 'resnet_9blocks_cond'
    normG = 'instance'; no_dropout = True; init_type = 'xavier'; init_gain = 0.02
    no_antialias = False; no_antialias_up = False; gpu_ids = [0]
    embedding_dim = 512; embedding_type = 'positional'
    num_timesteps = 5; stylegan2_G_num_downsampling = 1
    n_mlp = 3; style_dim = 512

fresh_opt = DummyOpt()
fresh_G = define_G(1, 1, 64, 'resnet_9blocks_cond', 'instance', False, 'xavier', 0.02, False, False, [0], fresh_opt)
fresh_state = torch.load(tmp_save, map_location='cuda:0')
fresh_G.load_state_dict(fresh_state)
fresh_G.eval()

# Test forward with the updated weights
seq = data["A"][0].unsqueeze(1).to('cuda:0')  # (20,1,64,64)
if seq.dim() == 3:
    seq = seq.unsqueeze(1)
# handle (T,H,W) -> (T,1,H,W)
if seq.dim() == 4 and seq.size(1) != 1:
    # It's (T,H,W) actually stored as (T,H,W) with size(1)=H
    pass
time_idx_t = torch.zeros(seq.size(0), dtype=torch.long, device='cuda:0')
z_t = torch.randn(seq.size(0), 4*64, device='cuda:0')

with torch.no_grad():
    fake_fresh = fresh_G(seq, time_idx_t, z_t)
mean_fresh = ((fake_fresh + 1) / 2 * 255).mean().item()
print(f"\nFresh model loaded from 2-step-updated weights:")
print(f"  fake_B mean (0-255): {mean_fresh:.1f}")
print(f"  Input mean: {((seq + 1) / 2 * 255).mean().item():.1f}")

"""
Replicate EXACTLY what train.py does during visualization:
1. Load model exactly as train.py does
2. Run optimize_parameters() on a batch  
3. Capture the fake_B_seq that would be displayed
Compare with just running forward() without optimization.
"""
import sys, os
sys.path.insert(0, '/workspace/WP-UNSB')

import torch
import numpy as np
import argparse
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# Parse options exactly like train.py
sys.argv = [
    'train.py',
    '--dataroot', '/workspace/data/org_data/moving_mnist',
    '--dataroot_B', '/workspace/data/preprocessed/bspline_transformed',
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
print(f"\n{'='*60}")
print("OPTIONS: isTrain={}, continue_train={}".format(opt.isTrain, opt.continue_train))
print(f"{'='*60}")

# Replicate train.py initialization
data_iter = iter(dataset)
data = next(data_iter)

model.data_dependent_initialize(data, None)
model.setup(opt)
model.parallelize()

print("\n--- Model loaded. Now testing different scenarios ---\n")

# Scenario A: Just forward (like my diagnostic)
print("=== Scenario A: Just forward() ===")
model.set_input(data)
with torch.no_grad():
    model.forward()
fake_A = model.fake_B.detach().cpu()
print(f"  fake_B shape: {fake_A.shape}")
print(f"  fake_B mean: {fake_A.mean().item():.4f} (pixel range [-1,1])")
print(f"  fake_B mean (0-255): {((fake_A + 1) / 2 * 255).mean().item():.1f}")
print(f"  real_A mean (0-255): {((model.real_A.detach().cpu() + 1) / 2 * 255).mean().item():.1f}")

# Scenario B: optimize_parameters then check (like train.py)  
print("\n=== Scenario B: optimize_parameters() then check ===")
model.set_input(data)
model.optimize_parameters()
fake_B = model.fake_B.detach().cpu()
print(f"  fake_B shape: {fake_B.shape}")
print(f"  fake_B mean: {fake_B.mean().item():.4f}")
print(f"  fake_B mean (0-255): {((fake_B + 1) / 2 * 255).mean().item():.1f}")
print(f"  real_A mean (0-255): {((model.real_A.detach().cpu() + 1) / 2 * 255).mean().item():.1f}")

# Check what tensor2im returns
from util.util import tensor2im
# Pick frame 0 of batch 0
f_b_frame = fake_B[0]  # (C,H,W)
im = tensor2im(f_b_frame)
print(f"  tensor2im of fake_B[0]: shape={im.shape}, mean={im.mean():.1f}, max={im.max()}")

# Scenario C: Do 10 optimization steps then check
print("\n=== Scenario C: 10 optimization steps ===")
for step in range(10):
    data = next(data_iter)
    model.set_input(data)
    model.optimize_parameters()
    fake_C = model.fake_B.detach().cpu()
    mean_val = ((fake_C + 1) / 2 * 255).mean().item()
    print(f"  Step {step}: fake_B mean (0-255): {mean_val:.1f}")

# Scenario D: Check the visual as train.py does
print("\n=== Scenario D: compute_visuals + get_current_visuals ===")
with torch.no_grad():
    model.compute_visuals()
visuals = model.get_current_visuals()
for name, tensor in visuals.items():
    if torch.is_tensor(tensor):
        arr = tensor.detach().cpu()
        mean255 = ((arr + 1) / 2 * 255).mean().item()
        print(f"  {name}: shape={tuple(arr.shape)}, mean(0-255)={mean255:.1f}")

# Scenario E: Use tensor2im on the seq visual (same as visualizer.py)
from util.visualizer import _pick_three_frames, _make_visuals_np
visuals_np = _make_visuals_np(visuals)
for name, arr in visuals_np.items():
    print(f"  NP {name}: shape={arr.shape}, mean={arr.mean():.1f}, max={arr.max()}")

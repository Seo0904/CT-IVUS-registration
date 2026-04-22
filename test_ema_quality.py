#!/usr/bin/env python3
"""
Compare original checkpoint vs 2000-step EMA weights quantitatively.
Tests on 20 sequences from test split.
"""
import sys, os
sys.path.insert(0, '/workspace/WP-UNSB')
os.chdir('/workspace/WP-UNSB')

import torch
import numpy as np
from options.test_options import TestOptions

cmd = (
    "--dataroot /workspace/data/org_data/moving_mnist "
    "--dataroot_B /workspace/data/preprocessed/bspline_transformed "
    "--data_file_B transformed_aligned.npy "
    "--name moving_mnist_paired_sb "
    "--model wpsb "
    "--dataset_mode moving_mnist_paired "
    "--input_nc 1 --output_nc 1 "
    "--load_size 64 --crop_size 64 "
    "--num_frames_per_seq 20 "
    "--train_ratio 0.7 --val_ratio 0.1 "
    "--checkpoints_dir /workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347 "
    "--phase test "
    "--epoch latest "
    "--gpu_ids 0 "
    "--eval --no_flip --serial_batches "
    "--num_test 20"
)

opt = TestOptions(cmd).parse()
opt.num_threads = 0
opt.batch_size = 1

from data import create_dataset
from models import create_model
from util.metrics import ssim, l2_loss, l1_loss, psnr

dataset = create_dataset(opt)
dataset2 = create_dataset(opt)

# ===== Test with original weights =====
print("\n" + "="*60)
print("TEST 1: Original checkpoint weights")
print("="*60)
model = create_model(opt)
orig_means = []
orig_ssims = []
orig_psnrs = []

for i, (data, data2) in enumerate(zip(dataset, dataset2)):
    if i == 0:
        model.data_dependent_initialize(data, data2)
        model.setup(opt)
        model.parallelize()
        model.eval()
    if i >= opt.num_test:
        break
    model.set_input(data, data2)
    model.test()
    fake_B = model.fake_B.detach()
    real_B = model.real_B.detach()
    mean_val = ((fake_B + 1) / 2 * 255).mean().item()
    orig_means.append(mean_val)
    orig_ssims.append(ssim(fake_B, real_B))
    orig_psnrs.append(psnr(fake_B, real_B))
    print(f"  Seq {i:3d}: fake_B mean={mean_val:.1f}, SSIM={orig_ssims[-1]:.4f}, PSNR={orig_psnrs[-1]:.2f}")

print(f"\nOriginal avg: mean={np.mean(orig_means):.1f}, SSIM={np.mean(orig_ssims):.4f}, PSNR={np.mean(orig_psnrs):.2f}")

# ===== Test with EMA weights =====
print("\n" + "="*60)
print("TEST 2: EMA weights (2000 steps, decay=0.9995)")
print("="*60)

# Reset datasets
dataset = create_dataset(opt)
dataset2 = create_dataset(opt)

model_ema = create_model(opt)
ema_means = []
ema_ssims = []
ema_psnrs = []

for i, (data, data2) in enumerate(zip(dataset, dataset2)):
    if i == 0:
        model_ema.data_dependent_initialize(data, data2)
        model_ema.setup(opt)
        loaded = model_ema.load_ema_as_G(opt.epoch)
        print(f"EMA loaded: {loaded}")
        model_ema.parallelize()
        model_ema.eval()
    if i >= opt.num_test:
        break
    model_ema.set_input(data, data2)
    model_ema.test()
    fake_B = model_ema.fake_B.detach()
    real_B = model_ema.real_B.detach()
    mean_val = ((fake_B + 1) / 2 * 255).mean().item()
    ema_means.append(mean_val)
    ema_ssims.append(ssim(fake_B, real_B))
    ema_psnrs.append(psnr(fake_B, real_B))
    print(f"  Seq {i:3d}: fake_B mean={mean_val:.1f}, SSIM={ema_ssims[-1]:.4f}, PSNR={ema_psnrs[-1]:.2f}")

print(f"\nEMA avg: mean={np.mean(ema_means):.1f}, SSIM={np.mean(ema_ssims):.4f}, PSNR={np.mean(ema_psnrs):.2f}")

# ===== Real data stats =====
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
# Get real B means
dataset = create_dataset(opt)
dataset2 = create_dataset(opt)
real_means = []
for i, (data, data2) in enumerate(zip(dataset, dataset2)):
    if i >= opt.num_test:
        break
    real_B = data['B']
    mean_val = ((real_B + 1) / 2 * 255).mean().item()
    real_means.append(mean_val)

print(f"  Real B avg mean:     {np.mean(real_means):.1f}")
print(f"  Original avg mean:   {np.mean(orig_means):.1f}  (ratio to real: {np.mean(orig_means)/np.mean(real_means):.3f})")
print(f"  EMA avg mean:        {np.mean(ema_means):.1f}  (ratio to real: {np.mean(ema_means)/np.mean(real_means):.3f})")
print(f"  Improvement:         {np.mean(ema_means)/np.mean(orig_means):.2f}x")
print(f"\n  Original avg SSIM:   {np.mean(orig_ssims):.4f}")
print(f"  EMA avg SSIM:        {np.mean(ema_ssims):.4f}")
print(f"  Original avg PSNR:   {np.mean(orig_psnrs):.2f}")
print(f"  EMA avg PSNR:        {np.mean(ema_psnrs):.2f}")
print("="*60)

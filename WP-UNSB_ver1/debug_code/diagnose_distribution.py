"""
モデル出力の品質分布を大量サンプルで確認する診断
"""
import sys
sys.path.insert(0, '/workspace/WP-UNSB')
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util as util_mod

def main():
    cmd = (
        "--dataroot /workspace/data/org_data/moving_mnist "
        "--dataroot_B /workspace/data/preprocessed/bspline_transformed "
        "--data_file_A mnist_test_seq.npy "
        "--data_file_B transformed_aligned.npy "
        "--name moving_mnist_paired_sb "
        "--model wpsb "
        "--mode sb "
        "--dataset_mode moving_mnist_paired "
        "--input_nc 1 --output_nc 1 --ngf 64 --ndf 64 "
        "--num_timesteps 5 "
        "--load_size 64 --crop_size 64 "
        "--gpu_ids 0 "
        "--checkpoints_dir /workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347 "
        "--phase train "
        "--epoch latest "
        "--num_test 100 "
        "--eval "
        "--no_flip --serial_batches"
    )
    
    opt = TestOptions(cmd).parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    
    model = create_model(opt)
    
    fake_means = []
    real_means = []
    ratios = []
    
    for i, (data, data2) in enumerate(zip(dataset, dataset2)):
        if i == 0:
            model.data_dependent_initialize(data, data2)
            model.setup(opt)
            model.parallelize()
        
        if i >= 50:
            break
        
        model.netG.train()
        model.set_input(data, data2)
        with torch.no_grad():
            model.forward()
        
        fake = model.fake_B.detach()
        real_A = model.real_A.detach()
        
        # tensor2im的な変換
        fake_im_mean = ((fake.cpu().float().mean().item() + 1) / 2 * 255)
        real_im_mean = ((real_A.cpu().float().mean().item() + 1) / 2 * 255)
        
        fake_means.append(fake_im_mean)
        real_means.append(real_im_mean)
        if real_im_mean > 1:
            ratios.append(fake_im_mean / real_im_mean)
    
    fake_means = np.array(fake_means)
    real_means = np.array(real_means)
    ratios = np.array(ratios)
    
    print("="*60)
    print(f"Model output distribution over {len(fake_means)} training samples")
    print("="*60)
    print(f"real_A image mean:  min={real_means.min():.1f}, max={real_means.max():.1f}, avg={real_means.mean():.1f}, std={real_means.std():.1f}")
    print(f"fake_B image mean:  min={fake_means.min():.1f}, max={fake_means.max():.1f}, avg={fake_means.mean():.1f}, std={fake_means.std():.1f}")
    print(f"fake/real ratio:    min={ratios.min():.3f}, max={ratios.max():.3f}, avg={ratios.mean():.3f}, std={ratios.std():.3f}")
    print()
    
    # 品質分布のヒストグラム
    print("fake_B im_mean distribution:")
    bins = [0, 2, 5, 8, 10, 15, 20, 50, 100, 256]
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = ((fake_means >= lo) & (fake_means < hi)).sum()
        pct = count / len(fake_means) * 100
        bar = '#' * int(pct / 2)
        print(f"  [{lo:3d}-{hi:3d}): {count:3d} ({pct:5.1f}%) {bar}")
    
    print()
    print("fake/real ratio distribution:")
    ratio_bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 2.0]
    for lo, hi in zip(ratio_bins[:-1], ratio_bins[1:]):
        count = ((ratios >= lo) & (ratios < hi)).sum()
        pct = count / len(ratios) * 100
        bar = '#' * int(pct / 2)
        print(f"  [{lo:.1f}-{hi:.1f}): {count:3d} ({pct:5.1f}%) {bar}")

    # z noise influence test
    print("\n" + "="*60)
    print("Z noise influence test (same input, different z)")
    print("="*60)
    
    # Use first sample
    for data, data2 in zip(dataset, dataset2):
        model.set_input(data, data2)
        break
    
    z_means = []
    for z_seed in range(20):
        torch.manual_seed(z_seed)
        with torch.no_grad():
            model.forward()
        fake = model.fake_B.detach()
        z_mean = ((fake.cpu().float().mean().item() + 1) / 2 * 255)
        z_means.append(z_mean)
    
    z_means = np.array(z_means)
    print(f"Same input, 20 different z: mean_of_means={z_means.mean():.2f}, std={z_means.std():.2f}")
    print(f"  min={z_means.min():.2f}, max={z_means.max():.2f}")
    print(f"  Values: {[f'{v:.1f}' for v in z_means]}")

if __name__ == '__main__':
    main()

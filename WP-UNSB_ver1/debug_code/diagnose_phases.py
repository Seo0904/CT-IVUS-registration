"""
train/test/val 各データでモデル出力を比較する診断スクリプト
"""
import sys
sys.path.insert(0, '/workspace/WP-UNSB')
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util as util_mod

def run_with_phase(phase_name, model=None):
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
        f"--phase {phase_name} "
        "--epoch latest "
        "--num_test 5 "
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
    
    need_init = (model is None)
    if need_init:
        model = create_model(opt)
    
    results = []
    for i, (data, data2) in enumerate(zip(dataset, dataset2)):
        if i == 0 and need_init:
            model.data_dependent_initialize(data, data2)
            model.setup(opt)
            model.parallelize()
        
        if i >= 3:
            break
        
        # TRAIN mode (not eval)
        model.netG.train()
        model.set_input(data, data2)
        
        torch.manual_seed(42 + i)
        with torch.no_grad():
            model.forward()
        
        fake = model.fake_B.detach()
        real_A = model.real_A.detach()
        real_B = model.real_B.detach()
        
        results.append({
            'i': i,
            'real_A_mean': real_A.mean().item(),
            'real_A_min': real_A.min().item(),
            'real_A_max': real_A.max().item(),
            'real_B_mean': real_B.mean().item(),
            'real_B_min': real_B.min().item(),
            'real_B_max': real_B.max().item(),
            'fake_mean': fake.mean().item(),
            'fake_min': fake.min().item(),
            'fake_max': fake.max().item(),
            'fake_near_minus1': (fake < -0.9).float().mean().item(),
            'fake_visible': (fake > -0.5).float().mean().item(),
            # tensor2imで画像化した時のstats
            'im_mean': util_mod.tensor2im(fake[0]).mean(),
            'im_max': util_mod.tensor2im(fake[0]).max(),
        })
    
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name} (dataset size: {len(dataset)})")
    print(f"{'='*60}")
    for r in results:
        print(f"  Sample {r['i']}:")
        print(f"    real_A  min/max/mean: {r['real_A_min']:.4f} / {r['real_A_max']:.4f} / {r['real_A_mean']:.4f}")
        print(f"    real_B  min/max/mean: {r['real_B_min']:.4f} / {r['real_B_max']:.4f} / {r['real_B_mean']:.4f}")
        print(f"    fake_B  min/max/mean: {r['fake_min']:.4f} / {r['fake_max']:.4f} / {r['fake_mean']:.4f}")
        print(f"    fake_B near_black%:   {r['fake_near_minus1']*100:.1f}%")
        print(f"    fake_B visible%:      {r['fake_visible']*100:.1f}%")
        print(f"    image mean/max:       {r['im_mean']:.1f} / {r['im_max']}")
    
    return model

def main():
    print("Comparing model output across train/val/test phases...")
    model = run_with_phase('train')
    run_with_phase('val', model)
    run_with_phase('test', model)
    
    # 追加: training中のvalidation (run_validation と同じ流れ)で生成品質を確認
    print(f"\n{'='*60}")
    print("Train mode forward (simulating training loop visuals):")
    print(f"{'='*60}")
    
    # trainデータを使ってtrain modeで生成
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
        "--num_test 5 "
        "--eval "
        "--no_flip --serial_batches"
    )
    
    opt_train = TestOptions(cmd).parse()
    opt_train.num_threads = 0
    opt_train.batch_size = 1
    opt_train.serial_batches = True
    opt_train.no_flip = True
    opt_train.display_id = -1
    
    dataset_train = create_dataset(opt_train)
    
    model.netG.train()
    for i, data in enumerate(dataset_train):
        if i >= 3:
            break
        model.set_input(data)
        torch.manual_seed(42 + i)
        with torch.no_grad():
            model.forward()
        fake = model.fake_B.detach()
        real_A = model.real_A.detach()
        
        # Save sample images for visual inspection
        if i == 0:
            import os
            save_dir = '/workspace/diag_images'
            os.makedirs(save_dir, exist_ok=True)
            
            # Save first frame from first sequence
            for j in range(min(3, fake.shape[0])):
                im_fake = util_mod.tensor2im(fake[j])
                im_realA = util_mod.tensor2im(real_A[j])
                im_realB = util_mod.tensor2im(model.real_B.detach()[j])
                
                # Concat horizontally: real_A | fake_B | real_B
                if im_fake.ndim == 3 and im_fake.shape[2] == 1:
                    im_fake = im_fake[:,:,0]
                    im_realA = im_realA[:,:,0]
                    im_realB = im_realB[:,:,0]
                
                concat = np.concatenate([im_realA, im_fake, im_realB], axis=1)
                util_mod.save_image(concat, os.path.join(save_dir, f'train_frame{j}_realA_fakeB_realB.png'))
            
            print(f"  Sample images saved to {save_dir}/")
        
        print(f"  Sample {i}: real_A mean={real_A.mean().item():.4f}, fake_B mean={fake.mean().item():.4f}, "
              f"im_mean={util_mod.tensor2im(fake[0]).mean():.1f}, im_max={util_mod.tensor2im(fake[0]).max()}")

if __name__ == '__main__':
    main()

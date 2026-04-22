"""
Validation script for Moving MNIST paired dataset.
Evaluates the model using paired GT data with SSIM, L2, L1, PSNR metrics.

Usage:
    python validate.py --name moving_mnist_exp --epoch latest --phase val
"""
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.metrics import MetricsCalculator, ssim, l2_loss, l1_loss, psnr
from util import util
import numpy as np
from tqdm import tqdm


def validate(opt):
    """Run validation and compute metrics."""
    # Create dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'Validation dataset size: {dataset_size}')
    
    # Create model
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    # Initialize metrics calculator
    metrics = MetricsCalculator()
    
    # Results storage
    results = {
        'ssim': [],
        'l2': [],
        'l1': [],
        'psnr': []
    }
    # Step変化量記録用
    step_diffs = []
    
    # Validate
    print('Running validation...')
    with torch.no_grad():
        prev_fake_B = None
        for i, data in enumerate(tqdm(dataset, desc='Validating')):
            model.set_input(data)
            model.forward()
            # Get generated and ground truth images
            fake_B = model.fake_B  # Generated from A
            real_B = model.real_B  # Ground truth (paired with A in val/test)
            # Compute metrics
            results['ssim'].append(ssim(fake_B, real_B))
            results['l2'].append(l2_loss(fake_B, real_B))
            results['l1'].append(l1_loss(fake_B, real_B))
            results['psnr'].append(psnr(fake_B, real_B))
            metrics.update(fake_B, real_B)
            # ステップ変化量計算
            if prev_fake_B is not None:
                diff = torch.norm(fake_B - prev_fake_B).item()
                step_diffs.append(diff)
            prev_fake_B = fake_B.clone()
            
            # 1seq（時系列画像全体）を保存
            if i == 0:
                save_dir = '/workspace/data/experiment_result/UNSB/moving-mnist/20260212_183400'
                os.makedirs(save_dir, exist_ok=True)
                real_A_seq = model.real_A
                fake_B_seq = model.fake_B
                real_B_seq = model.real_B
                real_A_img = util.tensor2im(real_A_seq[0])
                fake_B_img = util.tensor2im(fake_B_seq[0])
                real_B_img = util.tensor2im(real_B_seq[0])
                import numpy as np
                visual = np.concatenate([real_A_img, fake_B_img, real_B_img], axis=1)
                util.save_image(visual, os.path.join(save_dir, f'{opt.phase}_{opt.epoch}_seq.png'))
    
    # Print results
    print('\n' + '=' * 50)
    util.save_image(util.tensor2im(visual[0]), 
                                      os.path.join(save_dir, f'{opt.phase}_{opt.epoch}_sample.png'))
    print(metrics)
    # ステップ変化量統計
    if step_diffs:
        print(f"Step変化量: mean={np.mean(step_diffs):.4f}, std={np.std(step_diffs):.4f}, min={np.min(step_diffs):.4f}, max={np.max(step_diffs):.4f}")
    
    # Detailed statistics
    final_metrics = {
        'SSIM': {
            'mean': np.mean(results['ssim']),
            'std': np.std(results['ssim']),
            'min': np.min(results['ssim']),
            'max': np.max(results['ssim'])
        },
        'L2': {
            'mean': np.mean(results['l2']),
            'std': np.std(results['l2']),
            'min': np.min(results['l2']),
            'max': np.max(results['l2'])
        },
        'L1': {
            'mean': np.mean(results['l1']),
            'std': np.std(results['l1']),
            'min': np.min(results['l1']),
            'max': np.max(results['l1'])
        },
        'PSNR': {
            'mean': np.mean(results['psnr']),
            'std': np.std(results['psnr']),
            'min': np.min(results['psnr']),
            'max': np.max(results['psnr'])
        }
    }
    
    print('\nDetailed Statistics:')
    for metric_name, stats in final_metrics.items():
        print(f'{metric_name}:')
        print(f'  Mean: {stats["mean"]:.4f} ± {stats["std"]:.4f}')
        print(f'  Range: [{stats["min"]:.4f}, {stats["max"]:.4f}]')
    
    # Save results to file
    save_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, 'metrics.txt')
    with open(results_file, 'w') as f:
        f.write(f'Validation Results for {opt.name} (epoch {opt.epoch})\n')
        f.write('=' * 50 + '\n\n')
        for metric_name, stats in final_metrics.items():
            f.write(f'{metric_name}:\n')
            f.write(f'  Mean: {stats["mean"]:.4f} ± {stats["std"]:.4f}\n')
            f.write(f'  Range: [{stats["min"]:.4f}, {stats["max"]:.4f}]\n')
        f.write('\n')
        if step_diffs:
            f.write(f'Step変化量: mean={np.mean(step_diffs):.4f}, std={np.std(step_diffs):.4f}, min={np.min(step_diffs):.4f}, max={np.max(step_diffs):.4f}\n')
            f.write('Step変化量全リスト:\n')
            f.write(','.join([f'{d:.4f}' for d in step_diffs]) + '\n')
    print(f'\nResults saved to: {results_file}')
    return final_metrics


if __name__ == '__main__':
    opt = TestOptions().parse()
    
    # Override some options for validation
    if not hasattr(opt, 'phase') or opt.phase == 'test':
        opt.phase = 'val'  # Default to validation
    
    opt.num_threads = 0   # Test code only supports num_threads = 0
    opt.batch_size = 1    # Test code only supports batch_size = 1
    opt.serial_batches = True  # Disable data shuffling
    opt.no_flip = True    # No flip augmentation
    
    validate(opt)

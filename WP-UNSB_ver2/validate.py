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
from models.sequence_ot import get_valid_frame_idx
from util import util
import numpy as np
from tqdm import tqdm

from util.wandb_logger import WandbLogger


def validate(opt):
    """Run validation and compute metrics."""
    wandb = WandbLogger(opt)

    # epoch key for W&B x-axis (only if numeric)
    wandb_epoch = None
    try:
        if isinstance(opt.epoch, int):
            wandb_epoch = int(opt.epoch)
        elif isinstance(opt.epoch, str) and opt.epoch.isdigit():
            wandb_epoch = int(opt.epoch)
    except Exception:
        wandb_epoch = None

    save_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    os.makedirs(save_dir, exist_ok=True)

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

            # メトリクス計算用には、real_B のゼロ画素フレームを除外
            if real_B.dim() == 5:
                b, t, c, h, w = real_B.shape
                fake_flat = fake_B.view(b * t, c, h, w)
                real_flat = real_B.view(b * t, c, h, w)
            else:
                fake_flat = fake_B
                real_flat = real_B

            valid_idx = get_valid_frame_idx(real_flat)
            if valid_idx.numel() > 0 and valid_idx.numel() < real_flat.shape[0]:
                fake_flat = fake_flat[valid_idx]
                real_flat = real_flat[valid_idx]

            # Compute metrics on filtered frames
            results['ssim'].append(ssim(fake_flat, real_flat))
            results['l2'].append(l2_loss(fake_flat, real_flat))
            results['l1'].append(l1_loss(fake_flat, real_flat))
            results['psnr'].append(psnr(fake_flat, real_flat))
            metrics.update(fake_flat, real_flat)
            # ステップ変化量計算
            if prev_fake_B is not None:
                diff = torch.norm(fake_B - prev_fake_B).item()
                step_diffs.append(diff)
            prev_fake_B = fake_B.clone()
            
            # 1seq（時系列画像全体）を保存
            if i == 0:
                real_A_seq = model.real_A
                fake_B_seq = model.fake_B
                real_B_seq = model.real_B
                real_A_img = util.tensor2im(real_A_seq[0])
                fake_B_img = util.tensor2im(fake_B_seq[0])
                real_B_img = util.tensor2im(real_B_seq[0])
                visual = np.concatenate([real_A_img, fake_B_img, real_B_img], axis=1)
                util.save_image(visual, os.path.join(save_dir, f'{opt.phase}_{opt.epoch}_seq.png'))

                if wandb.enabled and wandb.wandb is not None:
                    try:
                        payload = {
                            'val/sample_seq': wandb.wandb.Image(os.path.join(save_dir, f'{opt.phase}_{opt.epoch}_seq.png'))
                        }
                        if wandb_epoch is not None:
                            payload['epoch'] = wandb_epoch
                        wandb.log(payload)
                    except Exception:
                        pass
    
    # Print results
    print('\n' + '=' * 50)
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

    # W&B logging (summary scalars)
    if wandb.enabled:
        try:
            payload = {}
            if wandb_epoch is not None:
                payload['epoch'] = wandb_epoch
            for metric_name, stats in final_metrics.items():
                payload[f'val/{metric_name}_mean'] = float(stats['mean'])
                payload[f'val/{metric_name}_std'] = float(stats['std'])
                payload[f'val/{metric_name}_min'] = float(stats['min'])
                payload[f'val/{metric_name}_max'] = float(stats['max'])
            if step_diffs:
                payload['val/step_diff_mean'] = float(np.mean(step_diffs))
                payload['val/step_diff_std'] = float(np.std(step_diffs))
                payload['val/step_diff_min'] = float(np.min(step_diffs))
                payload['val/step_diff_max'] = float(np.max(step_diffs))
            wandb.log(payload)
        except Exception:
            pass
        finally:
            wandb.finish()

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

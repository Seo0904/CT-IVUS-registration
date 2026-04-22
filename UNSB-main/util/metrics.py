"""
Evaluation metrics for image-to-image translation
SSIM, L2 (MSE), PSNR, MAE
"""
import torch
import torch.nn.functional as F
import numpy as np
from math import log10


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1: Tensor of shape (N, C, H, W) or (C, H, W), values in [-1, 1] or [0, 1]
        img2: Tensor of shape (N, C, H, W) or (C, H, W), values in [-1, 1] or [0, 1]
        window_size: Size of the Gaussian window
        size_average: If True, return mean SSIM over batch
    
    Returns:
        SSIM value (higher is better, max=1.0)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Normalize to [0, 1] if in [-1, 1]
    if img1.min() < 0:
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
    
    channel = img1.size(1)
    
    # Create Gaussian window
    gaussian = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * 1.5**2)) 
                            for x in range(window_size)])
    gaussian = gaussian / gaussian.sum()
    
    _1D_window = gaussian.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


def l2_loss(img1, img2, reduction='mean'):
    """
    Calculate L2 loss (MSE) between two images.
    
    Args:
        img1: Tensor of shape (N, C, H, W) or (C, H, W)
        img2: Tensor of shape (N, C, H, W) or (C, H, W)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        L2/MSE value (lower is better)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Normalize to [0, 1] if in [-1, 1]
    if img1.min() < 0:
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

    mse = F.mse_loss(img1, img2, reduction=reduction)
    if reduction == 'mean':
        return mse.item()
    return mse


def l1_loss(img1, img2, reduction='mean'):
    """
    Calculate L1 loss (MAE) between two images.
    
    Args:
        img1: Tensor of shape (N, C, H, W) or (C, H, W)
        img2: Tensor of shape (N, C, H, W) or (C, H, W)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        L1/MAE value (lower is better)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Normalize to [0, 1] if in [-1, 1]
    if img1.min() < 0:
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

    mae = F.l1_loss(img1, img2, reduction=reduction)
    if reduction == 'mean':
        return mae.item()
    return mae


def psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1: Tensor of shape (N, C, H, W) or (C, H, W), values in [0, max_val]
        img2: Tensor of shape (N, C, H, W) or (C, H, W), values in [0, max_val]
        max_val: Maximum value of the images
    
    Returns:
        PSNR value in dB (higher is better)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Normalize to [0, 1] if in [-1, 1]
    if img1.min() < 0:
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        max_val = 1.0
    
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    
    return 10 * log10(max_val ** 2 / mse)


class MetricsCalculator:
    """
    Accumulator for computing metrics over a dataset.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.ssim_sum = 0.0
        self.l2_sum = 0.0
        self.l1_sum = 0.0
        self.psnr_sum = 0.0
        self.count = 0
    
    def update(self, generated, ground_truth):
        """
        Update metrics with a batch of images.
        
        Args:
            generated: Generated images (N, C, H, W)
            ground_truth: Ground truth images (N, C, H, W)
        """
        batch_size = generated.size(0)
        
        self.ssim_sum += ssim(generated, ground_truth) * batch_size
        self.l2_sum += l2_loss(generated, ground_truth) * batch_size
        self.l1_sum += l1_loss(generated, ground_truth) * batch_size
        self.psnr_sum += psnr(generated, ground_truth) * batch_size
        self.count += batch_size
    
    def compute(self):
        """
        Compute average metrics.
        
        Returns:
            Dictionary with average metrics
        """
        if self.count == 0:
            return {'SSIM': 0, 'L2': 0, 'L1': 0, 'PSNR': 0}
        
        return {
            'SSIM': self.ssim_sum / self.count,
            'L2': self.l2_sum / self.count,
            'L1': self.l1_sum / self.count,
            'PSNR': self.psnr_sum / self.count
        }
    
    def __str__(self):
        metrics = self.compute()
        return (f"SSIM: {metrics['SSIM']:.4f} | "
                f"L2: {metrics['L2']:.6f} | "
                f"L1: {metrics['L1']:.4f} | "
                f"PSNR: {metrics['PSNR']:.2f} dB")


if __name__ == '__main__':
    # Test the metrics
    torch.manual_seed(42)
    
    # Create test images
    img1 = torch.rand(4, 1, 64, 64)  # Batch of 4 grayscale images
    img2 = img1 + 0.1 * torch.randn_like(img1)  # Add noise
    img2 = img2.clamp(0, 1)
    
    print("Testing metrics...")
    print(f"SSIM: {ssim(img1, img2):.4f}")
    print(f"L2 (MSE): {l2_loss(img1, img2):.6f}")
    print(f"L1 (MAE): {l1_loss(img1, img2):.4f}")
    print(f"PSNR: {psnr(img1, img2):.2f} dB")
    
    # Test with [-1, 1] range
    img1_norm = img1 * 2 - 1
    img2_norm = img2 * 2 - 1
    print("\nWith [-1, 1] range:")
    print(f"SSIM: {ssim(img1_norm, img2_norm):.4f}")
    print(f"PSNR: {psnr(img1_norm, img2_norm):.2f} dB")
    
    # Test MetricsCalculator
    print("\nTesting MetricsCalculator:")
    calc = MetricsCalculator()
    calc.update(img1, img2)
    print(calc)

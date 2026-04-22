"""FID computation utilities for tensors (real vs generated) during validation.

This module computes Frechet Inception Distance (FID) between two sets of
images given as PyTorch tensors, using the same InceptionV3 backbone as
vgg_sb.evaluations.
"""
import torch
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg

from vgg_sb.evaluations.inception import InceptionV3


def _get_inception_model(device: torch.device, dims: int = 2048) -> InceptionV3:
    """Create an InceptionV3 model for FID feature extraction."""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _get_activations_from_tensors(
    imgs: torch.Tensor,
    model: InceptionV3,
    device: torch.device,
    batch_size: int = 50,
    dims: int = 2048,
) -> np.ndarray:
    """Get Inception activations for a batch of images.

    Args:
        imgs: Tensor of shape (N, C, H, W), values in [-1, 1] or [0, 1].
        model: InceptionV3 network.
        device: Device to run on.
        batch_size: Batch size.
        dims: Feature dimension (default 2048).
    Returns:
        Numpy array of shape (N, dims).
    """
    if imgs.dim() != 4:
        raise ValueError(f"Expected 4D tensor (N,C,H,W), got {imgs.shape}")

    # Move to CPU first to avoid accidental graph ties, then normalize
    imgs = imgs.detach().cpu()

    # Normalize to [0,1] if currently in [-1,1]
    if imgs.min() < 0:
        imgs = (imgs + 1) / 2.0

    # Ensure 3 channels
    if imgs.size(1) == 1:
        imgs = imgs.repeat(1, 3, 1, 1)

    n_images = imgs.size(0)
    pred_arr = np.empty((n_images, dims), dtype=np.float32)

    for i in range(0, n_images, batch_size):
        batch = imgs[i : i + batch_size].to(device)
        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.view(pred.size(0), -1)
        pred_arr[i : i + pred.size(0)] = pred.cpu().numpy()

    return pred_arr


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Compute Frechet distance between two Gaussians.

    This follows the standard FID definition.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Training and test mean vectors have different lengths")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Training and test covariances have different dimensions")

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return fid


def calculate_fid_from_tensors(
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device = None,
    batch_size: int = 50,
    dims: int = 2048,
) -> float:
    """Calculate FID between real and fake images given as tensors.

    Args:
        real: Real images (N, C, H, W).
        fake: Generated images (N, C, H, W).
        device: Torch device. If None, uses CUDA when available.
        batch_size: Batch size for Inception forward.
        dims: Feature dimensionality.
    Returns:
        Scalar FID value (lower is better).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if real.size(0) == 0 or fake.size(0) == 0:
        return float("inf")

    model = _get_inception_model(device, dims=dims)

    act_real = _get_activations_from_tensors(real, model, device, batch_size, dims)
    act_fake = _get_activations_from_tensors(fake, model, device, batch_size, dims)

    mu_real, sigma_real = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu_fake, sigma_fake = act_fake.mean(axis=0), np.cov(act_fake, rowvar=False)

    fid = _frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid

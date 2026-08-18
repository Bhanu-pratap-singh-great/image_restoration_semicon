"""
losses.py
---------
Combined loss = L1 (pixel accuracy) + (1 - SSIM) (structural similarity).

Why not just L1/L2?
  Pure pixel losses (L1/L2) tend to produce slightly blurry results
  because the network learns to output the "average" plausible pixel
  value. Adding an SSIM term pushes the network to preserve local
  structure/contrast/edges, which is exactly what a semiconductor
  inspection use case cares about.

Why not add a GAN/adversarial loss?
  GANs can sharpen textures further, but they can also hallucinate
  fine detail that isn't really there. In a defect-inspection context
  that's risky (a hallucinated texture could hide or fake a defect),
  so this project intentionally sticks to a safe, faithful loss
  combination. This is a deliberate design decision - mention it in
  your slides as a reasoned trade-off, not a missing feature.

SSIM is implemented here from scratch (no extra dependency needed)
using a Gaussian-weighted local window, following the standard
Wang et al. 2004 formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)  # (window_size, window_size)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0):
    """Differentiable SSIM between two (N, C, H, W) tensors in [0, data_range]."""
    channels = img1.shape[1]
    window = _gaussian_window(window_size, sigma, channels, img1.device, img1.dtype)
    pad = window_size // 2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio between two (N, C, H, W) tensors."""
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return torch.tensor(100.0, device=pred.device)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_val = ssim(pred, target)
        ssim_loss = 1.0 - ssim_val
        total = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        # Return both the scalar loss (for backprop) and a dict of parts
        # (for logging) so training prints stay informative.
        return total, {"l1": l1_loss.item(), "ssim": ssim_val.item()}

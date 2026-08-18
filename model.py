"""
model.py
--------
RestorationNet: takes a noisy, low-resolution grayscale image (128x128)
and outputs a clean, full-resolution image (256x256).

Architecture idea (in plain words):
  1. A "head" conv extracts initial features from the noisy input.
  2. A stack of Residual-in-Residual Dense Blocks (RRDB, same building
     block used in ESRGAN) refines those features and removes noise.
     Dense connections inside each block let the network reuse
     low-level detail instead of throwing it away - important because
     we must NOT blur the image while denoising.
  3. A PixelShuffle upsampling layer doubles the spatial resolution
     (128 -> 256) in a learned way (avoids blocky/checkerboard
     artifacts that plain upsampling+conv can cause).
  4. We add the network's output on top of a simple bicubic-upsampled
     version of the input (a "global residual"). This means the
     network only has to learn the DIFFERENCE between a naive
     upsample and the true clean image, which is an easier and more
     stable thing to learn than generating the whole image from
     scratch.
  5. Final output is clamped to [0, 1] since that's the valid image
     range (ground truth is always in [0, 1], even though the noisy
     input can go slightly outside that range).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """One dense block: each conv sees the concatenation of all previous
    outputs in the block. Helps preserve fine detail during denoising."""

    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        # Small residual scaling (0.2) is a standard ESRGAN trick that
        # stabilizes training of deep residual stacks.
        return x + 0.2 * x5


class RRDB(nn.Module):
    """Residual in Residual Dense Block: 3 dense blocks + an outer skip."""

    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth)
        self.rdb2 = ResidualDenseBlock(channels, growth)
        self.rdb3 = ResidualDenseBlock(channels, growth)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


class RestorationNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, channels=64, n_blocks=6,
                 growth=32, scale=2):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(in_ch, channels, 3, 1, 1)

        self.body = nn.Sequential(*[RRDB(channels, growth) for _ in range(n_blocks)])
        self.body_conv = nn.Conv2d(channels, channels, 3, 1, 1)

        # Learned 2x upsampling via PixelShuffle (sub-pixel convolution).
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        feat = self.head(x)
        body_out = self.body_conv(self.body(feat))
        feat = feat + body_out          # global feature-level residual

        up = self.upsample(feat)
        out = self.tail(up)

        # Global residual in image space: predict the *correction* on
        # top of a naive bicubic upsample, not the whole image from zero.
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                              align_corners=False)
        out = out + base

        # IMPORTANT: only clamp to [0,1] at inference/eval time. During
        # training, a hard clamp gives ZERO gradient for any pixel that
        # goes outside [0,1] - since the input can go up to ~1.5, many
        # pixels get clamped from epoch 1 and the model gets stuck
        # (this caused the "frozen" training you saw). The L1+SSIM loss
        # already pulls values toward [0,1] on its own, so we let
        # gradients flow freely during training and only clamp for the
        # final displayed/evaluated image.
        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)


if __name__ == "__main__":
    # Quick sanity check: run a dummy batch through the model and
    # print shapes + parameter count. Run this file directly to verify
    # the architecture works on your machine before training:
    #     python model.py
    model = RestorationNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    dummy_input = torch.randn(2, 1, 128, 128)  # batch of 2, grayscale, 128x128
    out = model(dummy_input)
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (should be (2, 1, 256, 256))")
    assert out.shape == (2, 1, 256, 256), "Output shape mismatch!"
    print("Model sanity check passed.")

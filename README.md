# Semiconductor Inspection Image Restoration

Joint denoising + 2x super-resolution model for degraded (noisy, low-resolution)
semiconductor inspection images, trained on paired data provided by KLA for the
i4C hackathon (Problem Statement: AI-Based Restoration of Degraded Images).

## Repository Structure

```
.
├── model.py                    # Model architecture (RRDB-based restoration network)
├── losses.py                   # Combined L1 + SSIM loss, and PSNR/SSIM metric functions
├── dataset.py                  # Dataset loading + augmentation (used for training)
├── train.py                    # Reproduces training from scratch
├── infer.py                    # STANDALONE inference script (see "Running Inference" below)
├── visualize_test_outputs.py   # Helper: renders degraded-vs-restored PNGs for visual review
├── requirements.txt            # Exact pip freeze from the training environment
├── checkpoints/
│   ├── best_model.pth          # Final trained model weights (best validation SSIM)
│   ├── last_model.pth          # Final-epoch model weights
│   └── training_log.csv        # Per-epoch loss/PSNR/SSIM history
├── restored_test_outputs/      # Model outputs on the official 400-image test set
├── test_review/                # Sample degraded-vs-restored PNG comparisons (visual QA)
└── eval_outputs/                # Sample before/after/GT comparison images (validation split)
```

> Note: the raw `GT/` and `NoisyLR/` training data folders are not included in
> this repo (large, and provided directly by KLA) — see "Reproducing Training"
> below for how to point the training script at your own copy.

## Problem Summary

Input: a degraded grayscale image (speckle noise + downsampled resolution,
128x128, values may slightly exceed [0,1] due to noise).
Output: a restored grayscale image matching the clean, full-resolution ground
truth (256x256, values in [0,1]).

## Setup

```bash
git clone <this-repo-url>
cd <repo-folder>
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running Inference (for benchmarking)

This is the script that will be run as-is on the benchmarking test set:

```bash
python infer.py --input_dir /path/to/test/images --output_dir /path/to/save/restored
```

- `--input_dir`: folder containing degraded input images as `.npy` files
  (single-channel, shape `(H, W)`, float32).
- `--output_dir`: folder where restored images will be written, one `.npy`
  file per input file, same filename.
- The model checkpoint is loaded automatically from `checkpoints/best_model.pth`
  (relative to this script). Override with `--checkpoint <path>` if needed.
- The model is fully convolutional (aside from a fixed 2x upsample), so it
  accepts any input resolution, not just 128x128.

No manual edits are required — all paths are passed as command-line arguments.

## Reproducing Training

```bash
python train.py
```

Expects a `GT/` and `NoisyLR/` folder (paired `.npy` files, matched by
filename) in the same directory as `train.py`. Edit the `DATA_ROOT` constant
at the top of `train.py` if your data lives elsewhere. Training config
(epochs, batch size, learning rate) is also in the `CONFIG` block at the top
of the file.

Outputs:
- `checkpoints/best_model.pth` — best model by validation SSIM
- `checkpoints/last_model.pth` — final-epoch model
- `checkpoints/training_log.csv` — per-epoch loss/PSNR/SSIM history

## Model Architecture

RRDB (Residual-in-Residual Dense Block, from ESRGAN) backbone:
- Head conv (1→64 channels)
- 6x RRDB blocks (each with 3 residual dense blocks, growth rate 32)
- PixelShuffle 2x learned upsampling
- Global bicubic-upsample residual connection (model learns the correction
  on top of a naive upsample, not the whole image from scratch)
- Output clamped to [0,1] at inference time only (not during training —
  see note in `model.py` on why this matters for training stability)

~4.5M parameters.

## Loss Function

`Loss = 1.0 * L1(pred, target) + 0.5 * (1 - SSIM(pred, target))`

Deliberately excludes an adversarial (GAN) loss — see `losses.py` docstring
for the reasoning (risk of hallucinated texture in a defect-inspection
context).

## Results (on held-out validation split, 320 images)

| Metric | Score |
|---|---|
| PSNR | 28.88 dB |
| SSIM | 0.7948 |
| LPIPS | 0.2643 |
| Inference time | 11.69 ms/image (~85.5 images/sec) on RTX 4070 Super |

See `eval_outputs/` for before/after/ground-truth comparison images.

## Test Set Results

The official test set (400 images) was used to generate restored outputs:

```bash
python infer.py --input_dir <test_dir>/NoisyLR --output_dir restored_test_outputs
```

- Images processed: 400
- Average inference time: 20.40 ms/image (~49.0 images/sec) on RTX 4070 Super
- Restored outputs included in this repo: `restored_test_outputs/`
- Visual samples (degraded vs. restored): `test_review/` — generated with
  `python visualize_test_outputs.py --input_dir <test_dir>/NoisyLR --output_dir restored_test_outputs --n 50`

Note: the official test set has no ground truth, so PSNR/SSIM/LPIPS could not
be computed on it directly — the metrics in the Results table above are from
the held-out validation split, which does have ground truth.

## Known Limitations

- The train/validation split is a random split of the provided training
  data, not a source-held-out split — so the validation numbers above are a
  reasonable but possibly optimistic proxy for true out-of-distribution
  performance on KLA's separate test set.
- No synthetic degradation augmentation (randomized noise/downsampling
  applied on the fly) was used due to time constraints — only geometric
  augmentation (flips/rotations). This is a natural next step to further
  improve out-of-distribution robustness.

## References

See presentation slides (Slide 9) for full citations, including the ESRGAN
paper (RRDB architecture basis) and the SSIM paper (Wang et al., 2004).

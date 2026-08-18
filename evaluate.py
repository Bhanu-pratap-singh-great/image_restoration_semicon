"""
evaluate.py
-----------
Run this AFTER training finishes. It:
  1. Loads the best saved checkpoint.
  2. Computes PSNR, SSIM (and LPIPS if installed) on the validation set.
  3. Saves a handful of before/after/ground-truth comparison images to
     eval_outputs/ - use these directly in your Results slide (Slide 6).
  4. Measures average inference time per image - use this in your
     Technology & Feasibility slide (Slide 7).

HOW TO RUN:
    python evaluate.py

Optional: pip install lpips   (adds the LPIPS perceptual metric;
the script works fine without it, it'll just skip that number.)
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import RestorationNet
from dataset import make_datasets
from losses import ssim as ssim_fn, psnr as psnr_fn

# ============ CONFIG ============
DATA_ROOT = "."
CHECKPOINT = os.path.join("checkpoints", "best_model.pth")
N_VIS_SAMPLES = 6
OUT_DIR = "eval_outputs"
# =================================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_ds = make_datasets(DATA_ROOT, val_fraction=0.1)
    print(f"Evaluating on {len(val_ds)} validation samples.")

    model = RestorationNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # ---- Optional LPIPS ----
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        print("LPIPS available - will include it in results.")
    except ImportError:
        print("lpips not installed - skipping LPIPS metric "
              "(run: pip install lpips, then re-run this script if you want it).")

    total_psnr, total_ssim, total_lpips, n = 0.0, 0.0, 0.0, 0
    times = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            lr_img, gt_img = val_ds[i]
            lr_img = lr_img.unsqueeze(0).to(device)
            gt_img = gt_img.unsqueeze(0).to(device)

            t0 = time.time()
            pred = model(lr_img)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)

            total_psnr += psnr_fn(pred, gt_img).item()
            total_ssim += ssim_fn(pred, gt_img).item()

            if lpips_fn is not None:
                # lpips expects 3-channel input in [-1, 1]
                pred3 = pred.repeat(1, 3, 1, 1) * 2 - 1
                gt3 = gt_img.repeat(1, 3, 1, 1) * 2 - 1
                total_lpips += lpips_fn(pred3, gt3).item()

            n += 1

            if i < N_VIS_SAMPLES:
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(lr_img[0, 0].cpu().numpy(), cmap="gray")
                axes[0].set_title("Degraded Input (128x128)")
                axes[1].imshow(pred[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Restored Output")
                axes[2].imshow(gt_img[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Ground Truth (256x256)")
                for ax in axes:
                    ax.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, f"comparison_{i}.png"), dpi=150)
                plt.close()

    print("\n===== Results on validation set =====")
    print(f"Samples evaluated: {n}")
    print(f"Average PSNR:  {total_psnr / n:.2f} dB")
    print(f"Average SSIM:  {total_ssim / n:.4f}")
    if lpips_fn is not None:
        print(f"Average LPIPS: {total_lpips / n:.4f}")
    avg_time = float(np.mean(times))
    print(f"Average inference time: {avg_time * 1000:.2f} ms/image "
          f"({1.0/avg_time:.1f} images/sec)")
    print(f"\nComparison images saved to: {OUT_DIR}/")
    print("Use these numbers + images directly in Slide 6 (Results) and "
          "Slide 7 (Technology & Feasibility) of your submission deck.")


if __name__ == "__main__":
    main()

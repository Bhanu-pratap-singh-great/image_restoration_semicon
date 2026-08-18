"""
train.py
--------
Main training script. Run this on the Windows PC with the RTX 4070 Super.

HOW TO RUN:
    python train.py

All settings you might want to change are in the CONFIG block below -
you shouldn't need to touch anything else.

What this script does, step by step:
  1. Loads the dataset and splits it into train (90%) / val (10%).
  2. Builds the RestorationNet model and moves it to the GPU.
  3. Trains for NUM_EPOCHS epochs using mixed-precision (fp16) for
     speed, with the combined L1+SSIM loss.
  4. After every epoch, evaluates on the validation set (PSNR + SSIM)
     and saves the model if it's the best one seen so far
     (checkpoints/best_model.pth).
  5. Prints progress every epoch so you can watch it improve.
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from model import RestorationNet
from dataset import make_datasets
from losses import CombinedLoss, ssim as ssim_fn, psnr as psnr_fn

# ============ CONFIG - change these if needed ============
DATA_ROOT = "."             # folder containing GT/ and NoisyLR/
BATCH_SIZE = 8  # reduced from 64 due to CUDA OOM - GPU may be shared with other processes; check `nvidia-smi`
NUM_EPOCHS = 60
LEARNING_RATE = 2e-4
VAL_FRACTION = 0.1
NUM_WORKERS = 4             # set to 0 if you get multiprocessing errors on Windows
CHECKPOINT_DIR = "checkpoints"
# ===========================================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: CUDA GPU not detected. Training will be very slow on CPU. "
              "Check your PyTorch + CUDA install if you expected to use the RTX 4070.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds, val_ds = make_datasets(DATA_ROOT, VAL_FRACTION)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = RestorationNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = CombinedLoss(l1_weight=1.0, ssim_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_ssim = -1.0
    log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_psnr,val_ssim,epoch_time_sec\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for lr_img, gt_img in train_loader:
            lr_img = lr_img.to(device, non_blocking=True)
            gt_img = gt_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lr_img)
            loss, _parts = criterion(pred, gt_img)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: non-finite loss encountered, skipping this batch.")
                continue

            loss.backward()
            # Gradient clipping adds extra stability for this deep
            # residual network (no normalization layers), preventing
            # any single bad batch from causing a huge weight update.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_psnr_total, val_ssim_total = 0.0, 0.0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img = lr_img.to(device)
                gt_img = gt_img.to(device)
                pred = model(lr_img)
                val_psnr_total += psnr_fn(pred, gt_img).item() * lr_img.size(0)
                val_ssim_total += ssim_fn(pred, gt_img).item() * lr_img.size(0)

        val_psnr = val_psnr_total / len(val_ds)
        val_ssim = val_ssim_total / len(val_ds)
        dt = time.time() - t0

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_PSNR={val_psnr:.2f}dB | val_SSIM={val_ssim:.4f} | time={dt:.1f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_psnr:.4f},{val_ssim:.6f},{dt:.2f}\n")

        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  -> New best model saved (val_SSIM={val_ssim:.4f})")

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pth"))

    print(f"\nTraining complete. Best val SSIM: {best_val_ssim:.4f}")
    print(f"Best checkpoint: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    main()

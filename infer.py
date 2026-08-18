"""
infer.py
--------
STANDALONE evaluation/inference script — this is the file KLA's
benchmarking team will run AS-IS on the H100 GPU. It must run without
any manual edits.

USAGE:
    python infer.py --input_dir /path/to/test/images --output_dir /path/to/save/restored

What it does:
  1. Loads the trained model from checkpoints/best_model.pth (relative
     to this script's location — keep the checkpoints/ folder next to
     this file in the repo).
  2. Reads every .npy file in --input_dir. Each is expected to be a
     single-channel (grayscale) degraded image, shape (H, W), float32
     — matching the format of the training data (NoisyLR/*.npy).
  3. Runs the trained model on each image (joint denoising + 2x
     super-resolution).
  4. Saves the restored image as a .npy file with the SAME filename
     into --output_dir, ready for scoring against ground truth.

No hardcoded paths — everything needed is passed as a command-line
argument, as required by the submission spec.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from model import RestorationNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")


def load_model(checkpoint_path, device):
    model = RestorationNet()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run the trained restoration model on a folder of degraded images."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Path to folder containing degraded input images (.npy files).")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Path to folder where restored images will be saved.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                         help="Path to trained model checkpoint "
                              "(default: checkpoints/best_model.pth next to this script).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input_dir does not exist: {args.input_dir}")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params:,}")

    input_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".npy"))
    if len(input_files) == 0:
        print(f"ERROR: no .npy files found in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(input_files)} input images.")

    times = []
    with torch.no_grad():
        for i, fname in enumerate(input_files):
            in_path = os.path.join(args.input_dir, fname)
            arr = np.load(in_path).astype(np.float32)

            # Expect shape (H, W). Add batch + channel dims -> (1, 1, H, W).
            # The model is fully convolutional (aside from a fixed 2x
            # PixelShuffle), so it accepts any input H, W - not just
            # the 128x128 size seen during training.
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

            t0 = time.time()
            output = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            times.append(dt)

            out_arr = output.squeeze(0).squeeze(0).cpu().numpy()

            out_path = os.path.join(args.output_dir, fname)
            np.save(out_path, out_arr)

            if (i + 1) % 50 == 0 or (i + 1) == len(input_files):
                print(f"  Processed {i + 1}/{len(input_files)} images...")

    avg_time = float(np.mean(times))
    print(f"\nDone. Restored {len(input_files)} images -> {args.output_dir}")
    print(f"Average inference time: {avg_time * 1000:.2f} ms/image "
          f"({1.0 / avg_time:.1f} images/sec)")


if __name__ == "__main__":
    main()

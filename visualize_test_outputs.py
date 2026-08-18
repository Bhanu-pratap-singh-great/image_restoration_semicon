"""
visualize_test_outputs.py
--------------------------
Quick helper to visually inspect restored test-set outputs (since the
real test set has no ground truth to compare against, this just shows
Degraded Input vs Restored Output side by side).

USAGE:
    python visualize_test_outputs.py --input_dir <path to test NoisyLR folder> --output_dir restored_test_outputs --n 8

Saves PNG images to a new folder: test_review/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Folder with original degraded .npy test images.")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Folder with restored .npy outputs (from infer.py).")
    parser.add_argument("--n", type=int, default=8,
                         help="Number of sample images to visualize.")
    parser.add_argument("--save_dir", type=str, default="test_review",
                         help="Where to save the PNG comparison images.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.output_dir) if f.lower().endswith(".npy"))
    if len(files) == 0:
        print(f"No .npy files found in {args.output_dir}")
        return

    chosen = files[:args.n] if len(files) <= args.n else \
        [files[i] for i in np.linspace(0, len(files) - 1, args.n, dtype=int)]

    for fname in chosen:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)

        degraded = np.load(in_path)
        restored = np.load(out_path)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        axes[0].imshow(degraded, cmap="gray")
        axes[0].set_title(f"Degraded Input\n{degraded.shape}")
        axes[1].imshow(restored, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Restored Output\n{restored.shape}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()

        save_name = os.path.splitext(fname)[0] + "_review.png"
        plt.savefig(os.path.join(args.save_dir, save_name), dpi=150)
        plt.close()

    print(f"Saved {len(chosen)} comparison images to: {args.save_dir}/")
    print("Open that folder to view them (or upload a couple back here for me to check).")


if __name__ == "__main__":
    main()

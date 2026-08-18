"""
dataset.py
----------
Loads paired (degraded, clean) .npy images from the GT / NoisyLR folders.

Assumes the folder layout confirmed from your data:
    <DATA_ROOT>/GT/000000.npy       -> (256, 256) float32, range [0, 1]
    <DATA_ROOT>/NoisyLR/000000.npy  -> (128, 128) float32, range roughly
                                        [-0.05, 1.6] (can exceed [0,1] -
                                        this is expected, don't clip it)

Augmentation (only applied to the training split):
    - random horizontal flip
    - random vertical flip
    - random 90-degree rotation
  The SAME transform is applied to both the LR and GT image in a pair
  so they stay aligned.

A fixed random seed is used for the train/val split so re-running the
script always gives the same split (reproducible results for your
report).
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class RestorationDataset(Dataset):
    def __init__(self, gt_dir, lr_dir, indices, augment=True):
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        fname = f"{i:06d}.npy"

        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                gt = np.ascontiguousarray(np.flip(gt, axis=1))
                lr = np.ascontiguousarray(np.flip(lr, axis=1))
            if random.random() < 0.5:
                gt = np.ascontiguousarray(np.flip(gt, axis=0))
                lr = np.ascontiguousarray(np.flip(lr, axis=0))
            k = random.randint(0, 3)
            if k > 0:
                gt = np.ascontiguousarray(np.rot90(gt, k))
                lr = np.ascontiguousarray(np.rot90(lr, k))

        # Add channel dimension: (H, W) -> (1, H, W)
        gt_t = torch.from_numpy(gt).unsqueeze(0)
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        return lr_t, gt_t


def make_datasets(data_root, val_fraction=0.1, seed=42):
    gt_dir = os.path.join(data_root, "GT")
    lr_dir = os.path.join(data_root, "NoisyLR")

    n = len(os.listdir(gt_dir))
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_val = max(1, int(n * val_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ds = RestorationDataset(gt_dir, lr_dir, train_indices, augment=True)
    val_ds = RestorationDataset(gt_dir, lr_dir, val_indices, augment=False)
    return train_ds, val_ds


if __name__ == "__main__":
    # Quick sanity check - run this file directly to confirm the
    # dataset loads correctly and shapes match:
    #     python dataset.py
    train_ds, val_ds = make_datasets(".", val_fraction=0.1)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    lr_img, gt_img = train_ds[0]
    print(f"LR sample shape: {tuple(lr_img.shape)}  (expected (1, 128, 128))")
    print(f"GT sample shape: {tuple(gt_img.shape)}  (expected (1, 256, 256))")
    print("Dataset sanity check passed.")

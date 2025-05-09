# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainTumor2DSliceDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.image_dir, self.image_files[idx])).astype(np.float32)  # (128,128,4)
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx])).astype(np.int64)       # (128,128)

        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (4, 128, 128)
        mask = torch.from_numpy(mask)                     # (128, 128)

        if self.transform:
            image = self.transform(image)

        return image, mask

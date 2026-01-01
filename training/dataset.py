# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import random

class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for building segmentation.
    Automatically handles images in 8-bit PNG format.
    Masks are converted to 0/1.
    """
    def __init__(self, ids_file, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        with open(ids_file, "r") as f:
            self.ids = [line.strip() for line in f]

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load image
        img_path = self.images_dir / f"{img_id}.png"
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image, dtype=np.float32) / 255.0  # normalize 0-1
        image_tensor = torch.tensor(image_np).permute(2,0,1)  # C,H,W

        # Load mask
        mask_path = self.masks_dir / f"{img_id}.png"
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask, dtype=np.float32)
        mask_np = (mask_np > 127).astype(np.float32)  # threshold 0/1
        mask_tensor = torch.tensor(mask_np).unsqueeze(0)  # 1,H,W

        # Apply transform (optional)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor


def get_dataloaders(images_dir, masks_dir, train_txt, val_txt, batch_size=8, transform=None):
    """
    Helper to create PyTorch DataLoaders from split text files.
    """
    train_dataset = SegmentationDataset(
        ids_file=train_txt,
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=transform
    )

    val_dataset = SegmentationDataset(
        ids_file=val_txt,
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

# scripts/datasets.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path


class SegmentationDataset(Dataset):
    
    def __init__(self, ids_file, images_dir, masks_dir, transform=None):
       
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        with open(ids_file) as f:
            self.ids = [line.strip() for line in f]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load image
        img_path = self.images_dir / f"{img_id}.png"
        image = Image.open(img_path).convert("RGB")

        # Apply optional transform
        if self.transform:
            image = self.transform(image)

        # Convert to tensor (C,H,W) float32 in 0-1
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2,0,1)/255.0

        # Load mask
        mask_path = self.masks_dir / f"{img_id}.png"
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask) // 255  # 0 (background) or 1 (building)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # 1,H,W

        return image, mask


def get_dataloaders(images_dir, masks_dir, train_txt, val_txt, batch_size=8, transform=None):
    """
    Helper function to quickly create PyTorch DataLoaders for train/val splits.
    """
    from torch.utils.data import DataLoader

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

import os
from torch.utils.data import Dataset, DataLoader
import rasterio
import torch
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list):
        """
        images_dir: folder with RGB images
        masks_dir: folder with masks
        file_list: list of image filenames to use
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace(".tif","_mask.tif"))

        # Load image
        with rasterio.open(img_path) as src:
            img = src.read([1,2,3])                # RGB
            img = np.transpose(img, (1,2,0))       # H,W,C
            img = img.astype(np.float32) / 255.0   # normalize to 0-1

        # Load mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)  # H,W

        # Convert to torch tensors
        img = torch.from_numpy(img).permute(2,0,1)   # C,H,W
        mask = torch.from_numpy(mask).unsqueeze(0)   # 1,H,W

        return img, mask

def get_dataloaders(images_dir, masks_dir, train_list, val_list, batch_size=4):
    """
    Returns PyTorch dataloaders for training and validation
    """
    train_dataset = SegmentationDataset(images_dir, masks_dir, train_list)
    val_dataset = SegmentationDataset(images_dir, masks_dir, val_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

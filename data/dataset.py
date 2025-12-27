import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    

    def __init__(
        self,
        root_dir,                # Base folder of the dataset (e.g., "data/raw/spacenet")
        image_dir="images",      # Subfolder containing images
        mask_dir="masks",        # Subfolder containing masks
        transform=None,          # Optional transforms for images
        mask_transform=None,     # Optional transforms for masks
        mode="train"             # Mode: 'train', 'val', or 'test'
    ):
        
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.mode = mode

        # List all image files
        self.image_files = sorted(os.listdir(self.image_dir))

        # For train/val, list masks and check alignment
        if mode != "test":
            self.mask_files = sorted(os.listdir(self.mask_dir))
            assert len(self.image_files) == len(self.mask_files), \
                f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) must match!"

    def __len__(self):
        # Total number of samples
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Test mode: no mask
        if self.mode == "test":
            if self.transform:
                image = self.transform(image)
            return image

        # Load corresponding mask
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

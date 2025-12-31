import os
import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image
import numpy as np
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        # Load the TIFF image
        image_np = tifffile.imread(img_path)
        image = torch.tensor(image_np, dtype=torch.float32).permute(2,0,1)/255.0  # C,H,W

        # Load the mask PNG
        mask = Image.open(mask_path).convert("L")
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        # Optional transform on image
        if self.transform:
            image = self.transform(image)

        return image, mask

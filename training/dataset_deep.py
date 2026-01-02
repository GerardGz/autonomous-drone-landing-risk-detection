import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import random

class DeepGlobeDataset(Dataset):
    """
    Dataset for binary urban vs safe land masks.
    Assumes images and masks are in the same folder:
    119_sat.jpg → 119_mask.png
    """
    def __init__(self, folder, subset_size=None, resize=(256,256), shuffle=True):
        self.folder = Path(folder)
        self.resize = resize

        # Normalization for model input
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Collect images
        all_images = sorted(self.folder.glob("*_sat.jpg"))
        self.samples = []

        for img_path in all_images:
            key = img_path.stem.replace("_sat", "")
            mask_path = self.folder / f"{key}_mask.png"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if shuffle:
            random.shuffle(self.samples)
        if subset_size:
            self.samples = self.samples[:subset_size]

        if len(self.samples) == 0:
            raise RuntimeError(f"No image-mask pairs found in {folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # ---- Image ----
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.resize, Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)/255.0).permute(2,0,1).float()
        img_tensor = self.norm(img_tensor)

        # ---- Mask ----
        mask = Image.open(mask_path).convert("RGB")
        mask = mask.resize(self.resize, Image.NEAREST)
        mask_np = np.array(mask)

        # Binary: Urban = 1, Safe = 0
        urban_mask = (mask_np[:,:,0] == 0) & (mask_np[:,:,1] == 255) & (mask_np[:,:,2] == 255)
        mask_tensor = torch.from_numpy(urban_mask.astype(np.float32)).unsqueeze(0)

        return img_tensor, mask_tensor, img_path.name

# training/dataset_deep_multiclass.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import random

CLASS_COLORS = [
    (0, 255, 255),    # urban_land
    (255, 255, 0),    # agriculture_land
    (255, 0, 255),    # rangeland
    (0, 255, 0),      # forest_land
    (0, 0, 255),      # water
    (255, 255, 255),  # barren_land
    (0, 0, 0),        # unknown
]

class DeepGlobeDataset(Dataset):
    """DeepGlobe multi-class dataset with augmentations."""
    def __init__(self, folder, subset_size=None, resize=(256,256), augment=False, shuffle=True):
        self.folder = Path(folder)
        self.resize = resize
        self.augment = augment

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

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
        img = Image.open(img_path).convert("RGB").resize(self.resize, Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)/255.0).permute(2,0,1).float()
        img_tensor = self.norm(img_tensor)

        # ---- Mask ----
        mask = Image.open(mask_path).convert("RGB").resize(self.resize, Image.NEAREST)
        mask_np = np.array(mask)
        mask_class = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

        for i, color in enumerate(CLASS_COLORS):
            matches = (mask_np[:,:,0]==color[0]) & (mask_np[:,:,1]==color[1]) & (mask_np[:,:,2]==color[2])
            mask_class[matches] = i

        mask_tensor = torch.from_numpy(mask_class).long()

        # ---- Data Augmentation ----
        if self.augment:
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
                mask_tensor = torch.flip(mask_tensor, dims=[1])
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])
                mask_tensor = torch.flip(mask_tensor, dims=[0])

        return img_tensor, mask_tensor, img_path.name

import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        # Image and Mask Directory are subject to change once I actually upload data in the folders
        self,
        root_dir,
        image_dir="images",
        mask_dir="masks",
        transform=None,
        mask_transform=None,
        mode="train"
    ):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.mode = mode

        self.image_files = sorted(os.listdir(self.image_dir))

        if mode != "test":
            self.mask_files = sorted(os.listdir(self.mask_dir))
            assert len(self.image_files) == len(self.mask_files), \
                "Number of images and masks must match"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.mode == "test":
            if self.transform:
                image = self.transform(image)
            return image

        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

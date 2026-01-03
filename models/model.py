
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from training.dataset_deep import DeepGlobeDataset
from models.model import get_model


# Hyperparameters

BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 30
IMG_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Albumentations augmentations

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])


# Augmented Dataset Wrapper

class AugmentedDeepGlobeDataset(DeepGlobeDataset):
    def __init__(self, folder, transforms=None, **kwargs):
        super().__init__(folder, **kwargs)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, mask, img_name = super().__getitem__(idx)

        # Albumentations expects HWC uint8 for image and mask
        import numpy as np
        img_np = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        mask_np = mask.squeeze(0).numpy().astype(np.uint8)

        if self.transforms:
            augmented = self.transforms(image=img_np, mask=mask_np)
            img_tensor = augmented['image']
            mask_tensor = augmented['mask'].long().unsqueeze(0)
        else:
            img_tensor = img
            mask_tensor = mask

        return img_tensor, mask_tensor, img_name


# Training function

def train():
    # Full dataset
    dataset = AugmentedDeepGlobeDataset(
        repo_root / "data/deepglobe/raw/train",
        subset_size=None,
        resize=IMG_SIZE,
        shuffle=True,
        transforms=train_transforms
    )

    # Split train/val 90/10
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply validation transforms
    val_dataset.dataset.transforms = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = get_model(in_channels=3, out_channels=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
            imgs, masks = imgs.to(DEVICE), masks.squeeze(1).to(DEVICE)  # B x H x W
            optimizer.zero_grad()
            outputs = model(imgs)  # B x 7 x H x W
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.squeeze(1).to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save model
    save_path = repo_root / "checkpoints/deepglobe_multiclass_finetuned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")


# Main

if __name__ == "__main__":
    train()

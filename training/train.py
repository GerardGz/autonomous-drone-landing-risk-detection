import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import random

# ---------------------------
# Add repo root to sys.path
# ---------------------------
repo_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(repo_root))

from dataset import get_dataloaders
from models.model import get_model

if __name__ == "__main__":
    # ---------------------------
    # Configuration
    # ---------------------------
    DATA_DIR = repo_root / "data/spacenet/processed"  # folder with images & masks
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    CHECKPOINT_DIR = repo_root / "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # Get all images and create train/val split
    # ---------------------------
    all_images = [f for f in os.listdir(DATA_DIR / "images") if f.endswith(".tif")]
    if len(all_images) == 0:
        raise ValueError(f"No images found in {DATA_DIR / 'images'}")

    random.seed(42)  # reproducible split
    random.shuffle(all_images)

    split_idx = int(0.8 * len(all_images))  # 80% train, 20% val
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"Total images: {len(all_images)}, Train: {len(train_images)}, Val: {len(val_images)}")

    # ---------------------------
    # Dataloaders
    # ---------------------------
    train_loader, val_loader = get_dataloaders(
        images_dir=str(DATA_DIR / "images"),
        masks_dir=str(DATA_DIR / "masks"),
        train_list=train_images,
        val_list=val_images,
        batch_size=BATCH_SIZE
    )

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    # ---------------------------
    # Model, Loss, Optimizer
    # ---------------------------
    model = get_model(in_channels=3, out_channels=1, device=device)
    criterion = nn.BCEWithLogitsLoss()  # binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / train_size

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss /= val_size

        print(f"Epoch [{epoch}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

        # ---------------------------
        # Save checkpoint
        # ---------------------------
        checkpoint_path = CHECKPOINT_DIR / f"unet_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    print("Training completed!")

# train.py
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ---------------------------
# Ensure repo root is on sys.path
# ---------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(REPO_ROOT))

# Imports (now work regardless of working directory)
from dataset import get_dataloaders       # training/dataset.py
from models.model import get_model        # models/model.py

# ---------------------------
# Configuration
# ---------------------------
DATA_DIR = REPO_ROOT / "data/spacenet/processed"  # processed SpaceNet folder
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Dataloaders
# ---------------------------
train_loader, val_loader = get_dataloaders(
    images_dir=DATA_DIR / "images",
    masks_dir=DATA_DIR / "masks",
    train_txt=DATA_DIR / "splits/train.txt",
    val_txt=DATA_DIR / "splits/val.txt",
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

    print(f"Epoch [{epoch}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # ---------------------------
    # Save checkpoint
    # ---------------------------
    checkpoint_path = CHECKPOINT_DIR / f"unet_epoch_{epoch}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

print("Training completed!")

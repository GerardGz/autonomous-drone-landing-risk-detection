# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.loaders import SegmentationDataset
from data.transform import image_transform, mask_transform
from model import get_model

# ---------------------------
# Configuration
# ---------------------------
DATA_DIR = "./data"           # your dataset folder
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Dataset and Dataloaders
# ---------------------------
dataset = SegmentationDataset(
    images_dir=os.path.join(DATA_DIR, "images"),
    masks_dir=os.path.join(DATA_DIR, "masks"),
    image_transform=image_transform,
    mask_transform=mask_transform
)

# Train/Validation split
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unet_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

print("Training completed!")

# evaluate.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# -----------------------------
# Add repo root for imports
# -----------------------------
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from models.model import get_model

# -----------------------------
# Dataset class for evaluation
# -----------------------------
class EvalDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.files = sorted(self.images_dir.glob("*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2,0,1)/255.0

        mask_path = self.masks_dir / img_path.name
        mask = np.array(Image.open(mask_path).convert("L")) // 255  # binary mask
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return img_tensor, mask_tensor, img_path.name

# -----------------------------
# Metrics
# -----------------------------
def dice_coefficient(pred, target, eps=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + eps) / (pred_flat.sum() + target_flat.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + eps) / (union + eps)

# -----------------------------
# Configuration
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = repo_root / "checkpoints/unet_epoch_10.pth"
IMAGES_DIR = repo_root / "data/spacenet/processed/images"
MASKS_DIR = repo_root / "data/spacenet/processed/masks"
BATCH_SIZE = 4

# -----------------------------
# Load model
# -----------------------------
model = get_model(in_channels=3, out_channels=1, device=DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

# -----------------------------
# Dataloader
# -----------------------------
dataset = EvalDataset(IMAGES_DIR, MASKS_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Evaluation loop
# -----------------------------
all_dice = []
all_iou = []

for imgs, masks, names in dataloader:
    imgs = imgs.to(DEVICE)
    with torch.no_grad():
        preds = torch.sigmoid(model(imgs))
        preds_bin = (preds > 0.5).float().cpu()

    for pred_mask, true_mask in zip(preds_bin, masks):
        dice = dice_coefficient(pred_mask, true_mask)
        iou = iou_score(pred_mask, true_mask)
        all_dice.append(dice.item())
        all_iou.append(iou.item())

# -----------------------------
# Report metrics
# -----------------------------
print(f"Average Dice Coefficient: {np.mean(all_dice):.4f}")
print(f"Average IoU Score: {np.mean(all_iou):.4f}")

# -----------------------------
# Visualize a few samples
# -----------------------------
for i in range(min(5, len(dataset))):
    img_tensor, mask_tensor, name = dataset[i]
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
    pred_bin = pred > 0.5

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title(f"Original Image: {name}")
    plt.imshow(np.array(Image.open(IMAGES_DIR / name)))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title(f"Predicted Mask")
    plt.imshow(pred_bin, cmap="gray")
    plt.axis("off")
    plt.show()

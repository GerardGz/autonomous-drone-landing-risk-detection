# evaluate.py
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import sys

# -----------------------------
# Add repo root to Python path
# -----------------------------
repo_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(repo_root))

from models.model import get_model

# -----------------------------
# Dataset class
# -----------------------------
class EvalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, mask_suffix="_mask"):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix

        # Get all images that have a corresponding mask
        all_files = sorted(self.images_dir.glob("*.tif"))
        self.files = []
        for f in all_files:
            # Construct mask filename
            mask_name = f.name.replace(".tif", f"{self.mask_suffix}.tif") if mask_suffix else f.name
            mask_path = self.masks_dir / mask_name
            if mask_path.exists():
                self.files.append(f)
            else:
                print(f"⚠️ Skipping {f.name} — no mask found")

        assert len(self.files) > 0, f"❌ No images with masks found in {images_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        # ---- Load image ----
        with rasterio.open(img_path) as src:
            img = src.read([1,2,3]).astype(np.float32) / 255.0  # [C,H,W]
        img_tensor = torch.from_numpy(img)

        # ---- Load mask ----
        mask_name = img_path.name.replace(".tif", f"{self.mask_suffix}.tif") if self.mask_suffix else img_path.name
        mask_path = self.masks_dir / mask_name
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
        mask = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask)

        # ---- Debug first 3 ----
        if idx < 3:
            print(f"[DEBUG] {img_path.name}")
            print("  Mask unique values:", np.unique(mask))
            print("  Mask coverage (%):", mask.mean()*100)

        return img_tensor, mask_tensor, img_path.name

# -----------------------------
# Metrics
# -----------------------------
def dice_coefficient(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

# -----------------------------
# Config
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
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully!")

# -----------------------------
# Dataloader
# -----------------------------
dataset = EvalDataset(IMAGES_DIR, MASKS_DIR, mask_suffix="_mask")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Evaluation loop
# -----------------------------
all_dice = []
all_iou = []

for imgs, masks, names in dataloader:
    imgs = imgs.to(DEVICE)
    masks = masks.to(DEVICE)

    with torch.no_grad():
        preds = torch.sigmoid(model(imgs))
        preds_bin = (preds > 0.5).float()

    for p, t in zip(preds_bin, masks):
        all_dice.append(dice_coefficient(p, t).item())
        all_iou.append(iou_score(p, t).item())

# -----------------------------
# Report metrics
# -----------------------------
print("\n📊 Evaluation Results")
print(f"Average Dice Coefficient: {np.mean(all_dice):.4f}")
print(f"Average IoU Score:        {np.mean(all_iou):.4f}")

# -----------------------------
# Visualization (first 3 images)
# -----------------------------
for i in range(min(3, len(dataset))):
    img_tensor, mask_tensor, name = dataset[i]
    img_tensor_plot = img_tensor.permute(1,2,0).numpy()  # [H,W,C] for plotting

    with torch.no_grad():
        img_input = img_tensor.unsqueeze(0).to(DEVICE)
        pred = torch.sigmoid(model(img_input))[0,0].cpu().numpy()
        pred_bin = (pred > 0.5)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.imshow(img_tensor_plot)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask_tensor, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Predicted Mask")
    plt.imshow(pred_bin, cmap="gray")
    plt.axis("off")

    plt.show()

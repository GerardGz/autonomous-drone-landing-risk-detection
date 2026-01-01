# evaluate.py
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add repo root so imports work from anywhere
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from models.model import get_model

# ---------------------------
# Configuration
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = repo_root / "checkpoints/unet_epoch_10.pth"
IMAGES_DIR = repo_root / "data/spacenet/processed/images"
NUM_SAMPLES = 10  # number of images to visualize

# ---------------------------
# Load model
# ---------------------------
model = get_model(device=DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

# ---------------------------
# Get sample images
# ---------------------------
all_images = sorted(list(IMAGES_DIR.glob("*.png")))
sample_images = all_images[:NUM_SAMPLES]

# ---------------------------
# Loop over images and visualize
# ---------------------------
for img_path in sample_images:
    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255.0
    img_tensor = img_tensor.to(DEVICE)

    # Prediction
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()

    mask = pred > 0.5  # binary threshold

    # Plot
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.show()  # blocks until window closed

print("Finished visualizing sample images!")

import torch
from torch.utils.data import DataLoader

from models.model import SegmentationModel
from training.dataset import SegmentationDataset
from utils.general_helpers import compute_risk, safety_decision
from utils.image_utils import colorize_mask, overlay_mask  # optional

import cv2
import numpy as np

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load trained model
# -------------------------------
num_classes = 2
model = SegmentationModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# -------------------------------
# Load dataset
# -------------------------------
val_dataset = SegmentationDataset(split="val")  # customize as needed
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# -------------------------------
# Evaluation loop
# -------------------------------
for idx, (image, mask) in enumerate(val_loader):
    image = image.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(image)                   # [1, C, H, W]
        pred_mask = torch.argmax(output, dim=1) # [1, H, W]
        pred_mask = pred_mask.squeeze(0)        # [H, W]
    
    # Compute risk
    risk = compute_risk(pred_mask)
    decision = safety_decision(risk)
    
    print(f"Image {idx}: Risk {risk:.2%} → {decision}")
    
    # Optional: visualize
    # Convert tensor to numpy
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    color_mask = colorize_mask(pred_mask.cpu().numpy())
    overlay = overlay_mask(img_np, color_mask)
    
    # Show or save
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(100)  # press any key to advance

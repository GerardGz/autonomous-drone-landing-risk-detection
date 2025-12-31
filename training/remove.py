# remove_corrupted.py
import os
from PIL import Image

images_dir = "subset/images"
labels_dir = "subset/labels"
masks_dir = "subset/masks"

for img_file in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img_file)
    try:
        Image.open(img_path).convert("RGB")
    except Exception:
        print(f"Removing corrupted file: {img_file}")
        os.remove(img_path)

        # Remove corresponding label
        label_file = img_file.replace("RGB-PanSharpen_", "").replace(".tif", ".geojson")
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            os.remove(label_path)

        # Remove corresponding mask
        mask_file = img_file.replace("RGB-PanSharpen_", "").replace(".tif", ".png")
        mask_path = os.path.join(masks_dir, mask_file)
        if os.path.exists(mask_path):
            os.remove(mask_path)

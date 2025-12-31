print("Test script started")
from training.dataset import SegmentationDataset

dataset = SegmentationDataset("subset/images", "subset/masks")
print(f"Number of samples: {len(dataset)}")
image, mask = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Mask shape: {mask.shape}")

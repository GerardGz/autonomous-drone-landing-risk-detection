from data.dataset import SegmentationDataset
from data.transform import image_transform, mask_transform

def get_datasets(mode="train"):
    datasets = {}

    datasets['spacenet'] = SegmentationDataset(
        root_dir="data/raw/spacenet",
        image_dir="images",
        mask_dir="masks",
        transform=image_transform,
        mask_transform=mask_transform,
        mode=mode
    )

    datasets['deepglobe'] = SegmentationDataset(
        root_dir="data/raw/deepglobe",
        image_dir="images",
        mask_dir="masks",
        transform=image_transform,
        mask_transform=mask_transform,
        mode=mode
    )

    return datasets

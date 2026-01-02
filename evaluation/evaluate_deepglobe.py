
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from training.dataset_deep import DeepGlobeDataset
from models.model import get_model


# RGB colors for visualization

CLASS_COLORS = [
    (0, 255, 255),    # urban
    (255, 255, 0),    # agriculture
    (255, 0, 255),    # rangeland
    (0, 255, 0),      # forest
    (0, 0, 255),      # water
    (255, 255, 255),  # barren
    (0, 0, 0),        # unknown
]

CLASS_NAMES = [
    "urban", "agriculture", "rangeland",
    "forest", "water", "barren", "unknown"
]


# Threat scoring

CLASS_RISK = {
    "urban": 1.0,
    "agriculture": 0.5,
    "rangeland": 0.3,
    "forest": 0.2,
    "water": 0.1,
    "barren": 0.4,
    "unknown": 0.5,
}

def classify_threat(score):
    if score < 0.40:
        return "SAFE "
    elif score < 0.60:
        return "CAUTION "
    else:
        return "DANGEROUS "


# Evaluator

class Evaluator:
    def __init__(self, model, dataset, device="cpu", batch_size=1, visualize=True, max_images=8):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.visualize = visualize
        self.max_images = max_images

    @staticmethod
    def _compute_metrics(pred_classes, target_classes, num_classes=7, eps=1e-6):
        dice_list, iou_list = [], []
        for c in range(num_classes):
            pred_c = (pred_classes == c).float()
            target_c = (target_classes == c).float()
            inter = (pred_c * target_c).sum()
            dice = (2 * inter + eps) / (pred_c.sum() + target_c.sum() + eps)
            iou = (inter + eps) / (pred_c.sum() + target_c.sum() - inter + eps)
            dice_list.append(dice.item())
            iou_list.append(iou.item())
        acc = (pred_classes == target_classes).float().mean().item()
        return dice_list, iou_list, acc

    @staticmethod
    def _compute_threat(pred_classes):
        pred_flat = pred_classes.flatten()
        total_pixels = pred_flat.shape[0]
        threat = 0.0

        for i, name in enumerate(CLASS_NAMES):
            count = (pred_flat == i).sum().item()
            threat += CLASS_RISK[name] * (count / total_pixels)

        return threat

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()

        all_dice, all_iou, all_acc, all_threats = [], [], [], []
        images_shown = 0

        with torch.no_grad():
            for imgs, masks, _ in dataloader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                preds = self.model(imgs)
                pred_classes = torch.argmax(preds, dim=1)

                for pc, mc in zip(pred_classes, masks):
                    dice, iou, acc = self._compute_metrics(pc.cpu(), mc.cpu())
                    threat = self._compute_threat(pc.cpu())

                    all_dice.append(dice)
                    all_iou.append(iou)
                    all_acc.append(acc)
                    all_threats.append(threat)

                if self.visualize and images_shown < self.max_images:
                    show = min(self.max_images - images_shown, imgs.size(0))
                    self._visualize_predictions(
                        pred_classes[:show],
                        imgs[:show],
                        masks[:show]
                    )
                    images_shown += show

        results = {
            "dice_per_class": np.mean(all_dice, axis=0),
            "iou_per_class": np.mean(all_iou, axis=0),
            "pixel_acc": np.mean(all_acc),
            "threat_score_avg": np.mean(all_threats),
            "threat_label": classify_threat(np.mean(all_threats))
        }

        return results

    def _visualize_predictions(self, preds, imgs, masks):
        for i in range(preds.size(0)):
            img = imgs[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            pred = preds[i].cpu().numpy()
            mask = masks[i].cpu().numpy()

            pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
            mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

            for c, color in enumerate(CLASS_COLORS):
                pred_rgb[pred == c] = color
                mask_rgb[mask == c] = color

            threat = self._compute_threat(pred)
            label = classify_threat(threat)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(img)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(mask_rgb)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title(f"Prediction\nThreat: {threat:.2f} ({label})")
            plt.imshow(pred_rgb)
            plt.axis("off")

            plt.tight_layout()
            plt.show()


# Main

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(in_channels=3, out_channels=7).to(device)
    checkpoint = repo_root / "checkpoints/deepglobe_multiclass.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print("✅ Loaded DeepGlobe model")

    dataset = DeepGlobeDataset(
        repo_root / "data/deepglobe/raw/train",
        subset_size=8,
        resize=(256, 256)
    )

    evaluator = Evaluator(model, dataset, device=device)
    results = evaluator.evaluate()

    print("\n DeepGlobe Multi-class Evaluation Results:")
    for n, s in zip(CLASS_NAMES, results["dice_per_class"]):
        print(f"{n:12}: Dice = {s:.4f}")
    for n, s in zip(CLASS_NAMES, results["iou_per_class"]):
        print(f"{n:12}: IoU  = {s:.4f}")

    print(f"Pixel Accuracy: {results['pixel_acc']:.4f}")
    print(f"Average Threat Score: {results['threat_score_avg']:.4f}")
    print(f"Threat Level: {results['threat_label']}")

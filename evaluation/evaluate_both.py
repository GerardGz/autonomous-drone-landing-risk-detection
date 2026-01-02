# evaluate_both_fixed.py
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from training.dataset_deep import DeepGlobeDataset
from models.model import get_model

# -----------------------------
# Evaluator Class
# -----------------------------
class Evaluator:
    def __init__(self, model, dataset, device="cpu", batch_size=1, visualize=True, max_images=3):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.visualize = visualize
        self.max_images = max_images

    @staticmethod
    def _get_metrics(pred, target, eps=1e-6):
        pred_bin = (pred > 0.5).float().view(-1)
        target = target.view(-1)
        intersection = (pred_bin * target).sum()
        dice = (2. * intersection + eps) / (pred_bin.sum() + target.sum() + eps)
        iou = (intersection + eps) / (pred_bin.sum() + target.sum() - intersection + eps)
        acc = (pred_bin == target).float().mean().item()
        return dice.item(), iou.item(), acc

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.model.eval()

        all_dice, all_iou, all_acc = [], [], []
        all_preds_flat, all_targets_flat = [], []
        images_shown = 0

        with torch.no_grad():
            for batch in dataloader:
                # Detect whether dataset has masks
                if len(batch) == 3:
                    imgs, masks, _ = batch
                    masks = masks.to(self.device)
                else:
                    imgs = batch[0]
                    masks = None

                imgs = imgs.to(self.device)
                preds = torch.sigmoid(self.model(imgs))

                # Metrics for datasets with masks
                if masks is not None:
                    for p, t in zip(preds, masks):
                        d, i, a = self._get_metrics(p.cpu(), t.cpu())
                        all_dice.append(d)
                        all_iou.append(i)
                        all_acc.append(a)
                        all_preds_flat.append((p.cpu().numpy() > 0.5).astype(int).ravel())
                        all_targets_flat.append(t.cpu().numpy().astype(int).ravel())

                # Visualize only first few images
                if self.visualize and images_shown < self.max_images:
                    batch_imgs_to_show = min(self.max_images - images_shown, imgs.shape[0])
                    masks_to_show = masks[:batch_imgs_to_show] if masks is not None else None
                    self._visualize_predictions(preds[:batch_imgs_to_show], imgs[:batch_imgs_to_show], masks_to_show)
                    images_shown += batch_imgs_to_show

        # Prepare results
        results = {}
        if masks is not None and len(all_dice) > 0:
            results["dice"] = np.mean(all_dice)
            results["iou"] = np.mean(all_iou)
            results["pixel_acc"] = np.mean(all_acc)
            y_true = np.concatenate(all_targets_flat)
            y_pred = np.concatenate(all_preds_flat)
            results["classification_report"] = classification_report(
                y_true, y_pred, target_names=["Non-Urban", "Urban"], zero_division=0
            )

        return results

    def _visualize_predictions(self, preds, imgs, masks=None):
        num_to_show = preds.shape[0]
        for i in range(num_to_show):
            # Normalize image for plotting
            img = imgs[i].cpu().permute(1,2,0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            pred = preds[i,0].cpu().numpy()

            # Safely process mask
            mask_to_show = None
            if masks is not None:
                mask_to_show = masks[i].cpu().numpy()
                mask_to_show = np.squeeze(mask_to_show)
                if mask_to_show.ndim != 2:
                    raise ValueError(f"Mask has invalid shape {mask_to_show.shape}")

            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.title("Image")
            plt.imshow(img)
            plt.axis("off")

            plt.subplot(1,3,2)
            if mask_to_show is not None:
                plt.title("Ground Truth")
                plt.imshow(mask_to_show, cmap="gray")
            else:
                plt.title("Ground Truth")
                plt.axis("off")

            plt.subplot(1,3,3)
            plt.title("Prediction Heatmap")
            plt.imshow(pred, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(fraction=0.046, pad=0.04)

            # Threat assessment
            unsafe_ratio = (pred > 0.5).mean()
            status = "UNSAFE" if unsafe_ratio > 0.01 else "SAFE"
            plt.suptitle(f"Threat Assessment: {status} ({unsafe_ratio*100:.1f}% urban)", fontsize=14)

            plt.tight_layout()
            plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = get_model(in_channels=3, out_channels=1).to(device)
    checkpoint = repo_root / "checkpoints/deepglobe_finetuned_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print("✅ Loaded fine-tuned model")

    # Evaluate DeepGlobe train (with masks)
    train_dataset = DeepGlobeDataset(repo_root / "data/deepglobe/raw/train", resize=(256,256))
    train_subset = Subset(train_dataset, indices=[0,1,2])
    evaluator_train = Evaluator(model, train_subset, device=device, batch_size=1, visualize=True)
    results_train = evaluator_train.evaluate()
    if results_train:
        print("\n📊 Evaluation Summary (Train subset):")
        print(f"Mean Dice Score: {results_train['dice']:.4f}")
        print(f"Mean IoU Score: {results_train['iou']:.4f}")
        print(f"Pixel Accuracy: {results_train['pixel_acc']:.4f}")
        print("\nDetailed Classification Report:")
        print(results_train["classification_report"])

    # Evaluate DeepGlobe test (images only)
    test_dataset = DeepGlobeDataset(repo_root / "data/deepglobe/raw/test", resize=(256,256))
    # Wrap in subset to only pick first 3 images
    test_subset = Subset(test_dataset, indices=[0,1,2])
    evaluator_test = Evaluator(model, test_subset, device=device, batch_size=1, visualize=True)
    evaluator_test.evaluate()

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Add repo root to sys.path for imports anywhere
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from training.dataset_deep import DeepGlobeDataset
from models.model import get_model

class Evaluator:
    def __init__(self, model, dataset, device="cpu", batch_size=4, visualize=True, max_visualize=3):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.visualize = visualize
        self.max_visualize = max_visualize
        self.visualized_count = 0

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
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        self.model.eval()
        all_dice, all_iou, all_acc = [], [], []
        all_preds_flat, all_targets_flat = [], []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    imgs, masks, _ = batch
                    masks = masks.to(self.device)
                else:
                    imgs = batch[0]
                    masks = None

                imgs = imgs.to(self.device)
                preds = torch.sigmoid(self.model(imgs))

                # Compute metrics if masks exist
                if masks is not None:
                    for p, t in zip(preds, masks):
                        d, i, a = self._get_metrics(p.cpu(), t.cpu())
                        all_dice.append(d)
                        all_iou.append(i)
                        all_acc.append(a)

                        all_preds_flat.append((p.cpu().numpy() > 0.5).astype(int).ravel())
                        all_targets_flat.append(t.cpu().numpy().astype(int).ravel())

                # Visualize images
                if self.visualize and self.visualized_count < self.max_visualize:
                    self._visualize_predictions(preds, batch, masks_exist=(masks is not None))

        # Prepare results
        results = {}
        if len(all_dice) > 0:
            results["dice"] = np.mean(all_dice)
            results["iou"] = np.mean(all_iou)
            results["pixel_acc"] = np.mean(all_acc)
            y_true = np.concatenate(all_targets_flat)
            y_pred = np.concatenate(all_preds_flat)
            results["classification_report"] = classification_report(
                y_true, y_pred, target_names=["Non-Urban", "Urban"], zero_division=0
            )

            # Print summary
            print("\n📊 Evaluation Summary:")
            print(f"Mean Dice Score:    {results['dice']:.4f}")
            print(f"Mean IoU Score:     {results['iou']:.4f}")
            print(f"Pixel Accuracy:     {results['pixel_acc']:.4f}")
            print("\nDetailed Classification Report:")
            print(results["classification_report"])
        else:
            print("⚠️ No metrics computed (dataset may have no masks).")

        return results

    def _visualize_predictions(self, preds, batch, masks_exist=True):
        imgs = batch[0].cpu()
        if masks_exist:
            masks = batch[1].cpu()

        for i in range(min(3 - self.visualized_count, imgs.shape[0])):
            img = imgs[i].permute(1, 2, 0).numpy()
            pred = preds[i,0].cpu().numpy()

            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.title("Image")
            plt.imshow(img)

            if masks_exist:
                mask = masks[i,0].numpy()
                plt.subplot(1,3,2)
                plt.title("Ground Truth")
                plt.imshow(mask, cmap="gray")
            else:
                plt.subplot(1,3,2)
                plt.axis('off')

            plt.subplot(1,3,3)
            plt.title("Prediction")
            plt.imshow(pred, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()

            self.visualized_count += 1

# -----------------------------
# Windows-safe test block
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = get_model(in_channels=3, out_channels=1).to(device)
    checkpoint = repo_root / "checkpoints/deepglobe_finetuned_best.pth"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print("✅ Loaded fine-tuned model")
    else:
        print("⚠️ Checkpoint not found!")

    # Dataset with masks
    dataset = DeepGlobeDataset(repo_root / "data/deepglobe/raw/train", resize=(256,256))

    evaluator = Evaluator(model, dataset, device=device, batch_size=2, visualize=True)
    results = evaluator.evaluate()

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# -----------------------------
# Add repo root for imports
# -----------------------------
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from dataset_deep import DeepGlobeDataset
from models.model import get_model
from losses import BCEWithLogitsDiceLoss
from evaluation.evaluate_deepglobe import Evaluator

def main():
    # -----------------------------
    # Config
    # -----------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    LR = 1e-4
    RESIZE = (256, 256)
    CHECKPOINT_DIR = repo_root / "checkpoints"
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    SUBSET_SIZE = None
    VAL_SPLIT = 0.1  # 10% validation split

    # -----------------------------
    # Dataset & Loader
    # -----------------------------
    full_train_dataset = DeepGlobeDataset(
        repo_root / "data/deepglobe/raw/train",
        resize=RESIZE,
        subset_size=SUBSET_SIZE
    )

    val_size = int(len(full_train_dataset) * VAL_SPLIT)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"✅ Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # -----------------------------
    # Model
    # -----------------------------
    model = get_model(in_channels=3, out_channels=1).to(DEVICE)
    PRETRAINED_PATH = CHECKPOINT_DIR / "spacenet_pretrained.pth"
    if PRETRAINED_PATH.exists():
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
        print("✅ Loaded SpaceNet pre-trained weights")

    # -----------------------------
    # Loss & Optimizer
    # -----------------------------
    criterion = BCEWithLogitsDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -----------------------------
    # Training Loop
    # -----------------------------
    best_val_dice = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        evaluator = Evaluator(model, val_dataset, device=DEVICE, batch_size=BATCH_SIZE)
        results = evaluator.evaluate()
        val_dice = results.get("dice", 0.0)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Train Loss: {sum(train_losses)/len(train_losses):.4f} | Val Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint_path = CHECKPOINT_DIR / "deepglobe_finetuned_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved best model to {checkpoint_path}")

    print("✅ Fine-tuning complete.")


if __name__ == "__main__":
    main()

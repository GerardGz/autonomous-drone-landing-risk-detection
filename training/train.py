import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from models.model import SegmentationModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)               # [B, C, H, W]
        loss = criterion(outputs, masks.long())  # ensure masks are long for CrossEntropyLoss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.long())

            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Dummy segmentation dataset
    # --------------------------------------------------
    num_samples = 20
    num_classes = 2
    batch_size = 4
    image_size = 128

    X = torch.randn(num_samples, 3, image_size, image_size)
    y = torch.randint(0, num_classes, (num_samples, image_size, image_size))

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    # --------------------------------------------------
    # Model, loss, optimizer
    # --------------------------------------------------
    model = SegmentationModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --------------------------------------------------
    # Training loop with loss tracking
    # --------------------------------------------------
    num_epochs = 5
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    # --------------------------------------------------
    # Plot losses after training
    # --------------------------------------------------
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

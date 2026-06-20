from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset import MedicalImageDataset
from src.model import SimpleCNN


def train_model(
    annotations_csv: str,
    image_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    model_out: str = "models/simple_cnn.pt",
) -> None:
    dataset = MedicalImageDataset(annotations_csv, image_dir)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = correct / max(total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(total, 1)
        print(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

    model_out_path = Path(model_out)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out_path)
    print(f"Saved model to {model_out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="artifacts/annotations.csv")
    parser.add_argument("--image-dir", default="data/sample_images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-out", default="models/simple_cnn.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        annotations_csv=args.annotations,
        image_dir=args.image_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_out=args.model_out,
    )

"""Train the MastitisMLP on the tabular milk-sensor dataset.

Reads data/cow_milk_mastitis_dataset.csv, predicts the binary `class1` column
from the milk measurements.

Usage:
    uv run train-tabular [--csv data/cow_milk_mastitis_dataset.csv] [--epochs 100]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from uddernet.models import MastitisMLP, get_device

TARGET = "class1"
DROP_COLS = ["Cow_ID", TARGET]


def load_data(csv_path: Path, batch_size: int):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=DROP_COLS).to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int64)
    feature_names = [c for c in df.columns if c not in DROP_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(s) for s in (X_train, X_val, X_test))

    def to_loader(Xs, ys, shuffle=False):
        ds = TensorDataset(torch.from_numpy(Xs.astype(np.float32)), torch.from_numpy(ys))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    loaders = (
        to_loader(X_train, y_train, shuffle=True),
        to_loader(X_val, y_val),
        to_loader(X_test, y_test),
    )
    return loaders, feature_names, scaler


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(yb)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def main():
    parser = argparse.ArgumentParser(description="Train MastitisMLP on the milk-sensor CSV")
    parser.add_argument("--csv", type=Path, default=Path("data/cow_milk_mastitis_dataset.csv"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/mastitis_mlp.pt"))
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    (train_loader, val_loader, test_loader), feature_names, scaler = load_data(
        args.csv, args.batch_size
    )
    print(f"Features ({len(feature_names)}): {feature_names}")

    model = MastitisMLP(in_features=len(feature_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f}"
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_names": feature_names,
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                },
                args.out,
            )

    # Final evaluation on the held-out test set using the best checkpoint
    checkpoint = torch.load(args.out, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    preds, targets = evaluate(model, test_loader, device)
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print("Test set performance:")
    print(classification_report(targets, preds, target_names=["healthy (0)", "mastitis (1)"]))
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()

"""Train an image classifier on labelled udder images.

Expects a directory (default: data/images) with one subfolder per class:

    data/images/
        positive/   <- mastitis images
        negative/   <- healthy images

Two architectures are available via --model:
    resnet18 (default) -- pretrained ResNet-18, transfer learning (best for
                          small datasets like this one)
    cnn                -- the custom UdderCNN, trained from scratch (baseline)

Usage:
    uv run train-cnn [--model resnet18] [--epochs 20] [--batch-size 32]
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from uddernet.models import IMG_SIZE, build_model, get_device

# ImageNet stats, fine as generic defaults
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def make_loaders(data_dir: Path, batch_size: int, img_size: int, val_fraction: float = 0.2):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    # Two ImageFolder instances over the same files so train/val get different transforms.
    train_ds_full = datasets.ImageFolder(data_dir, transform=train_tf)
    val_ds_full = datasets.ImageFolder(data_dir, transform=val_tf)

    n_val = int(len(train_ds_full) * val_fraction)
    n_train = len(train_ds_full) - n_val
    generator = torch.Generator().manual_seed(42)
    train_split, val_split = random_split(range(len(train_ds_full)), [n_train, n_val], generator=generator)

    train_ds = torch.utils.data.Subset(train_ds_full, train_split.indices)
    val_ds = torch.utils.data.Subset(val_ds_full, val_split.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, train_ds_full.classes


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train an image classifier on positive/negative image folders")
    parser.add_argument("--model", choices=["resnet18", "cnn"], default="resnet18")
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="resnet18 only: train just the new classifier head")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/udder_cnn.pt"))
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(
            f"Data directory '{args.data_dir}' not found. "
            "Expected subfolders 'positive/' and 'negative/' containing images."
        )

    device = get_device()
    print(f"Using device: {device} | model: {args.model}")

    img_size = IMG_SIZE[args.model]
    train_loader, val_loader, classes = make_loaders(args.data_dir, args.batch_size, img_size)
    print(f"Classes: {classes} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model_kwargs = {"freeze_backbone": args.freeze_backbone} if args.model == "resnet18" else {}
    model = build_model(args.model, num_classes=len(classes), **model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
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
                    "classes": classes,
                    "arch": args.model,
                    "img_size": img_size,
                },
                args.out,
            )

    print(f"Best val accuracy: {best_val_acc:.3f} | model saved to {args.out}")


if __name__ == "__main__":
    main()

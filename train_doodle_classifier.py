import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def limit_classes_and_samples(full_dataset: datasets.ImageFolder, max_classes: int, samples_per_class: int):
    targets = full_dataset.targets
    class_to_indices = {}
    for i, t in enumerate(targets):
        class_to_indices.setdefault(t, []).append(i)

    all_class_ids = sorted(class_to_indices.keys())
    if max_classes > 0:
        all_class_ids = all_class_ids[:max_classes]

    keep_indices = []
    for class_id in all_class_ids:
        idxs = class_to_indices[class_id]
        if samples_per_class > 0:
            idxs = idxs[:samples_per_class]
        keep_indices.extend(idxs)

    keep_indices.sort()
    return Subset(full_dataset, keep_indices)


def flatten_subset_indices(ds):
    if isinstance(ds, Subset):
        base_ds, base_indices = flatten_subset_indices(ds.dataset)
        flattened = [base_indices[i] for i in ds.indices]
        return base_ds, flattened
    return ds, list(range(len(ds)))


def build_chunked_train_subset(train_subset: Subset, num_chunks: int, chunk_index: int):
    if num_chunks <= 1:
        return train_subset
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(f"chunk-index must be between 0 and {num_chunks - 1}")

    base_ds, base_indices = flatten_subset_indices(train_subset)
    if not hasattr(base_ds, "targets"):
        raise ValueError("Base dataset must expose targets for chunked training.")

    class_to_positions = {}
    for pos, base_idx in enumerate(base_indices):
        label = base_ds.targets[base_idx]
        class_to_positions.setdefault(label, []).append(pos)

    selected_positions = []
    for positions in class_to_positions.values():
        selected_positions.extend(positions[chunk_index::num_chunks])

    selected_positions.sort()
    if not selected_positions:
        raise ValueError("No training samples selected for this chunk. Reduce num-chunks.")

    return Subset(train_subset, selected_positions)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train doodle object classifier")
    parser.add_argument("--data-dir", type=str, default="doodle", help="Path to class folders")
    parser.add_argument("--output-dir", type=str, default="runs/doodle_classifier", help="Where model and labels are saved")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-classes", type=int, default=0, help="0 means all classes")
    parser.add_argument("--samples-per-class", type=int, default=0, help="0 means all samples")
    parser.add_argument("--num-chunks", type=int, default=1, help="Train data is split into this many chunks")
    parser.add_argument("--chunk-index", type=int, default=0, help="Current chunk index, 0-based")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in output-dir/latest_checkpoint.pt",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    full_dataset = datasets.ImageFolder(root=str(data_dir))
    if len(full_dataset.classes) < 2:
        raise ValueError("Need at least 2 classes to train.")

    working_subset = limit_classes_and_samples(full_dataset, args.max_classes, args.samples_per_class)

    total_size = len(working_subset)
    test_size = int(total_size * args.test_ratio)
    val_size = int(total_size * args.val_ratio)
    train_size = total_size - val_size - test_size
    if train_size <= 0:
        raise ValueError("Train split is empty. Reduce val/test ratios or increase data.")

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset, test_subset = random_split(
        working_subset, [train_size, val_size, test_size], generator=generator
    )
    chunked_train_subset = build_chunked_train_subset(train_subset, args.num_chunks, args.chunk_index)

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = TransformSubset(chunked_train_subset, train_tf)
    val_ds = TransformSubset(val_subset, eval_tf)
    test_ds = TransformSubset(test_subset, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_classes = len(full_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest_checkpoint.pt"
    start_epoch = 1

    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", -1.0)
        print(f"Resumed from {latest_path} at epoch {checkpoint['epoch']}.")

    print(f"Classes: {num_classes} | Total samples used: {total_size}")
    print(f"Train/Val/Test = {train_size}/{val_size}/{test_size}")
    print(f"Chunk: {args.chunk_index + 1}/{args.num_chunks} | Chunk train samples: {len(chunked_train_subset)}")
    print(f"Device: {device}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            },
            latest_path,
        )

    # Test with best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"Best val_acc={best_val_acc:.4f}")
    print(f"Test loss={test_loss:.4f}, test acc={test_acc:.4f}")

    labels_path = output_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(full_dataset.classes, f, ensure_ascii=False, indent=2)

    config_path = output_dir / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved model: {best_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()

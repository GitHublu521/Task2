"""Training and evaluation loops."""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, test_loader, optimizer, device,
                epochs: int = 20, verbose: bool = True):
    """Full training loop with history tracking.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        test_loader: Validation/test DataLoader.
        optimizer: Optimizer instance.
        device: torch.device.
        epochs: Number of training epochs.
        verbose: Whether to show progress bar.

    Returns:
        history: dict with keys "train_loss", "val_loss", "train_acc", "val_acc"
    """
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    epoch_iter = tqdm(range(1, epochs + 1), desc="Training") if verbose else range(1, epochs + 1)

    for epoch in epoch_iter:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_acc=f"{val_acc:.2%}"
            )

    return history

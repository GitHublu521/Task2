"""Task 2: Regularization Ablation Study.

Students must define 4 regularization configurations and train a SimpleCNN
(or custom model) under each to compare their effects.

Students implement: reg_configs list (below)
Provided: training orchestration, plotting, result saving.
"""

import pandas as pd
import torch
import torch.nn as nn

from dataset import get_cifar10_loaders
from models.cnn import SimpleCNN
from train import train_model
from utils import (
    get_device, set_seed, load_config, ensure_results_dir,
    plot_training_curves, plot_bar_comparison,
)


def get_reg_configs():
    """Return a list of 4 regularization configurations to compare.

    Each config is a dict with keys:
        - "name": str — descriptive name (e.g., "No Regularization")
        - "weight_decay": float — L2 regularization strength for optimizer
        - "dropout": float — dropout probability (0.0 = no dropout)

    Example config:
        {"name": "No Regularization", "weight_decay": 0.0, "dropout": 0.0}

    Returns:
        List of 4 config dicts.
    """
    # =========================================================================
    # TODO: Define 4 regularization configurations.
    #
    # Suggested experiments (you may modify):
    #   1. Baseline:       no dropout, no weight decay
    #   2. Dropout only:   dropout=0.3, no weight decay
    #   3. Weight decay:   no dropout, weight_decay=1e-3
    #   4. Both:           dropout=0.3, weight_decay=1e-3
    #
    # Return a list of 4 dicts, each with "name", "weight_decay", "dropout".
    # =========================================================================
    raise NotImplementedError("TODO: Define regularization configs in get_reg_configs()")


def build_model_with_dropout(dropout: float, num_classes: int = 10):
    """Build a SimpleCNN variant with dropout inserted before the final layer.

    This wraps SimpleCNN and inserts a Dropout layer. If you haven't
    implemented SimpleCNN yet, this will raise NotImplementedError.

    Args:
        dropout: Dropout probability.
        num_classes: Number of output classes.

    Returns:
        nn.Module
    """
    model = SimpleCNN(num_classes=num_classes)

    # Insert dropout before the last linear layer in the classifier
    if dropout > 0.0:
        layers = list(model.classifier.children())
        new_layers = []
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear) and i == len(layers) - 1:
                new_layers.append(nn.Dropout(p=dropout))
            new_layers.append(layer)
        model.classifier = nn.Sequential(*new_layers)

    return model


def run_task2():
    """Run the regularization ablation study."""
    print("=" * 60)
    print("Task 2: Regularization Ablation")
    print("=" * 60)

    config = load_config()
    cfg = config["task2"]
    device = get_device()
    ensure_results_dir()

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],
        subset_size=config["subset_size"],
        num_workers=config["num_workers"],
    )

    reg_configs = get_reg_configs()
    assert len(reg_configs) == 4, "Must define exactly 4 regularization configs"

    all_histories = {}
    summary_rows = []

    for reg in reg_configs:
        name = reg["name"]
        print(f"\n--- Config: {name} ---")
        set_seed(config["seed"])

        model = build_model_with_dropout(
            dropout=reg["dropout"],
            num_classes=10,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=reg["weight_decay"],
        )

        history = train_model(
            model, train_loader, test_loader, optimizer,
            device=device, epochs=cfg["epochs"],
        )
        all_histories[name] = history

        final_val_acc = history["val_acc"][-1]
        final_train_acc = history["train_acc"][-1]
        gap = final_train_acc - final_val_acc

        print(f"  Train acc: {final_train_acc:.2%}, Val acc: {final_val_acc:.2%}, Gap: {gap:.2%}")

        summary_rows.append({
            "config": name,
            "weight_decay": reg["weight_decay"],
            "dropout": reg["dropout"],
            "final_train_acc": round(final_train_acc, 4),
            "final_val_acc": round(final_val_acc, 4),
            "generalization_gap": round(gap, 4),
        })

    # Save results
    df = pd.DataFrame(summary_rows)
    df.to_csv("results/task2_summary.csv", index=False)
    print(f"\n  Saved: results/task2_summary.csv")

    # Plots
    plot_training_curves(all_histories, "loss", "results/task2_loss_curves.png",
                         title="Task 2: Training Loss by Regularization")
    plot_training_curves(all_histories, "acc", "results/task2_acc_curves.png",
                         title="Task 2: Accuracy by Regularization")
    plot_bar_comparison(
        [r["config"] for r in summary_rows],
        [r["final_val_acc"] for r in summary_rows],
        "Final Validation Accuracy",
        "results/task2_val_acc_comparison.png",
        title="Task 2: Regularization Comparison",
    )

    print("\nTask 2 complete!")
    return summary_rows

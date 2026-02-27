"""Task 1: Weight Initialization Ablation Study.

This task trains a 20-layer MLP with different weight initialization methods
and compares their activation statistics and training curves.

Students implement: models/mlp.py _init_weights()
Provided: orchestration, plotting, result saving (this file).
"""

import json
import pandas as pd
import torch

from dataset import get_cifar10_loaders
from models import DeepMLP
from train import train_model
from utils import (
    get_device, set_seed, load_config, ensure_results_dir,
    plot_activation_stds, plot_training_curves, plot_bar_comparison,
)


def run_task1():
    """Run the weight initialization ablation study."""
    print("=" * 60)
    print("Task 1: Weight Initialization Ablation")
    print("=" * 60)

    config = load_config()
    cfg = config["task1"]
    device = get_device()
    ensure_results_dir()

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],
        subset_size=config["subset_size"],
        num_workers=config["num_workers"],
    )

    init_methods = cfg["init_methods"]
    all_histories = {}
    all_activation_stds = {}
    summary_rows = []

    for method in init_methods:
        print(f"\n--- Init method: {method} ---")
        set_seed(config["seed"])

        model = DeepMLP(
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            init_method=method,
        )

        # Collect activation stds with a dummy forward pass
        dummy = torch.randn(1, 3, 32, 32).view(1, -1)
        with torch.no_grad():
            model.eval()
            model(dummy.to(next(model.parameters()).device if list(model.parameters()) else "cpu"))
        all_activation_stds[method] = model.activation_stds.copy()

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        history = train_model(
            model, train_loader, test_loader, optimizer,
            device=device, epochs=cfg["epochs"],
        )
        all_histories[method] = history

        final_val_acc = history["val_acc"][-1]
        final_train_loss = history["train_loss"][-1]
        print(f"  Final val acc: {final_val_acc:.2%}")

        summary_rows.append({
            "method": method,
            "final_train_loss": round(final_train_loss, 4),
            "final_val_acc": round(final_val_acc, 4),
        })

    # Save results
    df = pd.DataFrame(summary_rows)
    df.to_csv("results/task1_summary.csv", index=False)
    print(f"\n  Saved: results/task1_summary.csv")

    # Save activation stds
    with open("results/task1_activation_stds.json", "w") as f:
        json.dump(all_activation_stds, f, indent=2)

    # Plots
    plot_activation_stds(all_activation_stds, "results/task1_activation_stds.png")
    plot_training_curves(all_histories, "loss", "results/task1_loss_curves.png",
                         title="Task 1: Training Loss by Init Method")
    plot_training_curves(all_histories, "acc", "results/task1_acc_curves.png",
                         title="Task 1: Accuracy by Init Method")
    plot_bar_comparison(
        [r["method"] for r in summary_rows],
        [r["final_val_acc"] for r in summary_rows],
        "Final Validation Accuracy",
        "results/task1_val_acc_comparison.png",
        title="Task 1: Init Method Comparison",
    )

    print("\nTask 1 complete!")
    return summary_rows

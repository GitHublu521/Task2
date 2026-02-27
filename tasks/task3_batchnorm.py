"""Task 3: BatchNorm Ablation Study.

Compares SimpleCNN (no BN), DeeperCNN (no BN), and DeeperCNN (with BN)
to demonstrate the effect of BatchNorm on training deep CNNs.

Students implement: models/cnn.py SimpleCNN and DeeperCNN
Provided: orchestration, plotting, result saving (this file).
"""

import pandas as pd
import torch

from dataset import get_cifar10_loaders
from models.cnn import SimpleCNN, DeeperCNN
from train import train_model
from utils import (
    get_device, set_seed, load_config, ensure_results_dir,
    plot_training_curves, plot_bar_comparison,
)


def run_task3():
    """Run the BatchNorm ablation study."""
    print("=" * 60)
    print("Task 3: BatchNorm Ablation")
    print("=" * 60)

    config = load_config()
    cfg = config["task3"]
    device = get_device()
    ensure_results_dir()

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],
        subset_size=config["subset_size"],
        num_workers=config["num_workers"],
    )

    model_configs = cfg["configs"]
    all_histories = {}
    summary_rows = []

    for mcfg in model_configs:
        name = mcfg["name"]
        use_bn = mcfg["use_bn"]
        print(f"\n--- Model: {name} ---")
        set_seed(config["seed"])

        if "Simple" in name:
            model = SimpleCNN(num_classes=10)
        else:
            model = DeeperCNN(num_classes=10, use_bn=use_bn)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        history = train_model(
            model, train_loader, test_loader, optimizer,
            device=device, epochs=cfg["epochs"],
        )
        all_histories[name] = history

        final_val_acc = history["val_acc"][-1]
        print(f"  Final val acc: {final_val_acc:.2%}")

        summary_rows.append({
            "model": name,
            "use_bn": use_bn,
            "final_val_acc": round(final_val_acc, 4),
            "best_val_acc": round(max(history["val_acc"]), 4),
        })

    # Save results
    df = pd.DataFrame(summary_rows)
    df.to_csv("results/task3_summary.csv", index=False)
    print(f"\n  Saved: results/task3_summary.csv")

    # Plots
    plot_training_curves(all_histories, "loss", "results/task3_loss_curves.png",
                         title="Task 3: Training Loss — BatchNorm Ablation")
    plot_training_curves(all_histories, "acc", "results/task3_acc_curves.png",
                         title="Task 3: Accuracy — BatchNorm Ablation")
    plot_bar_comparison(
        [r["model"] for r in summary_rows],
        [r["final_val_acc"] for r in summary_rows],
        "Final Validation Accuracy",
        "results/task3_val_acc_comparison.png",
        title="Task 3: BatchNorm Comparison",
    )

    print("\nTask 3 complete!")
    return summary_rows

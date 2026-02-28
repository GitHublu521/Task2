"""Task 4: Robustness to Distribution Shift (CIFAR-10-C).

Students train two models — one with and one without data augmentation —
then evaluate both on CIFAR-10-C corruptions to compare robustness.

Students implement: the training section (marked TODO) below.
Provided: evaluation loop, plotting, result saving.
"""

import json

import numpy as np
import pandas as pd
import torch

from dataset import get_cifar10_loaders, load_cifar10c
from models.cnn import SimpleCNN
from train import train_model, evaluate
from utils import (
    get_device, set_seed, load_config, ensure_results_dir,
    plot_bar_comparison, plot_heatmap,
)


def run_task4():
    """Run the robustness evaluation on CIFAR-10-C."""
    print("=" * 60)
    print("Task 4: Robustness to Distribution Shift")
    print("=" * 60)

    config = load_config()
    cfg = config["task4"]
    device = get_device()
    ensure_results_dir()

    # =========================================================================
    # TODO: Train two SimpleCNN models:
    #
    # 1. model_no_aug: trained WITHOUT data augmentation
    #    - Use get_cifar10_loaders(..., augment=False)
    #
    # 2. model_aug: trained WITH data augmentation
    #    - Use get_cifar10_loaders(..., augment=True)
    #
    # For each model:
    #   - Call set_seed(config["seed"]) before creating the model
    #   - Create a SimpleCNN(num_classes=10)
    #   - Use Adam optimizer with lr=cfg["lr"]
    #   - Train for cfg["epochs"] epochs using train_model()
    #
    # Store the trained models in variables: model_no_aug, model_aug
    # Store their histories in: history_no_aug, history_aug
    #
    # Hint: This is nearly identical code twice, just change augment= flag.
    #
    # NOTE: You may also change the "corruptions" list in config.json to
    # choose which 3 corruption types to evaluate. Pick 3 from different
    # categories (e.g., one noise, one blur, one weather/digital).
    # See "all_corruptions" in config.json for the full list.
    # =========================================================================

    # 训练模型1：不带数据增强
    print("\n--- Training model WITHOUT augmentation ---")
    set_seed(config["seed"])  # 设置随机种子，保证可重复性

    # 获取不带增强的数据加载器
    train_loader_no_aug, test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],
        subset_size=config.get("subset_size"),  # 可能没有，用get避免错误
        num_workers=config["num_workers"],
        augment=False  # 关键：不带数据增强
    )

    # 创建模型
    model_no_aug = SimpleCNN(num_classes=10)

    # 定义优化器
    optimizer_no_aug = torch.optim.Adam(
        model_no_aug.parameters(),
        lr=cfg["lr"]
    )

    # 训练模型
    history_no_aug = train_model(
        model_no_aug,
        train_loader_no_aug,
        test_loader,
        optimizer_no_aug,
        device=device,
        epochs=cfg["epochs"]
    )

    # 训练模型2：带数据增强
    print("\n--- Training model WITH augmentation ---")
    set_seed(config["seed"])  # 同样设置种子，公平比较

    # 获取带增强的数据加载器
    train_loader_aug, _ = get_cifar10_loaders(
        batch_size=config["batch_size"],
        subset_size=config.get("subset_size"),
        num_workers=config["num_workers"],
        augment=True  # 关键：带数据增强
    )

    # 创建模型
    model_aug = SimpleCNN(num_classes=10)

    # 定义优化器
    optimizer_aug = torch.optim.Adam(
        model_aug.parameters(),
        lr=cfg["lr"]
    )

    # 训练模型
    history_aug = train_model(
        model_aug,
        train_loader_aug,
        test_loader,
        optimizer_aug,
        device=device,
        epochs=cfg["epochs"]
    )

    # 注意：这里要删除或注释掉下面的raise语句
    # raise NotImplementedError("TODO: Train model_no_aug and model_aug")


    # --- Evaluation on CIFAR-10-C (provided) ---
    print("\nEvaluating on CIFAR-10-C corruptions...")

    models = {
        "No Augmentation": model_no_aug,
        "With Augmentation": model_aug,
    }

    corruptions = cfg["corruptions"]
    severities = cfg["severities"]
    criterion = torch.nn.CrossEntropyLoss()

    results = {name: {} for name in models}

    for corruption in corruptions:
        for severity in severities:
            try:
                loader = load_cifar10c(
                    corruption, severity,
                    cifar10c_path=cfg["cifar10c_path"],
                    batch_size=config["batch_size"],
                    num_workers=config["num_workers"],
                )
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                continue

            for name, model in models.items():
                _, acc = evaluate(model, loader, criterion, device)
                key = f"{corruption}_s{severity}"
                results[name][key] = round(acc, 4)

        print(f"  Evaluated: {corruption}")

    # Save raw results
    with open("results/task4_cifar10c_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: results/task4_cifar10c_results.json")

    # Build summary: average accuracy per corruption across severities
    summary_rows = []
    for corruption in corruptions:
        for name in models:
            accs = []
            for severity in severities:
                key = f"{corruption}_s{severity}"
                if key in results[name]:
                    accs.append(results[name][key])
            if accs:
                summary_rows.append({
                    "corruption": corruption,
                    "model": name,
                    "avg_accuracy": round(np.mean(accs), 4),
                })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv("results/task4_summary.csv", index=False)
        print(f"  Saved: results/task4_summary.csv")

        # Heatmap: corruption × model (averaged over severities)
        model_names = list(models.keys())
        heatmap_data = []
        for corruption in corruptions:
            row = []
            for name in model_names:
                accs = [results[name].get(f"{corruption}_s{s}", 0)
                        for s in severities]
                row.append(np.mean(accs) * 100 if accs else 0)
            heatmap_data.append(row)

        plot_heatmap(
            np.array(heatmap_data),
            corruptions, model_names,
            "results/task4_heatmap.png",
            title="Task 4: CIFAR-10-C Accuracy (%) by Corruption",
        )

        # Bar chart: overall average
        for name in model_names:
            all_accs = [v for k, v in results[name].items()]
            if all_accs:
                avg = np.mean(all_accs)
                print(f"  {name} — Overall CIFAR-10-C avg: {avg:.2%}")

    # Save clean accuracy comparison
    clean_test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )[1]

    clean_accs = {}
    for name, model in models.items():
        _, acc = evaluate(model, clean_test_loader, criterion, device)
        clean_accs[name] = round(acc, 4)
        print(f"  {name} — Clean test acc: {acc:.2%}")

    plot_bar_comparison(
        list(clean_accs.keys()),
        list(clean_accs.values()),
        "Clean Test Accuracy",
        "results/task4_clean_acc.png",
        title="Task 4: Clean vs Augmented Model Accuracy",
    )

    print("\nTask 4 complete!")
    return results

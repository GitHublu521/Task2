"""Task 5: Hyperparameter Search with Optuna.

Students implement the objective function for Optuna to search over
learning rate, weight decay, dropout, and batch size.

Students implement: objective() function
Provided: Optuna study setup, best-model retraining, result saving.
"""

import json

import optuna
import pandas as pd
import torch

from dataset import get_cifar10_loaders
from models.cnn import SimpleCNN
from train import train_model
from utils import (
    get_device, set_seed, load_config, ensure_results_dir,
    plot_training_curves, plot_bar_comparison,
)


def objective(trial, config, device):
    """Optuna objective function for hyperparameter optimization.

    Use trial.suggest_* methods to sample hyperparameters, build a model,
    train it, and return the validation accuracy to maximize.

    Args:
        trial: An Optuna trial object.
        config: Full config dict (from config.json).
        device: torch.device to train on.

    Returns:
        float: Final validation accuracy (to be maximized).

    Hyperparameters to search:
        - lr: trial.suggest_float("lr", low, high, log=True)
        - weight_decay: trial.suggest_float("weight_decay", low, high, log=True)
        - dropout: trial.suggest_float("dropout", low, high)
        - batch_size: trial.suggest_categorical("batch_size", choices)

    Use config["task5"] to get the search ranges:
        - cfg["lr_range"] = [min_lr, max_lr]
        - cfg["weight_decay_range"] = [min_wd, max_wd]
        - cfg["dropout_range"] = [min_do, max_do]
        - cfg["batch_size_choices"] = [32, 64, 128, 256]
        - cfg["epochs_per_trial"] = number of epochs to train
    """
    cfg = config["task5"]

    # =========================================================================
    # TODO: Implement the Optuna objective function.
    #
    # Steps:
    #   1. Use trial.suggest_float / trial.suggest_categorical to sample:
    #      - lr (log scale)
    #      - weight_decay (log scale)
    #      - dropout
    #      - batch_size
    #
    #   2. Create data loaders with the sampled batch_size:
    #      train_loader, test_loader = get_cifar10_loaders(
    #          batch_size=batch_size,
    #          subset_size=config["subset_size"],
    #          num_workers=config["num_workers"],
    #      )
    #
    #   3. Build a SimpleCNN model (you may add dropout manually or use
    #      the build_model_with_dropout from task2 as reference)
    #
    #   4. Create an Adam optimizer with the sampled lr and weight_decay
    #
    #   5. Train using train_model(..., epochs=cfg["epochs_per_trial"])
    #
    #   6. Return the final validation accuracy: history["val_acc"][-1]
    #
    # =========================================================================
    raise NotImplementedError("TODO: Implement Optuna objective function")


def run_task5():
    """Run hyperparameter search with Optuna."""
    print("=" * 60)
    print("Task 5: Hyperparameter Search with Optuna")
    print("=" * 60)

    config = load_config()
    cfg = config["task5"]
    device = get_device()
    ensure_results_dir()
    set_seed(config["seed"])

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="cifar10_hparam_search",
    )

    # Suppress Optuna logging for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nRunning {cfg['n_trials']} trials...")
    study.optimize(
        lambda trial: objective(trial, config, device),
        n_trials=cfg["n_trials"],
        show_progress_bar=True,
    )

    # Report best trial
    best = study.best_trial
    print(f"\nBest trial #{best.number}:")
    print(f"  Value (val acc): {best.value:.4f}")
    print(f"  Params: {json.dumps(best.params, indent=4)}")

    # Save trial history
    trials_data = []
    for trial in study.trials:
        row = {"trial": trial.number, "value": trial.value}
        row.update(trial.params)
        trials_data.append(row)

    df = pd.DataFrame(trials_data)
    df.to_csv("results/task5_trials.csv", index=False)
    print(f"  Saved: results/task5_trials.csv")

    # Save best params
    best_params = {"best_value": best.value, **best.params}
    with open("results/task5_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved: results/task5_best_params.json")

    # Retrain best model for full epochs
    print(f"\nRetraining best model for {cfg['final_epochs']} epochs...")
    set_seed(config["seed"])

    batch_size = int(best.params.get("batch_size", 128))
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        subset_size=config["subset_size"],
        num_workers=config["num_workers"],
    )

    model = SimpleCNN(num_classes=10)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best.params["lr"],
        weight_decay=best.params["weight_decay"],
    )

    history = train_model(
        model, train_loader, test_loader, optimizer,
        device=device, epochs=cfg["final_epochs"],
    )

    final_acc = history["val_acc"][-1]
    print(f"  Best model final val acc: {final_acc:.2%}")

    # Plot final training curves
    plot_training_curves(
        {"Best HP Model": history}, "acc",
        "results/task5_best_model_curves.png",
        title="Task 5: Best Model Training Curves",
    )

    print("\nTask 5 complete!")
    return best_params

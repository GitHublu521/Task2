"""Utility functions: device detection, seeding, and plotting helpers."""

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_device():
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str = "config.json") -> dict:
    """Load experiment configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def ensure_results_dir():
    """Create results/ directory if it doesn't exist."""
    os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_activation_stds(activation_stds: dict, save_path: str):
    """Plot standard deviation of activations per layer for each init method.

    Args:
        activation_stds: {method_name: [std_layer1, std_layer2, ...]}
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, stds in activation_stds.items():
        ax.plot(range(1, len(stds) + 1), stds, marker="o", label=method)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Activation Std Dev")
    ax.set_title("Activation Standard Deviations by Layer")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(histories: dict, metric: str, save_path: str,
                         title: str = None):
    """Plot training curves for multiple experiments.

    Args:
        histories: {name: {"train_loss": [...], "val_loss": [...],
                           "train_acc": [...], "val_acc": [...]}}
        metric: One of "loss" or "acc".
        save_path: Path to save the figure.
        title: Optional title override.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories.items():
        axes[0].plot(hist[f"train_{metric}"], label=f"{name} (train)")
        axes[0].plot(hist[f"val_{metric}"], "--", label=f"{name} (val)")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(metric.capitalize())
    axes[0].set_title(title or f"Training & Validation {metric.capitalize()}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Second panel: validation metric only (cleaner comparison)
    for name, hist in histories.items():
        axes[1].plot(hist[f"val_{metric}"], label=name)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(f"Val {metric.capitalize()}")
    axes[1].set_title(f"Validation {metric.capitalize()} Comparison")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_bar_comparison(names: list, values: list, ylabel: str,
                        save_path: str, title: str = None):
    """Bar chart comparing final metrics across experiments.

    Args:
        names: List of experiment names.
        values: List of metric values.
        ylabel: Y-axis label.
        save_path: Path to save the figure.
        title: Optional title override.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2%}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_heatmap(data: np.ndarray, row_labels: list, col_labels: list,
                 save_path: str, title: str = None, fmt: str = ".1f",
                 cmap: str = "YlOrRd"):
    """Plot a heatmap (e.g., corruption type × severity).

    Args:
        data: 2D numpy array of values.
        row_labels: Labels for rows.
        col_labels: Labels for columns.
        save_path: Path to save the figure.
        title: Optional title.
        fmt: Number format string.
        cmap: Matplotlib colormap.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.2),
                                    max(6, len(row_labels) * 0.5)))
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = f"{data[i, j]:{fmt}}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7)

    ax.set_title(title or "Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

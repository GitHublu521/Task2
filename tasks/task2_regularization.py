"""Task 2: Regularization Ablation Study.

正则化消融研究任务

学生需要定义4种正则化配置，并使用 SimpleCNN 训练模型，
比较每种配置的效果。

学生需要实现: reg_configs 列表 (下面的函数)
已提供: 训练编排、绘图、结果保存
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
    """返回4种要比较的正则化配置列表。

    每个配置是一个字典，包含以下键:
        - "name": str — 描述性名称 (例如 "No Regularization")
        - "weight_decay": float — 优化器的 L2 正则化强度
        - "dropout": float — dropout 概率 (0.0 = 无 dropout)

    示例配置:
        {"name": "No Regularization", "weight_decay": 0.0, "dropout": 0.0}

    返回:
        包含4个配置字典的列表
    """
    # =========================================================================
    #   1. 基线:       无 dropout, 无 weight decay
    #   2. 仅 Dropout:   dropout=0.3, 无 weight decay
    #   3. 仅 Weight decay:   无 dropout, weight_decay=1e-3
    #   4. 两者结合:           dropout=0.3, weight_decay=1e-3
    #
    # 返回包含4个字典的列表，每个字典有 "name", "weight_decay", "dropout"
    # =========================================================================

    # 定义4种正则化配置
    configs = [
        {
            "name": "Baseline (No Regularization)",  # 基线：无正则化
            "weight_decay": 0.0,                     # 无 L2 正则化
            "dropout": 0.0                           # 无 Dropout
        },
        {
            "name": "Dropout Only",                  # 仅使用 Dropout
            "weight_decay": 0.0,                     # 无 L2 正则化
            "dropout": 0.3                           # dropout概率30%
        },
        {
            "name": "Weight Decay Only",             # 仅使用 L2 正则化
            "weight_decay": 1e-3,                    # L2正则化强度0.001
            "dropout": 0.0                           # 无 Dropout
        },
        {
            "name": "Both (Dropout + Weight Decay)", # 同时使用两种正则化
            "weight_decay": 1e-3,                    # L2正则化强度0.001
            "dropout": 0.3                           # dropout概率30%
        }
    ]

    return configs


def build_model_with_dropout(dropout: float, num_classes: int = 10):
    """构建一个在最后一层前插入 Dropout 的 SimpleCNN 变体。

    这个函数包装 SimpleCNN 并插入 Dropout 层。
    如果你还没有实现 SimpleCNN，这个函数会抛出 NotImplementedError。

    Args:
        dropout: Dropout 概率（0.0 表示不添加 Dropout）
        num_classes: 输出类别数量

    Returns:
        nn.Module: 构建好的模型
    """
    # 创建基础的 SimpleCNN 模型
    model = SimpleCNN(num_classes=num_classes)

    # 在最后一个线性层之前插入 Dropout
    if dropout > 0.0:
        # 获取分类器的所有层
        layers = list(model.classifier.children())
        new_layers = []  # 存储新的层序列
        
        # 遍历所有层
        for i, layer in enumerate(layers):
            # 如果是最后一个线性层，在此之前插入 Dropout
            if isinstance(layer, nn.Linear) and i == len(layers) - 1:
                new_layers.append(nn.Dropout(p=dropout))  # 插入 Dropout
            new_layers.append(layer)  # 添加原始层
        
        # 用新的层序列替换原来的分类器
        model.classifier = nn.Sequential(*new_layers)

    return model


def run_task2():
    """运行正则化消融研究"""
    print("=" * 60)
    print("Task 2: Regularization Ablation")
    print("=" * 60)

    # 加载配置
    config = load_config()
    cfg = config["task2"]              # 获取任务2的特定配置
    device = get_device()              # 获取设备 (CPU/GPU)
    ensure_results_dir()               # 确保 results 目录存在

    # 获取数据加载器
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],
        subset_size=config["subset_size"],
        num_workers=config["num_workers"],
    )

    # 获取正则化配置（学生需要实现的函数）
    reg_configs = get_reg_configs()
    assert len(reg_configs) == 4, "Must define exactly 4 regularization configs"  # 确保有4种配置

    all_histories = {}      # 存储每种配置的训练历史
    summary_rows = []        # 存储汇总结果

    # 遍历每种正则化配置
    for reg in reg_configs:
        name = reg["name"]
        print(f"\n--- Config: {name} ---")
        set_seed(config["seed"])      # 固定随机种子

        # 创建模型（带 Dropout）
        model = build_model_with_dropout(
            dropout=reg["dropout"],
            num_classes=10,
        )

        # 创建优化器（带 weight decay）
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=reg["weight_decay"],  # L2 正则化
        )

        # 训练模型
        history = train_model(
            model, train_loader, test_loader, optimizer,
            device=device, epochs=cfg["epochs"],
        )
        all_histories[name] = history

        # 获取最终准确率
        final_val_acc = history["val_acc"][-1]
        final_train_acc = history["train_acc"][-1]
        gap = final_train_acc - final_val_acc   # 泛化差距 = 训练准确率 - 验证准确率

        print(f"  Train acc: {final_train_acc:.2%}, Val acc: {final_val_acc:.2%}, Gap: {gap:.2%}")

        # 保存结果到汇总行
        summary_rows.append({
            "config": name,
            "weight_decay": reg["weight_decay"],
            "dropout": reg["dropout"],
            "final_train_acc": round(final_train_acc, 4),
            "final_val_acc": round(final_val_acc, 4),
            "generalization_gap": round(gap, 4),  # 泛化差距
        })

    # ========== 保存结果文件 ==========
    
    # 保存汇总表 (CSV格式)
    df = pd.DataFrame(summary_rows)
    df.to_csv("results/task2_summary.csv", index=False)
    print(f"\n  Saved: results/task2_summary.csv")

    # ========== 生成可视化图表 ==========
    
    # 图1: 训练损失曲线对比
    plot_training_curves(all_histories, "loss", "results/task2_loss_curves.png",
                         title="Task 2: Training Loss by Regularization")
    
    # 图2: 准确率曲线对比
    plot_training_curves(all_histories, "acc", "results/task2_acc_curves.png",
                         title="Task 2: Accuracy by Regularization")
    
    # 图3: 最终验证准确率柱状图对比
    plot_bar_comparison(
        [r["config"] for r in summary_rows],
        [r["final_val_acc"] for r in summary_rows],
        "Final Validation Accuracy",
        "results/task2_val_acc_comparison.png",
        title="Task 2: Regularization Comparison",
    )

    print("\nTask 2 complete!")
    return summary_rows


"""
代码结构说明:
1. get_reg_configs(): 定义4种正则化配置
   - 基线（无正则化）
   - 仅 Dropout（30% dropout）
   - 仅 Weight Decay（L2 = 0.001）
   - 两者结合（dropout=0.3, weight_decay=0.001）

2. build_model_with_dropout(): 在 SimpleCNN 的最后一层前插入 Dropout
   - 用于需要 Dropout 的配置
   - 不需要时 dropout=0.0 则不插入

3. run_task2(): 主函数
   - 对每种配置训练模型
   - 记录训练/验证准确率
   - 计算泛化差距（过拟合程度）
   - 生成对比图表

关键概念:
- Dropout: 训练时随机丢弃神经元，防止过拟合
- Weight Decay (L2 正则化): 惩罚大权重，限制模型复杂度
- 泛化差距: 训练准确率 - 验证准确率，越小表示过拟合越轻
- 正则化的目标: 缩小泛化差距，提高验证准确率
"""

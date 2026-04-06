"""Task 3: BatchNorm Ablation Study.

Compares SimpleCNN (no BN), DeeperCNN (no BN), and DeeperCNN (with BN)
to demonstrate the effect of BatchNorm on training deep CNNs.

Students implement: models/cnn.py SimpleCNN and DeeperCNN
Provided: orchestration, plotting, result saving (this file).
"""

"""
任务3：批归一化消融实验

比较 SimpleCNN（无BN）、DeeperCNN（无BN）和 DeeperCNN（有BN）
以展示批归一化对训练深层CNN的影响。

学生需要实现：models/cnn.py 中的 SimpleCNN 和 DeeperCNN
本文件提供：编排流程、绘图、结果保存功能
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
    """运行批归一化消融实验"""
    
    print("=" * 60)
    print("Task 3: BatchNorm Ablation")
    print("Task 3: 批归一化消融实验")
    print("=" * 60)

    # 加载配置文件
    config = load_config()
    
    # 获取task3的特定配置
    cfg = config["task3"]
    
    # 获取计算设备（CPU/CUDA/MPS）
    device = get_device()
    
    # 确保结果目录存在
    ensure_results_dir()

    # 加载CIFAR-10数据集
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],      # 批大小
        subset_size=config["subset_size"],    # 数据集子集大小（用于快速实验）
        num_workers=config["num_workers"],    # 数据加载线程数
    )

    # 获取模型配置列表
    model_configs = cfg["configs"]
    
    # 存储所有训练历史
    all_histories = {}
    
    # 存储汇总结果（用于CSV）
    summary_rows = []

    # 遍历每个模型配置进行训练
    for mcfg in model_configs:
        name = mcfg["name"]          # 模型名称
        use_bn = mcfg["use_bn"]      # 是否使用批归一化
        
        print(f"\n--- Model: {name} ---")
        
        # 设置随机种子（确保可重复性）
        set_seed(config["seed"])

        # 根据配置创建模型
        if "Simple" in name:
            # SimpleCNN：浅层网络，无BN
            model = SimpleCNN(num_classes=10)
        else:
            # DeeperCNN：深层网络，可选择是否使用BN
            model = DeeperCNN(num_classes=10, use_bn=use_bn)

        # 创建优化器（使用Adam）
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        # 训练模型
        history = train_model(
            model, train_loader, test_loader, optimizer,
            device=device, epochs=cfg["epochs"],
        )
        
        # 保存训练历史
        all_histories[name] = history

        # 获取最终验证准确率
        final_val_acc = history["val_acc"][-1]
        print(f"  Final val acc: {final_val_acc:.2%}")

        # 保存汇总信息
        summary_rows.append({
            "model": name,                          # 模型名称
            "use_bn": use_bn,                      # 是否使用BN
            "final_val_acc": round(final_val_acc, 4),   # 最终验证准确率
            "best_val_acc": round(max(history["val_acc"]), 4),  # 最佳验证准确率
        })

    # ==================== 保存结果 ====================
    
    # 保存CSV汇总表
    df = pd.DataFrame(summary_rows)
    df.to_csv("results/task3_summary.csv", index=False)
    print(f"\n  Saved: results/task3_summary.csv")

    # ==================== 绘制图表 ====================
    
    # 图1：训练损失曲线对比
    plot_training_curves(
        all_histories, 
        "loss",                                    # 绘制损失曲线
        "results/task3_loss_curves.png",
        title="Task 3: Training Loss — BatchNorm Ablation"
    )
    
    # 图2：准确率曲线对比
    plot_training_curves(
        all_histories, 
        "acc",                                     # 绘制准确率曲线
        "results/task3_acc_curves.png",
        title="Task 3: Accuracy — BatchNorm Ablation"
    )
    
    # 图3：最终验证准确率条形图对比
    plot_bar_comparison(
        [r["model"] for r in summary_rows],        # x轴：模型名称
        [r["final_val_acc"] for r in summary_rows], # y轴：最终准确率
        "Final Validation Accuracy",               # y轴标签
        "results/task3_val_acc_comparison.png",
        title="Task 3: BatchNorm Comparison",
    )

    print("\nTask 3 complete!")
    return summary_rows

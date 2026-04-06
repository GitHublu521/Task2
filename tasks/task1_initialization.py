"""Task 1: Weight Initialization Ablation Study.

权重初始化消融研究任务

本任务使用不同的权重初始化方法训练一个20层的MLP，
并比较它们的激活统计量和训练曲线。

需要实现: models/mlp.py _init_weights()
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
    """运行权重初始化消融研究"""
    print("=" * 60)
    print("Task 1: Weight Initialization Ablation")
    print("=" * 60)

    # 加载配置文件
    config = load_config()
    cfg = config["task1"]          # 获取任务1的特定配置
    device = get_device()          # 获取设备 (CPU/GPU)
    ensure_results_dir()           # 确保 results 目录存在

    # 获取数据加载器 (使用 CIFAR-10 的子集以加速实验)
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],
        subset_size=config["subset_size"],
        num_workers=config["num_workers"],
    )

    init_methods = cfg["init_methods"]   # 要测试的初始化方法列表
    all_histories = {}                    # 存储每种方法的训练历史
    all_activation_stds = {}              # 存储每种方法的激活标准差
    summary_rows = []                     # 存储汇总结果

    # 遍历每种初始化方法
    for method in init_methods:
        print(f"\n--- Init method: {method} ---")
        set_seed(config["seed"])          # 固定随机种子，确保可复现

        # 创建模型，指定初始化方法
        model = DeepMLP(
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            init_method=method,
        )

        # 收集激活标准差：使用一个虚拟的前向传播
        # 将输入从 (1,3,32,32) 展平为 (1, 3*32*32)
        dummy = torch.randn(1, 3, 32, 32).view(1, -1)
        with torch.no_grad():              # 不需要梯度，只做前向传播
            model.eval()                   # 切换到评估模式
            # 将虚拟输入移动到与模型参数相同的设备
            model(dummy.to(next(model.parameters()).device if list(model.parameters()) else "cpu"))
        # 保存该方法的激活标准差
        all_activation_stds[method] = model.activation_stds.copy()

        # 训练模型
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        history = train_model(
            model, train_loader, test_loader, optimizer,
            device=device, epochs=cfg["epochs"],
        )
        all_histories[method] = history

        # 获取最终验证准确率和最终训练损失
        final_val_acc = history["val_acc"][-1]
        final_train_loss = history["train_loss"][-1]
        print(f"  Final val acc: {final_val_acc:.2%}")

        # 保存到汇总行
        summary_rows.append({
            "method": method,
            "final_train_loss": round(final_train_loss, 4),
            "final_val_acc": round(final_val_acc, 4),
        })

    # ========== 保存结果文件 ==========
    
    # 保存汇总表 (CSV格式)
    df = pd.DataFrame(summary_rows)
    df.to_csv("results/task1_summary.csv", index=False)
    print(f"\n  Saved: results/task1_summary.csv")

    # 保存激活标准差 (JSON格式)
    with open("results/task1_activation_stds.json", "w") as f:
        json.dump(all_activation_stds, f, indent=2)

    # ========== 生成可视化图表 ==========
    
    # 图1: 各层激活标准差对比
    plot_activation_stds(all_activation_stds, "results/task1_activation_stds.png")
    
    # 图2: 训练损失曲线对比
    plot_training_curves(all_histories, "loss", "results/task1_loss_curves.png",
                         title="Task 1: Training Loss by Init Method")
    
    # 图3: 验证准确率曲线对比
    plot_training_curves(all_histories, "acc", "results/task1_acc_curves.png",
                         title="Task 1: Accuracy by Init Method")
    
    # 图4: 最终验证准确率柱状图对比
    plot_bar_comparison(
        [r["method"] for r in summary_rows],
        [r["final_val_acc"] for r in summary_rows],
        "Final Validation Accuracy",
        "results/task1_val_acc_comparison.png",
        title="Task 1: Init Method Comparison",
    )

    print("\nTask 1 complete!")
    return summary_rows


"""
代码结构说明:
1. 加载配置 → 获取数据 → 创建模型
2. 对每种初始化方法:
   - 创建模型 (使用指定的初始化方法)
   - 执行虚拟前向传播收集激活标准差
   - 训练模型并记录历史
   - 保存结果
3. 生成可视化图表:
   - 激活标准差图: 展示梯度传播情况
   - 损失曲线: 展示收敛速度
   - 准确率曲线: 展示学习效果
   - 柱状图: 直观对比最终性能

关键概念:
- 激活标准差: 衡量每层输出的方差，太小说明梯度消失，太大说明梯度爆炸
- Kaiming初始化: 针对ReLU优化的初始化方法
- Xavier初始化: 针对tanh/sigmoid优化的初始化方法
- 默认初始化: 均匀分布[-0.05,0.05]，在深层网络中效果很差
"""

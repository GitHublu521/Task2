"""Task 4: Robustness to Distribution Shift (CIFAR-10-C).

Students train two models — one with and one without data augmentation —
then evaluate both on CIFAR-10-C corruptions to compare robustness.

Students implement: the training section (marked TODO) below.
Provided: evaluation loop, plotting, result saving.
"""

"""
任务4：对分布偏移的鲁棒性（CIFAR-10-C）

训练两个模型 — 一个带数据增强，一个不带数据增强 —
然后在CIFAR-10-C的扰动数据上评估，比较两者的鲁棒性。

本文件提供：评估循环、绘图、结果保存
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
    """运行CIFAR-10-C的鲁棒性评估"""
    
    print("=" * 60)
    print("Task 4: Robustness to Distribution Shift")
    print("任务4：对分布偏移的鲁棒性")
    print("=" * 60)

    # ==================== 初始化配置 ====================
    config = load_config()
    cfg = config["task4"]          # 任务4的特定配置
    device = get_device()           # 自动选择计算设备
    ensure_results_dir()            # 确保结果目录存在

    # ==================== 训练部分（学生需要实现） ====================
    """
    TODO: 训练两个SimpleCNN模型：
    
    1. model_no_aug: 不带数据增强训练
       - 使用 get_cifar10_loaders(..., augment=False)
    
    2. model_aug: 带数据增强训练
       - 使用 get_cifar10_loaders(..., augment=True)
    
    每个模型需要：
       - 创建模型前调用 set_seed(config["seed"])
       - 创建 SimpleCNN(num_classes=10)
       - 使用 Adam 优化器，lr=cfg["lr"]
       - 训练 cfg["epochs"] 个epoch，使用 train_model()
    
    将训练好的模型存储在变量：model_no_aug, model_aug
    将训练历史存储在：history_no_aug, history_aug
    """
    
    # 训练模型1：不带数据增强
    print("\n--- Training model WITHOUT augmentation ---")
    print("--- 训练不带数据增强的模型 ---")
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
    print("--- 训练带数据增强的模型 ---")
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

    # ==================== CIFAR-10-C 评估（已提供） ====================
    print("\nEvaluating on CIFAR-10-C corruptions...")
    print("在CIFAR-10-C扰动数据上评估...")

    # 定义要评估的两个模型
    models = {
        "No Augmentation": model_no_aug,      # 无数据增强
        "With Augmentation": model_aug,        # 有数据增强
    }

    # 从配置中读取扰动类型和严重程度
    corruptions = cfg["corruptions"]
    severities = cfg["severities"]
    
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 存储结果的嵌套字典
    results = {name: {} for name in models}

    # 遍历每种扰动类型和每个严重程度
    for corruption in corruptions:
        for severity in severities:
            try:
                # 加载对应扰动和严重程度的CIFAR-10-C数据
                loader = load_cifar10c(
                    corruption, severity,
                    cifar10c_path=cfg["cifar10c_path"],
                    batch_size=config["batch_size"],
                    num_workers=config["num_workers"],
                )
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                continue

            # 对两个模型分别评估
            for name, model in models.items():
                _, acc = evaluate(model, loader, criterion, device)
                key = f"{corruption}_s{severity}"  # 例如: "gaussian_noise_s3"
                results[name][key] = round(acc, 4)

        print(f"  Evaluated: {corruption}")

    # ==================== 保存结果 ====================
    
    # 保存原始JSON结果
    with open("results/task4_cifar10c_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: results/task4_cifar10c_results.json")

    # 构建汇总表：每种扰动类型下各模型平均准确率
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
                    "corruption": corruption,      # 扰动类型
                    "model": name,                 # 模型名称
                    "avg_accuracy": round(np.mean(accs), 4),  # 平均准确率
                })

    if summary_rows:
        # 保存CSV汇总表
        df = pd.DataFrame(summary_rows)
        df.to_csv("results/task4_summary.csv", index=False)
        print(f"  Saved: results/task4_summary.csv")

        # ==================== 绘制热力图 ====================
        # 热力图：扰动类型 × 模型（按严重程度平均）
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
            title_cn="任务4：各扰动类型下的CIFAR-10-C准确率(%)"
        )

        # 打印总体平均准确率
        for name in model_names:
            all_accs = [v for k, v in results[name].items()]
            if all_accs:
                avg = np.mean(all_accs)
                print(f"  {name} — Overall CIFAR-10-C avg: {avg:.2%}")

    # ==================== 干净数据准确率对比 ====================
    # 在干净测试集上评估，对比基准性能
    clean_test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )[1]  # [1] 是 test_loader

    clean_accs = {}
    for name, model in models.items():
        _, acc = evaluate(model, clean_test_loader, criterion, device)
        clean_accs[name] = round(acc, 4)
        print(f"  {name} — Clean test acc: {acc:.2%}")

    # 绘制干净数据准确率条形图
    plot_bar_comparison(
        list(clean_accs.keys()),
        list(clean_accs.values()),
        "Clean Test Accuracy",
        "results/task4_clean_acc.png",
        title="Task 4: Clean vs Augmented Model Accuracy",
        title_cn="任务4：干净测试集准确率对比"
    )

    print("\nTask 4 complete!")
    return results

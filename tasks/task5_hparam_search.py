"""Task 5: Hyperparameter Search with Optuna.

Students implement the objective function for Optuna to search over
learning rate, weight decay, dropout, and batch size.

Students implement: objective() function
Provided: Optuna study setup, best-model retraining, result saving.
"""

"""
任务5：使用Optuna进行超参数搜索

学生需要为Optuna实现目标函数，搜索以下超参数：
学习率、权重衰减、Dropout比率、批大小

学生需要实现：objective() 函数
本文件提供：Optuna研究设置、最佳模型重新训练、结果保存
"""

import json

import optuna
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


def objective(trial, config, device):
    
    """
    Optuna超参数优化的目标函数
    
    使用 trial.suggest_* 方法采样超参数，构建模型，
    训练模型，并返回验证准确率（需要最大化）。
    
    参数:
        trial: Optuna试验对象
        config: 完整配置字典（来自config.json）
        device: 训练用的torch设备
    
    返回:
        float: 最终验证准确率（需要最大化）
    
    需要搜索的超参数:
        - lr: 学习率，对数尺度采样
        - weight_decay: 权重衰减，对数尺度采样
        - dropout: Dropout比率
        - batch_size: 批大小，从候选值中选择
    
    使用 config["task5"] 获取搜索范围:
        - cfg["lr_range"] = [最小学习率, 最大学习率]
        - cfg["weight_decay_range"] = [最小权重衰减, 最大权重衰减]
        - cfg["dropout_range"] = [最小dropout, 最大dropout]
        - cfg["batch_size_choices"] = [32, 64, 128, 256]
        - cfg["epochs_per_trial"] = 每次试验的训练轮数
    """
    cfg = config["task5"]

    # ==================== 超参数采样 ====================
    # 1. 使用 trial.suggest_float / trial.suggest_categorical 采样：
    #    - lr（对数尺度）
    #    - weight_decay（对数尺度）
    #    - dropout
    #    - batch_size
    
    # 采样学习率（对数尺度：在log空间均匀采样）
    lr = trial.suggest_float(
        "lr", 
        cfg["lr_range"][0], 
        cfg["lr_range"][1], 
        log=True  # 对数尺度，因为学习率通常跨越几个数量级
    )
    
    # 采样权重衰减（对数尺度）
    weight_decay = trial.suggest_float(
        "weight_decay", 
        cfg["weight_decay_range"][0], 
        cfg["weight_decay_range"][1],
        log=True  # 对数尺度，权重衰减也跨越多个数量级
    )
    
    # 采样Dropout比率（线性尺度，范围0~0.5）
    dropout = trial.suggest_float(
        "dropout", 
        cfg["dropout_range"][0], 
        cfg["dropout_range"][1]
    )
    
    # 采样批大小（从候选值中选择）
    batch_size = trial.suggest_categorical(
        "batch_size", 
        cfg["batch_size_choices"]
    )

    # ==================== 创建数据加载器 ====================
    # 2. 使用采样的batch_size创建数据加载器
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        subset_size=config.get("subset_size"),  # 可选：使用子集加快搜索
        num_workers=config["num_workers"],
        augment=True  # 通常数据增强有助于泛化
    )

    # ==================== 构建带Dropout的模型 ====================
    # 3. 构建SimpleCNN模型（手动添加Dropout）
    
    # 创建基础模型
    model = SimpleCNN(num_classes=10)

    # 手动添加Dropout（如果dropout > 0）
    # 参考task2的build_model_with_dropout的实现方式
    if dropout > 0.0:
        # 获取分类器层
        layers = list(model.classifier.children())
        new_layers = []
        for i, layer in enumerate(layers):
            # 在最后一个Linear层前添加Dropout
            if isinstance(layer, nn.Linear) and i == len(layers) - 1:
                new_layers.append(nn.Dropout(p=dropout))
            new_layers.append(layer)
        model.classifier = nn.Sequential(*new_layers)

    # ==================== 创建优化器 ====================
    # 4. 使用采样的lr和weight_decay创建Adam优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # ==================== 训练模型 ====================
    # 5. 训练模型（只训练epochs_per_trial轮，加快搜索速度）
    history = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        device=device,
        epochs=cfg["epochs_per_trial"]
    )

    # ==================== 返回结果 ====================
    # 6. 返回最终验证准确率（Optuna将最大化这个值）
    final_val_acc = history["val_acc"][-1]

    # 打印本次试验的结果（便于观察搜索进度）
    print(
        f"Trial {trial.number}: lr={lr:.6f}, wd={weight_decay:.6f}, "
        f"dropout={dropout:.2f}, batch={batch_size}, val_acc={final_val_acc:.4f}"
    )

    return final_val_acc


def run_task5():
    """Run hyperparameter search with Optuna."""
    """使用Optuna运行超参数搜索"""
    
    print("=" * 60)
    print("Task 5: Hyperparameter Search with Optuna")
    print("任务5：使用Optuna进行超参数搜索")
    print("=" * 60)

    # ==================== 初始化 ====================
    config = load_config()
    cfg = config["task5"]          # 任务5的特定配置
    device = get_device()           # 自动选择计算设备
    ensure_results_dir()            # 确保结果目录存在
    set_seed(config["seed"])        # 设置随机种子

    # ==================== 创建Optuna研究 ====================
    # 创建Optuna研究对象（目标：最大化验证准确率）
    study = optuna.create_study(
        direction="maximize",                    # 最大化目标值
        study_name="cifar10_hparam_search",     # 研究名称
    )

    # 抑制Optuna的日志输出，让输出更简洁
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ==================== 运行超参数搜索 ====================
    print(f"\nRunning {cfg['n_trials']} trials...")
    print(f"正在运行 {cfg['n_trials']} 次试验...")
    
    study.optimize(
        lambda trial: objective(trial, config, device),  # 目标函数
        n_trials=cfg["n_trials"],                         # 试验次数
        show_progress_bar=True,                           # 显示进度条
    )

    # ==================== 报告最佳结果 ====================
    best = study.best_trial
    print(f"\nBest trial #{best.number}:")
    print(f"  Value (val acc): {best.value:.4f}")  # 最佳验证准确率
    print(f"  Params: {json.dumps(best.params, indent=4)}")  # 最佳超参数

    # ==================== 保存试验历史 ====================
    trials_data = []
    for trial in study.trials:
        row = {"trial": trial.number, "value": trial.value}
        row.update(trial.params)  # 添加所有超参数
        trials_data.append(row)

    df = pd.DataFrame(trials_data)
    df.to_csv("results/task5_trials.csv", index=False)
    print(f"  Saved: results/task5_trials.csv")

    # ==================== 保存最佳参数 ====================
    best_params = {"best_value": best.value, **best.params}
    with open("results/task5_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved: results/task5_best_params.json")

    # ==================== 重新训练最佳模型 ====================
    # 使用找到的最佳超参数，训练更多轮数，获得最终模型
    print(f"\nRetraining best model for {cfg['final_epochs']} epochs...")
    print(f"使用最佳超参数重新训练模型，共 {cfg['final_epochs']} 轮...")
    set_seed(config["seed"])

    # 获取最佳批大小（转换为int）
    batch_size = int(best.params.get("batch_size", 128))
    
    # 创建数据加载器
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        subset_size=config["subset_size"],
        num_workers=config["num_workers"],
    )

    # 创建模型
    model = SimpleCNN(num_classes=10)
    
    # 创建优化器（使用最佳学习率和权重衰减）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best.params["lr"],
        weight_decay=best.params["weight_decay"],
    )

    # 训练最终模型（使用更多轮数）
    history = train_model(
        model, train_loader, test_loader, optimizer,
        device=device, epochs=cfg["final_epochs"],
    )

    final_acc = history["val_acc"][-1]
    print(f"  Best model final val acc: {final_acc:.2%}")
    print(f"  最佳模型最终验证准确率: {final_acc:.2%}")

    # ==================== 绘制训练曲线 ====================
    plot_training_curves(
        {"Best HP Model": history},  # 最佳超参数模型
        "acc",                        # 绘制准确率曲线
        "results/task5_best_model_curves.png",
        title="Task 5: Best Model Training Curves",
        title_cn="任务5：最佳模型训练曲线",
    )

    print("\nTask 5 complete!")
    return best_params

{
    // ========== 全局配置 ==========
    "seed": 42,                    // 随机种子，确保实验可复现
    "subset_size": 5000,           // CIFAR-10 子集大小（加速训练，原数据集50000张）
    "batch_size": 128,             // 默认批次大小（Task5 会动态调整）
    "num_workers": 2,              // 数据加载的并行进程数

    // ========== Task 1: 权重初始化消融研究 ==========
    "task1": {
        "description": "Weight Initialization Ablation",   // 任务描述
        "num_layers": 20,          // MLP 的层数（深度网络，测试梯度传播）
        "hidden_dim": 256,         // 隐藏层维度（每层神经元数量）
        "epochs": 20,              // 训练轮数
        "lr": 0.001,               // 学习率（Adam优化器）
        "init_methods": [          // 要测试的5种初始化方法
            "default",             // PyTorch 默认初始化
            "xavier_uniform",      // Xavier 均匀分布初始化
            "xavier_normal",       // Xavier 正态分布初始化
            "kaiming_uniform",     // Kaiming 均匀分布初始化（针对ReLU）
            "kaiming_normal"       // Kaiming 正态分布初始化（针对ReLU）
        ]
    },

    // ========== Task 2: 正则化消融研究 ==========
    "task2": {
        "description": "Regularization Ablation",   // 任务描述
        "epochs": 30,              // 训练轮数（需要更多轮次观察过拟合）
        "lr": 0.001                // 学习率
    },

    // ========== Task 3: BatchNorm 消融研究 ==========
    "task3": {
        "description": "BatchNorm Ablation",        // 任务描述
        "epochs": 30,              // 训练轮数
        "lr": 0.001,               // 学习率
        "configs": [               // 三种模型配置
            {
                "name": "SimpleCNN (no BN)",   // 简单CNN，无BN（基线）
                "use_bn": false
            },
            {
                "name": "DeeperCNN (no BN)",   // 深层CNN，无BN（测试深度的影响）
                "use_bn": false
            },
            {
                "name": "DeeperCNN (with BN)", // 深层CNN，带BN（测试BN的效果）
                "use_bn": true
            }
        ]
    },

    // ========== Task 4: 分布偏移鲁棒性研究 ==========
    "task4": {
        "description": "Robustness to Distribution Shift",   // 任务描述
        "epochs": 30,              // 训练轮数
        "lr": 0.001,               // 学习率
        "cifar10c_path": "data/CIFAR-10-C",   // CIFAR-10-C 数据集路径（损坏数据）
        "all_corruptions": [       // 所有可用的损坏类型（共15种）
            "gaussian_noise",      // 高斯噪声（噪声类）
            "shot_noise",          // 散粒噪声（噪声类）
            "impulse_noise",       // 脉冲噪声（噪声类）
            "defocus_blur",        // 散焦模糊（模糊类）
            "glass_blur",          // 玻璃模糊（模糊类）
            "motion_blur",         // 运动模糊（模糊类）
            "zoom_blur",           // 缩放模糊（模糊类）
            "snow",                // 雪（天气类）
            "frost",               // 霜（天气类）
            "fog",                 // 雾（天气类）
            "brightness",          // 亮度（数字类）
            "contrast",            // 对比度（数字类）
            "elastic_transform",   // 弹性变换（数字类）
            "pixelate",            // 像素化（数字类）
            "jpeg_compression"     // JPEG压缩（数字类）
        ],
        "corruptions": [           // 选择的3种损坏类型（每个类别选一种）
            "gaussian_noise",      // 噪声类 - 测试对噪声的鲁棒性
            "frost",               // 天气类 - 测试对天气效果的鲁棒性
            "brightness"           // 数字类 - 测试对亮度变化的鲁棒性
        ],
        "severities": [1, 3, 5]    // 损坏严重程度级别（1=最轻，5=最重）
    },

    // ========== Task 5: Optuna 超参数搜索 ==========
    "task5": {
        "description": "Hyperparameter Search with Optuna",   // 任务描述
        "n_trials": 20,            // Optuna 搜索试验次数
        "epochs_per_trial": 10,    // 每次试验的训练轮数（快速评估）
        "final_epochs": 30,        // 最终模型的训练轮数（使用最佳参数重新训练）
        "lr_range": [1e-5, 1e-1],  // 学习率搜索范围（对数均匀分布）
        "weight_decay_range": [1e-6, 1e-2],  // 权重衰减搜索范围（对数均匀分布）
        "batch_size_choices": [32, 64, 128, 256]  // 批次大小可选值（分类选择）
    }
}

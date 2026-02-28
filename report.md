# Lab 2: Ablation Study on CIFAR-10

**Author:** Wang Wenlu (st136325)  
**Date:** February 28, 2026  

## Abstract
This experiment conducted a systematic ablation study on the CIFAR-10 dataset to investigate the impact of five key design choices in deep learning: weight initialization, regularization, batch normalization, data augmentation, and hyperparameter optimization. The experimental results demonstrate that:  
1) Kaiming initialization achieves the best performance on a 20-layer MLP (38.39% validation accuracy), while default initialization completely fails (10% accuracy, equivalent to random guessing);  
2) Combining Dropout and L2 regularization reduces the generalization gap from 40.66% to 33.97%;  
3) Batch normalization improves DeeperCNN's validation accuracy by 10.05% (from 58.51% to 68.56%);  
4) Data augmentation significantly enhances model robustness to brightness variations (+3.37%) but shows limited effectiveness for noise-type corruptions;  
5) Optimal hyperparameters identified via Optuna (lr=0.001014, weight_decay=2.48e-4, dropout=0.3034, batch_size=32) yield a validation accuracy of 56.43% over 10 epochs.
These findings provide practical guidelines for training deep neural networks on image classification tasks.

## 1. Introduction
In deep learning research, understanding the individual contribution of each design choice to model performance is critical for building effective and robust neural networks. Ablation studies quantify the importance of specific components by systematically removing or modifying them, enabling researchers to isolate and evaluate their impact.  

This experiment focuses on five core design choices in deep learning:  
- **Weight Initialization**: Influences gradient propagation and convergence speed in deep networks  
- **Regularization**: Mitigates overfitting via Dropout and L2 regularization  
- **Batch Normalization**: Accelerates training convergence and improves generalization  
- **Data Augmentation**: Enhances model robustness to distribution shifts in real-world scenarios  
- **Hyperparameter Optimization**: Automatically searches for optimal training configurations  

This report analyzes the performance of each technique, discusses their synergistic effects, and provides actionable guidance for training deep neural networks on image datasets.

## 2. Experimental Setup
### 2.1 Dataset
CIFAR-10 (10 classes, 32×32 color images) – A subset of 5,000 images was used to accelerate experimental iteration while maintaining statistical representativeness.

### 2.2 Model Architectures
- **Task 1**: 20-layer MLP with 256 neurons per layer (ReLU activation, fully connected layers)  
- **Tasks 2-5**: SimpleCNN and DeeperCNN (detailed architecture in `models/cnn.py`):  
  - SimpleCNN: 2 convolutional layers + 2 fully connected layers  
  - DeeperCNN: 4 convolutional layers + 2 fully connected layers  

### 2.3 Training Configuration
- Optimizer: Adam (β₁=0.9, β₂=0.999, ε=1e-8)  
- Loss Function: Cross-entropy loss  
- Training Epochs: 20–30 (task-specific, see `config.json` for details)  
- Batch Size: 128 (default; dynamically adjusted in Task 5)  
- Hardware: CPU training (no GPU acceleration)  
- Random Seed: 42 (ensures full reproducibility of results)  

## 3. Task 1: Weight Initialization
### 3.1 Experimental Design
Five weight initialization methods were compared on a 20-layer MLP to evaluate their impact on gradient propagation and convergence:  
- Default (uniform random initialization, range [-0.05, 0.05])  
- Xavier uniform (Glorot 2010)  
- Xavier normal (Glorot 2010)  
- Kaiming uniform (He et al. 2015)  
- Kaiming normal (He et al. 2015)  

### 3.2 Results
| Initialization Method | Training Loss | Validation Accuracy | Rank |
|-----------------------|---------------|---------------------|------|
| default               | 2.3023        | 10.00%              | 5    |
| xavier_uniform        | 1.4601        | 29.79%              | 4    |
| xavier_normal         | 1.4005        | 29.86%              | 3    |
| kaiming_uniform       | 1.1298        | 38.39%              | 1    |
| kaiming_normal        | 1.2154        | 37.04%              | 2    |

### 3.3 Analysis
- **Theoretical Foundation**:  
  Xavier initialization assumes linear activation functions (suitable for tanh/sigmoid), while Kaiming initialization is optimized for ReLU (accounts for 50% neuron zero-out).  
- **Key Findings**:  
  1. Kaiming uniform outperforms all other methods (8.5% higher accuracy than Xavier variants), confirming its suitability for ReLU-based deep networks.  
  2. Default initialization leads to complete gradient vanishing (10% accuracy = random guessing), highlighting the necessity of proper initialization for deep MLPs.  
  3. Uniform variants outperform normal variants for Kaiming, while normal slightly outperforms uniform for Xavier.  

## 4. Task 2: Regularization
### 4.1 Experimental Design
Four regularization configurations were tested on SimpleCNN to quantify their ability to reduce overfitting:  
- Baseline (no regularization)  
- Dropout Only (dropout rate = 0.3)  
- Weight Decay Only (L2 regularization, λ = 0.001)  
- Both (Dropout + Weight Decay)  

### 4.2 Results
| Configuration       | weight_decay | dropout | Training Accuracy | Validation Accuracy | Generalization Gap |
|---------------------|--------------|---------|-------------------|---------------------|--------------------|
| Baseline            | 0.0          | 0.0     | 100.0%            | 59.34%              | 40.66%             |
| Dropout Only        | 0.0          | 0.3     | 98.08%            | 60.29%              | 37.79%             |
| Weight Decay Only   | 0.001        | 0.0     | 100.0%            | 58.74%              | 41.26%             |
| Both                | 0.001        | 0.3     | 94.90%            | 60.93%              | 33.97%             |

### 4.3 Analysis
- Baseline exhibits severe overfitting (100% training accuracy vs. 59.34% validation accuracy).  
- Dropout alone reduces the generalization gap by 2.87% and improves validation accuracy by 0.95%.  
- Weight decay alone has a negligible/negative effect (validation accuracy drops by 0.6%), likely due to over-regularization.  
- **Synergistic Effect**: Combining Dropout and L2 regularization achieves the smallest generalization gap (33.97%) and highest validation accuracy (60.93%), demonstrating complementary effects.  

## 5. Task 3: BatchNorm
### 5.1 Experimental Design
Batch normalization (BN) was added to SimpleCNN and DeeperCNN to evaluate its impact on deep network performance:  
- SimpleCNN (no BN)  
- DeeperCNN (no BN)  
- DeeperCNN (with BN, applied after convolutional layers)  

### 5.2 Results
| Model                  | use_bn | Validation Accuracy | Best Accuracy |
|------------------------|--------|---------------------|---------------|
| SimpleCNN (no BN)      | False  | 59.34%              | 59.43%        |
| DeeperCNN (no BN)      | False  | 58.51%              | 59.42%        |
| DeeperCNN (with BN)    | True   | 68.56%              | 68.56%        |

### 5.3 Analysis
- **Performance Gain**: BN improves DeeperCNN's accuracy by 10.05%, the largest single improvement in this study.  
- **Depth vs. BN**: Without BN, deeper networks perform slightly worse (58.51% vs. 59.34%), indicating gradient degradation; BN unlocks the potential of deep networks by stabilizing training.  
- **Theoretical Basis**: BN mitigates internal covariate shift by normalizing layer inputs, enabling higher learning rates and providing mild regularization.  

## 6. Task 4: Robustness to Distribution Shift
### 6.1 Experimental Design
Model robustness was evaluated on clean and corrupted CIFAR-10-C data (Hendrycks & Dietterich 2019) with/without data augmentation (random crop, horizontal flip, brightness jitter):  
- Models compared:
  - No Augmentation: SimpleCNN trained without data augmentation 
  - With Augmentation: SimpleCNN trained with data augmentation (from Task 2 Both configuration, validation accuracy 60.93%)
 
- Corruption Types (one per category):  
  - Noise: gaussian_noise (severity = 3)  
  - Weather: frost (severity = 3)  
  - Digital: brightness (severity = 3)  

### 6.2 Results
*Clean test accuracy baseline (from Task 2): 59.34% (No Augmentation) / 60.93% (With Augmentation)*

| Corruption Type   | No Augmentation | With Augmentation | Difference |
|-------------------|-----------------|-------------------|------------|
| gaussian_noise    | 55.68%          | 54.59%            | -1.09%     |
| frost             | 48.14%          | 49.80%            | +1.66%     |
| brightness        | 54.22%          | 57.59%            | +3.37%     |

### 6.3 Analysis
- Brightness augmentation directly improves robustness to brightness corruptions (+3.37%), confirming alignment between augmentation and corruption type.  
- Gaussian noise corruption shows no benefit (accuracy drops by 1.09%), as augmentation does not include noise injection.  
- Frost corruption shows mild improvement (+1.66%), likely due to indirect augmentation effects (e.g., flip/crop simulating partial occlusion).  
- Note: Clean test accuracy (59.34% for no augmentation, 60.93% with augmentation) is reported in Task 2 results and serves as the baseline for corrupted data comparison.
## 7. Task 5: Hyperparameter Search
### 7.1 Experimental Design
Optuna (TPE sampler) was used to search for optimal hyperparameters on SimpleCNN (20 trials):  
- Learning rate: log-uniform [1e-5, 1e-1]  
- Weight decay: log-uniform [1e-6, 1e-2]  
- Dropout rate: uniform [0.0, 0.5]  
- Batch size: categorical [32, 64, 128, 256]  

### 7.2 Results
#### Top 3 Trials (10-epoch accuracy)
| Rank | Trial | Learning Rate | Weight Decay | Dropout | Batch Size | Accuracy  |
|------|-------|---------------|--------------|---------|------------|-----------|
| 1    | 9     | 0.001014      | 2.48e-4      | 0.3034  | 32         | 56.43%    |
| 2    | 18    | 3.76e-4       | 2.20e-4      | 0.1866  | 32         | 55.86%    |
| 3    | 19    | 3.55e-4       | 1.79e-5      | 0.1761  | 32         | 54.70%    |
| 4    | 12    | 0.001482      | 4.97e-4      | 0.2320  | 64         | 54.19%    |
| 5    | 11    | 8.45e-4       | 4.70e-4      | 0.3159  | 64         | 54.04%    |

#### Optimal Configuration (10-epoch training)
```json
{
  "lr": 0.001013601266456614,
  "weight_decay": 0.000248045162060495,
  "dropout": 0.30339097494238687,
  "batch_size": 32,
  "best_value": 0.5643
}
```
Rounded values for reporting: lr = 0.001014, weight_decay = 2.48e-4, dropout = 0.3034
#### Failure Cases (Non-convergent Trials)
| Trial | Learning Rate | Accuracy | Failure Reason                     |
|-------|---------------|----------|------------------------------------|
| 0     | 0.07857       | 10.45%   | Learning rate too high (>0.01)      |
| 14    | 0.01708       | 10.00%   | Learning rate too high (>0.01)      |
| 7     | 0.01117       | 33.49%   | Learning rate moderately high       |
These failure cases clearly demonstrate the typical problems caused by an excessively large learning rate, which have important guiding significance for practice.

### 7.3 Key Observations
- **Learning Rate**: Most critical hyperparameter – values >0.01 lead to non-convergence (10% accuracy in Trials 0/14), while values <1e-4 result in slow convergence and suboptimal performance.
- **Batch Size**: Smaller batches (32) consistently outperform larger ones (128/256) across all trials, likely due to more frequent weight updates and stronger regularization effects from mini-batch statistics.
- **Dropout**: Optimal rate centers around 30% – rates <5% fail to mitigate overfitting, while rates >45% lead to underfitting (excessive information loss during training).
- **Weight Decay**: Optimal value is approximately 2.5e-4 (moderate regularization strength) – higher values (>1e-3) cause parameter shrinkage and accuracy degradation, while lower values (<1e-5) provide insufficient overfitting protection.

## 8. Discussion
### 8.1 Synthesized Findings
| Technique                  | Impact Magnitude | Key Insight |
|----------------------------|------------------|-------------|
| Batch Normalization        | +10.05%          | Unlocks deep network performance by mitigating internal covariate shift |
| Kaiming Initialization     | +8.5% (vs Xavier)| Critical for maintaining gradient flow in deep ReLU-based networks |
| Regularization (Combined)  | -6.69% (gap)     | Synergistic effect of Dropout + L2 regularization outperforms single methods |
| Data Augmentation          | +3.37% (brightness) | Effectiveness is highly dependent on alignment between augmentation strategies and corruption types |
| Hyperparameter Tuning      | Optimized to 56.43% | Automated search identifies superior configurations beyond default settings |

### 8.2 Practical Guidelines
1. **Initialization**: Always use Kaiming uniform initialization for ReLU-based deep networks (default random initialization leads to complete gradient vanishing in 20-layer MLPs).
2. **Normalization**: Integrate Batch Normalization after convolutional/fully connected layers – this single change delivers the largest performance gain and stabilizes training dynamics.
3. **Regularization**: Combine Dropout (0.2–0.3) with L2 regularization (1e-4–5e-4) to minimize generalization gap; avoid using either technique in isolation for complex architectures.
4. **Augmentation**: Diversify augmentation strategies to match real-world distribution shifts – include brightness/contrast jitter for digital corruptions, and noise injection for noise-type corruptions.
5. **Hyperparameters**: Prioritize learning rate tuning (1e-4–1e-3 range for Adam) and use automated tools like Optuna (20+ trials) – smaller batch sizes (32) are preferred for small-scale datasets like CIFAR-10 subsets.

### 8.3 Limitations
- **Dataset Constraints**: Using a 5,000-image subset may not fully reflect performance on the full CIFAR-10 dataset (10x larger), and results may not generalize to larger image datasets (e.g., ImageNet).
- **Architectural Scope**: Findings are limited to MLP and basic CNN architectures – transformer-based models or modern CNN variants (e.g., ResNet) may exhibit different behavior for the tested techniques.
- **Search Space Limitations**: The hyperparameter search was constrained to 20 trials (computational efficiency) – a larger search space (50+ trials) or extended search range (e.g., batch size 16/512) may yield better configurations.
- **Robustness Testing**: Only three corruption types were evaluated – a broader range of corruptions (e.g., motion blur, pixelation) would provide more comprehensive robustness insights.

## 9. Conclusion
This ablation study systematically quantified the impact of five core design choices in deep learning for CIFAR-10 image classification. Key conclusions from the experiments include:  
1. **Initialization**: Kaiming uniform initialization is essential for training deep ReLU networks – default initialization results in complete failure (10% accuracy, equivalent to random guessing).  
2. **Regularization**: Combining Dropout and L2 regularization effectively reduces the generalization gap from 40.66% to 33.97%, demonstrating synergistic effects between different regularization strategies.  
3. **Batch Normalization**: Delivers the largest performance gain (+10.05% for DeeperCNN) by mitigating internal covariate shift, unlocking the potential of deeper network architectures.  
4. **Data Augmentation**: Improves robustness to specific distribution shifts (e.g., +3.37% for brightness variations) but has limited effectiveness for unaligned corruption types (e.g., Gaussian noise).  
5. **Hyperparameter Optimization**: Automated search with Optuna identifies optimal configurations (lr=0.001014, weight_decay=2.48e-4, dropout=0.3034, batch_size=32) achieving 56.43% validation accuracy over 10 epochs.

These findings confirm that deep learning design choices are interdependent and collectively determine model performance. The practical guidelines derived from this study (e.g., Kaiming initialization, combined regularization, Batch Normalization adoption) can be directly applied to improve training efficiency, generalization, and robustness for image classification tasks. Future work should extend these experiments to full datasets, modern architectures, and broader corruption sets to further validate the generalizability of these conclusions.

## References
1. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*.  
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.  
3. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.  
4. Hendrycks, D., & Dietterich, T. G. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *Proceedings of the International Conference on Learning Representations (ICLR)*.  
5. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD)*.

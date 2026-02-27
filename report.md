# Lab 2: Ablation Study on CIFAR-10

**Author:** <!-- TODO: Your name and student ID -->

**Date:** <!-- TODO: Date -->

## Abstract

<!-- TODO: Summarize the purpose of this lab, the experiments conducted (initialization, regularization, BatchNorm, robustness, HPO), key findings, and main takeaways in one paragraph (150–250 words). -->

## 1. Introduction

<!-- TODO: Introduce the motivation for ablation studies in deep learning. Why is it important to understand the individual contribution of design choices (initialization, regularization, normalization, data augmentation, hyperparameter tuning)? Briefly outline the structure of this report. -->

## 2. Experimental Setup

<!-- TODO: Describe the shared experimental setup: dataset (CIFAR-10, subset size), model architectures used, training procedure (optimizer, loss function), hardware, and any other relevant details. Reference config.json where appropriate. This section should give the reader enough context to understand all subsequent experiments. -->

## 3. Task 1: Weight Initialization

<!-- TODO: Present your weight initialization experiments. Include:
- The initialization methods compared and their theoretical motivation
- Quantitative results (table or inline)
- Analysis of activation statistics across layers (reference your figures)
- Discussion of which methods succeeded/failed and why
-->

## 4. Task 2: Regularization

<!-- TODO: Present your regularization experiments. Include:
- The configurations you tested and your rationale for choosing them
- Quantitative results comparing training accuracy, validation accuracy, and generalization gap
- Analysis of how dropout and weight decay affect overfitting (reference your figures)
- Discussion of whether combining techniques yields additive benefits
-->

## 5. Task 3: BatchNorm

<!-- TODO: Present your BatchNorm experiments. Include:
- The three model configurations compared
- Quantitative results (accuracy, convergence speed)
- Analysis of training dynamics with and without BatchNorm (reference your figures)
- Discussion connecting your observations to the theoretical role of BatchNorm
-->

## 6. Task 4: Robustness to Distribution Shift

<!-- TODO: Present your robustness experiments. Include:
- Which 3 corruption types you selected and why
- Clean accuracy vs. corrupted accuracy for both models
- Analysis of how data augmentation affects robustness across corruption types (reference your heatmap/figures)
- Discussion of why certain corruptions are more/less affected by augmentation
-->

## 7. Task 5: Hyperparameter Search

<!-- TODO: Present your hyperparameter search experiments. Include:
- Search space definition and methodology (Optuna TPE sampler)
- Best hyperparameters found and their validation accuracy
- Analysis of hyperparameter importance and interactions (reference trial history)
- Comparison of the optimized model vs. default settings from earlier tasks
-->

## 8. Discussion

<!-- TODO: Synthesize your findings across all five tasks. Consider:
- Which design choices had the largest impact on performance?
- How do these techniques interact with each other?
- What practical guidelines would you derive for training deep networks?
- Limitations of your experiments (e.g., subset size, architecture choices, number of trials)
-->

## 9. Conclusion

<!-- TODO: Summarize the key takeaways from this lab in 1–2 paragraphs. What did you learn about the importance of initialization, regularization, normalization, data augmentation, and hyperparameter tuning for training deep neural networks? -->

## References

<!-- TODO: Cite any papers, textbooks, or resources you referenced (e.g., Glorot & Bengio 2010, He et al. 2015, Ioffe & Szegedy 2015, Hendrycks & Dietterich 2019). Use a consistent citation format. -->

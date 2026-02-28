"""CNN models for Tasks 3 and 4.

Students must implement:
1. SimpleCNN.__init__: Define convolutional feature extractor and classifier.
2. DeeperCNN.__init__: Define a deeper CNN with conditional BatchNorm.
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """A simple CNN for CIFAR-10 classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # =================================================================
        # TODO: Define self.features as an nn.Sequential with 3 conv blocks.
        # =================================================================
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 32x16x16

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 64x8x8

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 128x4x4
        )
        # 注意：这里不要有 raise NotImplementedError

        # =================================================================
        # TODO: Define self.classifier as an nn.Sequential
        # =================================================================
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),  # 128*4*4 = 2048
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        # 注意：这里不要有 raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeeperCNN(nn.Module):
    """A deeper CNN with optional BatchNorm for Task 3."""

    def __init__(self, num_classes: int = 10, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn

        # =================================================================
        # TODO: Define self.features as nn.Sequential with 3 conv blocks.
        # =================================================================
        layers = []

        # Block 1: 3 -> 64 -> 64
        # Conv1
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        # Conv2
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))  # 输出: 64x16x16

        # Block 2: 64 -> 128 -> 128
        # Conv3
        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())

        # Conv4
        layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))  # 输出: 128x8x8

        # Block 3: 128 -> 256 -> 256
        # Conv5
        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())

        # Conv6
        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))  # 输出: 256x4x4

        self.features = nn.Sequential(*layers)
        # 注意：这里不要有 raise NotImplementedError

        # =================================================================
        # TODO: Define self.classifier as nn.Sequential
        # =================================================================
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),  # 256*4*4 = 4096
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        # 注意：这里不要有 raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

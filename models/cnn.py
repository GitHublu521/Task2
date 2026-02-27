"""CNN models for Tasks 3 and 4.

Students must implement:
1. SimpleCNN.__init__: Define convolutional feature extractor and classifier.
2. DeeperCNN.__init__: Define a deeper CNN with conditional BatchNorm.
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """A simple CNN for CIFAR-10 classification.

    Target architecture:
        Conv2d(3, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Conv2d(64, 128, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Flatten -> Linear(128*4*4, 256) -> ReLU -> Linear(256, num_classes)

    Args:
        num_classes: Number of output classes (default: 10).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # =================================================================
        # TODO: Define self.features as an nn.Sequential with 3 conv blocks.
        #
        # Each conv block: Conv2d -> ReLU -> MaxPool2d(2, 2)
        #   Block 1: 3  -> 32 channels, kernel_size=3, padding=1
        #   Block 2: 32 -> 64 channels, kernel_size=3, padding=1
        #   Block 3: 64 -> 128 channels, kernel_size=3, padding=1
        #
        # After 3 rounds of MaxPool2d(2), a 32x32 image becomes 4x4.
        # =================================================================
        self.features = None  # Replace with nn.Sequential(...)
        raise NotImplementedError("TODO: Define SimpleCNN feature extractor")

        # =================================================================
        # TODO: Define self.classifier as an nn.Sequential:
        #   nn.Flatten()
        #   nn.Linear(128 * 4 * 4, 256)
        #   nn.ReLU()
        #   nn.Linear(256, num_classes)
        # =================================================================
        self.classifier = None  # Replace with nn.Sequential(...)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeeperCNN(nn.Module):
    """A deeper CNN with optional BatchNorm for Task 3.

    Target architecture (with use_bn=True, BatchNorm2d goes after Conv2d):
        Conv2d(3, 64, 3, padding=1)  -> [BN] -> ReLU -> Conv2d(64, 64, 3, padding=1)  -> [BN] -> ReLU -> MaxPool2d(2)
        Conv2d(64, 128, 3, padding=1) -> [BN] -> ReLU -> Conv2d(128, 128, 3, padding=1) -> [BN] -> ReLU -> MaxPool2d(2)
        Conv2d(128, 256, 3, padding=1) -> [BN] -> ReLU -> Conv2d(256, 256, 3, padding=1) -> [BN] -> ReLU -> MaxPool2d(2)
        Flatten -> Linear(256*4*4, 512) -> ReLU -> Dropout(0.5) -> Linear(512, num_classes)

    When use_bn=False, omit the BatchNorm2d layers.

    Args:
        num_classes: Number of output classes (default: 10).
        use_bn: Whether to include BatchNorm2d layers (default: True).
    """

    def __init__(self, num_classes: int = 10, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn

        # =================================================================
        # TODO: Define self.features as nn.Sequential with 3 conv blocks.
        #
        # Each block has TWO conv layers followed by MaxPool2d(2).
        # If use_bn is True, add BatchNorm2d(channels) after each Conv2d.
        #
        # Hint: Build a list of layers and conditionally append
        #       nn.BatchNorm2d(out_channels) when use_bn is True.
        #       Then pass the list to nn.Sequential(*layers).
        #
        # Block 1: 3   -> 64  -> 64   + MaxPool
        # Block 2: 64  -> 128 -> 128  + MaxPool
        # Block 3: 128 -> 256 -> 256  + MaxPool
        # =================================================================
        self.features = None  # Replace with nn.Sequential(...)
        raise NotImplementedError("TODO: Define DeeperCNN feature extractor")

        # =================================================================
        # TODO: Define self.classifier as nn.Sequential:
        #   nn.Flatten()
        #   nn.Linear(256 * 4 * 4, 512)
        #   nn.ReLU()
        #   nn.Dropout(0.5)
        #   nn.Linear(512, num_classes)
        # =================================================================
        self.classifier = None  # Replace with nn.Sequential(...)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

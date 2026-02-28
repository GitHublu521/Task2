"""20-layer MLP for weight initialization experiments (Task 1).

Students must implement _init_weights() to apply different initialization
strategies to the linear layers of this deep MLP.
"""

import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    """A deep MLP with configurable number of hidden layers.

    Architecture: Input -> [Linear -> ReLU] * num_layers -> Linear -> Output

    Args:
        input_dim: Flattened input dimension (3*32*32 = 3072 for CIFAR-10).
        hidden_dim: Number of neurons in each hidden layer.
        num_classes: Number of output classes.
        num_layers: Number of hidden layers.
        init_method: Initialization method name (used in _init_weights).
    """

    def __init__(self, input_dim: int = 3072, hidden_dim: int = 256,
                 num_classes: int = 10, num_layers: int = 20,
                 init_method: str = "default"):
        super().__init__()

        self.init_method = init_method

        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Apply custom initialization
        if init_method != "default":
            self.apply(self._init_weights)

        # Storage for activation statistics (populated during forward)
        self.activation_stds = []

    def _init_weights(self, module):
        """Initialize weights of Linear layers based on self.init_method."""
        # 检查是否是线性层
        if isinstance(module, nn.Linear):
            # 根据初始化方法选择
            if self.init_method == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif self.init_method == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif self.init_method == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            elif self.init_method == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

            # 偏置初始化为0
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass with optional activation statistics recording."""
        x = x.view(x.size(0), -1)  # Flatten

        # Record activation stds per layer (for analysis)
        self.activation_stds = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self.activation_stds.append(x.std().item())

        return self.classifier(x)
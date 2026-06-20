import torch.nn as nn


class LeNet(nn.Module):
    """LeNet-5 adapted for arbitrary input size via AdaptiveAvgPool before FC."""

    def __init__(self, in_channels: int = 1, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        # Global avg pool (output 1×1) avoids MPS divisibility constraint
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_lenet(in_channels: int = 1, num_classes: int = 8) -> nn.Module:
    return LeNet(in_channels=in_channels, num_classes=num_classes)

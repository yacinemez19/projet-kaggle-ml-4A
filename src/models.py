import torch.nn as nn


# ── LeNet ──────────────────────────────────────────────────────────────────────

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


# ── ResNet from scratch ────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """Standard ResNet residual block (no bottleneck)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetScratch(nn.Module):
    """ResNet-18 or ResNet-34 trained from scratch on 1-channel inputs."""

    _CONFIGS = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
    }

    def __init__(self, depth: int = 18, in_channels: int = 1, num_classes: int = 8):
        super().__init__()
        assert depth in self._CONFIGS, f"depth must be 18 or 34, got {depth}"
        layers = self._CONFIGS[depth]

        self._in_ch = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,  layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self._in_ch != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_ch, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        blocks = [BasicBlock(self._in_ch, out_channels, stride=stride, downsample=downsample)]
        self._in_ch = out_channels
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


def build_resnet_scratch(
    depth: int = 18,
    in_channels: int = 1,
    num_classes: int = 8,
) -> nn.Module:
    return ResNetScratch(depth=depth, in_channels=in_channels, num_classes=num_classes)


# ── VGG-like compact ───────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGGLike(nn.Module):
    """Compact VGG-style CNN for 1-channel TILDA inputs.

    Four conv blocks (32→64→128→256), global avg pool, dropout + FC head.
    ~2.4 M parameters — lighter than ResNet-18 to limit overfitting.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 8, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(in_channels, 32), _conv_bn_relu(32, 32),
            nn.MaxPool2d(2, 2),
            _conv_bn_relu(32, 64), _conv_bn_relu(64, 64),
            nn.MaxPool2d(2, 2),
            _conv_bn_relu(64, 128), _conv_bn_relu(128, 128),
            nn.MaxPool2d(2, 2),
            _conv_bn_relu(128, 256), _conv_bn_relu(256, 256),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_vgg_like(
    in_channels: int = 1,
    num_classes: int = 8,
    dropout: float = 0.5,
) -> nn.Module:
    return VGGLike(in_channels=in_channels, num_classes=num_classes, dropout=dropout)

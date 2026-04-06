import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Baseline(nn.Module):
    """Simple ResNet18 baseline for direct 3D joint regression."""

    def __init__(self, num_joints: int):
        super().__init__()
        backbone = resnet18(weights=None)

        # Depth images are single-channel.
        backbone.conv1 = nn.Conv2d(
            1,
            backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            bias=False,
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_joints * 3)

        self.backbone = backbone
        self.num_joints = num_joints

    def forward(self, x):
        out = self.backbone(x)
        return out.view(out.shape[0], self.num_joints, 3)

# /point_pillar/pillar_voxelizer.py
# Author: Yonghao Li (Paul)
# The backbonme (2D CNN)
# This is a simplified version with fixed structures. 
# Because we are doing classification instead of detection, the Detection Head is not needed. The backbone will output the logit for classes

# Reference:
# PointPillars: Fast Encoders for Object Detection from Point Clouds
# https://arxiv.org/pdf/1812.05784
# Section 2.2: Backbone

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplePillarBackbone(nn.Module):
    """
    2D CNN Backbone for classification

    Input:
        BEV: (B, C, H, W)
    Output:
        logits: (B, num_classes)
    """
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, fc1_dim: int = 256, dropout_p: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
                                                                                                                # input (B, C_in, H, W)
        # Block 1: C_in,H,W -> C, H, W
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)                # (B, C_base, H, W)
        self.bn1   = nn.BatchNorm2d(base_channels)

        # Block 2: C, H/2, W/2 -> C*2, H/2, W/2
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(base_channels * 2)

        # Block 3: C * 2, H/4, W/4 -> C*4, H/4, W/4
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(base_channels * 4)

        # Global average pooling → (B, C_base*4) -> (B, num_classes)
        self.fc1   = nn.Linear(base_channels * 4, fc1_dim)
        self.fc2   = nn.Linear(fc1_dim, num_classes)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # Block 1                                                                                               # input (B, C_in, H, W)
        x = F.relu(self.bn1(self.conv1(x)))                                                                     # (B, C_base, H, W)
        x = F.max_pool2d(x, kernel_size=2, stride=2)   # 20x20 → 10x10                                          # (B, C_base, H/2, W/2)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))                                                                     # (B, C_base * 2, H/2, W/2)
        x = F.max_pool2d(x, kernel_size=2, stride=2)   # 10x10 → 5x5                                            # (B, C_base * 2, H/4, W/4)      

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))                                                                     # (B, C_base * 4, H/4, H/4)
        x = F.max_pool2d(x, kernel_size=2, stride=2)   # 5x5 → 2x2                                              # (B, C_base * 4, H/8, H/4)

        # Global average pool over spatial dims
        x = x.mean(dim=(2, 3))                                                                                  # (B, C_base*4)

        # Classifier MLP
        x = F.relu(self.fc1(x))                                                                                 # (B, Hidden)
        x = self.dropout(x)
        logits = self.fc2(x)                                                                                    # (B, num_classes)

        return logits
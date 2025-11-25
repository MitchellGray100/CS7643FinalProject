# /point_pillar/pillar_feature_net.py
# Author: Yonghao Li (Paul)
# encode the pillar features

# Reference:
# PointPillars: Fast Encoders for Object Detection from Point Clouds
# https://arxiv.org/pdf/1812.05784
# Section 2.1: Pointcloud to Pseudo-Image

import torch
import torch.nn as nn
import torch.nn.functional as F

class PillarFeatureNet(nn.Module):
    """
    A simplified PointNet that encodes the pillar features
    """
    def __init__(self, in_dim=8, out_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pillar_features, pillar_mask):
        B, P, N, D = pillar_features.shape
        x = pillar_features.view(B * P * N, D)    # (BPN, D)

        x = self.linear(x)                        # (BPN, C)
        x = self.bn(x)                            # BN over feature dim
        x = self.relu(x)

        x = x.view(B, P, N, self.out_dim)         # (B, P, N, C)
        x = x.max(dim=2).values                   # max over N → (B, P, C)

        # mask out padded pillars
        mask = pillar_mask.unsqueeze(-1)          # (B, P, 1)
        x = x * mask                              # (B, P, C)

        return x
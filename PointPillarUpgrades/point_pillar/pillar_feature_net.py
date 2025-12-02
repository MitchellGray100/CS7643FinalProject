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

class PFNBlock(nn.Module):
    """
    MLP block with residual connections
    """
    def __init__(self, in_dim: int, out_dim: int, use_residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

        self.proj = None
        if use_residual and in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)

        if self.use_residual:
            if self.proj is not None:
                identity = self.proj(identity)
            
            out = out + identity
            out = self.relu(out)

        return out

class PillarFeatureNet(nn.Module):
    """
    A deep PointNet that encodes the pillar features
    """
    def __init__(self, in_dim=8, out_dim=64, hidden_dims: list[int] | None = None, use_residual: bool = True):
        super().__init__()
        self.in_dim = in_dim

        if hidden_dims is None:
            hidden_dims = [out_dim]
        
        dims = [in_dim] + hidden_dims

        blocks = []
        
        for i in range(len(dims) - 1):
            blocks.append(
                PFNBlock(
                    in_dim=dims[i],
                    out_dim=dims[i+1],
                    use_residual=use_residual
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.out_dim = dims[-1]

    def forward(self, pillar_features, pillar_mask):
        B, P, N, D = pillar_features.shape
        x = pillar_features.view(B * P * N, D)    # (BPN, D)

        # iterate through all PFNBlocks
        for block in self.blocks:
            x = block(x)

        x = x.view(B, P, N, self.out_dim)         # (B, P, N, C)
        x = x.max(dim=2).values                   # max over N → (B, P, C)

        # mask out padded pillars
        mask = pillar_mask.unsqueeze(-1).float()  # (B, P, 1)
        x = x * mask                              # (B, P, C)

        return x
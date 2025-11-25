# /point_pillar/pillar_scatter.py
# Author: Yonghao Li (Paul)
# scatter the encoded features back to the original location of pillars to create a pseudo image

# Reference:
# PointPillars: Fast Encoders for Object Detection from Point Clouds
# https://arxiv.org/pdf/1812.05784
# Section 2.1: Pointcloud to Pseudo-Image

import torch
import torch.nn as nn
import torch.nn.functional as F

class PillarScatter(nn.Module):
    """
    scatter the encoded features back to the original location of pillars to create a pseudo image
    """
    def __init__(self, nx, ny):
        super().__init__()
        self.nx = nx
        self.ny = ny

    def forward(self, pillar_embeddings, pillar_coords, pillar_mask):
        device = pillar_embeddings.device
        B, P, C = pillar_embeddings.shape

        bev = torch.zeros(B, C, self.ny, self.nx, device=device)

        for b in range(B):
            valid = pillar_mask[b] > 0
            coords = pillar_coords[b, valid]          # (P_valid, 2)
            feats  = pillar_embeddings[b, valid]      # (P_valid, C)

            for (ix, iy), f in zip(coords, feats):
                ix = int(ix.item())
                iy = int(iy.item())
                bev[b, :, iy, ix] = f

        return bev                                      # (B, C, H, W)
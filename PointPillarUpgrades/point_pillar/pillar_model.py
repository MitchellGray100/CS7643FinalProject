# /point_pillar/pillar_model.py
# Author: Yonghao Li (Paul)
# The pipeline that put the components together

# Reference:
# PointPillars: Fast Encoders for Object Detection from Point Clouds
# https://arxiv.org/pdf/1812.05784
# Section 2: PointPillars Network

# point_pillar/model.py
import torch
import torch.nn as nn

from .pillar_voxelizer import PillarVoxelizer
from .pillar_feature_net import PillarFeatureNet
from .pillar_scatter import PillarScatter
from .pillar_simple_backbone import SimplePillarBackbone

class PointPillarsClassifier(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        ds_cfg  = config["dataset"]
        vox_cfg = config["voxelizer"]
        pfn_cfg = config["pfn"]
        bb_cfg  = config["backbone"]

        num_classes = ds_cfg["num_classes"]

        self.voxelizer = PillarVoxelizer(
            x_range=vox_cfg["x_range"],
            y_range=vox_cfg["y_range"],
            z_range=vox_cfg["z_range"],
            pillar_size=vox_cfg["pillar_size"],
            max_pillars=vox_cfg["max_pillars"],
            max_points_per_pillar=vox_cfg["max_points_per_pillar"],
            device=device,
        )

        self.pfn = PillarFeatureNet(
            in_dim=pfn_cfg["in_dim"],
            out_dim=pfn_cfg["out_dim"],
            hidden_dims=pfn_cfg.get("hidden_dims"),
            use_residual=pfn_cfg.get("use_residual", True)
        )

        self.scatter = PillarScatter(
            nx=self.voxelizer.nx,
            ny=self.voxelizer.ny,
        )

        self.backbone = SimplePillarBackbone(
            in_channels=self.pfn.out_dim,
            num_classes=num_classes,
            base_channels=bb_cfg["base_channels"],
            fc1_dim=bb_cfg["fc1_dim"],
            dropout_p=bb_cfg["dropout_p"]
        )

        self.to(device)

    def forward(self, points):
        """
        points: (B, N, 3) on self.device
        returns: logits: (B, num_classes)
        """
        # voxelize
        pillar_features, pillar_coords, pillar_mask = \
            self.voxelizer.voxelize_batch(points)

        # to device
        pillar_features = pillar_features.to(self.device)
        pillar_coords   = pillar_coords.to(self.device)
        pillar_mask     = pillar_mask.to(self.device)

        # PFN
        pillar_embeddings = self.pfn(pillar_features, pillar_mask)

        # scatter
        bev = self.scatter(pillar_embeddings, pillar_coords, pillar_mask)

        # 2D backbone -> logits
        logits = self.backbone(bev)
        return logits
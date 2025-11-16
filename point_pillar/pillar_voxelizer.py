# /point_pillar/pillar_voxelizer.py
# Author: Yonghao Li (Paul)
# Convert the point cloud to pillars

# Reference:
# PointPillars: Fast Encoders for Object Detection from Point Clouds
# https://arxiv.org/pdf/1812.05784
# Section 2.1: Pointcloud to Pseudo-Image

import torch
from collections import defaultdict
import matplotlib.pyplot as plt

class PillarVoxelizer:
    def __init__(
            self,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            z_range=(-1.0, 1.0),
            pillar_size=(0.1, 0.1),
            max_pillars=1024,           # the number if non_empty pillars per sample (P)
            max_points_per_pillar=32,   # the number of points per pllar (N) 
            device='cpu'
            ):
        self.device = device
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range

        self.x_size, self.y_size = pillar_size

        # grid size
        self.nx = int((self.x_max - self.x_min) / self.x_size)
        self.ny = int((self.y_max - self.y_min) / self.y_size)

        self.max_pillars = max_pillars
        self.max_points_per_pillar= max_points_per_pillar
        # In the paper each point has x, y, z, r and augmented with xc, yc, zc, and xp, yp, so that's 9 dimensions
        # here we dont have r: reflectance so we only have 8 dimensions
        self.in_dim = 8            # the dim (D)  

    def voxelize_single(self, points: torch.Tensor):
        """
        process a single batch of points and create the pillars
        """
        points = points.to(self.device)             # points: (N, 3).

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        # clamp
        in_range = (
            (x >= self.x_min) & (x < self.x_max) &
            (y >= self.y_min) & (y < self.y_max) &
            (z >= self.z_min) & (z < self.z_max)
        )
        points = points[in_range]
        N = points.size(0)
        
        # discretize
        ix = torch.floor((x - self.x_min) / self.x_size).long()
        iy = torch.floor((y - self.y_min) / self.y_size).long()
        ix = torch.clamp(ix, 0, self.nx - 1)
        iy = torch.clamp(iy, 0, self.ny - 1)


        # group by ix, iy
        # pillar is a dict of list-keys [ix, iy] -> list-items [x, y, z]
        pillars = defaultdict(list)
        for idx in range(N):
            key = (ix[idx].item(), iy[idx].item())
            pillars[key].append(points[idx])

        # limit number of pillars
        pillar_keys = list(pillars.keys())
        if len(pillar_keys) > self.max_pillars:
            perm = torch.randperm(len(pillar_keys))[:self.max_pillars]
            pillar_keys = [pillar_keys[i] for i in perm.tolist()]

        # init outputs
        pillar_features = torch.zeros(
            (self.max_pillars, self.max_points_per_pillar, self.in_dim),            # (P, N, D)
            device=self.device
        )
        pillar_coords = torch.zeros(
            (self.max_pillars, 2), dtype=torch.long,                                # (P, 2)
            device=self.device
        )
        pillar_mask = torch.zeros(
            (self.max_pillars,), dtype=torch.float32, device=self.device            # (P, )
        )
        
        # build augmented features for each pillar
        for p_idx, (ix_cell, iy_cell) in enumerate(pillar_keys):
            pts_list = pillars[(ix_cell, iy_cell)]
            pts = torch.stack(pts_list, dim=0)
            n_pts = pts.size(0)

            # limit the number of points in the pillar
            if n_pts > self.max_points_per_pillar:
                perm = torch.randperm(n_pts)[:self.max_points_per_pillar]
                pts = pts[perm]
            else:
                pass # zero padded

            x_center = self.x_min + (ix_cell + 0.5) * self.x_size
            y_center = self.y_min + (iy_cell + 0.5) * self.y_size
            x_mean = pts[:, 0].mean()
            y_mean = pts[:, 1].mean()
            z_mean = pts[:, 2].mean()

            xp = pts[:, 0] - x_center
            yp = pts[:, 1] - y_center
            xc = pts[:, 0] - x_mean
            yc = pts[:, 1] - y_mean
            zc = pts[:, 2] - z_mean

            feat = torch.cat([pts, torch.stack([xc, yc, zc, xp, yp], dim=-1)], dim=-1)  # (n_pts, 8)

            # write into output tensors
            K_eff = min(n_pts, self.max_points_per_pillar)
            pillar_features[p_idx, :K_eff, :] = feat[:K_eff]
            pillar_coords[p_idx, 0] = ix_cell
            pillar_coords[p_idx, 1] = iy_cell
            pillar_mask[p_idx] = 1.0

        return pillar_features, pillar_coords, pillar_mask
    
    def voxelize_batch(self, points_batch: torch.Tensor):
        B, _, _ = points_batch.shape
        feats_list, coords_list, mask_list = [], [], []

        for b in range(B):
            f, c, m, = self.voxelize_single(points_batch[b])
            feats_list.append(f)
            coords_list.append(c)
            mask_list.append(m)

        pillar_features = torch.stack(feats_list, dim=0)    # (B, P, N, D)
        pillar_coords = torch.stack(coords_list, dim=0)     # (B, P, 2)
        pillar_mask = torch.stack(mask_list, dim=0)         # (B, P)

        return pillar_features, pillar_coords, pillar_mask





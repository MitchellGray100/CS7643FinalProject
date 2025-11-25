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
        points = points.to(self.device) # points: (N, 3). 
        x, y, z = points[:, 0], points[:, 1], points[:, 2] 
        # clamp 
        in_range = ( 
            (x >= self.x_min) & (x < self.x_max) 
            & (y >= self.y_min) & (y < self.y_max) 
            & (z >= self.z_min) & (z < self.z_max) 
            ) 
        
        points = points[in_range] 
        x, y, z = x[in_range], y[in_range], z[in_range] 
        N = points.size(0) 
        
        # discretize 
        ix = torch.floor((x - self.x_min) / self.x_size).long() 
        iy = torch.floor((y - self.y_min) / self.y_size).long() 
        ix = torch.clamp(ix, 0, self.nx - 1) 
        iy = torch.clamp(iy, 0, self.ny - 1) 
        
        # vectorized 
        # # get unique pillars 
        coords = torch.stack([ix, iy], dim=-1) # (N, 2) 
        uniq_coords, inv = torch.unique(coords, dim=0, return_inverse=True) # inv : inverse indices 
        
        # limit numbero f pillars 
        P = uniq_coords.shape[0] 
        if P > self.max_pillars: 
            perm = torch.randperm(P, device=self.device)[:self.max_pillars] 
            uniq_coords = uniq_coords[perm] 
            # pillars kept 
            mask = (inv[:, None] == perm[None, :]).any(dim=1) # (N, P) -> (N,) 
            inv = inv[mask] 
            
            # mapping points -> pillars 
            points = points[mask] 
            # points in the pillars kept 
            coords = coords[mask] 
            # coords of points in the pillars kept 
            P = self.max_pillars 

        # sort points by pillar 
        sort_idx = torch.argsort(inv) 
        points_sorted = points[sort_idx] 
        inv_sorted = inv[sort_idx] 
        
        # counts points per pillar 
        counts = torch.bincount(inv_sorted, minlength=P).clamp(max=self.max_points_per_pillar) 
        
        # cum to split : get starting indices of each pillar in the sorted points 
        splits = torch.cat([torch.tensor([0], device=self.device), counts.cumsum(dim=0)]) 
        
        # init outputs 
        pillar_features = torch.zeros( 
            (self.max_pillars, self.max_points_per_pillar, self.in_dim), # (P, N, D) 
            device=self.device 
            ) 
        pillar_coords = torch.zeros( 
            (self.max_pillars, 2), dtype=torch.long, # (P, 2) 
            device=self.device 
            ) 
        pillar_mask = torch.zeros(
             (self.max_pillars,), dtype=torch.float32, device=self.device # (P, ) 
            )
        
        # build augmented features for each pillar
        for p in range(P):
            start = splits[p].item() 
            end = splits[p+1].item() 
            pts = points_sorted[start:end]

            n_pts = pts.size(0)

            # limit the number of points in the pillar
            if n_pts > self.max_points_per_pillar:
                perm = torch.randperm(n_pts)[:self.max_points_per_pillar]
                pts = pts[perm]
            else:
                pass # zero padded

            x_center = self.x_min + (uniq_coords[p, 0].float() + 0.5) * self.x_size 
            y_center = self.y_min + (uniq_coords[p, 1].float() + 0.5) * self.y_size
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
            pillar_features[p, :K_eff, :] = feat[:K_eff]
        pillar_coords[:P] = uniq_coords 
        pillar_mask[:P] = 1.0

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





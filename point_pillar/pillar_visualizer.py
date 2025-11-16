# /point_pillar/pillar_voxelizer.py
# Author: Yonghao Li (Paul)
# helper functions to visualize the data and features learned in PointPillar model

import torch
import matplotlib.pyplot as plt
import math

def visualize_point_cloud(points, title=None, elev=20, azim=45):
  """
  Visualize a point cloud tensor or numpy as a 3D scatter plot
  """
  # change tensor back to array
  if isinstance(points, torch.Tensor):
    points = points.detach().cpu().numpy()

  fig = plt.figure(figsize=(4, 4))
  ax = fig.add_subplot(111, projection='3d')
  xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
  c = xs + ys + zs
  ax.scatter(xs, ys, zs, c=c, cmap='viridis', s=10)  
  ax.view_init(elev=elev, azim=azim)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  if title is not None:
      ax.set_title(title)

  max_range = (xs.max() - xs.min(),
                ys.max() - ys.min(),
                zs.max() - zs.min())
  max_range = max(max_range)
  for axis, data in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim],
                        [(xs, ys, zs)[0], (xs, ys, zs)[1], (xs, ys, zs)[2]]):
      mid = (data.max() + data.min()) / 2.0
      axis(mid - max_range / 2, mid + max_range / 2)
  plt.show()


def visualize_pillar_occupancy(c, m, nx, ny):
    """
    create a Bird's Eyes View (BEV) from a single instance of object.
    c: (P, 2)           coords
    m: (p,)             mask
    nx, ny: size of the map
    """
    occ = torch.zeros((ny, nx))
    active_coords = c[m.bool()]
    for ix, iy in active_coords:
        occ[iy, ix] = 1.0

    plt.figure(figsize=(4, 4))
    plt.imshow(occ.cpu().numpy(), origin="lower")
    plt.title("Pillar occupancy (BEV)")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.colorbar(label="occupied")
    plt.show()

def visualize_pillar_strength(f, c, m, nx, ny):
    """
    create a heat map from a single instance of object.
    f: (P, N, D)        features
    c: (P, 2)           coords
    m: (p,)             mask
    nx, ny: size of the map
    """
    pillar_feat_norm = f.norm(dim=-1)           # (P, N)
    pillar_feat_norm_max, _ = pillar_feat_norm.max(dim=-1)  # (P,)

    P, N, D = f.shape
    heat = torch.zeros((ny, nx))

    for p_idx in range(P):
        if m[p_idx] == 0:
                continue
        ix, iy = c[p_idx]
        heat[iy, ix] = pillar_feat_norm_max[p_idx]

    plt.figure(figsize=(4, 4))
    plt.imshow(heat.cpu().numpy(), origin="lower")
    plt.title("Pillar feature strength (max norm)")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.colorbar(label="max ||feature||")
    plt.show()

def visualize_pfn_channels_grid(bev, k=16, title_prefix="BEV channel"):
    """
    create a grid of all the channels generated from Pillar Feature Net
    """
    if isinstance(bev, torch.Tensor):
        bev = bev.detach().cpu()

    C, ny, nx = bev.shape
    k = min(k, C)

    cols = int(math.ceil(math.sqrt(k)))
    rows = int(math.ceil(k / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))

    for i in range(k):
        ax = plt.subplot(rows, cols, i + 1)
        ch = bev[i]  # (ny, nx)
        im = ax.imshow(ch, origin='lower')
        ax.set_title(f"{title_prefix} {i}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
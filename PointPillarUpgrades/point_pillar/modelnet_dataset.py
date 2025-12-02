# /point_pillar/modelnet_dataset.py
# Author: Yonghao Li (Paul)
# Utils to read ModelNet data

# Reference:
# https://segeval.cs.princeton.edu/public/off_format.html
# https://trimesh.org/


import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from torch.utils.data import Dataset

# only to verify the files
def read_off(path):
    """
    Read .off file and return a raw vertices np array (float32) and a faces np array (int16)
    """
    with open(path, 'r') as f:
        # First row is always "OFF"
        header = f.readline().strip()
        assert header == 'OFF'

        # Second row is 
        # numVertices numFaces numEdges
        counts = f.readline().strip().split()
        n_verts = int(counts[0])
        n_faces = int(counts[1])

        # next are verts
        verts = []
        for _ in range(n_verts):
            line = f.readline()
            if not line:
                break
            x, y, z = map(float, line.strip().split()[:3])
            verts.append([x, y, z])

        # the rest are faces.
        faces = []
        for _ in range(n_faces):
            line = f.readline()
            if not line:
                break
            n, a, b, c = map(int, line.strip().split()[:4])
            faces.append([n, a, b, c])

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int16)

def sample_off_surface(path, n_points: int=2048, seed: int | None = None):
    """
    using trimesh to read an OFF file and sample the surfaces. return np array float32
    """
    mesh = trimesh.load(path, process=True)
    if seed is None:
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    else:
        points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)

    return points.astype(np.float32)

def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    center and scale the point cloud
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points = points / max_dist

    return points.astype(np.float32)
    


class ModelNetDataset(Dataset):
    """
    pytorch dataset for ModelNet10 and ModelNet40.

    expected folder structure:
    root/
        class1/
                train/
                        *.off
                test/
                        *.off
        class2/
        ...

    To use Dataset class:
        from torch.utils.data import DataLoader
        train_dataset = ModelNetDataset(root, 'train')
        train_loader = DataLoader(train_dataset, batch_size)
    """
    def __init__(
            self, root:str, split:str='train', num_points:int=2048, normalize:bool=True, 
            precomputed_root: str | None = None,
            cache_mode: str = "none",
            base_seed: int = 42,
            transform=None
            ):
        super().__init__()
        self.base_seed = base_seed
        self.root = root
        self.num_points = num_points
        self.normalize = normalize
        self.split = split

        self.precomputed_root = precomputed_root
        self.cache_mode = cache_mode

        self.transform = transform

        class_dirs = sorted(
                            d for d in os.listdir(self.root)
                            if os.path.isdir(os.path.join(self.root, d))
                    )
        self.class_names = class_dirs
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        self.samples = []
        for cls in self.class_names:
            split_dir = os.path.join(self.root, cls, self.split)
            off_files = glob.glob(os.path.join(split_dir, "*.off"))
            for fpath in off_files:
                self.samples.append((fpath, self.class_to_idx[cls]))

        # this won't validate the data
        print(f"Found {len(self.class_names)} classes, {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label_idx = self.samples[index]
        #print(self.cache_mode + self.precomputed_root)
        seed = self.base_seed + index

        use_cache = (
            self.precomputed_root is not None
            and self.cache_mode in ("write", "load")
        )
        
        if use_cache:
            # keep class/split folder structure under precomputed_root
            # e.g. original: root/chair/train/chair_0001.off
            # precomputed: precomputed_root/chair/train/chair_0001.npy
            rel_path = os.path.relpath(path, self.root)
            npy_path = os.path.splitext(os.path.join(self.precomputed_root, rel_path))[0] + ".npy"

            if os.path.exists(npy_path) and (self.cache_mode == "load"):
                # load mode
                points = np.load(npy_path)
            elif self.cache_mode == "load":
                # attempt to load but no file path
                raise FileNotFoundError(
                    f"Precomputed point cloud not found in 'load' mode: {npy_path}"
                )
            else:
                # 'write' mode: compute and save, allow rewrite
                print('write mode')
                points = sample_off_surface(path=path, n_points=self.num_points, seed=seed)
                if self.normalize:
                    points = normalize_point_cloud(points=points)

                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, points)
                print(f'saved {npy_path}')
        else:
            # dont use cache (sample the points on the fly)
            points = sample_off_surface(path=path, n_points=self.num_points)
            if self.normalize:
                points = normalize_point_cloud(points=points)

        if self.transform is not None and self.split == 'train':
            points = self.transform(points)

        points = torch.from_numpy(points.astype(np.float32))
        label = torch.tensor(label_idx, dtype=torch.long)

        return points, label
    
def augment_pointcloud(points: np.ndarray) -> np.ndarray:
    # small rotations
    theta = np.random.uniform(-np.pi / 12, np.pi / 12) # ( Rotations between -15 to 15 degres)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [c,  -s,  0.0],
            [s,   c,  0.0],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float32,
    )

    points = points @ R.T # Performs a (N,3)@(3,3)

    # small translations
    t = np.random.normal(loc=0.0, scale=0.02, size=(1,3)).astype(np.float32)
    points = points + t

    return points.astype(np.float32)
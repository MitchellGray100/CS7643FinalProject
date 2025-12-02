config = {
    "dataset": {
        "name": "ModelNet10",
        "num_classes": 10,
        "num_points": 1024,
    },
    "voxelizer": {
        "x_range": (-1.0, 1.0),
        "y_range": (-1.0, 1.0),
        "z_range": (-1.0, 1.0),
        "pillar_size": (0.1, 0.1),
        "max_pillars": 1024,
        "max_points_per_pillar": 32,
    },
    "pfn": {
        "in_dim": 8,
        "out_dim": 64,
        "hidden_dim": [64, 64],
        "use_residual": True
    },
    "backbone": {
        "base_channels": 32,
        "fc1_dim": 256,
        "dropout_p": 0.3
    },
    "train": {
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 10,
    },
}

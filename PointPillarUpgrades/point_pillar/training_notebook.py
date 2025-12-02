# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %%
import torch
from point_pillar.pillar_model import PointPillarsClassifier
from point_pillar.pillar_trainer import Trainer
from torch.utils.data import DataLoader
from point_pillar.modelnet_dataset import ModelNetDataset
from point_pillar.config import config

# %%
import random
import numpy as np

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Extra safety for dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    import numpy as np
    import random
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
print(config)

# %%
DATA_DIR = "./data" + "/" + config["dataset"]["name"]
PREDATA_DIR = "./data" + "/" + config["dataset"]["name"] + "_precomputed_" + str(config["dataset"]["num_points"])
# 1. build datasets / loaders using config

train_dataset = ModelNetDataset(
    root=DATA_DIR,
    split="train",
    num_points=config["dataset"]["num_points"],
    normalize=True,
    precomputed_root=PREDATA_DIR,
    cache_mode="write",
)
val_dataset = ModelNetDataset(
    root=DATA_DIR,
    split="test",
    num_points=config["dataset"]["num_points"],
    normalize=True,
    precomputed_root=PREDATA_DIR,
    cache_mode="write",
)

# %%
# for i in range(len(train_dataset)):
#     _ = train_dataset[i]
# for i in range(len(val_dataset)):
#     _ = val_dataset[i]

# %%
# 1. build datasets / loaders using config
train_dataset = ModelNetDataset(
    root=DATA_DIR,
    split="train",
    num_points=config["dataset"]["num_points"],
    normalize=True,
    precomputed_root=PREDATA_DIR,
    cache_mode="load",
)
val_dataset = ModelNetDataset(
    root=DATA_DIR,
    split="test",
    num_points=config["dataset"]["num_points"],
    normalize=True,
    precomputed_root=PREDATA_DIR,
    cache_mode="load",
)

# %%
# # 1. build datasets / loaders using config
# train_dataset = ModelNetDataset(
#     root=DATA_DIR,
#     split="train",
#     num_points=config["dataset"]["num_points"],
#     normalize=True,
#     precomputed_root=None,
#     cache_mode="load",
# )
# val_dataset = ModelNetDataset(
#     root=DATA_DIR,
#     split="test",
#     num_points=config["dataset"]["num_points"],
#     normalize=True,
#     precomputed_root=None,
#     cache_mode="load",
# )

# %%
g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=0,
    worker_init_fn=seed_worker,
    generator=g,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=False,
    num_workers=0,
    worker_init_fn=seed_worker,
    generator=g,
)

# %%
#config["train"]["num_epochs"] = 1

# %%
# 2. build model from config
model = PointPillarsClassifier(config=config, device=device)

# %%

# 3. build trainer from config
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    config=config,
)

# %%
# 4. train
# 9m 53s without cache 1 epoch
# 9m 13s with cache 1 epoch
# 5m 18s with voxelizer vectorize level 1
dataset_name = config["dataset"]["name"]
# add timestamp to the name of the output directory
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_{timestamp}"
os.makedirs(f"{output_dir}", exist_ok=True)
trainer.fit(save_path=f"{output_dir}/pointpillars_{dataset_name}.pth")

# %%
# 5. plots for report
trainer.plot_history()

# %%
# 6. save artifacts

trainer.save_curves_and_config(output_dir=output_dir)

# %%
import csv

new_row = {

    "timestamp": timestamp,
    "name": config["dataset"]["name"],
    "num_classes": config["dataset"]["num_classes"],
    "num_points": config["dataset"]["num_points"],
    "pillar_size": config["voxelizer"]["pillar_size"],
    "max_pillars": config["voxelizer"]["max_pillars"],
    "max_points_per_pillar": config["voxelizer"]["max_points_per_pillar"],
    "pfn_out_dim": config["pfn"]["out_dim"],
    "bb_base_channels": config["backbone"]["base_channels"],
    "bb_fc1_dim": config["backbone"]["fc1_dim"],
    "bb_dropout_p": config["backbone"]["dropout_p"],
    "batch_size": config["train"]["batch_size"],
    "lr": config["train"]["lr"],
    "weight_decay": config["train"]["weight_decay"],
    "num_epochs": config["train"]["num_epochs"],
    "min_train_loss": min(trainer.history["train_loss"]),
    "min_val_loss": min(trainer.history["val_loss"]),
    "max_train_acc": max(trainer.history["train_acc"]),
    "max_val_acc": max(trainer.history["val_acc"])
}

csv_path = "config_log.csv"

# Append mode
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_row.keys())

    # Write header only if file is empty
    if f.tell() == 0:
        writer.writeheader()

    writer.writerow(new_row)


# %%




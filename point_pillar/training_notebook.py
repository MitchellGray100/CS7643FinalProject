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
for i in range(len(train_dataset)):
    _ = train_dataset[i]
for i in range(len(val_dataset)):
    _ = val_dataset[i]

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


# %%




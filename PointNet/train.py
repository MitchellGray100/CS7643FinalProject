# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from loss import t_net_regularization_loss
from PointNetClassification import PointNetClassification

from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import SamplePoints
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_batch

import os
import matplotlib.pyplot as plt
import traceback

# detect colab
import sys
is_colab = "google.colab" in sys.modules


def plot_curves(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
    config_name,
    out_dir="plots",
):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))
    wrapped_config_name = config_name.replace("_aug", "_\naug")

    # loss
    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, test_losses, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"loss vs epoch\n({wrapped_config_name})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_{config_name}.png"))
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, label="train accuracy")
    plt.plot(epochs, test_accuracies, label="test accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy percent")
    plt.title(f"accuracy vs epoch\n({wrapped_config_name})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"accuracy_{config_name}.png"))
    plt.close()


def log_result(
    config_name,
    model_name,
    batch_size,
    num_epochs,
    learning_rate,
    learning_rate_step_size,
    learning_rate_decay_factor,
    min_learning_rate,
    regularization_loss_weight,
    dropout_prob,
    adam_weight_decay,
    augment_training_data,
    num_points,
    batch_norm_init_decay,
    batch_norm_decay_rate,
    batch_norm_decay_step,
    batch_norm_decay_clip,
    best_test_accuracy,
    path="log/results.csv",
):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    header = (
        "config_name,"
        "model_name,"
        "batch_size,"
        "num_epochs,"
        "learning_rate,"
        "learning_rate_step_size,"
        "learning_rate_decay_factor,"
        "min_learning_rate,"
        "regularization_loss_weight,"
        "dropout_prob,"
        "adam_weight_decay,"
        "augment_training_data,"
        "num_points,"
        "batch_norm_init_decay,"
        "batch_norm_decay_rate,"
        "batch_norm_decay_step,"
        "batch_norm_decay_clip,"
        "best_test_accuracy\n"
    )

    file_exists = os.path.exists(path)
    with open(path, "a") as f:
        if not file_exists:
            f.write(header)

        f.write(
            f"{config_name},"
            f"{model_name},"
            f"{batch_size},"
            f"{num_epochs},"
            f"{learning_rate},"
            f"{learning_rate_step_size},"
            f"{learning_rate_decay_factor},"
            f"{min_learning_rate},"
            f"{regularization_loss_weight},"
            f"{dropout_prob},"
            f"{adam_weight_decay},"
            f"{int(augment_training_data)},"
            f"{num_points},"
            f"{batch_norm_init_decay},"
            f"{batch_norm_decay_rate},"
            f"{batch_norm_decay_step},"
            f"{batch_norm_decay_clip},"
            f"{best_test_accuracy}\n"
        )


def get_batch(batch, device):
    # to_dense_batch returns (batch_size, num_nodes, features) and a mask
    # batch.batch is the index vector [0, 0, ... 1, 1, ...]
    points, mask = to_dense_batch(batch.pos, batch=batch.batch)

    # Transpose to (Batch, 3, Num_points) for PointNet
    points = points.transpose(2, 1)

    points_on_device = points.to(device)
    labels_on_device = batch.y.to(device)
    return points_on_device, labels_on_device


def count_correct(pred, labels):
    return (pred == labels).sum().item()


def get_batch_norm_momentum(
    step_index, batch_size, init_decay, decay_rate, decay_step, decay_clip
):
    total_samples_processed = step_index * batch_size
    num_decay_periods = float(total_samples_processed) / float(decay_step)
    batch_norm_momentum_deduction = init_decay * (decay_rate**num_decay_periods)
    batch_norm_decay_original = 1.0 - batch_norm_momentum_deduction
    batch_norm_decay_original = min(decay_clip, batch_norm_decay_original)
    new_momentum = 1.0 - batch_norm_decay_original
    return new_momentum


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    regularization_loss_weight=0.001,
    global_step_count=0,
    batch_size=32,
    batch_norm_init_decay=0.5,
    batch_norm_decay_rate=0.5,
    batch_norm_decay_step=200000,
    batch_norm_decay_clip=0.99,
    grad_clip=1.0,
    label_smoothing=0.1,
):
    model.train()
    total_loss = 0
    correct_pred_num = 0
    total_samples_num = 0
    step_index = global_step_count

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:

        # clear
        optimizer.zero_grad()

        # points
        points, labels = get_batch(batch, device)

        # forward
        logits, trans_feat = model(points)

        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        # Regularization loss for feature transform
        regularization_loss = t_net_regularization_loss(trans_feat)
        loss = loss + regularization_loss_weight * regularization_loss

        # backward
        loss.backward()

        # Gradient clipping for training stability
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        # loss
        total_loss += loss.item()
        pred = logits.argmax(dim=1)

        correct_pred_num += count_correct(pred, labels)
        total_samples_num += labels.size(0)

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            {
                "loss": loss.item(),
                "accuracy": (correct_pred_num / total_samples_num) * 100,
                "lr": current_lr,
            }
        )

        step_index += 1

    average_loss = total_loss / len(loader)
    average_accuracy = 100 * correct_pred_num / total_samples_num

    return average_loss, average_accuracy, step_index


def evaluate_one_epoch(model, loader, device, label_smoothing=0.0):
    model.eval()

    correct_pred_num = 0
    total_samples_num = 0
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            points, labels = get_batch(batch, device)

            logits, transform_matrix_feature = model(points)

            pred = logits.argmax(dim=1)
            correct_pred_num += count_correct(pred, labels)
            total_samples_num += labels.size(0)

            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            total_loss += loss.item()

    average_loss = total_loss / len(loader)
    accuracy = (correct_pred_num / total_samples_num) * 100
    return average_loss, accuracy


def train(
    model_name="ModelNet40",  # ModelNet10 or ModelNet40
    batch_size=32,
    num_epochs=250,  # 250
    learning_rate=0.001,  # 0.01, 0.001
    min_learning_rate=1e-6,
    regularization_loss_weight=0.001,
    dropout_prob=0.4,
    adam_weight_decay=1e-4,  # 0, 1e-4
    augment_training_data=True,
    num_points=1024,  # sample points from 3d models
    batch_norm_init_decay=0.5,
    batch_norm_decay_rate=0.5,
    batch_norm_decay_step=200000,
    batch_norm_decay_clip=0.99,
    label_smoothing=0.1,
    grad_clip=1.0,
):
    model_path = None
    number_of_classes = 0

    default_local_root = "../data/"
    default_colab_root = "/content/data/"
    file_path = os.environ.get(
        "MODELNET_ROOT", default_colab_root if is_colab else default_local_root
    )
    print(f"MODELNET_ROOT env var: {os.environ.get('MODELNET_ROOT', 'NOT SET')}")
    print(f"Using data path: {file_path}")
    print(f"Model path: {model_path}")

    # reload data all the time
    will_force_reload = False

    config_name = (
        f"{model_name}_bs{batch_size}_ep{num_epochs}"
        f"_lr{learning_rate}_minlr{min_learning_rate}"
        f"_reg{regularization_loss_weight}_drop{dropout_prob}"
        f"_wd{adam_weight_decay}"
        f"_aug{int(augment_training_data)}_pts{num_points}"
        f"_bnID{batch_norm_init_decay}"
        f"_bnDR{batch_norm_decay_rate}"
        f"_bnDS{batch_norm_decay_step}"
        f"_bnDC{batch_norm_decay_clip}"
    )

    if model_name == "ModelNet10":
        model_path = os.path.join(file_path, "ModelNet10")
        model_name_short = "10"
        number_of_classes = 10
    elif model_name == "ModelNet40":
        model_path = os.path.join(file_path, "ModelNet40")
        model_name_short = "40"
        number_of_classes = 40

    os.makedirs("model", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if augment_training_data:
        train_transform = T.Compose(
            [
                SamplePoints(num_points),
                T.RandomRotate(15, axis='y'),
                T.RandomScale((0.8, 1.2)),
                T.RandomJitter(translate=0.01),
                T.NormalizeScale(),
            ]
        )
    else:
        train_transform = T.Compose([SamplePoints(num_points), T.NormalizeScale()])

    test_transform = T.Compose([SamplePoints(num_points), T.NormalizeScale()])
    print("Creating dataset.")
    try:
        train_dataset = ModelNet(
            root=model_path,
            name=model_name_short,
            train=True,
            pre_transform=None,
            force_reload=will_force_reload,
            transform=train_transform,
        )

        test_dataset = ModelNet(
            root=model_path,
            name=model_name_short,
            train=False,
            pre_transform=None,
            force_reload=will_force_reload,
            transform=test_transform,
        )
    except Exception as e:
        print(f"Error creating datasets: {e}")
        traceback.print_exc()
        raise

    print("Loading data.")
    try:
        num_workers = 4
        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        traceback.print_exc()
        raise

    print(f"number of classes {train_dataset.num_classes}")

    # PointNetClassification
    model = PointNetClassification(
        num_classes=number_of_classes,
        dropout_probability=dropout_prob,
    )
    model = model.to(device)

    # optimizer with adam and weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=adam_weight_decay
    )

    # Create CosineAnnealingLR scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=min_learning_rate
    )

    # training loop
    best_test_accuracy = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []
    global_step_count = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        current_momentum = get_batch_norm_momentum(
            global_step_count,
            batch_size,
            batch_norm_init_decay,
            batch_norm_decay_rate,
            batch_norm_decay_step,
            batch_norm_decay_clip,
        )
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.momentum = current_momentum

        # train
        train_loss, train_accuracy, global_step_count = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            regularization_loss_weight,
            global_step_count=global_step_count,
            batch_size=batch_size,
            batch_norm_init_decay=batch_norm_init_decay,
            batch_norm_decay_rate=batch_norm_decay_rate,
            batch_norm_decay_step=batch_norm_decay_step,
            batch_norm_decay_clip=batch_norm_decay_clip,
            grad_clip=grad_clip,
            label_smoothing=label_smoothing,
        )

        # evaluate (no label smoothing for evaluation)
        test_loss, test_accuracy = evaluate_one_epoch(model, test_loader, device, label_smoothing=0.0)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

        current_learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"\nlearning rate: {current_learning_rate:.6f}, train loss: {train_loss}, test loss: {test_loss}, train accuracy: {train_accuracy}%, test accuracy: {test_accuracy}%"
        )

        # save for plot
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    print(f"best test accuracy {best_test_accuracy}%")
    plot_curves(
        train_losses, test_losses, train_accuracies, test_accuracies, config_name
    )
    log_result(
        config_name=config_name,
        model_name=model_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        learning_rate_step_size=total_steps,  # T_max for CosineAnnealingLR
        learning_rate_decay_factor=0.0,  # Not applicable for CosineAnnealingLR
        min_learning_rate=min_learning_rate,
        regularization_loss_weight=regularization_loss_weight,
        dropout_prob=dropout_prob,
        adam_weight_decay=adam_weight_decay,
        augment_training_data=augment_training_data,
        num_points=num_points,
        batch_norm_init_decay=batch_norm_init_decay,
        batch_norm_decay_rate=batch_norm_decay_rate,
        batch_norm_decay_step=batch_norm_decay_step,
        batch_norm_decay_clip=batch_norm_decay_clip,
        best_test_accuracy=best_test_accuracy,
    )


if __name__ == "__main__":
    train(model_name="ModelNet10")

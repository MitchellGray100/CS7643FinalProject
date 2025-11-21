# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from loss import t_net_regularization_loss
from PointNetClassification import PointNetClassification
import augment_train_data

from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import Compose

import os
import matplotlib.pyplot as plt

# detect colab
try:
    import google.colab  # type: ignore
    is_colab = True
except ImportError:
    is_colab = False

def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies, config_name, out_dir="plots"):
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

def log_result(config_name, model_name, batch_size, num_epochs, learning_rate, learning_rate_step_size, learning_rate_decay_factor, min_learning_rate, regularization_loss_weight, dropout_prob, adam_weight_decay, augment_training_data, num_points,
               batch_norm_init_decay, batch_norm_decay_rate, batch_norm_decay_step, batch_norm_decay_clip, best_test_accuracy,
               path="log/results.csv"):

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

    batch_size = batch.y.size(0)
    total_num_points = batch.pos.size(0) // batch_size
    points = batch.pos.reshape(batch_size, total_num_points, 3)
    points = points.transpose(2, 1)
    points = points.contiguous()
    points_on_device = points.to(device)
    labels_on_device = batch.y.to(device)
    return points_on_device, labels_on_device

def count_correct(pred, labels):
    return (pred == labels).sum().item()

def get_batch_norm_momentum(step_index, batch_size, init_decay, decay_rate, decay_step, decay_clip):
    total_samples_processed = step_index * batch_size
    num_decay_periods = float(total_samples_processed)/float(decay_step)
    batch_norm_momentum_deduction = init_decay * (decay_rate**num_decay_periods)
    batch_norm_decay_original = 1.0 - batch_norm_momentum_deduction
    batch_norm_decay_original = min(decay_clip, batch_norm_decay_original)
    new_momentum = 1.0 - batch_norm_decay_original
    return new_momentum

def get_learning_rate(step_index, batch_size, base_learning_rate, decay_rate, decay_step, min_learning_rate):
    total_samples_processed  = step_index * batch_size
    num_decay_steps = total_samples_processed  // decay_step
    new_learning_rate = base_learning_rate * (decay_rate ** num_decay_steps)
    if new_learning_rate < min_learning_rate:
        new_learning_rate = min_learning_rate
    else:
        new_learning_rate = new_learning_rate
    return new_learning_rate

def normalize_unit_sphere(data):
    pos = data.pos
    point_centroid = pos.mean(dim=0, keepdim=True)
    pos = pos - point_centroid
    distances_from_origin = torch.sqrt((pos ** 2).sum(dim=1))
    max_distance_from_origin = distances_from_origin.max()
    normalized_pos = pos/max_distance_from_origin
    data.pos = normalized_pos
    return data

def train_one_epoch(model, loader, optimizer, device, regularization_loss_weight=0.001,
                    global_step_count=0, batch_size=32, batch_norm_init_decay=0.5, batch_norm_decay_rate=0.5, batch_norm_decay_step=200000, batch_norm_decay_clip=0.99,
                    base_learning_rate=0.001, learning_rate_decay_rate=0.7, learning_rate_decay_step=200000, learning_rate_min=0.0 ):

    model.train()
    total_loss = 0
    correct_pred_num = 0
    total_samples_num = 0
    step_index = global_step_count

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        batch_norm_momentum = get_batch_norm_momentum(step_index, batch_size, batch_norm_init_decay, batch_norm_decay_rate, batch_norm_decay_step, batch_norm_decay_clip)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.momentum = batch_norm_momentum
        learning_rate = get_learning_rate(step_index, batch_size, base_learning_rate=base_learning_rate, decay_rate=learning_rate_decay_rate, decay_step=learning_rate_decay_step, min_learning_rate=learning_rate_min)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        # clear
        optimizer.zero_grad()

        # points
        points, labels = get_batch(batch, device)

        # forward
        logits, trans_feat = model(points)

        # classification loss
        loss = F.cross_entropy(logits, labels)

        # Regularization loss for feature transform
        regularization_loss = t_net_regularization_loss(trans_feat)
        loss = loss + regularization_loss_weight * regularization_loss

        # backward
        loss.backward()
        optimizer.step()

        # loss
        total_loss += loss.item()
        pred = logits.argmax(dim=1)

        correct_pred_num += count_correct(pred, labels)
        total_samples_num += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'accuracy': (correct_pred_num / total_samples_num)*100})

        step_index += 1

    average_loss = total_loss/len(loader)
    average_accuracy = 100 * correct_pred_num/total_samples_num

    return average_loss, average_accuracy, step_index

def evaluate_one_epoch( model, loader, device):
    model.eval()

    correct_pred_num = 0
    total_samples_num = 0
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            points, labels = get_batch(batch, device)

            logits, transform_matrix_feature = model(points)

            pred = logits.argmax(dim=1)
            correct_pred_num += count_correct(pred, labels)
            total_samples_num += labels.size(0)

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

    average_loss = total_loss / len(loader)
    accuracy = (correct_pred_num/total_samples_num) * 100
    return average_loss, accuracy


def train(
    model_name = "ModelNet40",  # ModelNet10 or ModelNet40
    batch_size = 32,
    num_epochs = 250, # 250
    learning_rate = 0.001, # 0.01, 0.001
    learning_rate_decay_step = 200000,
    learning_rate_decay_factor = 0.7,
    min_learning_rate = 1e-5,
    regularization_loss_weight = 0.001,
    dropout_prob = 0.3,
    adam_weight_decay = 0 ,# 0, 1e-4
    augment_training_data = True,
    num_points = 1024, # sample points from 3d models
    batch_norm_init_decay = 0.5,
    batch_norm_decay_rate = 0.5,
    batch_norm_decay_step = 200000,
    batch_norm_decay_clip = 0.99):

    model_path = None
    number_of_classes = 0

    default_local_root = "../../data/"
    default_colab_root = "/content/data/"
    file_path = os.environ.get("MODELNET_ROOT", default_colab_root if is_colab else default_local_root)

    # # params!
    # model_name = "ModelNet40",  # ModelNet10 or ModelNet40
    # batch_size = 32,
    # num_epochs = 250,  # 250
    # learning_rate = 0.001,  # 0.01, 0.001
    # learning_rate_decay_steps = 200000,
    # learning_rate_decay_factor = 0.7,
    # min_learning_rate = 1e-5,
    # regularization_loss_weight = 0.001,
    # dropout_prob = 0.3,
    # adam_weight_decay = 0,  # 0, 1e-4
    # augment_training_data = True,
    # num_points = 1024,  # sample points from 3d models
    # batch_norm_init_decay = 0.5,
    # batch_norm_decay_rate = 0.5,
    # batch_norm_decay_step = 200000,
    # batch_norm_decay_clip = 0.99,

    # reload data all the time
    will_force_reload = False

    config_name = (
        f"{model_name}_bs{batch_size}_ep{num_epochs}"
        f"_lr{learning_rate}_schedS{learning_rate_decay_step}"
        f"_schedD{learning_rate_decay_factor}"
        f"_reg{regularization_loss_weight}_drop{dropout_prob}"
        f"_wd{adam_weight_decay}"
        f"_aug{int(augment_training_data)}_pts{num_points}"
        f"_bnID{batch_norm_init_decay}"
        f"_bnDR{batch_norm_decay_rate}"
        f"_bnDS{batch_norm_decay_step}"
        f"_bnDC{batch_norm_decay_clip}")

    if model_name == "ModelNet10":
        model_path = os.path.join(file_path, "ModelNet10")
        model_name_short = "10"
        number_of_classes = 10
    elif model_name == "ModelNet40":
        model_path = os.path.join(file_path, "ModelNet40")
        model_name_short = "40"
        number_of_classes = 40

    os.makedirs('model', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if augment_training_data:
        train_transform = Compose([
            SamplePoints(num_points),
            normalize_unit_sphere,
            augment_train_data.apply_random_y_rotation,
            augment_train_data.apply_random_jitter])
    else:
        train_transform = Compose([
            SamplePoints(num_points),
            normalize_unit_sphere])

    test_transform = Compose([
        SamplePoints(num_points),
        normalize_unit_sphere])

    train_dataset = ModelNet(
        root=model_path,
        name=model_name_short,
        train=True,
        pre_transform=None,
        force_reload=will_force_reload,
        transform=train_transform)

    test_dataset = ModelNet(
        root=model_path,
        name=model_name_short,
        train=False,
        pre_transform=None,
        force_reload=will_force_reload,
        transform=test_transform)

    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"number of classes {train_dataset.num_classes}")

    # PointNetClassification
    model = PointNetClassification(num_classes=number_of_classes, dropout_probability=dropout_prob)
    model = model.to(device)

    # optimizer with adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=adam_weight_decay)

    # training loop
    best_test_accuracy = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []
    global_step_count = 0

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')

        # train
        train_loss, train_accuracy, global_step_count = train_one_epoch(model, train_loader, optimizer, device, regularization_loss_weight,
                                                                        global_step_count=global_step_count, batch_size=batch_size, batch_norm_init_decay=batch_norm_init_decay, batch_norm_decay_rate=batch_norm_decay_rate, batch_norm_decay_step=batch_norm_decay_step, batch_norm_decay_clip=batch_norm_decay_clip,
                                                                        base_learning_rate=learning_rate, learning_rate_decay_rate=learning_rate_decay_factor, learning_rate_decay_step=learning_rate_decay_step, learning_rate_min=min_learning_rate)

        # evaluate
        test_loss, test_accuracy = evaluate_one_epoch(model, test_loader, device)

        current_learning_rate = optimizer.param_groups[0]['lr']
        print(f"\nlearning rate: {current_learning_rate:.6f}, train loss: {train_loss}, test loss: {test_loss}, train accuracy: {train_accuracy}%, test accuracy: {test_accuracy}%")

        # save for plot
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)


    print(f"best test accuracy {best_test_accuracy}%")
    plot_curves(train_losses, test_losses, train_accuracies, test_accuracies, config_name)
    log_result(config_name=config_name, model_name=model_name, batch_size=batch_size, num_epochs=num_epochs,
               learning_rate=learning_rate, learning_rate_step_size=learning_rate_decay_step, learning_rate_decay_factor=learning_rate_decay_factor, min_learning_rate=min_learning_rate,
               regularization_loss_weight=regularization_loss_weight, dropout_prob=dropout_prob,
               adam_weight_decay=adam_weight_decay, augment_training_data=augment_training_data, num_points=num_points,
               batch_norm_init_decay=batch_norm_init_decay, batch_norm_decay_rate=batch_norm_decay_rate, batch_norm_decay_step=batch_norm_decay_step, batch_norm_decay_clip=batch_norm_decay_clip, best_test_accuracy=best_test_accuracy)

if __name__ == '__main__':
    train(model_name = "ModelNet40")
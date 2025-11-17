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

def plot_curves(train_losses, train_accuracies, test_accuracies, config_name, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    # loss
    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"train loss vs epoch ({config_name})")
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
    plt.title(f"accuracy vs epoch ({config_name})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"accuracy_{config_name}.png"))
    plt.close()

def log_result(config_name, model_name, batch_size, num_epochs, learning_rate, learning_rate_step_size, learning_rate_decay_factor, min_learning_rate, regularization_weight, dropout_prob, adam_weight_decay, augment_training_data, num_points, best_test_accuracy, path="results.csv"):
    header = (
        "config_name,"
        "model_name,"
        "batch_size,"
        "num_epochs,"
        "learning_rate,"
        "learning_rate_step_size,"
        "learning_rate_decay_factor,"
        "min_learning_rate,"
        "regularization_weight,"
        "dropout_prob,"
        "adam_weight_decay,"
        "augment_training_data,"
        "num_points,"
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
            f"{regularization_weight},"
            f"{dropout_prob},"
            f"{adam_weight_decay},"
            f"{int(augment_training_data)},"
            f"{num_points},"
            f"{best_test_accuracy}\n"
        )

def get_batch(batch, device):

    batch_size = batch.y.size(0)
    total_num_points = batch.pos.size(0) // batch_size
    points = batch.pos.reshape(batch_size, total_num_points, 3)
    points = points.transpose(2, 1).contiguous()

    return points.to(device), batch.y.to(device)

def count_correct(pred, labels):
    return (pred == labels).sum().item()

def one_epoch(model, loader, optimizer, device, regularization_loss_weight=0.001):

    model.train()
    total_loss = 0
    correct_pred_num = 0
    total_samples_num = 0
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        # points
        points, labels = get_batch(batch, device)

        # clear
        optimizer.zero_grad()

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

        pbar.set_postfix({'loss': loss.item(),
                          'accuracy': (correct_pred_num / total_samples_num) * 100})

    average_loss = total_loss / len(loader)
    average_accuracy = 100. * correct_pred_num / total_samples_num

    return average_loss, average_accuracy

def evaluate(model, loader, device):
    model.eval()
    correct_pred_num = 0
    total_samples_num = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            points, labels = get_batch(batch, device)

            logits, transform_matrix_feature = model(points)
            pred = logits.argmax(dim=1)

            correct_pred_num += count_correct(pred, labels)
            total_samples_num += labels.size(0)

    accuracy = (correct_pred_num / total_samples_num) * 100
    return accuracy

def set_batch_norm_momentum(model, new_momentum):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = new_momentum

def train():
    model_path = None
    number_of_classes = 0

    file_path = '../../data/'
    # params!
    model_name = "ModelNet10"  # ModelNet10 or ModelNet40
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001
    learning_rate_step_size = 20
    learning_rate_decay_factor = 0.5
    min_learning_rate = 1e-5
    regularization_weight = 0.001
    dropout_prob = 0.3
    adam_weight_decay = 1e-4
    augment_training_data = True
    num_points = 1024 # sample points from models

    config_name = (
        f"{model_name}_bs{batch_size}_lr{learning_rate}"
        f"_reg{regularization_weight}_drop{dropout_prob}"
        f"_aug{int(augment_training_data)}_pts{num_points}"
    )

    if model_name == "ModelNet10":
        model_path = file_path+"ModelNet10/"
        model_name = "10"
        number_of_classes = 10
    elif model_name == "ModelNet40":
        model_path = file_path+"ModelNet40/"
        model_name = "40"
        number_of_classes = 40

    os.makedirs('model', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    processed_dir = os.path.join(model_path, 'processed')
    need_reload = False
    if os.path.exists(processed_dir):
        try:
            temp_dataset = ModelNet(root=model_path, name=model_name, train=True)
            current_points = temp_dataset[0].pos.shape[0]
            if current_points != num_points:
                need_reload = True
        except:
            need_reload = True

    # augment train data
    if augment_training_data:
        train_transform = Compose([augment_train_data.apply_random_y_rotation,augment_train_data.apply_random_jitter])
    else:
        train_transform = None

    train_dataset = ModelNet(root=model_path, name=model_name, train=True, pre_transform=SamplePoints(1024), force_reload=need_reload, transform=train_transform)
    test_dataset = ModelNet(root=model_path, name=model_name, train=False, pre_transform=SamplePoints(1024), force_reload=need_reload)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"number of classes {train_dataset.num_classes}")
    print(f"train n samples: {len(train_dataset)}")
    print(f"test n samples: {len(test_dataset)}")


    # PointNetClassification
    model = PointNetClassification(num_classes=number_of_classes, dropout_probability=dropout_prob)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step_size, gamma=learning_rate_decay_factor)

    # training loop
    batch_norm_momentum_decay_start = 0.5
    batch_norm_momentum_decay_end = 0.99
    best_test_accuracy = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')

        # change batch norm momentum
        percent_epoch = (epoch - 1) / (num_epochs - 1)
        decay = batch_norm_momentum_decay_start + ((batch_norm_momentum_decay_end-batch_norm_momentum_decay_start)*percent_epoch)
        new_momentum = 1 - decay
        set_batch_norm_momentum(model, new_momentum)

        # train
        train_loss, train_accuracy = one_epoch(model, train_loader, optimizer, device, regularization_weight)

        # evaluate
        test_accuracy = evaluate(model, test_loader, device)

        # learning rate schedule step
        scheduler.step()

        # have floor for learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_learning_rate)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"learning rate: {current_lr:.6f}, train loss: {train_loss}, train accuracy: {train_accuracy}%, test accuracy: {test_accuracy}%")

        # save for plot
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # save best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), 'model/pointnet_model.pth')
            print(f"save model test accuracy: {best_test_accuracy:}%)")

    print(f"best test accuracy {best_test_accuracy}%")
    plot_curves(train_losses, train_accuracies, test_accuracies, config_name)
    log_result(config_name=config_name, model_name=model_name, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, learning_rate_step_size=learning_rate_step_size, learning_rate_decay_factor=learning_rate_decay_factor, min_learning_rate=min_learning_rate, regularization_weight=regularization_weight, dropout_prob=dropout_prob, adam_weight_decay=adam_weight_decay, augment_training_data=augment_training_data, num_points=num_points, best_test_accuracy=best_test_accuracy)

if __name__ == '__main__':
    train()
import torch
from torch.nn.functional import dropout
from tqdm import tqdm
import torch.nn.functional as F

from loss import t_net_regularization_loss
from PointNetClassification import PointNetClassification

from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import SamplePoints

import os

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


def train():
    model_path = None
    number_of_classes = 0

    # params!
    file_path = '../../data/'
    model_name = "ModelNet10"  # ModelNet10 or ModelNet40
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    reg_weight = 0.001
    dropout_prob = 0.3

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
    num_points = 1024
    if os.path.exists(processed_dir):
        try:
            temp_dataset = ModelNet(root=model_path, name=model_name, train=True)
            current_points = temp_dataset[0].pos.shape[0]
            if current_points != num_points:
                need_reload = True
        except:
            need_reload = True

    train_dataset = ModelNet(root=model_path, name=model_name, train=True, pre_transform=SamplePoints(1024), force_reload=need_reload)
    test_dataset = ModelNet(root=model_path, name=model_name, train=False, pre_transform=SamplePoints(1024), force_reload=need_reload)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"number of classes {train_dataset.num_classes}")
    print(f"train n samples: {len(train_dataset)}")
    print(f"test n samples: {len(train_dataset)}")


    # PointNetClassification
    model = PointNetClassification(num_classes=number_of_classes, dropout_probability=dropout_prob)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # training loop
    best_test_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')

        # train
        train_loss, train_acc = one_epoch(model, train_loader, optimizer, device, reg_weight)

        # evaluate
        test_accuracy = evaluate(model, test_loader, device)

        # learning rate schedule step
        scheduler.step()

        print(f"train loss: {train_loss}, train accuracy: {train_acc}%, Test Acc: {test_accuracy}%")

        # save best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), 'model/pointnet_model.pth')
            print(f"save model test accuracy: {best_test_accuracy:}%)")

    print(f"best test accuracy {best_test_accuracy}%")

if __name__ == '__main__':
    train()
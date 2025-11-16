# /point_pillar/pillar_trainer.py
# Author: Yonghao Li (Paul)
# The training that run the training loop and get the loss/accuracy for reports

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, val_loader, device,
                 lr=1e-3, weight_decay=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for i, (points, labels) in enumerate(self.train_loader):
            points = points.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(points)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # stats
            running_loss += loss.item() * points.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            # light verbose
            if i % 20 == 0:
                batch_acc = (preds == labels).float().mean().item()
                print(f"[Epoch {epoch} Batch {i}] "
                      f"Loss={loss.item():.4f} Acc={batch_acc:.3f}")

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        print(f"--> Train Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
        self.history["train_loss"].append(epoch_loss)
        self.history["train_acc"].append(epoch_acc)

        return epoch_loss, epoch_acc
    
    def evaluate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with torch.no_grad():
            for points, labels in self.val_loader:
                points = points.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(points)
                loss = self.criterion(logits, labels)

                running_loss += loss.item() * points.size(0)
                preds = logits.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_total += labels.size(0)

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        print(f"--> Val   Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
        self.history["val_loss"].append(epoch_loss)
        self.history["val_acc"].append(epoch_acc)

        return epoch_loss, epoch_acc
    
    def fit(self, num_epochs, save_path=None):
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate(epoch)

            if save_path is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"  [*] Saved new best model (val_acc={val_acc:.4f})")


    def plot_history(self, save_path=None):
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # loss
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(f"{save_path}_loss.png", dpi=300, bbox_inches="tight")

        plt.show()

        # accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_acc"], label="Train Acc")
        plt.plot(epochs, self.history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(f"{save_path}_acc.png", dpi=300, bbox_inches="tight")

        plt.show()

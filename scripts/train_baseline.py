import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random
import time

import os as _os
import sys as _sys
from pathlib import Path as _Path

ROOT = _Path(_os.environ.get("PRUNING_ROOT", _Path(__file__).resolve().parents[1])).resolve()
_sys.path.insert(0, str(ROOT / "src"))

from models.resnet import get_resnet18
from data.dataset import AerialLandscapeDataset, get_transforms
from utils.split_data import split_and_save_dataset, load_split_from_file


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=10):
    """
    Trains the model with validation and early stopping.
    """
    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    no_improve_count = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        epoch_time = time.time() - start_time

        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Time: {epoch_time:.2f}s')

        # --- 验证阶段 ---
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val

        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

        # 学习率调度
        scheduler.step()

        # 早停机制
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = model.state_dict().copy()
            no_improve_count = 0
            # 保存最佳模型
            torch.save(best_model_wts, os.path.join(ROOT, 'checkpoints', 'best_resnet18_aerial.pth'))
            print("  -> Best model saved.")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    print(f'Best val Acc: {best_val_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model


def main():
    # --- 配置 ---
    DATA_ROOT = os.environ.get("AERIAL_DATA_ROOT", str(ROOT / "data" / "Aerial_Landscapes"))
    BATCH_SIZE = 256
    NUM_EPOCHS = 90
    LEARNING_RATE = 0.1
    PATIENCE = 10
    SPLIT_FILE_PREFIX = str(ROOT / "splits" / "aerial_split")

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 数据划分与加载 ---
    train_split_file = f"{SPLIT_FILE_PREFIX}_train.txt"
    val_split_file = f"{SPLIT_FILE_PREFIX}_val.txt"

    if os.path.exists(train_split_file) and os.path.exists(val_split_file):
        print("Loading existing dataset splits...")
        train_samples = load_split_from_file(train_split_file)
        val_samples = load_split_from_file(val_split_file)
    else:
        print("Creating new dataset splits...")
        train_samples, val_samples = split_and_save_dataset(DATA_ROOT, train_ratio=0.8,
                                                            output_file_prefix=SPLIT_FILE_PREFIX)

    # --- 获取数据变换 ---
    train_transform, val_transform = get_transforms()

    # --- 创建数据集和加载器 ---
    train_dataset = AerialLandscapeDataset(DATA_ROOT, train_samples, transform=train_transform)
    val_dataset = AerialLandscapeDataset(DATA_ROOT, val_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # --- 模型、损失函数、优化器、调度器 ---
    model = get_resnet18(num_classes=15)  # 使用函数创建模型
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    # --- 开始训练 ---
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device,
                                patience=PATIENCE)
    print("Training completed.")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    main()
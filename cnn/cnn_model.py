import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                       'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

DATA_DIR  = '/Volumes/T9/ml-project-data/CRC-VAL-HE-7K'
SAVE_PATH = '/Volumes/T9/ml-project-models/cnn_weights.pth'
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
EPOCHS    = 30
BATCH     = 64
LR        = 1e-3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # simulates noise/blur variation
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# Model — ResNet18 pretrained backbone with a custom classifier head
class ColonCancerCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    is_valid = lambda p: not os.path.basename(p).startswith('._')
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform, is_valid_file=is_valid)
    CLASS_NAMES  = full_dataset.classes
    NUM_CLASSES  = len(CLASS_NAMES)
    print(f'Classes ({NUM_CLASSES}): {CLASS_NAMES}')
    print(f'Total images: {len(full_dataset)}')

    n_train = int(0.8 * len(full_dataset))
    n_test  = len(full_dataset) - n_train
    train_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply val transforms to test split
    test_dataset.dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform, is_valid_file=is_valid)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)
    print(f'Train: {n_train} | Test: {n_test}')

    model = ColonCancerCNN(num_classes=NUM_CLASSES).to(DEVICE)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

  
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    train_losses, train_accs = [], []
    test_losses,  test_accs  = [], []
    best_acc = 0.0

    def evaluate(loader):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                total_loss += criterion(outputs, labels).item() * imgs.size(0)
                correct    += (outputs.argmax(1) == labels).sum().item()
                total      += imgs.size(0)
        return total_loss / total, correct / total

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  

            run_loss    += loss.item() * imgs.size(0)
            run_correct += (outputs.argmax(1) == labels).sum().item()
            run_total   += imgs.size(0)

        tr_loss, tr_acc = run_loss / run_total, run_correct / run_total
        te_loss, te_acc = evaluate(test_loader)

        train_losses.append(tr_loss);  train_accs.append(tr_acc)
        test_losses.append(te_loss);   test_accs.append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f'Epoch {epoch:2d}/{EPOCHS} | '
              f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | '
              f'Test  Loss: {te_loss:.4f} Acc: {te_acc:.4f}')

    print(f'\nBest test accuracy: {best_acc:.4f}')

    # Results
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, train_losses, label='Train', color='steelblue')
    ax1.plot(epochs_range, test_losses,  label='Test',  color='tomato')
    ax1.set_title('Loss per Epoch'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()

    ax2.plot(epochs_range, train_accs, label='Train', color='steelblue')
    ax2.plot(epochs_range, test_accs,  label='Test',  color='tomato')
    ax2.set_title('Accuracy per Epoch'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'training_curves.png'), dpi=150)
    plt.show()

    # Confusion matrix
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title('Normalized Confusion Matrix (Test Set)')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'), dpi=150)
    plt.show()

    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

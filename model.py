import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Large_Weights

# --- CONFIGURATION ---
TRAIN_DIR = "dataset/final-new/train"
VAL_DIR = "dataset/final-new/valid"

BATCH_SIZE = 128
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=10
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train" and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    print(f"Stage Complete. Best Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def main():
    print(f"Using device: {DEVICE}")

    # --- STRONG AUGMENTATION ---
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),  # Increased rotation slightly
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3
                ),  # Increased jitter
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("ERROR: Train or Validation dataset directory not found")
        return

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms["val"])

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    num_workers = min(8, os.cpu_count() if os.cpu_count() else 4)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    print("Loading MobileNetV3-Large...")
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    # FIX 2: Increased Dropout to 0.5 (Strong Regularization)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(num_ftrs, num_classes)
    )

    model = model.to(DEVICE)

    # FIX 1: Label Smoothing (Prevents model from being "too confident")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n--- STAGE 1: WARM UP (Head Only) ---")
    # FIX 3: Added Weight Decay (L2 Regularization)
    optimizer_head = optim.Adam(
        model.classifier.parameters(), lr=0.001, weight_decay=1e-4
    )
    model = train_model(model, dataloaders, criterion, optimizer_head, num_epochs=10)

    print("\n--- STAGE 2: FINE TUNING (Backbone Unfrozen) ---")

    # Unfreeze the last 6 blocks
    for param in model.features[-6:].parameters():
        param.requires_grad = True

    # Weight decay added here too
    optimizer_fine = optim.Adam(
        [
            {"params": model.features[-6:].parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_fine, T_max=40)

    model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer_fine,
        scheduler=scheduler,
        num_epochs=40,
    )

    save_path = "breed_classifier_mobilenetv3.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    print("Converting to TorchScript...")
    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(DEVICE)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("breed_classifier_mobile.pt")
    print("Mobile-ready model saved to breed_classifier_mobile.pt")


if __name__ == "__main__":
    main()

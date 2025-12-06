import copy
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Large_Weights

warnings.filterwarnings("ignore", category=UserWarning)

CONFIG = {
    "DATA_DIR": "dataset/Indian_bovine_breeds",
    "BATCH_SIZE": 64,  # Reduced slightly to be safe with SGD momentum
    "EPOCHS": 50,
    "IMG_SIZE": (224, 224),
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def safe_pil_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except OSError:
        return Image.new("RGB", (224, 224))


def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            # SGD Strategy: Harder augmentation to prevent memorization
            transforms.RandomResizedCrop(CONFIG["IMG_SIZE"], scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize(CONFIG["IMG_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def get_dataloaders(data_dir, train_tf, val_tf):
    print(f"Loading data from: {data_dir}")

    full_train_dataset = datasets.ImageFolder(
        data_dir, transform=train_tf, loader=safe_pil_loader
    )
    full_val_dataset = datasets.ImageFolder(
        data_dir, transform=val_tf, loader=safe_pil_loader
    )

    targets = full_train_dataset.targets
    class_names = full_train_dataset.classes

    train_idx, temp_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        shuffle=True,
        stratify=targets,
        random_state=42,
    )

    temp_targets = np.array(targets)[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, shuffle=True, stratify=temp_targets, random_state=42
    )

    train_ds = Subset(full_train_dataset, train_idx)
    val_ds = Subset(full_val_dataset, val_idx)
    test_ds = Subset(full_val_dataset, test_idx)

    print(f"Stats: {len(train_ds)} Train | {len(val_ds)} Val | {len(test_ds)} Test")
    print(f"Classes: {class_names}")

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        ),
    }

    return dataloaders, len(class_names)


def build_model(num_classes):
    print("Building MobileNetV3-Large...")
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    # Unfreeze all for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    num_ftrs = model.classifier[-1].in_features
    # Moderate dropout for SGD
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(num_ftrs, num_classes)
    )

    return model.to(CONFIG["DEVICE"])


def train_loop(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
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
                inputs = inputs.to(CONFIG["DEVICE"])
                labels = labels.to(CONFIG["DEVICE"])

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                # Step scheduler based on accuracy plateau usually better for SGD
                if scheduler:
                    scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


def save_model(model):
    save_path = "breed_classifier_large.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")

    print("Converting to TorchScript for mobile...")
    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(CONFIG["DEVICE"])
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("breed_classifier_large_mobile.pt")
    print("Saved mobile model to breed_classifier_large_mobile.pt")


def main():
    print(f"Using device: {CONFIG['DEVICE']}")

    train_tf, val_tf = get_transforms()
    dataloaders, num_classes = get_dataloaders(CONFIG["DATA_DIR"], train_tf, val_tf)
    model = build_model(num_classes)

    # SWITCH TO SGD (Better generalization for Vision)
    optimizer = optim.SGD(
        [
            {"params": model.features.parameters(), "lr": 0.001},  # Body LR
            {"params": model.classifier.parameters(), "lr": 0.01},  # Head LR
        ],
        momentum=0.9,
        weight_decay=1e-4,
    )

    # Smooth label loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Patience 5 allows SGD to settle before cutting LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    print("\n--- STARTING TRAINING (SGD) ---")
    model = train_loop(
        model, dataloaders, criterion, optimizer, scheduler, CONFIG["EPOCHS"]
    )
    save_model(model)


if __name__ == "__main__":
    main()

import copy
import os
import time
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

warnings.filterwarnings("ignore", category=UserWarning)

CONFIG = {
    "DATA_DIR": "dataset/Indian_bovine_breeds",
    "BATCH_SIZE": 128,
    "WARMUP_EPOCHS": 5,
    "MAIN_EPOCHS": 45,
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
            transforms.RandomResizedCrop(CONFIG["IMG_SIZE"], scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
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

    print("Calculating class weights for Sampler...")
    train_targets = np.array(targets)[train_idx]
    class_counts = Counter(train_targets)

    weights = []
    for t in train_targets:
        weights.append(1.0 / class_counts[t])

    sample_weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    print(f"Stats: {len(train_ds)} Train | {len(val_ds)} Val | {len(test_ds)} Test")

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=CONFIG["BATCH_SIZE"],
            sampler=sampler,
            shuffle=False,
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

    return dataloaders, class_names


def build_model(num_classes):
    print("Building MobileNetV3-Small...")
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.2), nn.Linear(num_ftrs, num_classes)
    )

    return model.to(CONFIG["DEVICE"])


def save_confusion_matrix(labels, preds, class_names, epoch, phase_name):
    try:
        cm = confusion_matrix(labels, preds)
        fig_size = max(10, len(class_names) // 2)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(
            cmap="Blues",
            ax=ax,
            xticks_rotation="vertical",
            values_format="d",
            colorbar=False,
        )
        plt.title(f"Confusion Matrix - {phase_name} Epoch {epoch + 1}")
        plt.tight_layout()
        filename = f"confusion_matrix_epoch_{epoch + 1}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Failed to save confusion matrix: {e}")


def train_loop(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    class_names,
    phase_name="Training",
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

    for epoch in range(num_epochs):
        print(f"{phase_name} Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        val_preds = []
        val_labels = []

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

                if phase == "val" and (epoch + 1) % 10 == 0:
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                if scheduler:
                    scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 and val_preds:
            save_confusion_matrix(val_labels, val_preds, class_names, epoch, phase_name)

        print()

    time_elapsed = time.time() - since
    print(
        f"{phase_name} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Best Val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


def save_model(model):
    save_path = "breed_classifier_small.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")

    print("Converting to TorchScript for mobile...")
    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(CONFIG["DEVICE"])
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("breed_classifier_small_mobile.pt")
    print("Saved mobile model to breed_classifier_small_mobile.pt")


def main():
    print(f"Using device: {CONFIG['DEVICE']}")

    train_tf, val_tf = get_transforms()
    dataloaders, class_names = get_dataloaders(CONFIG["DATA_DIR"], train_tf, val_tf)
    model = build_model(len(class_names))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n--- PHASE 1: WARMUP ---")
    optimizer_warmup = optim.AdamW(
        model.classifier.parameters(), lr=0.001, weight_decay=0.01
    )

    model = train_loop(
        model,
        dataloaders,
        criterion,
        optimizer_warmup,
        scheduler=None,
        num_epochs=CONFIG["WARMUP_EPOCHS"],
        class_names=class_names,
        phase_name="Warmup",
    )

    print("\n--- PHASE 2: MAIN TRAINING ---")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_main = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 5e-5},
            {"params": model.classifier.parameters(), "lr": 5e-4},
        ],
        weight_decay=0.02,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_main, mode="min", factor=0.5, patience=3
    )

    model = train_loop(
        model,
        dataloaders,
        criterion,
        optimizer_main,
        scheduler,
        num_epochs=CONFIG["MAIN_EPOCHS"],
        class_names=class_names,
        phase_name="Main",
    )

    save_model(model)


if __name__ == "__main__":
    main()

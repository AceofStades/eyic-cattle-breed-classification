import copy
import os
import time
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Large_Weights

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
CONFIG = {
    "DATA_DIR": "dataset/Indian_bovine_breeds_balanced",
    "BATCH_SIZE": 64,  # 64 is safe for 16GB VRAM with Unfrozen MobileNet
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
    """Defines the augmentation pipeline."""
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

    # 1. Load Dataset TWICE (Augmented for Train, Clean for Val/Test)
    full_train_dataset = datasets.ImageFolder(
        data_dir, transform=train_tf, loader=safe_pil_loader
    )
    full_val_dataset = datasets.ImageFolder(
        data_dir, transform=val_tf, loader=safe_pil_loader
    )

    targets = full_train_dataset.targets
    class_names = full_train_dataset.classes

    # 2. Stratified Split (80% Train, 10% Val, 10% Test)
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

    # 3. Create WeightedRandomSampler for Class Imbalance
    print("Calculating class weights for Sampler...")

    # Get the targets for ONLY the training indices
    train_targets = np.array(targets)[train_idx]
    class_counts = Counter(train_targets)

    # Weight = 1 / class_count
    weights = []
    for t in train_targets:
        weights.append(1.0 / class_counts[t])

    sample_weights = torch.DoubleTensor(weights)

    # Sampler picks samples based on weight. Rare classes get picked more often.
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    print(f"Stats: {len(train_ds)} Train | {len(val_ds)} Val | {len(test_ds)} Test")

    # 4. Create Loaders
    # NOTE: shuffle=False is MANDATORY when using a sampler
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

    return dataloaders, len(class_names)


def build_model(num_classes):
    print("Building MobileNetV3-Large...")
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    # Initial state: Freeze Backbone
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[-1].in_features
    # Moderate dropout (0.3)
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(num_ftrs, num_classes)
    )

    return model.to(CONFIG["DEVICE"])


def train_loop(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    phase_name="Training",
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

    for epoch in range(num_epochs):
        print(f"{phase_name} Epoch {epoch + 1}/{num_epochs}")
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
                if scheduler:
                    scheduler.step(epoch_loss)  # ReduceLROnPlateau step

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        f"{phase_name} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Best Val Acc: {best_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_model(model):
    save_path = "breed_classifier_large.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved weights to {save_path}")

    print("Converting to TorchScript for mobile...")
    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(CONFIG["DEVICE"])
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("breed_classifier_large_mobile.pt")
    print("✅ Saved mobile model to breed_classifier_large_mobile.pt")


def main():
    print(f"Using device: {CONFIG['DEVICE']}")

    train_tf, val_tf = get_transforms()
    dataloaders, num_classes = get_dataloaders(CONFIG["DATA_DIR"], train_tf, val_tf)
    model = build_model(num_classes)

    # Use Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- PHASE 1: WARMUP (Head Only) ---
    print("\n--- PHASE 1: WARMUP (Frozen Backbone) ---")

    # Only optimize the head parameters
    optimizer_warmup = optim.AdamW(
        model.classifier.parameters(), lr=0.001, weight_decay=0.05
    )

    model = train_loop(
        model,
        dataloaders,
        criterion,
        optimizer_warmup,
        scheduler=None,
        num_epochs=CONFIG["WARMUP_EPOCHS"],
        phase_name="Warmup",
    )

    # --- PHASE 2: MAIN TRAINING (Unfrozen) ---
    print("\n--- PHASE 2: MAIN TRAINING (Unfrozen) ---")

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    # Differential Learning Rates: Slow for body, Fast for head
    optimizer_main = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 5e-5},
            {"params": model.classifier.parameters(), "lr": 5e-4},
        ],
        weight_decay=0.02,
    )  # Higher decay to prevent overfitting on rare classes

    # Scheduler for fine-tuning
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
        phase_name="Main",
    )

    # Save Final Model
    save_model(model)


if __name__ == "__main__":
    main()

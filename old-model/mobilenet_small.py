import copy
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

# --- CONFIGURATION ---
TRAIN_DIR = "dataset/final-new/train"
VAL_DIR = "dataset/final-new/valid"
BATCH_SIZE = 128
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MAGENTA COLOR CODE
COLOR = "\033[95m"
RESET = "\033[0m"

logging.basicConfig(
    filename="training_log.txt", level=logging.INFO, format="%(message)s"
)


def log(msg):
    print(f"{COLOR}[SMALL] {msg}{RESET}")
    logging.info(f"[SMALL] {msg}")


def train_loop(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

    for epoch in range(num_epochs):
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                pass
            else:
                if scheduler:
                    scheduler.step(epoch_loss)
                log(
                    f"Epoch {epoch + 1}/{num_epochs} - {phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
                )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    log(f"Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def train_small():
    log("STARTING TRAINING (V8: High Decay + AutoAug)")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1),
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

    if not os.path.exists(TRAIN_DIR):
        log("Dataset not found!")
        return

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms["val"])
    num_classes = len(train_dataset.classes)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        ),
    }

    log("Loading MobileNetV3-Small...")
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(num_ftrs, num_classes)
    )

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    log("STAGE 1: WARM UP (5 Epochs)")
    optimizer_head = optim.AdamW(
        model.classifier.parameters(), lr=0.001, weight_decay=0.01
    )
    model = train_loop(model, dataloaders, criterion, optimizer_head, num_epochs=5)

    log("STAGE 2: FULL FINE TUNING (50 Epochs)")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_fine = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 2e-5},
            {"params": model.classifier.parameters(), "lr": 5e-4},
        ],
        weight_decay=0.01,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_fine, mode="min", factor=0.1, patience=5
    )

    model = train_loop(
        model,
        dataloaders,
        criterion,
        optimizer_fine,
        scheduler=scheduler,
        num_epochs=50,
    )

    save_path = "breed_classifier_mobilenetv3_small.pth"
    torch.save(model.state_dict(), save_path)
    log(f"Saved to {save_path}")

    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(DEVICE)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("breed_classifier_mobile_small.pt")
    log("Mobile model saved.")


if __name__ == "__main__":
    train_small()

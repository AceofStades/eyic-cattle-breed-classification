import argparse
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
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Large_Weights

# Optional: timm for more/backbone choices
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

# Optional: Albumentations for on-the-fly augmentation
try:
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2
    ALB_AVAILABLE = True
except Exception:
    ALB_AVAILABLE = False

# AMP utilities
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=UserWarning)

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


class AlbumentationsTransform:
    """Wrapper to use albumentations with torchvision ImageFolder (returns torch.Tensor)."""

    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        # img is PIL Image
        arr = np.array(img)  # HWC RGB
        out = self.aug(image=arr)
        return out["image"]


def get_transforms(img_size, use_alb=False):
    """Defines the augmentation pipeline. Supports albumentations when requested."""
    if use_alb and ALB_AVAILABLE:
        print("Using Albumentations on-the-fly transforms")
        train_aug = A.Compose(
            [
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Cutout(num_holes=1, max_h_size=int(img_size * 0.2), max_w_size=int(img_size * 0.2), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            p=1.0,
        )

        val_aug = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], p=1.0)

        return AlbumentationsTransform(train_aug), AlbumentationsTransform(val_aug)

    # Fallback to torchvision transforms
    train_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
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

    print("Calculating class weights for Sampler...")

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


def build_model(num_classes, backbone="mobile", device=CONFIG["DEVICE"]):
    """Builds the model. backbone: 'mobile' or 'efficientnet_b3' (accuracy)"""
    print(f"Building model backbone: {backbone}")

    if backbone == "mobile":
        # Prefer timm variant if available for consistency
        if TIMM_AVAILABLE:
            model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=num_classes)
        else:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            model = models.mobilenet_v3_large(weights=weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, num_classes))
    elif backbone == "efficientnet_b3":
        if TIMM_AVAILABLE:
            model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes)
        else:
            # Fallback to torchvision if present (may not be pretrained)
            try:
                model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            except Exception:
                raise RuntimeError("EfficientNet_B3 not available and timm not installed. Install timm for best models.")
    else:
        raise ValueError("Unknown backbone: choose 'mobile' or 'efficientnet_b3'")

    # Initial state: Freeze Backbone
    for param in model.parameters():
        param.requires_grad = False

    return model.to(device)


def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_loop(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    phase_name="Training",
    use_amp=False,
    mixup_alpha=0.0,
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

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

            # For per-class metrics on validation
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(CONFIG["DEVICE"])
                labels = labels.to(CONFIG["DEVICE"])

                optimizer.zero_grad()

                if phase == "train" and mixup_alpha > 0:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)
                else:
                    targets_a, targets_b, lam = labels, None, 1.0

                with torch.set_grad_enabled(phase == "train"):
                    with autocast(enabled=(use_amp and torch.cuda.is_available())):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        if targets_b is not None:
                            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                        else:
                            loss = criterion(outputs, labels)

                    if phase == "train":
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        # Step OneCycleLR per batch when used
                        if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                            scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                # Scheduler that expects epoch metrics
                if scheduler and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                    scheduler.step(epoch_loss)

                # Per-class F1
                try:
                    f1_per_class = f1_score(all_labels, all_preds, average=None)
                    print("Val per-class F1:", np.round(f1_per_class, 4))
                except Exception:
                    pass

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


def save_model(model, name="breed_classifier"):
    save_path = f"{name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved weights to {save_path}")

    # Try to export TorchScript (best-effort). Catch failures so training isn't interrupted.
    try:
        print("Converting to TorchScript for mobile...")
        model.eval()
        size = CONFIG.get("IMG_SIZE", (224, 224))[0]
        example_input = torch.rand(1, 3, size, size).to(CONFIG["DEVICE"])
        traced_script_module = torch.jit.trace(model, example_input)
        mobile_path = f"{name}_mobile.pt"
        traced_script_module.save(mobile_path)
        print(f"✅ Saved mobile model to {mobile_path}")
    except Exception as e:
        print(f"⚠️ TorchScript conversion failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train breed classifier")
    parser.add_argument("--backbone", choices=["mobile", "efficientnet_b3"], default="mobile", help="Backbone: mobile (fast) or efficientnet_b3 (higher accuracy)")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--use-alb", action="store_true", help="Use albumentations for on-the-fly augmentation (requires albumentations)")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0 to disable)")
    parser.add_argument("--batch-size", type=int, default=CONFIG["BATCH_SIZE"], help="Batch size")
    parser.add_argument("--main-epochs", type=int, default=CONFIG["MAIN_EPOCHS"], help="Main training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=CONFIG["WARMUP_EPOCHS"], help="Warmup epochs")
    parser.add_argument("--max-lr", type=float, default=5e-4, help="Max LR for OneCycleLR")
    args = parser.parse_args()

    print(f"Using device: {CONFIG['DEVICE']}")

    # Update CONFIG if CLI overrides
    CONFIG["BATCH_SIZE"] = args.batch_size
    CONFIG["MAIN_EPOCHS"] = args.main_epochs
    CONFIG["WARMUP_EPOCHS"] = args.warmup_epochs

    # If user requested alb transforms but it's missing, warn and fall back
    if args.use_alb and not ALB_AVAILABLE:
        print("Warning: --use-alb requested but albumentations is not installed. Falling back to torchvision transforms.")

    train_tf, val_tf = get_transforms(CONFIG["IMG_SIZE"][0], use_alb=args.use_alb)
    dataloaders, num_classes = get_dataloaders(CONFIG["DATA_DIR"], train_tf, val_tf)

    model = build_model(num_classes, backbone=args.backbone)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- PHASE 1: WARMUP (Head Only) ---
    print("\n--- PHASE 1: WARMUP (Frozen Backbone) ---")

    # Only optimize the head parameters
    # Works for most timm/torchvision models: find classifier params
    head_params = [p for n, p in model.named_parameters() if "classifier" in n or "head" in n or "linear" in n]
    if len(head_params) == 0:
        head_params = model.parameters()

    optimizer_warmup = optim.AdamW(head_params, lr=1e-3, weight_decay=0.05)

    model = train_loop(
        model,
        dataloaders,
        criterion,
        optimizer_warmup,
        scheduler=None,
        num_epochs=CONFIG["WARMUP_EPOCHS"],
        phase_name="Warmup",
        use_amp=args.use_amp,
        mixup_alpha=0.0,
    )

    # --- PHASE 2: MAIN TRAINING (Unfrozen) ---
    print("\n--- PHASE 2: MAIN TRAINING (Unfrozen) ---")

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    # Differential Learning Rates: Slow for body, Fast for head (best-effort)
    body_params = [p for n, p in model.named_parameters() if not ("classifier" in n or "head" in n or "linear" in n)]
    head_params = [p for n, p in model.named_parameters() if ("classifier" in n or "head" in n or "linear" in n)]

    optimizer_main = optim.AdamW(
        [
            {"params": body_params, "lr": 5e-5},
            {"params": head_params, "lr": args.max_lr},
        ],
        weight_decay=0.02,
    )

    # OneCycleLR for better convergence
    steps_per_epoch = len(dataloaders["train"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer_main,
        max_lr=args.max_lr,
        epochs=CONFIG["MAIN_EPOCHS"],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    model = train_loop(
        model,
        dataloaders,
        criterion,
        optimizer_main,
        scheduler,
        num_epochs=CONFIG["MAIN_EPOCHS"],
        phase_name="Main",
        use_amp=args.use_amp,
        mixup_alpha=args.mixup,
    )

    # Save model name according to backbone
    save_model(model, name=f"breed_classifier_{args.backbone}.pth")


if __name__ == "__main__":
    main()

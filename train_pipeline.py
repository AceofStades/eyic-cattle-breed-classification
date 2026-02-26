import os
import sys
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from ultralytics import YOLO
import onnx
import onnxruntime as ort


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / "dataset" / "Indian_bovine_breeds"
CSV_PATH = BASE_DIR / "bovine_breeds_metadata.csv"
AUG_DIR = SCRIPT_DIR / "dataset" / "augmented"
AUG_CSV_PATH = AUG_DIR / "augmented_metadata.csv"
LABEL_MAP_PATH = SCRIPT_DIR / "breed_labels.json"
BEST_MODEL_PATH = SCRIPT_DIR / "best_breed_classifier.pth"
LAST_CKPT_PATH = SCRIPT_DIR / "last_checkpoint.pth"
ONNX_MODEL_PATH = SCRIPT_DIR / "breed_classifier.onnx"
TORCHSCRIPT_MODEL_PATH = SCRIPT_DIR / "breed_classifier.ptl"

MODEL_NAME = "efficientnet_b2"
IMG_SIZE = 260
RESIZE_SIZE = 292
BATCH_SIZE = 64
FREEZE_EPOCHS = 5
FINETUNE_EPOCHS = 50
EPOCHS = FREEZE_EPOCHS + FINETUNE_EPOCHS
FREEZE_LR = 3e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.4
MIXUP_ALPHA = 0.3
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
NUM_WORKERS = 4
AUGMENT_COPIES = 10
HARD_NEG_EXTRA_COPIES = 5
VAL_SIZE = 0.2
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
SCALER = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

HARD_NEGATIVE_PAIRS = [
    ("Red_Sindhi", "Sahiwal"),
    ("Hariana", "Tharparkar"),
    ("Mehsana", "Murrah"),
    ("Gir", "Sahiwal"),
    ("Bhadawari", "Toda"),
    ("Murrah", "Nagpuri"),
    ("Banni", "Murrah"),
    ("Nimari", "Rathi"),
    ("Nili_Ravi", "Murrah"),
    ("Amritmahal", "Khillari"),
    ("Hallikar", "Khillari"),
    ("Jersey", "Holstein_Friesian"),
    ("Red_Dane", "Jersey"),
    ("Ayrshire", "Holstein_Friesian"),
    ("Kasargod", "Malnad_gidda"),
    ("Deoni", "Ongole"),
    ("Kenkatha", "Kherigarh"),
    ("Kenkatha", "Hariana"),
    ("Kangayam", "Pulikulam"),
    ("Pulikulam", "Umblachery"),
    ("Kasargod", "Vechur"),
    ("Malnad_gidda", "Vechur"),
    ("Gir", "Red_Sindhi"),
    ("Krishna_Valley", "Hallikar"),
    ("Nagori", "Kenkatha"),
    ("Surti", "Mehsana"),
]

HARD_NEGATIVE_BREEDS = set()
for a, b in HARD_NEGATIVE_PAIRS:
    HARD_NEGATIVE_BREEDS.add(a)
    HARD_NEGATIVE_BREEDS.add(b)


def load_metadata():
    df = pd.read_csv(CSV_PATH)
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    df["full_path"] = df["path"].apply(lambda p: str(BASE_DIR / p))
    valid_mask = df["full_path"].apply(lambda p: os.path.isfile(p))
    df = df[valid_mask].reset_index(drop=True)
    return df


def encode_labels(df):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["breed"])
    label_map = {int(idx): name for idx, name in enumerate(le.classes_)}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    return df, le, label_map


def split_data(df):
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


def get_augmentation_transform():
    return A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 20), hole_width_range=(8, 20), p=0.3),
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.4),
    ])


def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def generate_augmented_dataset(train_df, force=False):
    if AUG_DIR.exists() and any(AUG_DIR.iterdir()) and not force:
        print(f"aug dir exists at {AUG_DIR} -> skip | use --force-augment to regen")
        aug_df = pd.read_csv(AUG_CSV_PATH)
        return aug_df

    if AUG_DIR.exists():
        shutil.rmtree(AUG_DIR)
    AUG_DIR.mkdir(parents=True, exist_ok=True)

    aug_transform = get_augmentation_transform()
    records = []

    print(f"gen aug dataset | {AUGMENT_COPIES}x copies/img")
    for idx in tqdm(range(len(train_df)), desc="Augmenting"):
        row = train_df.iloc[idx]
        src_path = row["full_path"]
        breed = row["breed"]
        label = row["label"]
        original_ext = Path(src_path).suffix
        original_stem = Path(src_path).stem

        breed_dir = AUG_DIR / breed
        breed_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(src_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_filename = f"{original_stem}{original_ext}"
        orig_save_path = breed_dir / orig_filename
        cv2.imwrite(str(orig_save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        records.append({
            "image_id": orig_filename,
            "breed": breed,
            "path": f"{breed}/{orig_filename}",
            "label": label,
        })

        n_copies = AUGMENT_COPIES
        if breed in HARD_NEGATIVE_BREEDS:
            n_copies = AUGMENT_COPIES + HARD_NEG_EXTRA_COPIES

        for i in range(n_copies):
            augmented = aug_transform(image=img)
            aug_img = augmented["image"]
            aug_filename = f"{original_stem}_aug{i}{original_ext}"
            aug_save_path = breed_dir / aug_filename
            cv2.imwrite(str(aug_save_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            records.append({
                "image_id": aug_filename,
                "breed": breed,
                "path": f"{breed}/{aug_filename}",
                "label": label,
            })

    aug_df = pd.DataFrame(records)
    aug_df.to_csv(AUG_CSV_PATH, index=False)
    print(f"aug saved | {len(aug_df)} imgs -> {AUG_DIR}")
    return aug_df


class AugmentedBreedDataset(Dataset):
    def __init__(self, dataframe, base_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(self.base_dir / row["path"])
        label = int(row["label"])

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        return img, label


class BreedDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["full_path"]
        label = int(row["label"])

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        return img, label


def build_model(num_classes):
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes, drop_rate=DROPOUT_RATE)
    model = model.to(DEVICE)
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  backbone frozen | trainable {trainable:,}/{total:,}")


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  backbone unfrozen | trainable {trainable:,}")


def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    _, _, h, w = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, dataloader, criterion, optimizer, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            if use_mixup and np.random.rand() < MIXUP_PROB:
                if np.random.rand() < 0.5:
                    images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                else:
                    images, y_a, y_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

        SCALER.scale(loss).backward()
        SCALER.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        SCALER.step(optimizer)
        SCALER.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="val", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_classes, resume_ckpt=None):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best_val_acc = 0.0
    patience_counter = 0
    start_epoch = 1
    skip_phase1 = False

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        best_val_acc = resume_ckpt.get("best_val_acc", resume_ckpt.get("val_acc", 0.0))
        patience_counter = resume_ckpt.get("patience_counter", 0)
        start_epoch = resume_ckpt["epoch"] + 1
        print(f"  resumed from ep {start_epoch - 1} | best val acc {best_val_acc:.4f} | patience {patience_counter}")
        if start_epoch > FREEZE_EPOCHS:
            skip_phase1 = True

    if not skip_phase1:
        print(f"\n--- phase 1: frozen backbone ({FREEZE_EPOCHS} ep) ---")
        freeze_backbone(model)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=FREEZE_LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)

        if resume_ckpt is not None and not skip_phase1 and "optimizer_state_dict" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            except Exception:
                pass
            if "scheduler_state_dict" in resume_ckpt:
                try:
                    scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
                except Exception:
                    pass

        phase1_start = max(start_epoch, 1)
        for epoch in range(phase1_start, FREEZE_EPOCHS + 1):
            print(f"\nep {epoch}/{EPOCHS} [freeze] | lr {optimizer.param_groups[0]['lr']:.6f}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, use_mixup=False)
            val_loss, val_acc = validate(model, val_loader, criterion)
            scheduler.step()
            print(f"  trn loss {train_loss:.4f} acc {train_acc:.4f}")
            print(f"  val loss {val_loss:.4f} acc {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "patience_counter": 0,
                    "phase": "freeze",
                    "num_classes": num_classes,
                }, BEST_MODEL_PATH)
                print(f"  best saved | val acc {best_val_acc:.4f}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "patience_counter": 0,
                "phase": "freeze",
                "num_classes": num_classes,
            }, LAST_CKPT_PATH)

    print(f"\n--- phase 2: full finetune ({FINETUNE_EPOCHS} ep) ---")
    unfreeze_backbone(model)
    optimizer = optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": FINETUNE_LR * 0.1},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": FINETUNE_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    if resume_ckpt is not None and skip_phase1 and "optimizer_state_dict" in resume_ckpt:
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        except Exception:
            pass
        if "scheduler_state_dict" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            except Exception:
                pass

    phase2_start = max(start_epoch, FREEZE_EPOCHS + 1)
    for epoch in range(phase2_start, EPOCHS + 1):
        print(f"\nep {epoch}/{EPOCHS} [finetune] | lr bb {optimizer.param_groups[0]['lr']:.6f} head {optimizer.param_groups[1]['lr']:.6f}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, use_mixup=True)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        print(f"  trn loss {train_loss:.4f} acc {train_acc:.4f}")
        print(f"  val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "patience_counter": 0,
                "phase": "finetune",
                "num_classes": num_classes,
            }, BEST_MODEL_PATH)
            print(f"  best saved | val acc {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  no improv | patience {patience_counter}/{EARLY_STOP_PATIENCE}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  early stop triggered")
                break
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "patience_counter": patience_counter,
            "phase": "finetune",
            "num_classes": num_classes,
        }, LAST_CKPT_PATH)

    print(f"\ntraining done | best val acc {best_val_acc:.4f}")
    return model


HARD_NEG_EPOCHS = 10
HARD_NEG_LR = 1e-5
HARD_NEG_WEIGHT = 1.5


def train_hard_negatives(model, aug_df, val_loader, num_classes, label_map, hard_neg_epochs=None):
    if hard_neg_epochs is None:
        hard_neg_epochs = HARD_NEG_EPOCHS

    print(f"\n--- phase 3: hard-negative mining ({hard_neg_epochs} ep) ---")

    if BEST_MODEL_PATH.exists():
        ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("best_val_acc", ckpt.get("val_acc", 0.0))
        print(f"  loaded best ckpt | val acc {best_val_acc:.4f}")
    else:
        best_val_acc = 0.0

    name_to_idx = {v: int(k) for k, v in label_map.items()}
    hard_breed_indices = set()
    for a, b in HARD_NEGATIVE_PAIRS:
        if a in name_to_idx:
            hard_breed_indices.add(name_to_idx[a])
        if b in name_to_idx:
            hard_breed_indices.add(name_to_idx[b])

    class_weights = torch.ones(num_classes, device=DEVICE)
    for idx in hard_breed_indices:
        class_weights[idx] = HARD_NEG_WEIGHT
    print(f"  {len(hard_breed_indices)} hard-neg breeds weighted {HARD_NEG_WEIGHT}x | {num_classes - len(hard_breed_indices)} normal")

    from torch.utils.data import WeightedRandomSampler
    sample_weights = []
    for _, row in aug_df.iterrows():
        lbl = int(row["label"])
        sample_weights.append(HARD_NEG_WEIGHT if lbl in hard_breed_indices else 1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(aug_df), replacement=True)

    hn_dataset = AugmentedBreedDataset(aug_df, AUG_DIR, transform=get_train_transform())
    hn_loader = DataLoader(
        hn_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    print(f"  full aug ds {len(aug_df)} samples | weighted sampler active")

    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(k in name for k in ["blocks.5", "blocks.6", "conv_head", "bn2", "classifier"]):
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {trainable}/{total} params ({trainable*100//total}%)")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=HARD_NEG_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hard_neg_epochs)

    patience_counter = 0
    for epoch in range(1, hard_neg_epochs + 1):
        print(f"\nep {epoch}/{hard_neg_epochs} [hard-neg] | lr {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(model, hn_loader, criterion, optimizer, use_mixup=True)
        val_loss, val_acc = validate(model, val_loader, nn.CrossEntropyLoss(label_smoothing=0.15))
        scheduler.step()

        print(f"  hn trn loss {train_loss:.4f} acc {train_acc:.4f}")
        print(f"  full val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "phase": "hard_neg",
                "num_classes": num_classes,
            }, BEST_MODEL_PATH)
            print(f"  best saved | val acc {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  no improv | patience {patience_counter}/5")
            if patience_counter >= 5:
                print(f"  early stop hard-neg")
                break

    print(f"\nhard-neg training done | best val acc {best_val_acc:.4f}")
    return model


def export_to_torchscript(model, num_classes):
    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
    model_cpu = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    model_cpu.load_state_dict(checkpoint["model_state_dict"])
    model_cpu.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    traced_model = torch.jit.trace(model_cpu, dummy_input)
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized = optimize_for_mobile(traced_model)
    optimized._save_for_lite_interpreter(str(TORCHSCRIPT_MODEL_PATH))

    with torch.no_grad():
        pth_out = model_cpu(dummy_input)
        ts_out = traced_model(dummy_input)
        diff = (pth_out - ts_out).abs().max().item()

    import os
    file_mb = os.path.getsize(TORCHSCRIPT_MODEL_PATH) / (1024 * 1024)
    print(f"torchscript exported ok -> {TORCHSCRIPT_MODEL_PATH}")
    print(f"  file size       {file_mb:.2f} mb")
    print(f"  max output diff {diff:.10f}")
    print(f"  in [1 3 {IMG_SIZE} {IMG_SIZE}] | out [1 {num_classes}]")


def export_to_onnx(model, num_classes):
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(str(ONNX_MODEL_PATH))
    onnx.checker.check_model(onnx_model)
    print(f"onnx exported ok -> {ONNX_MODEL_PATH}")
    print(f"  in [batch 3 {IMG_SIZE} {IMG_SIZE}] | out [batch {num_classes}]")


def detect_and_crop_cows(image_path, confidence_threshold=0.5):
    yolo_model = YOLO("yolov8n.pt")
    results = yolo_model(image_path, verbose=False)
    crops = []

    img = Image.open(image_path).convert("RGB")

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 19 and conf >= confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cropped = img.crop((x1, y1, x2, y2))
                crops.append((cropped, conf))

    return crops


def predict_breed(image_path, model, label_map, transform):
    crops = detect_and_crop_cows(image_path)

    if not crops:
        img = Image.open(image_path).convert("RGB")
        crops = [(img, 1.0)]

    predictions = []
    model.eval()

    for crop_img, det_conf in crops:
        img_np = np.array(crop_img)
        transformed = transform(image=img_np)
        img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = probabilities.max(1)

        breed_name = label_map[str(predicted_idx.item())]
        predictions.append({
            "breed": breed_name,
            "confidence": confidence.item(),
            "detection_confidence": det_conf,
        })

    return predictions


def compute_val_metrics(model, val_loader, label_map):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="eval metrics", leave=False):
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    target_names = [label_map[str(i)] for i in range(len(label_map))]
    cls_report = classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True
    )

    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "prec_macro": prec_macro,
        "rec_macro": rec_macro,
        "cls_report": cls_report,
        "preds": all_preds,
        "labels": all_labels,
    }


def print_model_diagnostics(model, num_classes, label_map, metrics=None, tag="pytorch"):
    print(f"\n{'='*50}")
    print(f"model diagnostics [{tag}]")
    print(f"{'='*50}")

    if tag == "pytorch":
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        buffer_size_mb = sum(b.nelement() * b.element_size() for b in model.buffers()) / (1024 * 1024)

        print(f"  arch            {MODEL_NAME}")
        print(f"  backend         timm")
        print(f"  device          {DEVICE}")
        print(f"  num classes     {num_classes}")
        print(f"  input shape     [1 3 {IMG_SIZE} {IMG_SIZE}]")
        print(f"  total params    {total_params:,}")
        print(f"  trainable       {trainable_params:,}")
        print(f"  frozen          {frozen_params:,}")
        print(f"  param size      {model_size_mb:.2f} mb")
        print(f"  buffer size     {buffer_size_mb:.2f} mb")
        print(f"  total size      {model_size_mb + buffer_size_mb:.2f} mb")

        if hasattr(model, "classifier"):
            cls_layer = model.classifier
            print(f"  head layer      classifier")
            print(f"  head type       {cls_layer.__class__.__name__}")
            if hasattr(cls_layer, "in_features"):
                print(f"  head in feat    {cls_layer.in_features}")
            if hasattr(cls_layer, "out_features"):
                print(f"  head out feat   {cls_layer.out_features}")

        layer_counts = {}
        for name, module in model.named_modules():
            mtype = module.__class__.__name__
            layer_counts[mtype] = layer_counts.get(mtype, 0) + 1
        print(f"  layer breakdown")
        for ltype, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {ltype:<25} {count}")

    elif tag == "onnx":
        onnx_model = onnx.load(str(ONNX_MODEL_PATH))
        file_size_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
        print(f"  file            {ONNX_MODEL_PATH}")
        print(f"  file size       {file_size_mb:.2f} mb")
        print(f"  opset ver       {onnx_model.opset_import[0].version}")
        print(f"  ir ver          {onnx_model.ir_version}")
        print(f"  graph name      {onnx_model.graph.name}")
        print(f"  num nodes       {len(onnx_model.graph.node)}")
        print(f"  num initializers {len(onnx_model.graph.initializer)}")

        for inp in onnx_model.graph.input:
            shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
            print(f"  input           {inp.name} shape {shape}")
        for out in onnx_model.graph.output:
            shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
            print(f"  output          {out.name} shape {shape}")

        op_counts = {}
        for node in onnx_model.graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        print(f"  op breakdown")
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {op:<25} {count}")

        sess = ort.InferenceSession(str(ONNX_MODEL_PATH))
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        out = sess.run(None, {"input": dummy})
        print(f"  ort test run    ok | out shape {out[0].shape}")

    if metrics:
        print(f"  --- val metrics ---")
        print(f"  accuracy        {metrics['acc']:.4f}")
        print(f"  f1 macro        {metrics['f1_macro']:.4f}")
        print(f"  f1 weighted     {metrics['f1_weighted']:.4f}")
        print(f"  precision macro {metrics['prec_macro']:.4f}")
        print(f"  recall macro    {metrics['rec_macro']:.4f}")
        print(f"  --- per class f1 top 10 ---")
        cls_report = metrics["cls_report"]
        breed_f1s = []
        for name in cls_report:
            if name in ["accuracy", "macro avg", "weighted avg"]:
                continue
            breed_f1s.append((name, cls_report[name]["f1-score"], cls_report[name]["support"]))
        breed_f1s.sort(key=lambda x: -x[1])
        for name, f1, sup in breed_f1s[:10]:
            print(f"    {name:<22} f1 {f1:.4f} n={int(sup)}")
        print(f"  --- per class f1 bottom 5 ---")
        for name, f1, sup in breed_f1s[-5:]:
            print(f"    {name:<22} f1 {f1:.4f} n={int(sup)}")

    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-augment", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hard-neg-epochs", type=int, default=0)
    parser.add_argument("--hard-neg-only", action="store_true")
    parser.add_argument("--demo-image", type=str, default=None)
    args = parser.parse_args()

    print(f"bovine breed clf pipeline | dev {DEVICE}")

    print(f"\n[1/6] load meta")
    df = load_metadata()
    print(f"  imgs {len(df)} | breeds {df['breed'].nunique()}")

    print(f"\n[2/6] encode + split")
    df, le, label_map = encode_labels(df)
    num_classes = len(label_map)
    train_df, val_df = split_data(df)
    print(f"  trn {len(train_df)} | val {len(val_df)} | cls {num_classes}")
    print(f"  labels -> {LABEL_MAP_PATH}")

    print(f"\n[3/6] gen aug dataset")
    aug_df = generate_augmented_dataset(train_df, force=args.force_augment)
    print(f"  aug trn samples {len(aug_df)}")

    if not args.skip_training:
        print(f"\n[4/6] train effnetb0")

        train_dataset = AugmentedBreedDataset(aug_df, AUG_DIR, transform=get_train_transform())
        val_dataset = BreedDataset(val_df, transform=get_val_transform())

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        model = build_model(num_classes)

        resume_ckpt = None
        if args.resume and LAST_CKPT_PATH.exists():
            resume_ckpt = torch.load(LAST_CKPT_PATH, map_location=DEVICE, weights_only=True)
            print(f"  resuming from last ckpt ep {resume_ckpt['epoch']} phase {resume_ckpt.get('phase', 'unknown')}")
        elif args.resume and BEST_MODEL_PATH.exists():
            resume_ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
            print(f"  resuming from best ckpt ep {resume_ckpt['epoch']}")
        elif args.resume:
            print(f"  no ckpt found -> training from scratch")

        model = train_model(model, train_loader, val_loader, num_classes, resume_ckpt=resume_ckpt)

        if args.hard_neg_epochs > 0:
            model = train_hard_negatives(
                model, aug_df, val_loader, num_classes, label_map,
                hard_neg_epochs=args.hard_neg_epochs
            )
    else:
        print(f"\n[4/6] skip train")
        model = build_model(num_classes)
        val_dataset = BreedDataset(val_df, transform=get_val_transform())
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        if args.hard_neg_only and BEST_MODEL_PATH.exists():
            ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            hn_epochs = args.hard_neg_epochs if args.hard_neg_epochs > 0 else HARD_NEG_EPOCHS
            model = train_hard_negatives(
                model, aug_df, val_loader, num_classes, label_map,
                hard_neg_epochs=hn_epochs
            )

    print(f"\n[5/6] model diagnostics")
    if BEST_MODEL_PATH.exists():
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    with open(LABEL_MAP_PATH, "r") as f:
        label_map_loaded = json.load(f)

    metrics = compute_val_metrics(model, val_loader, label_map_loaded)
    print_model_diagnostics(model, num_classes, label_map_loaded, metrics=metrics, tag="pytorch")

    if not args.skip_export:
        print(f"\n[6/6] export models")
        if BEST_MODEL_PATH.exists():
            export_to_onnx(model, num_classes)
            print_model_diagnostics(model, num_classes, label_map_loaded, metrics=metrics, tag="onnx")
            print(f"\n  exporting torchscript for android...")
            export_to_torchscript(model, num_classes)
        else:
            print(f"  no ckpt found -> skip export")
    else:
        print(f"\n[6/6] skip export")

    if args.demo_image:
        print(f"\ndemo infer -> {args.demo_image}")
        if BEST_MODEL_PATH.exists():
            preds = predict_breed(args.demo_image, model, label_map_loaded, get_val_transform())
            for i, pred in enumerate(preds):
                print(f"  det {i+1} | breed {pred['breed']} | cls conf {pred['confidence']:.4f} | det conf {pred['detection_confidence']:.4f}")
        else:
            print(f"  no ckpt -> skip demo")

    print(f"\ndone")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models

# Import data tools from your main script so splits are identical
from model import (
    CONFIG,
    get_dataloaders,
    get_transforms,
    safe_pil_loader,
)


def load_saved_model(path, num_classes, device):
    """Loads the model architecture and weights."""
    print(f"Loading model from {path}...")
    model = models.mobilenet_v3_large(weights=None)  # No ImageNet weights needed

    # Recreate the head exactly as defined in training
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(num_ftrs, num_classes)
    )

    # Load weights
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    print(f"Using device: {CONFIG['DEVICE']}")

    # 1. Get Data
    train_tf, val_tf = get_transforms()
    # FIXED: Capture the integer 'num_classes' directly
    dataloaders, num_classes = get_dataloaders(CONFIG["DATA_DIR"], train_tf, val_tf)

    # FIXED: Extract the actual list of class names from the underlying dataset
    # dataloaders['test'].dataset is a 'Subset'
    # .dataset is the original 'ImageFolder'
    # .classes is the list of strings ['Gir', 'Sahiwal', etc.]
    class_names = dataloaders["test"].dataset.dataset.classes

    print(f"Successfully loaded {num_classes} classes.")

    # 2. Load Model
    model_path = "breed_classifier_large.pth"
    # FIXED: Pass 'num_classes' (int) directly, don't use len()
    model = load_saved_model(model_path, num_classes, CONFIG["DEVICE"])

    # 3. Run Evaluation
    print("\nRunning Evaluation on Test Set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(CONFIG["DEVICE"])
            labels = labels.to(CONFIG["DEVICE"])

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Accuracy
    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    accuracy = correct / len(all_labels)
    print(f"\nFinal Test Accuracy: {accuracy:.2%}")

    # 5. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    # Dynamic figure size based on number of classes
    fig_size = max(10, len(class_names) // 2)
    plt.figure(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Final Test Set Confusion Matrix (Acc: {accuracy:.2%})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("final_test_confusion_matrix.png")
    print("✅ Saved final_test_confusion_matrix.png")

    # 6. Detailed Report
    print("\nDetailed Classification Report:")
    # zero_division=0 handles cases where a class might not appear in the test set
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        )
    )


if __name__ == "__main__":
    main()

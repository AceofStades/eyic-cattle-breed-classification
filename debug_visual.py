import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# PATH TO YOUR TRAIN DATA
DATA_DIR = "dataset/final/train"

# The Exact Transforms we were using (to see the damage)
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # Note: We skip Normalize for visualization so colors look normal-ish
        transforms.RandomErasing(p=0.5),
    ]
)


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis("off")


if __name__ == "__main__":
    # Load Data
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Get a batch
    inputs, classes = next(iter(loader))

    # Plot grid
    fig = plt.figure(figsize=(15, 15))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        imshow(inputs[i], title=dataset.classes[classes[i]])

    plt.tight_layout()
    plt.savefig("debug_augmentation.png")
    print("Saved 'debug_augmentation.png'. Open it and look at the cows!")

import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

DATA_DIR = "dataset/final/train"
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = (224, 224)
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda")


def main():
    print("Using device: ", DEVICE)

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": (
            [
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),
            ]
        ),
    }


if __name__ == "__main__":
    main()

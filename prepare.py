import os

import cv2
import yaml
from tqdm import tqdm

BASE_INPUT_DIR = "dataset/new"
BASE_OUTPUT_DIR = "dataset/final-new"

TRAIN_IMAGES_DIR = os.path.join(BASE_INPUT_DIR, "train/images")
TRAIN_LABELS_DIR = os.path.join(BASE_INPUT_DIR, "train/labels")
VALID_IMAGES_DIR = os.path.join(BASE_INPUT_DIR, "valid/images")
VALID_LABELS_DIR = os.path.join(BASE_INPUT_DIR, "valid/labels")
TEST_IMAGES_DIR = os.path.join(BASE_INPUT_DIR, "test/images")
TEST_LABELS_DIR = os.path.join(BASE_INPUT_DIR, "test/labels")
DATA_YAML_PATH = os.path.join(BASE_INPUT_DIR, "data.yaml")

TRAIN_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "train")
VALID_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "valid")
TEST_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "test")


def process_folder(images_dir, labels_dir, output_dir, classes):
    if not os.path.exists(images_dir):
        print(f"Skipping {images_dir} (Not found)")
        return

    for breed in classes:
        folder_name = breed.strip().replace(" ", "_")
        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)

    image_files = [
        f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"Processing {len(image_files)} images from {images_dir}...")

    count_crops = 0

    for img_file in tqdm(image_files):
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        height, width, _ = img.shape

        label_file = img_file.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        crop_idx = 0
        for line in lines:
            try:
                parts = line.strip().split()
                class_id = int(parts[0])

                if class_id >= len(classes):
                    continue

                x_center, y_center, w, h = map(float, parts[1:5])

                x = int((x_center - w / 2) * width)
                y = int((y_center - h / 2) * height)
                w_px = int(w * width)
                h_px = int(h * height)

                x = max(0, x)
                y = max(0, y)
                w_px = min(w_px, width - x)
                h_px = min(h_px, height - y)

                crop = img[y : y + h_px, x : x + w_px]

                if crop.size == 0:
                    continue

                breed_name = classes[class_id].strip().replace(" ", "_")
                save_name = f"{img_file.rsplit('.', 1)[0]}_crop{crop_idx}.jpg"
                save_path = os.path.join(output_dir, breed_name, save_name)

                cv2.imwrite(save_path, crop)
                crop_idx += 1
                count_crops += 1

            except ValueError:
                continue

    print(f"-> Created {count_crops} crops in {output_dir}")


CLASSES = []
if os.path.exists(DATA_YAML_PATH):
    print(f"Reading {DATA_YAML_PATH}...")
    with open(DATA_YAML_PATH, "r") as f:
        try:
            data = yaml.safe_load(f)
            if "names" in data:
                names = data["names"]
                if isinstance(names, dict):
                    CLASSES = [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    CLASSES = names
        except Exception as e:
            print(f"Error parsing YAML: {e}")

if not CLASSES:
    print("ERROR: Could not find class names in data.yaml. Exiting.")
    exit()

print(f"✅ Found {len(CLASSES)} classes: {CLASSES}")

process_folder(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TRAIN_OUTPUT_DIR, CLASSES)
process_folder(VALID_IMAGES_DIR, VALID_LABELS_DIR, VALID_OUTPUT_DIR, CLASSES)
process_folder(TEST_IMAGES_DIR, TEST_LABELS_DIR, TEST_OUTPUT_DIR, CLASSES)

print("\nAll done! Ready for training.")

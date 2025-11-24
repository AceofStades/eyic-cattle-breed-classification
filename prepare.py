import os

import cv2
import yaml  # Make sure to pip install PyYAML
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_IMAGES_DIR = "dataset/mobilenetv3/train/images"
SOURCE_LABELS_DIR = "dataset/mobilenetv3/train/labels"
DATA_YAML_PATH = "dataset/mobilenetv3/data.yaml"

OUTPUT_DIR = "dataset/final/train"

# --- 1. GET CLASS NAMES ---
CLASSES = []

if os.path.exists(DATA_YAML_PATH):
    print(f"Reading {DATA_YAML_PATH}...")
    with open(DATA_YAML_PATH, "r") as f:
        try:
            data = yaml.safe_load(f)
            # Check for standard YOLO 'names' key
            if "names" in data:
                names = data["names"]
                # Handle if 'names' is a Dictionary {0: 'Gir', 1: 'Sahiwal'}
                if isinstance(names, dict):
                    # Sort by ID to ensure correct order
                    CLASSES = [names[i] for i in sorted(names.keys())]
                # Handle if 'names' is a List ['Gir', 'Sahiwal']
                elif isinstance(names, list):
                    CLASSES = names
            else:
                print("Error: 'names' key not found in yaml.")
        except Exception as e:
            print(f"Error parsing YAML: {e}")

# FALLBACK: If YAML fails, hardcode your classes here in the exact order IDs 0, 1, 2...
if not CLASSES:
    print("⚠️ YAML reading failed. Using manual classes.")
    # UNCOMMENT AND FILL THIS IF THE ABOVE FAILS
    # CLASSES = ["Gir", "Sahiwal", "Holstein", "Jersey", "Red Sindhi"]

if not CLASSES:
    print("ERROR: Could not determine class names. Script stopped.")
    exit()

print(f"✅ Found {len(CLASSES)} classes: {CLASSES}")

# --- 2. CREATE OUTPUT FOLDERS ---
for breed in CLASSES:
    # Clean breed name to be folder-safe (remove spaces/special chars)
    folder_name = breed.strip().replace(" ", "_")
    os.makedirs(os.path.join(OUTPUT_DIR, folder_name), exist_ok=True)

# --- 3. PROCESS IMAGES ---
if not os.path.exists(SOURCE_IMAGES_DIR):
    print(f"Error: Image directory not found at {SOURCE_IMAGES_DIR}")
    exit()

image_files = [
    f for f in os.listdir(SOURCE_IMAGES_DIR) if f.endswith((".jpg", ".jpeg", ".png"))
]

print(f"Processing {len(image_files)} images from {SOURCE_IMAGES_DIR}...")

count_crops = 0

for img_file in tqdm(image_files):
    # Load Image
    img_path = os.path.join(SOURCE_IMAGES_DIR, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    height, width, _ = img.shape

    # Load Label File
    label_file = img_file.rsplit(".", 1)[0] + ".txt"
    label_path = os.path.join(SOURCE_LABELS_DIR, label_file)

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    # Process each box in the file
    crop_idx = 0
    for line in lines:
        parts = line.strip().split()
        try:
            class_id = int(parts[0])

            # Safety check for weird class IDs
            if class_id >= len(CLASSES):
                continue

            # YOLO format is normalized: center_x, center_y, w, h
            x_center, y_center, w, h = map(float, parts[1:5])

            # Un-normalize to pixels
            x = int((x_center - w / 2) * width)
            y = int((y_center - h / 2) * height)
            w_px = int(w * width)
            h_px = int(h * height)

            # Clip to image boundaries
            x = max(0, x)
            y = max(0, y)
            w_px = min(w_px, width - x)
            h_px = min(h_px, height - y)

            # CROP!
            crop = img[y : y + h_px, x : x + w_px]

            if crop.size == 0:
                continue

            # Save to the breed folder
            breed_name = CLASSES[class_id].strip().replace(" ", "_")
            save_name = f"{img_file.rsplit('.', 1)[0]}_crop{crop_idx}.jpg"
            save_path = os.path.join(OUTPUT_DIR, breed_name, save_name)

            cv2.imwrite(save_path, crop)
            crop_idx += 1
            count_crops += 1

        except ValueError:
            continue

print(f"Done! Created {count_crops} cropped images in '{OUTPUT_DIR}'.")

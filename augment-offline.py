import os
import random
import shutil
from collections import Counter

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "dataset/Indian_bovine_breeds"
OUTPUT_DIR = "dataset/Indian_bovine_breeds_balanced"
TARGET_COUNT_PER_CLASS = 500  # Target number of images per breed


def augment_image(img):
    """Applies random distinct augmentations for offline storage."""

    # 1. Random Horizontal Flip (50%)
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # 2. Random Rotation (-15 to 15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    # 3. Random Brightness (0.8x to 1.2x)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # 4. Random Contrast (0.8x to 1.2x)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # 5. Random Sharpness (0.8x to 1.5x) - Helps with texture
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.5))

    return img


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return

    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Warning: Output directory '{OUTPUT_DIR}' already exists.")
        # shutil.rmtree(OUTPUT_DIR) # Uncomment to delete and restart clean

    print(f"🚀 Starting Offline Balancing...")
    print(f"Target: {TARGET_COUNT_PER_CLASS} images per breed")

    classes = [
        d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]
    print(f"Found {len(classes)} classes.")

    total_generated = 0

    for breed in classes:
        src_folder = os.path.join(INPUT_DIR, breed)
        dst_folder = os.path.join(OUTPUT_DIR, breed)

        os.makedirs(dst_folder, exist_ok=True)

        # Get list of valid images
        images = [
            f
            for f in os.listdir(src_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        current_count = len(images)

        print(f"Processing '{breed}': {current_count} original images...")

        # 1. Copy Originals first
        for img_name in images:
            src_path = os.path.join(src_folder, img_name)
            dst_path = os.path.join(dst_folder, img_name)
            if not os.path.exists(dst_path):
                try:
                    # Convert to RGB to fix palette/alpha issues immediately
                    with Image.open(src_path) as img:
                        img.convert("RGB").save(dst_path)
                except Exception as e:
                    print(f"Skipped corrupt file {img_name}: {e}")

        # 2. Augment if needed
        needed = TARGET_COUNT_PER_CLASS - current_count

        if needed > 0:
            print(f"  -> Generating {needed} augmented images...")

            for i in tqdm(range(needed), desc=f"Augmenting {breed}", leave=False):
                # Pick a random original image to augment
                source_img_name = random.choice(images)
                source_img_path = os.path.join(src_folder, source_img_name)

                try:
                    with Image.open(source_img_path) as img:
                        img = img.convert("RGB")

                        # Apply Augmentation
                        aug_img = augment_image(img)

                        # Save with unique name
                        save_name = f"aug_{i}_{source_img_name}"
                        aug_img.save(os.path.join(dst_folder, save_name))
                        total_generated += 1
                except Exception as e:
                    pass
        else:
            print(f"  -> Class full (Count: {current_count}). Skipping augmentation.")

    print(f"\n✅ Done! Generated {total_generated} new images.")
    print(f"Balanced dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

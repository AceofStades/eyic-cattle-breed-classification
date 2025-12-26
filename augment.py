import argparse
import random
import shutil
import uuid
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm

# Handle Pillow version differences for Resampling
if hasattr(Image, 'Resampling'):
    RESAMPLE_METHOD = Image.Resampling.LANCZOS
else:
    # Fallback for older Pillow versions
    RESAMPLE_METHOD = Image.LANCZOS

# Optional: use albumentations for richer, faster offline augmentations
ALB_AVAILABLE = False
try:
    import albumentations as A
    import cv2
    ALB_AVAILABLE = True
except ImportError:
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_args():
    parser = argparse.ArgumentParser(description="Offline dataset balancer and augmenter")
    parser.add_argument("--input", required=True, help="Input dataset directory (class subfolders)")
    parser.add_argument("--output", required=True, help="Output directory for balanced dataset")
    parser.add_argument("--target", type=int, default=500, help="Target images per class")
    parser.add_argument("--img-size", type=int, default=224, help="Output square image size (px)")
    parser.add_argument("--use-alb", action="store_true", help="Use albumentations when available (faster/stronger)")
    parser.add_argument("--force", action="store_true", help="Delete and recreate output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for saved images")
    
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def get_image_list(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]

def make_alb_transform(img_size):
    """
    Creates an Albumentations pipeline.
    Updated for Albumentations 2.0+ (requires 'size' tuple for RandomResizedCrop).
    """
    return A.Compose(
        [
            # FIX: Used 'size' tuple instead of separate height/width
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), p=1.0),
            
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            
            # CoarseDropout (Replaces deprecated Cutout)
            A.CoarseDropout(max_holes=1, max_height=int(img_size * 0.2), max_width=int(img_size * 0.2), 
                            min_holes=1, min_height=1, min_width=1, fill_value=128, p=0.3),
            
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.5),
        ],
        p=1.0,
    )

def apply_alb(img: Image.Image, aug, img_size):
    """
    Applies Albumentations to a PIL image.
    """
    # Convert PIL to Numpy (RGB)
    arr = np.array(img)
    
    # Apply Augmentation
    out = aug(image=arr)
    out_img = out["image"]
    
    # Convert back to PIL
    pil = Image.fromarray(out_img)
    return pil

def apply_pil(img: Image.Image, img_size):
    # Resize first for consistency
    img = img.resize((img_size, img_size), RESAMPLE_METHOD)

    # Random horizontal flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    # Random rotate
    if random.random() < 0.6:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC)

    # Color jitter-like operations
    if random.random() < 0.6:
        enh = ImageEnhance.Brightness(img)
        img = enh.enhance(random.uniform(0.8, 1.2))
        enh = ImageEnhance.Contrast(img)
        img = enh.enhance(random.uniform(0.8, 1.2))
        enh = ImageEnhance.Sharpness(img)
        img = enh.enhance(random.uniform(0.8, 1.5))

    # Random blur
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Random cutout-like
    if random.random() < 0.25:
        w, h = img.size
        max_size = int(min(w, h) * 0.2)
        x0 = random.randint(0, w - max_size)
        y0 = random.randint(0, h - max_size)
        rect = Image.new("RGB", (max_size, max_size), color=(128, 128, 128))
        img.paste(rect, (x0, y0))

    return img

def save_image(img: Image.Image, path: Path, quality: int = 95):
    try:
        # Ensure RGB before saving (removes Alpha channel if present)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(path, quality=quality)
    except Exception as e:
        logging.error(f"Error saving {path}: {e}")

def main():
    args = parse_args()
    setup_logging()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    target = args.target
    img_size = args.img_size

    logging.info(f"Input: {input_dir}")
    logging.info(f"Output: {output_dir}")
    logging.info(f"Target: {target} | Size: {img_size}")

    if not input_dir.exists():
        logging.error(f"Input directory {input_dir} does not exist")
        return

    if output_dir.exists() and args.force:
        logging.info("Removing existing output directory...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [d for d in input_dir.iterdir() if d.is_dir()]
    logging.info(f"Found {len(classes)} classes")

    total_generated = 0
    skipped = 0

    # Prepare albumentations if requested and available
    aug = None
    if args.use_alb:
        if ALB_AVAILABLE:
            aug = make_alb_transform(img_size)
            logging.info("Using Albumentations")
        else:
            logging.warning("Albumentations requested but not installed. Falling back to PIL.")

    for cls in classes:
        src_files = get_image_list(cls)
        if not src_files:
            logging.warning(f"No images found in {cls.name}")
            continue

        dst_cls = output_dir / cls.name
        dst_cls.mkdir(parents=True, exist_ok=True)

        logging.info(f"Processing Class: {cls.name} | Found: {len(src_files)}")

        # 1) Copy and normalize originals
        for src in src_files:
            dst = dst_cls / src.name
            if dst.exists():
                continue
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im = im.resize((img_size, img_size), RESAMPLE_METHOD)
                    save_image(im, dst, quality=args.quality)
            except Exception as e:
                logging.warning(f"Skipping corrupt file {src}: {e}")
                skipped += 1

        # 2) Calculate need
        existing_count = len([p for p in dst_cls.iterdir() if is_image_file(p)])
        needed = max(0, target - existing_count)

        if needed == 0:
            logging.info(f"Class {cls.name} has {existing_count} images. Target met.")
            continue

        # 3) Generate augmented images
        logging.info(f"Augmenting {cls.name}: Generating {needed} new images...")
        
        for _ in tqdm(range(needed), desc=f"Augmenting {cls.name}"):
            src = random.choice(src_files)
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    
                    if aug:
                        im_aug = apply_alb(im, aug, img_size)
                    else:
                        im_aug = apply_pil(im, img_size)

                    save_name = f"aug_{uuid.uuid4().hex[:8]}_{src.stem}.jpg"
                    save_path = dst_cls / save_name
                    save_image(im_aug, save_path, quality=args.quality)
                    total_generated += 1
            except Exception as e:
                logging.warning(f"Failed to augment {src.name}: {e}")
                skipped += 1

    logging.info("-" * 50)
    logging.info(f"Processing Complete.")
    logging.info(f"Total Augmented Generated: {total_generated}")
    logging.info(f"Total Skipped/Errors: {skipped}")
    logging.info(f"Balanced dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()
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

if hasattr(Image, 'Resampling'):
    RESAMPLE_METHOD = Image.Resampling.LANCZOS
else:

    RESAMPLE_METHOD = Image.LANCZOS

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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--use-alb", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=95)

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
 
    return A.Compose(
        [

            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), p=1.0),

            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),

            A.CoarseDropout(max_holes=1, max_height=int(img_size * 0.2), max_width=int(img_size * 0.2), 
                            min_holes=1, min_height=1, min_width=1, fill_value=128, p=0.3),

            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.5),
        ],
        p=1.0,
    )

def apply_alb(img: Image.Image, aug, img_size):
   

    arr = np.array(img)

    out = aug(image=arr)
    out_img = out["image"]

    pil = Image.fromarray(out_img)
    return pil

def apply_pil(img: Image.Image, img_size):

    img = img.resize((img_size, img_size), RESAMPLE_METHOD)

    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    if random.random() < 0.6:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC)

    if random.random() < 0.6:
        enh = ImageEnhance.Brightness(img)
        img = enh.enhance(random.uniform(0.8, 1.2))
        enh = ImageEnhance.Contrast(img)
        img = enh.enhance(random.uniform(0.8, 1.2))
        enh = ImageEnhance.Sharpness(img)
        img = enh.enhance(random.uniform(0.8, 1.5))

    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

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

        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(path, quality=quality)
    except Exception as e:
        logging.error(f"err saving {path}: {e}")

def main():
    args = parse_args()
    setup_logging()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    target = args.target
    img_size = args.img_size

    logging.info(f"inp: {input_dir}")
    logging.info(f"out: {output_dir}")
    logging.info(f"tgt: {target} | Size: {img_size}")

    if not input_dir.exists():
        return

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [d for d in input_dir.iterdir() if d.is_dir()]
    logging.info(f"fnd {len(classes)} classes")

    total_generated = 0
    skipped = 0

    aug = None
    if args.use_alb:
        if ALB_AVAILABLE:
            aug = make_alb_transform(img_size)
            logging.info("albumten")
        else:
            logging.warning("lol")

    for cls in classes:
        src_files = get_image_list(cls)
        if not src_files:
            continue

        dst_cls = output_dir / cls.name
        dst_cls.mkdir(parents=True, exist_ok=True)

        logging.info(f"proc Class: {cls.name} | Found: {len(src_files)}")

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
                skipped += 1

        existing_count = len([p for p in dst_cls.iterdir() if is_image_file(p)])
        needed = max(0, target - existing_count)

        if needed == 0:
            logging.info(f"class {cls.name} has {existing_count} img")
            continue

        logging.info(f"auging {cls.name}: gen {needed} new img...")

        for _ in tqdm(range(needed), desc=f"auging {cls.name}"):
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
    logging.info(f"aug gen: {total_generated}")
    logging.info(f"skip: {skipped}")

if __name__ == "__main__":
    main()
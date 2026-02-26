"""
Run the ONNX model on ALL breed images and sort them into
correct / incorrect folders. Also generates a CSV report.

Usage:
    python eval_all_breeds.py

Output structure:
    eval_results/
        correct/
            Gir/
                Gir_1.JPG  (symlink)
                ...
            Deoni/
                ...
        incorrect/
            Gir/                        

                Gir_42.jpg -> Deoni     

                ...
        eval_report.csv
        eval_summary.txt
"""

import os
import json
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import albumentations as A
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = SCRIPT_DIR / "breed_classifier.onnx"
LABEL_MAP_PATH = SCRIPT_DIR / "breed_labels.json"
DATASET_DIR = SCRIPT_DIR / "dataset" / "Indian_bovine_breeds"
RESULTS_DIR = SCRIPT_DIR / "eval_results"
IMG_SIZE = 260
RESIZE_SIZE = 292

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def load_label_map():
    with open(LABEL_MAP_PATH) as f:
        return json.load(f)

def get_val_transform():
    return A.Compose([
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    transformed = transform(image=img_np)["image"]

    tensor = np.transpose(transformed, (2, 0, 1)).astype(np.float32)
    return tensor[np.newaxis, ...]

def main():
    label_map = load_label_map()

    name_to_idx = {v: int(k) for k, v in label_map.items()}
    num_classes = len(label_map)

    print(f"Loading ONNX model: {ONNX_MODEL_PATH}")
    sess = ort.InferenceSession(str(ONNX_MODEL_PATH))
    transform = get_val_transform()

    correct_dir = RESULTS_DIR / "correct"
    incorrect_dir = RESULTS_DIR / "incorrect"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    breed_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name != ".ipynb_checkpoints"])
    for breed_dir in breed_dirs:
        breed_name = breed_dir.name
        if breed_name not in name_to_idx:
            print(f"  WARNING: folder '{breed_name}' not in label map, skipping")
            continue
        for img_file in sorted(breed_dir.iterdir()):
            if img_file.suffix.lower() in IMAGE_EXTS:
                all_images.append((img_file, breed_name))

    print(f"Found {len(all_images)} images across {len(breed_dirs)} breeds\n")

    breed_stats = {}  

    results_rows = []

    for img_path, true_breed in tqdm(all_images, desc="Evaluating"):
        try:
            input_data = preprocess(img_path, transform)
            output = sess.run(None, {"input": input_data})[0]

            exp_out = np.exp(output - np.max(output, axis=1, keepdims=True))
            probs = exp_out / exp_out.sum(axis=1, keepdims=True)

            pred_idx = int(np.argmax(probs, axis=1)[0])
            pred_breed = label_map[str(pred_idx)]
            confidence = float(probs[0, pred_idx])

            is_correct = pred_breed == true_breed

            if is_correct:
                dest_dir = correct_dir / true_breed
            else:
                dest_dir = incorrect_dir / true_breed
            dest_dir.mkdir(parents=True, exist_ok=True)

            if is_correct:
                link_name = img_path.name
            else:
                stem = img_path.stem
                ext = img_path.suffix
                link_name = f"{stem}__pred_{pred_breed}{ext}"

            link_path = dest_dir / link_name
            if not link_path.exists():
                link_path.symlink_to(img_path.resolve())

            if true_breed not in breed_stats:
                breed_stats[true_breed] = {"correct": 0, "total": 0, "incorrect_as": {}}
            breed_stats[true_breed]["total"] += 1
            if is_correct:
                breed_stats[true_breed]["correct"] += 1
            else:
                breed_stats[true_breed]["incorrect_as"][pred_breed] = \
                    breed_stats[true_breed]["incorrect_as"].get(pred_breed, 0) + 1

            results_rows.append({
                "image": str(img_path.relative_to(SCRIPT_DIR)),
                "true_breed": true_breed,
                "predicted_breed": pred_breed,
                "confidence": f"{confidence:.6f}",
                "correct": is_correct,
            })

        except Exception as e:
            print(f"  ERROR on {img_path.name}: {e}")
            results_rows.append({
                "image": str(img_path.relative_to(SCRIPT_DIR)),
                "true_breed": true_breed,
                "predicted_breed": "ERROR",
                "confidence": "0",
                "correct": False,
            })

    csv_path = RESULTS_DIR / "eval_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "true_breed", "predicted_breed", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(results_rows)

    total_correct = sum(s["correct"] for s in breed_stats.values())
    total_images = sum(s["total"] for s in breed_stats.values())
    overall_acc = total_correct / total_images if total_images > 0 else 0

    summary_lines = []
    summary_lines.append(f"{'='*65}")
    summary_lines.append(f"  ONNX Model Evaluation Summary")
    summary_lines.append(f"{'='*65}")
    summary_lines.append(f"  Total images:    {total_images}")
    summary_lines.append(f"  Correct:         {total_correct}")
    summary_lines.append(f"  Incorrect:       {total_images - total_correct}")
    summary_lines.append(f"  Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    summary_lines.append(f"")
    summary_lines.append(f"  {'Breed':<22} {'Correct':>8} {'Total':>8} {'Acc':>8}  Top confusions")
    summary_lines.append(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}  {'-'*25}")

    for breed in sorted(breed_stats.keys()):
        s = breed_stats[breed]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0

        confusions = sorted(s["incorrect_as"].items(), key=lambda x: -x[1])[:3]
        conf_str = ", ".join(f"{b}({c})" for b, c in confusions) if confusions else "-"
        summary_lines.append(f"  {breed:<22} {s['correct']:>8} {s['total']:>8} {acc:>7.2%}  {conf_str}")

    summary_lines.append(f"{'='*65}")
    summary_lines.append(f"")
    summary_lines.append(f"  Output folders:")
    summary_lines.append(f"    eval_results/correct/    -> correctly classified images (symlinks)")
    summary_lines.append(f"    eval_results/incorrect/  -> misclassified images (symlinks)")
    summary_lines.append(f"    eval_results/eval_report.csv -> full per-image results")
    summary_lines.append(f"{'='*65}")

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    summary_path = RESULTS_DIR / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\nResults saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()




import json
import argparse
from pathlib import Path

import numpy as np
import torch
import timm
import onnxruntime as ort
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

SCRIPT_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = SCRIPT_DIR / "best_breed_classifier.pth"
ONNX_MODEL_PATH = SCRIPT_DIR / "breed_classifier.onnx"
LABEL_MAP_PATH = SCRIPT_DIR / "breed_labels.json"
IMG_SIZE = 260
RESIZE_SIZE = 292
DEVICE = torch.device("cpu")  

def load_label_map():
    with open(LABEL_MAP_PATH) as f:
        return json.load(f)

def get_val_transform():
    return A.Compose([
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def load_pytorch_model(num_classes):
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes)
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def run_pytorch(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
    return output.numpy(), probs.numpy()

def run_onnx(input_tensor):
    sess = ort.InferenceSession(str(ONNX_MODEL_PATH))
    input_np = input_tensor.numpy()
    output = sess.run(None, {"input": input_np})[0]

    exp_out = np.exp(output - np.max(output, axis=1, keepdims=True))
    probs = exp_out / exp_out.sum(axis=1, keepdims=True)
    return output, probs

def compare(label_map, input_tensor, tag=""):
    num_classes = len(label_map)
    model = load_pytorch_model(num_classes)

    pth_logits, pth_probs = run_pytorch(model, input_tensor)
    onnx_logits, onnx_probs = run_onnx(input_tensor)

    pth_idx = int(np.argmax(pth_probs, axis=1)[0])
    onnx_idx = int(np.argmax(onnx_probs, axis=1)[0])
    pth_conf = float(pth_probs[0, pth_idx])
    onnx_conf = float(onnx_probs[0, onnx_idx])

    logit_diff = np.abs(pth_logits - onnx_logits)
    prob_diff = np.abs(pth_probs - onnx_probs)

    print(f"\n{'='*55}")
    print(f"  PTH vs ONNX comparison {tag}")
    print(f"{'='*55}")
    print(f"  PTH  prediction:  {label_map[str(pth_idx)]:20s}  conf {pth_conf:.6f}")
    print(f"  ONNX prediction:  {label_map[str(onnx_idx)]:20s}  conf {onnx_conf:.6f}")
    print(f"  Labels match:     {'YES' if pth_idx == onnx_idx else 'NO  <<<< MISMATCH!'}")
    print(f"  Logit max diff:   {logit_diff.max():.8f}")
    print(f"  Logit mean diff:  {logit_diff.mean():.8f}")
    print(f"  Prob  max diff:   {prob_diff.max():.8f}")
    print(f"  Prob  mean diff:  {prob_diff.mean():.8f}")

    pth_top5 = np.argsort(pth_probs[0])[::-1][:5]
    onnx_top5 = np.argsort(onnx_probs[0])[::-1][:5]

    print(f"\n  PTH  top-5:")
    for rank, idx in enumerate(pth_top5):
        print(f"    {rank+1}. {label_map[str(idx)]:20s}  {pth_probs[0, idx]:.6f}")
    print(f"  ONNX top-5:")
    for rank, idx in enumerate(onnx_top5):
        print(f"    {rank+1}. {label_map[str(idx)]:20s}  {onnx_probs[0, idx]:.6f}")

    return pth_idx == onnx_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to a test image")
    args = parser.parse_args()

    label_map = load_label_map()

    print("\n[Test 1] Random tensor input")
    torch.manual_seed(42)
    random_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    match1 = compare(label_map, random_input, tag="(random)")

    if args.image:
        print(f"\n[Test 2] Real image: {args.image}")
        img = Image.open(args.image).convert("RGB")
        img_np = np.array(img)
        transform = get_val_transform()
        transformed = transform(image=img_np)
        img_tensor = transformed["image"].unsqueeze(0)
        match2 = compare(label_map, img_tensor, tag=f"({args.image})")
    else:

        gir_dir = SCRIPT_DIR / "dataset" / "Indian_bovine_breeds" / "Gir"
        deoni_dir = SCRIPT_DIR / "dataset" / "Indian_bovine_breeds" / "Deoni"

        transform = get_val_transform()
        mismatches = 0
        tested = 0

        for breed_dir, breed_name in [(gir_dir, "Gir"), (deoni_dir, "Deoni")]:
            if breed_dir.exists():
                imgs = sorted(breed_dir.glob("*"))[:3]  

                for img_path in imgs:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_np = np.array(img)
                        transformed = transform(image=img_np)
                        img_tensor = transformed["image"].unsqueeze(0)
                        matched = compare(label_map, img_tensor, tag=f"({breed_name}: {img_path.name})")
                        tested += 1
                        if not matched:
                            mismatches += 1
                    except Exception as e:
                        print(f"  skip {img_path.name}: {e}")

        print(f"\n{'='*55}")
        print(f"  sum: {tested} images test, {mismatches} PTH/ONNX mismatches")
        if mismatches == 0:
            print(f"  -> onyx i ok")
        else:
            print(f"  ->wtf")
        print(f"{'='*55}")

if __name__ == "__main__":
    main()



import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
import albumentations as A

SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = SCRIPT_DIR / "breed_classifier.onnx"
LABEL_MAP_PATH = SCRIPT_DIR / "breed_labels.json"
IMG_SIZE = 260
RESIZE_SIZE = 292

@st.cache_resource
def load_model():
    sess = ort.InferenceSession(str(ONNX_MODEL_PATH))
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    return sess, label_map

def get_transform():
    return A.Compose([
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms():
    norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    margin = RESIZE_SIZE - IMG_SIZE
    return [
        ("center", A.Compose([
            A.Resize(RESIZE_SIZE, RESIZE_SIZE),
            A.CenterCrop(IMG_SIZE, IMG_SIZE),
            norm,
        ])),
        ("hflip", A.Compose([
            A.Resize(RESIZE_SIZE, RESIZE_SIZE),
            A.CenterCrop(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            norm,
        ])),
        ("top_left", A.Compose([
            A.Resize(RESIZE_SIZE, RESIZE_SIZE),
            A.Crop(x_min=0, y_min=0, x_max=IMG_SIZE, y_max=IMG_SIZE),
            norm,
        ])),
        ("top_right", A.Compose([
            A.Resize(RESIZE_SIZE, RESIZE_SIZE),
            A.Crop(x_min=margin, y_min=0, x_max=RESIZE_SIZE, y_max=IMG_SIZE),
            A.Resize(IMG_SIZE, IMG_SIZE),
            norm,
        ])),
        ("bottom_left", A.Compose([
            A.Resize(RESIZE_SIZE, RESIZE_SIZE),
            A.Crop(x_min=0, y_min=margin, x_max=IMG_SIZE, y_max=RESIZE_SIZE),
            A.Resize(IMG_SIZE, IMG_SIZE),
            norm,
        ])),
        ("bottom_right", A.Compose([
            A.Resize(RESIZE_SIZE, RESIZE_SIZE),
            A.Crop(x_min=margin, y_min=margin, x_max=RESIZE_SIZE, y_max=RESIZE_SIZE),
            A.Resize(IMG_SIZE, IMG_SIZE),
            norm,
        ])),
        ("bright+", A.Compose([
            A.Resize(RESIZE_SIZE, RESIZE_SIZE),
            A.CenterCrop(IMG_SIZE, IMG_SIZE),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0),
            norm,
        ])),
        ("bright-", A.Compose([
            A.Resize(RESIZE_SIZE, RESIZE_SIZE),
            A.CenterCrop(IMG_SIZE, IMG_SIZE),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=0, p=1.0),
            norm,
        ])),
    ]


def run_inference(sess, img_np, transform):
    transformed = transform(image=img_np)["image"]
    tensor = np.transpose(transformed, (2, 0, 1)).astype(np.float32)[np.newaxis, ...]
    return sess.run(None, {"input": tensor})[0]


def softmax(logits):
    exp_out = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_out / exp_out.sum(axis=1, keepdims=True)


def predict(image: Image.Image, sess, label_map, transform):
    img_np = np.array(image.convert("RGB"))
    output = run_inference(sess, img_np, transform)
    probs = softmax(output)[0]

    top5_idx = np.argsort(probs)[::-1][:5]
    results = []
    for idx in top5_idx:
        results.append((label_map[str(idx)], float(probs[idx])))
    return results


def predict_tta(image: Image.Image, sess, label_map):
    img_np = np.array(image.convert("RGB"))
    tta_transforms = get_tta_transforms()

    all_probs = []
    for name, t in tta_transforms:
        output = run_inference(sess, img_np, t)
        probs = softmax(output)[0]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)

    top5_idx = np.argsort(avg_probs)[::-1][:5]
    results = []
    for idx in top5_idx:
        results.append((label_map[str(idx)], float(avg_probs[idx])))
    return results, len(tta_transforms)


st.set_page_config(page_title="Cattle Breed Classifier", page_icon="🐄", layout="centered")
st.title("🐄 Cattle Breed Classifier")
st.caption("Upload or paste an image to classify using the ONNX model")

sess, label_map = load_model()
transform = get_transform()

use_tta = st.toggle("Enable TTA (Test-Time Augmentation)", value=True,
                     help="Averages predictions over 8 augmented views for better accuracy. Slightly slower.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Input image", use_container_width=True)

    with st.spinner("Classifying..." + (" (TTA: 8 views)" if use_tta else "")):
        if use_tta:
            results, n_views = predict_tta(image, sess, label_map)
        else:
            results = predict(image, sess, label_map, transform)
            n_views = 1

    st.subheader("Predictions")
    top_breed, top_conf = results[0]
    st.metric("Top prediction", top_breed.replace("_", " "), f"{top_conf:.1%}")
    if use_tta:
        st.caption(f"Averaged over {n_views} augmented views")

    st.divider()
    st.caption("Top 5")
    for breed, conf in results:
        st.progress(conf, text=f"{breed.replace('_', ' ')}  —  {conf:.2%}")


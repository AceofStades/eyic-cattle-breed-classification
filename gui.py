import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import traceback
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except Exception:
    ONNXRUNTIME_AVAILABLE = False
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models, transforms

from model import CONFIG, get_dataloaders, get_transforms, safe_pil_loader

st.set_page_config(page_title="Herd-Link Classifier GUI", page_icon=None, layout="wide")


@st.cache_resource
def load_trained_model(path=None, num_classes=None, device=None, choice="auto"):
    """Try to load an ONNX model (based on `choice`) or fall back to a PyTorch .pth model.

    Returns a tuple (kind, model) where kind is 'onnx' or 'torch' or None on failure.
    """
    # If user provided a specific onnx path, try it first
    if isinstance(choice, str) and os.path.exists(choice) and choice.lower().endswith(".onnx"):
        if ONNXRUNTIME_AVAILABLE:
            try:
                sess = onnxruntime.InferenceSession(choice)
                return ("onnx", sess)
            except Exception as e:
                st.warning(f"Failed to load ONNX model '{choice}': {e}")

    # Find experiment directories (newest last)
    try:
        exp_dirs = sorted(glob.glob("experiments/*/"))
    except Exception:
        exp_dirs = []

    onnx_path = None
    choice_key = (choice or "").lower()

    if "pth" not in choice_key:
        # search experiments in reverse chronological order
        for d in reversed(exp_dirs):
            files = glob.glob(os.path.join(d, "*.onnx"))
            for f in files:
                fname = os.path.basename(f).lower()
                if choice_key.startswith("auto"):
                    onnx_path = f
                    break
                if "mobile" in choice_key and "mobile" in fname:
                    onnx_path = f
                    break
                if ("b3" in choice_key or "efficientnet" in choice_key) and (
                    "efficientnet" in fname or "b3" in fname
                ):
                    onnx_path = f
                    break
            if onnx_path:
                break

    if onnx_path and ONNXRUNTIME_AVAILABLE:
        try:
            sess = onnxruntime.InferenceSession(onnx_path)
            return ("onnx", sess)
        except Exception as e:
            st.warning(f"Failed to load ONNX model '{onnx_path}': {e}")

    # Fallback to PyTorch checkpoint
    if path is None:
        path = "breed_classifier_large.pth"
    try:
        if not os.path.exists(path):
            return (None, None)

        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(num_ftrs, num_classes)
        )

        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return ("torch", model)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return (None, None)


@st.cache_resource
def load_data_info():
    # get_transforms expects an integer img size (width). Use configured size.
    train_tf, val_tf = get_transforms(CONFIG["IMG_SIZE"][0])
    dataloaders, num_classes = get_dataloaders(CONFIG["DATA_DIR"], train_tf, val_tf)

    if hasattr(dataloaders["test"].dataset, "dataset"):
        class_names = dataloaders["test"].dataset.dataset.classes
    else:
        class_names = [str(i) for i in range(num_classes)]

    return dataloaders, class_names


st.sidebar.title("Herd-Link AI")
st.sidebar.info("MobileNetV3-Large Breed Classifier")
def discover_onnx_models():
    """Return list of discovered ONNX model paths (experiments, quantization folders)."""
    candidates = []
    # search experiments/*/*.onnx
    for p in glob.glob(os.path.join("experiments", "**", "*.onnx"), recursive=True):
        candidates.append(p)
    # search quantization folder
    for p in glob.glob(os.path.join("quantization", "**", "*.onnx"), recursive=True):
        candidates.append(p)
    # search onnyx folder
    for p in glob.glob(os.path.join("onnyx", "**", "*.onnx"), recursive=True):
        candidates.append(p)
    # dedupe and sort newest first by mtime
    uniq = list({os.path.abspath(x): x for x in candidates}.values())
    uniq.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    return uniq

# Model source selector (auto or explicit discovered ONNX files or pth)
found_onnx = discover_onnx_models()
options = ["Auto (ONNX preferred)"]
options += [os.path.relpath(p) for p in found_onnx]
options += ["PyTorch (.pth)"]
model_choice = st.sidebar.selectbox("Model Source", options)
page = st.sidebar.radio(
    "Navigate", ["Project Info", "Live Classification", "Model Evaluation"]
)

DEVICE = CONFIG["DEVICE"]
MODEL_PATH = "breed_classifier_large.pth"

with st.spinner("Loading dataset information..."):
    dataloaders, class_names = load_data_info()

# Map choice to a direct path when user picks a discovered ONNX
chosen_path = None
if model_choice not in ("Auto (ONNX preferred)", "PyTorch (.pth)"):
    # model_choice is a relative path string from options; convert to absolute
    candidate = os.path.abspath(model_choice)
    # if not absolute, try join with repo root
    if not os.path.exists(candidate):
        candidate = os.path.abspath(os.path.join(".", model_choice))
    if os.path.exists(candidate) and candidate.lower().endswith(".onnx"):
        chosen_path = candidate

# Load model according to selected source
model_kind, model = load_trained_model(MODEL_PATH, len(class_names), DEVICE, choice=chosen_path or model_choice)

if page == "Project Info":
    st.title("About Herd-Link AI")
    st.markdown("""
    ### The Mission
    **Herd-Link** uses advanced Computer Vision to identify indigenous Indian cattle breeds.
    Accurate identification is the first step towards better health monitoring, genetic tracking, and productivity improvement.

    ### The Brain: MobileNetV3-Large
    We use **MobileNetV3-Large**, a state-of-the-art Convolutional Neural Network designed specifically for mobile devices.
    * **Efficient:** Runs offline on Android/iOS.
    * **Accurate:** Fine-tuned on a balanced dataset of 40 Indian breeds.
    * **Robust:** Trained with AutoAugment to handle diverse lighting and angles.

    ### Dataset Details
    * **Source:** Indian Bovine Breeds Dataset
    * **Classes:** 40 Breeds (Gir, Sahiwal, Red Sindhi, etc.)
    * **Technique:** Stratified Split + Weighted Sampling to handle rare breeds.
    """)

    if model:
        st.success(f"Model loaded successfully ({model_kind}).")
        st.code(
            f"Device: {DEVICE}\nInput Size: {CONFIG['IMG_SIZE']}\nClasses: {len(class_names)}"
        )
    else:
        st.error(
            "Model file 'breed_classifier_large.pth' not found. Please train the model first."
        )

elif page == "Live Classification":
    st.title("Live Breed Classifier")
    st.write("Upload an image of a cow to identify its breed.")

    if model is None:
        st.error("Model not found. Please train the model first.")
    else:
        uploaded_file = st.file_uploader(
            "Choose a cow image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                    image = Image.open(uploaded_file).convert("RGB")
                    # Use width='stretch' to make the image fill the column (replaces deprecated use_container_width=True).
                    # Do NOT use width=True (that sets width to 1 pixel).
                    st.image(image, caption="Uploaded Image", width='stretch')
            with col2:
                st.write("### Analysis")
                with st.spinner("Analyzing..."):
                    _, val_tf = get_transforms(CONFIG["IMG_SIZE"][0])
                    img_tensor = val_tf(image).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        if model_kind == "onnx":
                            # ONNX runtime expects a contiguous float32 numpy array
                            inp_name = model.get_inputs()[0].name
                            np_in = img_tensor.cpu().numpy()
                            try:
                                np_in = np.ascontiguousarray(np_in.astype(np.float32))
                                ort_outs = model.run(None, {inp_name: np_in})
                                outputs = torch.from_numpy(ort_outs[0])
                            except Exception as e:
                                st.error(f"ONNX runtime error: {e}")
                                st.text(traceback.format_exc())
                                outputs = torch.zeros((1, len(class_names)))
                        else:
                            outputs = model(img_tensor)

                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        top_p, top_class = probabilities.topk(1, dim=1)

                        top3_p, top3_class = probabilities.topk(3, dim=1)

                    breed = class_names[top_class.item()]
                    score = top_p.item() * 100

                    if score > 80:
                        st.success(f"**Result:** {breed}")
                    elif score > 50:
                        st.warning(f"**Result:** {breed} (Uncertain)")
                    else:
                        st.error(f"**Result:** {breed} (Low Confidence)")

                    st.metric("Confidence", f"{score:.2f}%")

                    st.write("#### Top 3 Predictions")
                    probs = top3_p.cpu().numpy()[0]
                    classes = [class_names[idx] for idx in top3_class.cpu().numpy()[0]]

                    # Use a pandas Series for bar_chart for consistent rendering
                    st.bar_chart(pd.Series(data=probs, index=classes))

elif page == "Model Evaluation":
    st.title("Model Performance")

    if "test_acc" not in st.session_state:
        st.session_state.test_acc = None
        st.session_state.report = None
        st.session_state.cm_fig = None

    if st.button("Run Evaluation on Test Set"):
        if model is None:
            st.error("Model not found.")
        else:
            with st.spinner("Running evaluation... This may take a moment."):
                all_preds = []
                all_labels = []

                # Use a batch-size-1 loader for ONNX to avoid Reshape errors in some exported graphs
                test_loader = dataloaders["test"]
                if model_kind == "onnx":
                    st.info("ONNX selected: evaluation will run with batch_size=1 to avoid reshape errors (slower).")
                    test_loader = torch.utils.data.DataLoader(
                        test_loader.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
                    )

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(DEVICE)
                        labels = labels.to(DEVICE)
                        if model_kind == "onnx":
                            inp_name = model.get_inputs()[0].name
                            np_in = inputs.cpu().numpy()
                            try:
                                np_in = np.ascontiguousarray(np_in.astype(np.float32))
                                ort_outs = model.run(None, {inp_name: np_in})
                                outputs = torch.from_numpy(ort_outs[0])
                            except Exception as e:
                                # Reshape issues may happen when ONNX model expects batch size 1 or different intermediate shapes.
                                err_msg = str(e).lower()
                                if "reshape" in err_msg or "cannot be reshaped" in err_msg or "resize" in err_msg:
                                    st.warning("ONNX runtime reshape error; falling back to per-sample inference.")
                                    outs_list = []
                                    for i in range(np_in.shape[0]):
                                        try:
                                            ort_out = model.run(None, {inp_name: np_in[i : i + 1]})
                                            outs_list.append(ort_out[0])
                                        except Exception as e2:
                                            st.warning(f"ONNX per-sample failure: {e2}")
                                            outs_list.append(np.zeros((1, len(class_names)), dtype=np.float32))
                                    merged = np.vstack(outs_list)
                                    outputs = torch.from_numpy(merged)
                                else:
                                    st.error(f"ONNX runtime error during evaluation: {e}")
                                    st.text(traceback.format_exc())
                                    outputs = torch.zeros((inputs.size(0), len(class_names)))
                        else:
                            outputs = model(inputs)

                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                correct = sum([p == l for p, l in zip(all_preds, all_labels)])
                accuracy = correct / len(all_labels)
                st.session_state.test_acc = accuracy

                st.session_state.report = classification_report(
                    all_labels, all_preds, target_names=class_names, output_dict=True
                )

                cm = confusion_matrix(all_labels, all_preds)
                fig_size = max(10, len(class_names) // 2)
                fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Greens",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax,
                )
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                st.session_state.cm_fig = fig

    if st.session_state.test_acc is not None:
        st.metric("Test Set Accuracy", f"{st.session_state.test_acc:.2%}")

        st.subheader("Confusion Matrix")
        st.pyplot(st.session_state.cm_fig)

        st.subheader("Classification Report")

        report_df = pd.DataFrame(st.session_state.report).transpose()

        def highlight_summary(s):
            is_summary = s.name in ["accuracy", "macro avg", "weighted avg"]
            return ["background-color: #262730" if is_summary else "" for _ in s]

        # Streamlit expects an int height; use a reasonable default.
        st.dataframe(
            report_df.style.apply(highlight_summary, axis=1).format("{:.2f}"),
            height=400,
        )

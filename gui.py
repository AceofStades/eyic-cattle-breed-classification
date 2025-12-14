import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models, transforms

from model import CONFIG, get_dataloaders, get_transforms, safe_pil_loader

st.set_page_config(page_title="Herd-Link Classifier GUI", page_icon=None, layout="wide")


@st.cache_resource
def load_trained_model(path, num_classes, device):
    try:
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(num_ftrs, num_classes)
        )

        if not os.path.exists(path):
            return None

        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_data_info():
    train_tf, val_tf = get_transforms()
    dataloaders, num_classes = get_dataloaders(CONFIG["DATA_DIR"], train_tf, val_tf)

    if hasattr(dataloaders["test"].dataset, "dataset"):
        class_names = dataloaders["test"].dataset.dataset.classes
    else:
        class_names = [str(i) for i in range(num_classes)]

    return dataloaders, class_names


st.sidebar.title("Herd-Link AI")
st.sidebar.info("MobileNetV3-Large Breed Classifier")
page = st.sidebar.radio(
    "Navigate", ["Project Info", "Live Classification", "Model Evaluation"]
)

DEVICE = CONFIG["DEVICE"]
MODEL_PATH = "breed_classifier_large.pth"

with st.spinner("Loading dataset information..."):
    dataloaders, class_names = load_data_info()

model = load_trained_model(MODEL_PATH, len(class_names), DEVICE)

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
        st.success("Model loaded successfully!")
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
                # FIX: use_container_width=True ensures image fits column.
                # Do NOT use width=True (that sets width to 1 pixel).
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.write("### Analysis")
                with st.spinner("Analyzing..."):
                    _, val_tf = get_transforms()
                    img_tensor = val_tf(image).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
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

                    st.bar_chart(data={c: p for c, p in zip(classes, probs)})

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

                with torch.no_grad():
                    for inputs, labels in dataloaders["test"]:
                        inputs = inputs.to(DEVICE)
                        labels = labels.to(DEVICE)
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

        st.dataframe(report_df.style.apply(highlight_summary, axis=1).format("{:.2f}"))

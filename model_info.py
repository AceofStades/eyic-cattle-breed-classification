import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
import onnx
import onnxruntime as ort
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.figsize": (12, 7),
})

SCRIPT_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = SCRIPT_DIR / "best_breed_classifier.pth"
ONNX_MODEL_PATH = SCRIPT_DIR / "breed_classifier.onnx"
LABEL_MAP_PATH = SCRIPT_DIR / "breed_labels.json"
CSV_PATH = SCRIPT_DIR / "dataset" / "Indian_bovine_breeds" / "bovine_breeds_metadata.csv"
GRAPHS_DIR = SCRIPT_DIR / "graphs"
IMG_SIZE = 260
RESIZE_SIZE = 292
MODEL_NAME = "efficientnet_b2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_map():
    with open(LABEL_MAP_PATH, "r") as f:
        return json.load(f)


def pytorch_info(model_path, num_classes):
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_params - trainable
    param_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    buf_mb = sum(b.nelement() * b.element_size() for b in model.buffers()) / (1024 * 1024)
    file_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"\n{'='*55}")
    print(f"pytorch model info")
    print(f"{'='*55}")
    print(f"  path            {model_path}")
    print(f"  file size       {file_mb:.2f} mb")
    print(f"  arch            {MODEL_NAME}")
    print(f"  backend         timm")
    print(f"  num classes     {num_classes}")
    print(f"  input shape     [1 3 {IMG_SIZE} {IMG_SIZE}]")
    print(f"  total params    {total_params:,}")
    print(f"  trainable       {trainable:,}")
    print(f"  frozen          {frozen:,}")
    print(f"  param mem       {param_mb:.2f} mb")
    print(f"  buffer mem      {buf_mb:.2f} mb")
    print(f"  total mem       {param_mb + buf_mb:.2f} mb")

    if "epoch" in ckpt:
        print(f"  best epoch      {ckpt['epoch']}")
    if "val_acc" in ckpt:
        print(f"  best val acc    {ckpt['val_acc']:.4f}")

    if hasattr(model, "classifier"):
        cls_layer = model.classifier
        print(f"  head layer      classifier")
        print(f"  head type       {cls_layer.__class__.__name__}")
        if hasattr(cls_layer, "in_features"):
            print(f"  head in feat    {cls_layer.in_features}")
        if hasattr(cls_layer, "out_features"):
            print(f"  head out feat   {cls_layer.out_features}")

    if hasattr(model, "conv_stem"):
        stem = model.conv_stem
        print(f"  stem type       {stem.__class__.__name__}")
        if hasattr(stem, "in_channels"):
            print(f"  stem in ch      {stem.in_channels}")
        if hasattr(stem, "out_channels"):
            print(f"  stem out ch     {stem.out_channels}")

    if hasattr(model, "bn1"):
        bn = model.bn1
        print(f"  stem bn         {bn.__class__.__name__} feat={bn.num_features}")

    if hasattr(model, "global_pool"):
        pool = model.global_pool
        print(f"  global pool     {pool.__class__.__name__}")

    layer_counts = {}
    for name, module in model.named_modules():
        mtype = module.__class__.__name__
        layer_counts[mtype] = layer_counts.get(mtype, 0) + 1
    print(f"  layer breakdown")
    for ltype, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {ltype:<28} {count}")

    param_by_block = {}
    for name, param in model.named_parameters():
        block = name.split(".")[0]
        param_by_block[block] = param_by_block.get(block, 0) + param.numel()
    print(f"  params by block")
    for block, count in sorted(param_by_block.items(), key=lambda x: -x[1]):
        pct = count / total_params * 100
        print(f"    {block:<28} {count:>10,} ({pct:.1f}%)")

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(dummy)
    print(f"  fwd pass test   ok | out shape {list(out.shape)}")
    print(f"{'='*55}")

    return model


def onnx_info(onnx_path, num_classes):
    if not os.path.exists(onnx_path):
        print(f"onnx file not found -> {onnx_path}")
        return

    onnx_model = onnx.load(str(onnx_path))
    file_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    try:
        onnx.checker.check_model(onnx_model)
        valid = "pass"
    except Exception as e:
        valid = f"fail ({e})"

    print(f"\n{'='*55}")
    print(f"onnx model info")
    print(f"{'='*55}")
    print(f"  path            {onnx_path}")
    print(f"  file size       {file_mb:.2f} mb")
    print(f"  valid           {valid}")
    print(f"  opset ver       {onnx_model.opset_import[0].version}")
    print(f"  ir ver          {onnx_model.ir_version}")
    print(f"  producer        {onnx_model.producer_name} {onnx_model.producer_version}")
    print(f"  graph name      {onnx_model.graph.name}")
    print(f"  num nodes       {len(onnx_model.graph.node)}")
    print(f"  num initializers {len(onnx_model.graph.initializer)}")
    print(f"  num params      {sum(np.prod(i.dims) for i in onnx_model.graph.initializer):,}")

    total_param_bytes = sum(
        np.prod(i.dims) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(i.data_type)).itemsize
        for i in onnx_model.graph.initializer
    )
    print(f"  param bytes     {total_param_bytes / (1024*1024):.2f} mb")

    for inp in onnx_model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
        print(f"  input           {inp.name} | shape {shape} | dtype {dtype}")

    for out in onnx_model.graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(out.type.tensor_type.elem_type)
        print(f"  output          {out.name} | shape {shape} | dtype {dtype}")

    op_counts = {}
    for node in onnx_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print(f"  op breakdown ({len(op_counts)} unique ops)")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op:<28} {count}")

    print(f"  --- ort inference test ---")
    sess = ort.InferenceSession(str(onnx_path))
    providers = sess.get_providers()
    print(f"  ort providers   {providers}")

    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    out = sess.run(None, {"input": dummy})
    print(f"  ort fwd test    ok | out shape {out[0].shape} | dtype {out[0].dtype}")
    print(f"  ort out range   [{out[0].min():.4f} {out[0].max():.4f}]")

    pytorch_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    if os.path.exists(BEST_MODEL_PATH):
        ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
        pytorch_model.load_state_dict(ckpt["model_state_dict"])
        pytorch_model.eval()
        dummy_torch = torch.from_numpy(dummy)
        with torch.no_grad():
            pt_out = pytorch_model(dummy_torch).numpy()
        diff = np.abs(pt_out - out[0]).max()
        print(f"  pt vs onnx diff {diff:.8f}")

    print(f"{'='*55}")


def compare_models(num_classes):
    if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(ONNX_MODEL_PATH):
        print(f"need both .pth and .onnx to compare")
        return

    pt_mb = os.path.getsize(BEST_MODEL_PATH) / (1024 * 1024)
    ox_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
    ratio = ox_mb / pt_mb if pt_mb > 0 else 0

    print(f"\n{'='*55}")
    print(f"pytorch vs onnx comparison")
    print(f"{'='*55}")
    print(f"  pytorch size    {pt_mb:.2f} mb")
    print(f"  onnx size       {ox_mb:.2f} mb")
    print(f"  size ratio      {ratio:.2f}x")

    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    pytorch_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
    pytorch_model.load_state_dict(ckpt["model_state_dict"])
    pytorch_model.eval()

    dummy_torch = torch.from_numpy(dummy)
    torch.no_grad().__enter__()
    for _ in range(5):
        pytorch_model(dummy_torch)
    t0 = time.perf_counter()
    for _ in range(100):
        pytorch_model(dummy_torch)
    pt_time = (time.perf_counter() - t0) / 100

    sess = ort.InferenceSession(str(ONNX_MODEL_PATH))
    for _ in range(5):
        sess.run(None, {"input": dummy})
    t0 = time.perf_counter()
    for _ in range(100):
        sess.run(None, {"input": dummy})
    ort_time = (time.perf_counter() - t0) / 100

    print(f"  pytorch latency {pt_time*1000:.2f} ms (cpu avg 100 runs)")
    print(f"  onnx latency    {ort_time*1000:.2f} ms (cpu avg 100 runs)")
    print(f"  speedup         {pt_time/ort_time:.2f}x")

    pt_out = pytorch_model(dummy_torch).detach().numpy()
    ox_out = sess.run(None, {"input": dummy})[0]
    max_diff = np.abs(pt_out - ox_out).max()
    mean_diff = np.abs(pt_out - ox_out).mean()
    print(f"  max abs diff    {max_diff:.8f}")
    print(f"  mean abs diff   {mean_diff:.8f}")
    print(f"  numerically eq  {'yes' if max_diff < 1e-5 else 'no (>1e-5)'}") 
    print(f"{'='*55}")

    return {
        "pt_mb": pt_mb,
        "ox_mb": ox_mb,
        "pt_time_ms": pt_time * 1000,
        "ort_time_ms": ort_time * 1000,
    }


def _ensure_graphs_dir():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def plot_dataset_distribution(label_map):
    if not CSV_PATH.exists():
        print(f"  csv not found -> {CSV_PATH} | skip")
        return

    _ensure_graphs_dir()
    df = pd.read_csv(CSV_PATH)
    counts = df["breed"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = sns.color_palette("viridis", n_colors=len(counts))
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of Images")
    ax.set_title("Dataset Distribution: Samples per Breed")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for bar, val in zip(bars, counts.values):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=7)

    plt.tight_layout()
    path = GRAPHS_DIR / "dataset_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_parameter_distribution(model):
    _ensure_graphs_dir()
    total_params = sum(p.numel() for p in model.parameters())
    param_by_block = {}
    for name, param in model.named_parameters():
        block = name.split(".")[0]
        param_by_block[block] = param_by_block.get(block, 0) + param.numel()

    blocks = sorted(param_by_block.keys(), key=lambda b: param_by_block[b])
    counts = [param_by_block[b] for b in blocks]
    pcts = [c / total_params * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("coolwarm", n_colors=len(blocks))
    bars = ax.barh(blocks, counts, color=colors)
    ax.set_xlabel("Number of Parameters")
    ax.set_title("Parameter Distribution by Model Block")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.1f}K"))

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + total_params * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    path = GRAPHS_DIR / "parameter_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_layer_type_distribution(model):
    _ensure_graphs_dir()
    layer_counts = {}
    for _, module in model.named_modules():
        mtype = module.__class__.__name__
        layer_counts[mtype] = layer_counts.get(mtype, 0) + 1

    sorted_layers = sorted(layer_counts.items(), key=lambda x: -x[1])
    top_n = 10
    top_layers = sorted_layers[:top_n]
    other_count = sum(c for _, c in sorted_layers[top_n:])
    if other_count > 0:
        top_layers.append(("Other", other_count))

    labels = [l for l, _ in top_layers]
    sizes = [c for _, c in top_layers]

    fig, ax = plt.subplots(figsize=(9, 9))
    colors = sns.color_palette("Set3", n_colors=len(labels))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        pctdistance=0.85, startangle=140,
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Layer Type Distribution")

    plt.tight_layout()
    path = GRAPHS_DIR / "layer_type_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_confusion_matrix(all_labels, all_preds, label_map):
    _ensure_graphs_dir()
    num_classes = len(label_map)
    target_names = [label_map[str(i)] for i in range(num_classes)]
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
        ax=ax, linewidths=0.3, linecolor="gray",
        annot_kws={"fontsize": 6},
        cbar_kws={"shrink": 0.7},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    plt.tight_layout()
    path = GRAPHS_DIR / "confusion_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_normalised_confusion_matrix(all_labels, all_preds, label_map):
    _ensure_graphs_dir()
    num_classes = len(label_map)
    target_names = [label_map[str(i)] for i in range(num_classes)]
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=target_names, yticklabels=target_names,
        ax=ax, linewidths=0.3, linecolor="gray",
        annot_kws={"fontsize": 6},
        vmin=0, vmax=1,
        cbar_kws={"shrink": 0.7, "label": "Recall"},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalised Confusion Matrix (Row = Recall)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    plt.tight_layout()
    path = GRAPHS_DIR / "confusion_matrix_normalised.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_per_class_f1(cls_report, label_map):
    _ensure_graphs_dir()
    breed_f1s = []
    for name in cls_report:
        if name in ("accuracy", "macro avg", "weighted avg"):
            continue
        breed_f1s.append((name, cls_report[name]["f1-score"]))
    breed_f1s.sort(key=lambda x: x[1])

    names = [b for b, _ in breed_f1s]
    scores = [s for _, s in breed_f1s]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = sns.color_palette("RdYlGn", n_colors=len(names))
    bars = ax.barh(names, scores, color=colors)
    ax.set_xlabel("F1 Score")
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-Class F1 Scores")
    ax.axvline(x=cls_report["macro avg"]["f1-score"], color="red",
               linestyle="--", linewidth=1.2, label=f"Macro Avg ({cls_report['macro avg']['f1-score']:.3f})")
    ax.legend(loc="lower right")

    for bar, val in zip(bars, scores):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    path = GRAPHS_DIR / "per_class_f1.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_per_class_precision_recall(cls_report, label_map):
    _ensure_graphs_dir()
    breed_metrics = []
    for name in cls_report:
        if name in ("accuracy", "macro avg", "weighted avg"):
            continue
        breed_metrics.append((
            name,
            cls_report[name]["precision"],
            cls_report[name]["recall"],
        ))
    breed_metrics.sort(key=lambda x: x[2])

    names = [b for b, _, _ in breed_metrics]
    precisions = [p for _, p, _ in breed_metrics]
    recalls = [r for _, _, r in breed_metrics]

    y = np.arange(len(names))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(y - bar_height / 2, precisions, bar_height, label="Precision", color=sns.color_palette("muted")[0])
    ax.barh(y + bar_height / 2, recalls, bar_height, label="Recall", color=sns.color_palette("muted")[2])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-Class Precision and Recall")
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = GRAPHS_DIR / "per_class_precision_recall.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_model_size_comparison(comparison_data):
    if comparison_data is None:
        return
    _ensure_graphs_dir()

    labels = ["PyTorch (.pth)", "ONNX (.onnx)"]
    sizes = [comparison_data["pt_mb"], comparison_data["ox_mb"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [sns.color_palette("muted")[0], sns.color_palette("muted")[1]]
    bars = ax.bar(labels, sizes, color=colors, width=0.5)
    ax.set_ylabel("File Size (MB)")
    ax.set_title("Model File Size Comparison")

    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f} MB", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    path = GRAPHS_DIR / "model_size_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_latency_comparison(comparison_data):
    if comparison_data is None:
        return
    _ensure_graphs_dir()

    labels = ["PyTorch (CPU)", "ONNX Runtime (CPU)"]
    latencies = [comparison_data["pt_time_ms"], comparison_data["ort_time_ms"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [sns.color_palette("muted")[0], sns.color_palette("muted")[1]]
    bars = ax.bar(labels, latencies, color=colors, width=0.5)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Comparison (CPU, avg 100 runs)")

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f} ms", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    path = GRAPHS_DIR / "latency_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_metrics_summary(metrics_dict):
    _ensure_graphs_dir()

    metric_names = ["Accuracy", "F1 (Macro)", "F1 (Weighted)", "Precision (Macro)", "Recall (Macro)"]
    metric_vals = [
        metrics_dict["acc"],
        metrics_dict["f1_macro"],
        metrics_dict["f1_weighted"],
        metrics_dict["prec_macro"],
        metrics_dict["rec_macro"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("viridis", n_colors=len(metric_names))
    bars = ax.bar(metric_names, metric_vals, color=colors, width=0.55)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Aggregate Evaluation Metrics")

    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    path = GRAPHS_DIR / "metrics_summary.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_top_bottom_classes(cls_report, label_map, top_n=10, bottom_n=10):
    _ensure_graphs_dir()
    breed_f1s = []
    for name in cls_report:
        if name in ("accuracy", "macro avg", "weighted avg"):
            continue
        breed_f1s.append((name, cls_report[name]["f1-score"], int(cls_report[name]["support"])))
    breed_f1s.sort(key=lambda x: -x[1])

    top = breed_f1s[:top_n]
    bottom = breed_f1s[-bottom_n:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    names_t = [b for b, _, _ in top]
    scores_t = [s for _, s, _ in top]
    support_t = [n for _, _, n in top]
    ax1.barh(names_t[::-1], scores_t[::-1], color=sns.color_palette("Greens_d", n_colors=top_n))
    ax1.set_xlabel("F1 Score")
    ax1.set_xlim(0, 1.05)
    ax1.set_title(f"Top {top_n} Classes by F1")
    for i, (s, n) in enumerate(zip(scores_t[::-1], support_t[::-1])):
        ax1.text(s + 0.005, i, f"{s:.3f} (n={n})", va="center", fontsize=8)

    names_b = [b for b, _, _ in bottom]
    scores_b = [s for _, s, _ in bottom]
    support_b = [n for _, _, n in bottom]
    ax2.barh(names_b[::-1], scores_b[::-1], color=sns.color_palette("Reds_d", n_colors=bottom_n))
    ax2.set_xlabel("F1 Score")
    ax2.set_xlim(0, 1.05)
    ax2.set_title(f"Bottom {bottom_n} Classes by F1")
    for i, (s, n) in enumerate(zip(scores_b[::-1], support_b[::-1])):
        ax2.text(s + 0.005, i, f"{s:.3f} (n={n})", va="center", fontsize=8)

    plt.tight_layout()
    path = GRAPHS_DIR / "top_bottom_classes.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def generate_all_graphs(model, label_map, metrics_dict=None, comparison_data=None):
    _ensure_graphs_dir()
    print(f"\n{'='*55}")
    print(f"generating graphs -> {GRAPHS_DIR}")
    print(f"{'='*55}")

    print(f"  [1/9] dataset distribution")
    plot_dataset_distribution(label_map)

    print(f"  [2/9] parameter distribution")
    plot_parameter_distribution(model)

    print(f"  [3/9] layer type distribution")
    plot_layer_type_distribution(model)

    if metrics_dict is not None:
        cls_report = metrics_dict["cls_report"]
        all_labels = metrics_dict["labels"]
        all_preds = metrics_dict["preds"]

        print(f"  [4/9] confusion matrix")
        plot_confusion_matrix(all_labels, all_preds, label_map)

        print(f"  [5/9] normalised confusion matrix")
        plot_normalised_confusion_matrix(all_labels, all_preds, label_map)

        print(f"  [6/9] per-class F1 scores")
        plot_per_class_f1(cls_report, label_map)

        print(f"  [7/9] per-class precision & recall")
        plot_per_class_precision_recall(cls_report, label_map)

        print(f"  [8/9] metrics summary")
        plot_metrics_summary(metrics_dict)

        print(f"  [9/9] top/bottom classes")
        plot_top_bottom_classes(cls_report, label_map)
    else:
        print(f"  [4-9] skipped | no metrics data -> use --eval")

    if comparison_data is not None:
        print(f"  [+] model size comparison")
        plot_model_size_comparison(comparison_data)
        print(f"  [+] latency comparison")
        plot_latency_comparison(comparison_data)

    print(f"{'='*55}")
    print(f"all graphs saved -> {GRAPHS_DIR}")


def _compute_eval_metrics(model, label_map, num_classes):
    from torch.utils.data import DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import cv2

    BASE_DIR = SCRIPT_DIR / "dataset" / "Indian_bovine_breeds"

    if not CSV_PATH.exists():
        print(f"  csv not found -> skip eval")
        return None

    df = pd.read_csv(CSV_PATH)
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    df["full_path"] = df["path"].apply(lambda p: str(BASE_DIR / p))
    valid_mask = df["full_path"].apply(lambda p: os.path.isfile(p))
    df = df[valid_mask].reset_index(drop=True)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["breed"])

    _, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_df = val_df.reset_index(drop=True)

    val_transform = A.Compose([
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    class _ValDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, transform):
            self.df = dataframe.reset_index(drop=True)
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = cv2.imread(row["full_path"])
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=img)
            return transformed["image"], int(row["label"])

    val_dataset = _ValDataset(val_df, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model.to(DEVICE)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    target_names = [label_map[str(i)] for i in range(num_classes)]
    cls_report = classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True
    )

    return {
        "acc": (all_preds == all_labels).mean(),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "prec_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "rec_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "cls_report": cls_report,
        "preds": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-only", action="store_true")
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--graphs-only", action="store_true")
    args = parser.parse_args()

    if not LABEL_MAP_PATH.exists():
        print(f"label map not found -> {LABEL_MAP_PATH}")
        print(f"run train_pipeline.py first")
        sys.exit(1)

    label_map = load_label_map()
    num_classes = len(label_map)
    print(f"loaded {num_classes} classes from {LABEL_MAP_PATH}")

    model = None
    comparison_data = None
    metrics_dict = None

    if not args.graphs_only:
        if args.pytorch_only or (not args.onnx_only and not args.compare):
            if BEST_MODEL_PATH.exists():
                model = pytorch_info(BEST_MODEL_PATH, num_classes)
            else:
                print(f"pytorch ckpt not found -> {BEST_MODEL_PATH}")

        if args.onnx_only or (not args.pytorch_only and not args.compare):
            if ONNX_MODEL_PATH.exists():
                onnx_info(ONNX_MODEL_PATH, num_classes)
            else:
                print(f"onnx model not found -> {ONNX_MODEL_PATH}")

        if args.compare or (not args.pytorch_only and not args.onnx_only):
            comparison_data = compare_models(num_classes)

    if (args.graphs or args.graphs_only) and model is None and BEST_MODEL_PATH.exists():
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
        ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()

    if (args.eval or args.graphs or args.graphs_only) and model is not None:
        print(f"\ncomputing val metrics")
        metrics_dict = _compute_eval_metrics(model, label_map, num_classes)
        if metrics_dict is not None:
            print(f"  accuracy        {metrics_dict['acc']:.4f}")
            print(f"  f1 macro        {metrics_dict['f1_macro']:.4f}")
            print(f"  f1 weighted     {metrics_dict['f1_weighted']:.4f}")

    if (args.graphs or args.graphs_only) and comparison_data is None:
        if os.path.exists(BEST_MODEL_PATH) and os.path.exists(ONNX_MODEL_PATH):
            comparison_data = compare_models(num_classes)

    if args.graphs or args.graphs_only:
        if model is not None:
            generate_all_graphs(model, label_map, metrics_dict=metrics_dict, comparison_data=comparison_data)
        else:
            print(f"no model loaded -> cannot gen graphs")


if __name__ == "__main__":
    main()

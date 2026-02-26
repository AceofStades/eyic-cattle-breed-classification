import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import timm
import onnx
import onnxruntime as ort
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

SCRIPT_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = SCRIPT_DIR / "best_breed_classifier.pth"
ONNX_MODEL_PATH = SCRIPT_DIR / "breed_classifier.onnx"
LABEL_MAP_PATH = SCRIPT_DIR / "breed_labels.json"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_map():
    with open(LABEL_MAP_PATH, "r") as f:
        return json.load(f)


def pytorch_info(model_path, num_classes):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
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
    print(f"  arch            efficientnet_b0")
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
        np.prod(i.dims) * np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[i.data_type]).itemsize
        for i in onnx_model.graph.initializer
    )
    print(f"  param bytes     {total_param_bytes / (1024*1024):.2f} mb")

    for inp in onnx_model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param)
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(inp.type.tensor_type.elem_type, "unknown")
        print(f"  input           {inp.name} | shape {shape} | dtype {dtype}")

    for out in onnx_model.graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param)
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(out.type.tensor_type.elem_type, "unknown")
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

    pytorch_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
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

    pytorch_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
    pytorch_model.load_state_dict(ckpt["model_state_dict"])
    pytorch_model.eval()

    import time

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-only", action="store_true")
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if not LABEL_MAP_PATH.exists():
        print(f"label map not found -> {LABEL_MAP_PATH}")
        print(f"run train_pipeline.py first")
        sys.exit(1)

    label_map = load_label_map()
    num_classes = len(label_map)
    print(f"loaded {num_classes} classes from {LABEL_MAP_PATH}")

    if args.pytorch_only or (not args.onnx_only and not args.compare):
        if BEST_MODEL_PATH.exists():
            pytorch_info(BEST_MODEL_PATH, num_classes)
        else:
            print(f"pytorch ckpt not found -> {BEST_MODEL_PATH}")

    if args.onnx_only or (not args.pytorch_only and not args.compare):
        if ONNX_MODEL_PATH.exists():
            onnx_info(ONNX_MODEL_PATH, num_classes)
        else:
            print(f"onnx model not found -> {ONNX_MODEL_PATH}")

    if args.compare or (not args.pytorch_only and not args.onnx_only):
        compare_models(num_classes)


if __name__ == "__main__":
    main()

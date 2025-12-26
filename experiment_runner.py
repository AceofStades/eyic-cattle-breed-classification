import json
import os
import time
from pathlib import Path

from model import (
    CONFIG,
    build_model,
    get_transforms,
    get_dataloaders,
    train_loop,
    save_model,
)
import torch
import torch.nn as nn
import torch.optim as optim

def run_experiment(
    backbone="mobile",
    output_dir="experiments",
    use_amp=True,
    use_alb=True,
    mixup=0.2,
    warmup_epochs=1,
    main_epochs=3,
    batch_size=64,
    max_lr=5e-4,
):
    os.makedirs(output_dir, exist_ok=True)
    exp_name = f"exp_{backbone}_{int(time.time())}"
    out_path = Path(output_dir) / exp_name
    out_path.mkdir(parents=True, exist_ok=True)

    CONFIG["BATCH_SIZE"] = batch_size
    CONFIG["WARMUP_EPOCHS"] = warmup_epochs
    CONFIG["MAIN_EPOCHS"] = main_epochs

    print(f"startin exp: {exp_name}")

    train_tf, val_tf = get_transforms(CONFIG["IMG_SIZE"][0], use_alb=use_alb)
    dataloaders, num_classes = get_dataloaders(CONFIG["DATA_DIR"], train_tf, val_tf)

    try:
        model = build_model(num_classes, backbone=backbone)
    except Exception as e:
        print(f"fked up{backbone}: {e}")
        return {"error": str(e)}

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    head_params = [p for n, p in model.named_parameters() if "classifier" in n or "head" in n or "linear" in n]
    if len(head_params) == 0:
        head_params = model.parameters()

    optimizer_warmup = optim.AdamW(head_params, lr=1e-3, weight_decay=0.05)
    model = train_loop(
        model,
        dataloaders,
        criterion,
        optimizer_warmup,
        scheduler=None,
        num_epochs=warmup_epochs,
        phase_name="Warmup",
        use_amp=use_amp,
        mixup_alpha=0.0,
    )

    for param in model.parameters():
        param.requires_grad = True

    body_params = [p for n, p in model.named_parameters() if not ("classifier" in n or "head" in n or "linear" in n)]
    head_params = [p for n, p in model.named_parameters() if ("classifier" in n or "head" in n or "linear" in n)]

    optimizer_main = optim.AdamW(
        [
            {"params": body_params, "lr": 5e-5},
            {"params": head_params, "lr": max_lr},
        ],
        weight_decay=0.02,
    )

    steps_per_epoch = len(dataloaders["train"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer_main,
        max_lr=max_lr,
        epochs=main_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    model = train_loop(
        model,
        dataloaders,
        criterion,
        optimizer_main,
        scheduler,
        num_epochs=main_epochs,
        phase_name="Main",
        use_amp=use_amp,
        mixup_alpha=mixup,
    )

    model_name = f"breed_classifier_{backbone}"
    save_model(model, name=str(out_path / model_name))

    summary = {
        "exp_name": exp_name,
        "backbone": backbone,
        "use_amp": use_amp,
        "use_alb": use_alb,
        "mixup": mixup,
        "warmup_epochs": warmup_epochs,
        "main_epochs": main_epochs,
        "batch_size": batch_size,
        "model_path": str(out_path / f"{model_name}.pth"),
    }

    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"exp {exp_name} fk. see{out_path}")
    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="tung tung sahur")
    parser.add_argument("--backbones", nargs="+", default=["mobile", "efficientnet_b3"])
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--use-alb", action="store_true")
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--main-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=CONFIG["BATCH_SIZE"])
    parser.add_argument("--max-lr", type=float, default=5e-4)
    args = parser.parse_args()

    results = []
    for b in args.backbones:
        res = run_experiment(
            backbone=b,
            output_dir="experiments",
            use_amp=args.use_amp,
            use_alb=args.use_alb,
            mixup=args.mixup,
            warmup_epochs=args.warmup_epochs,
            main_epochs=args.main_epochs,
            batch_size=args.batch_size,
            max_lr=args.max_lr,
        )
        results.append(res)

    print("bombastic")
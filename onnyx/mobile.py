import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import timm
from torchvision import models
import os
import glob

NUM_CLASSES = 41
IMG_SIZE = 224

try:

    exp_dir = sorted(glob.glob("experiments/exp_mobile_*"))[-1]
    MODEL_PATH = os.path.join(exp_dir, "breed_classifier_mobile.pth")
    ONNX_PATH = os.path.join(exp_dir, "breed_classifier_mobile.onnx")
    print(f"Targeting Experiment: {exp_dir}")
except IndexError:
    exit()

def build_mobile_model(num_classes):
    try:

        model = timm.create_model("mobilenetv3_large_100", pretrained=False)
        model.reset_classifier(num_classes)
    except:

        model = models.mobilenet_v3_large()
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(num_ftrs, num_classes)
        )
    return model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    device = torch.device("cpu") 

    if not os.path.exists(MODEL_PATH):
        return

    model = build_mobile_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, requires_grad=True).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17, 

        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("\n berify")
    ort_session = onnxruntime.InferenceSession(ONNX_PATH)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    torch_out = model(dummy_input)

    try:
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-02)
        print("yeah gtg")
    except AssertionError:
        print("hmm")

if __name__ == "__main__":
    main()
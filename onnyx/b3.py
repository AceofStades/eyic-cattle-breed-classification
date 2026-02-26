import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import timm
from torchvision import models

NUM_CLASSES = 41
IMG_SIZE = 224
MODEL_PATH = "experiments/exp_efficientnet_b3_1766733994/breed_classifier_efficientnet_b3.pth"
ONNX_PATH = "experiments/exp_efficientnet_b3_1766733994/breed_classifier.onnx"

def build_model(num_classes):
    try:
        model = timm.create_model("efficientnet_b3", pretrained=False)
        model.reset_classifier(num_classes)
    except:
        model = models.efficientnet_b3()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():

    device = torch.device("cpu")

    model = build_model(NUM_CLASSES)

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
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    print(f"export {ONNX_PATH}")

    print("\n verify")

    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("graph pass")

    ort_session = onnxruntime.InferenceSession(ONNX_PATH)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    torch_out = model(dummy_input)

    try:
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-02)
    except AssertionError as e:
        print(f"diff error {np.max(np.abs(to_numpy(torch_out) - ort_outs[0]))}")

    torch_cls = np.argmax(to_numpy(torch_out))
    onnx_cls = np.argmax(ort_outs[0])

    print(f"ptorch class: {torch_cls}")
    print(f"onnx class: {onnx_cls}")

    if torch_cls == onnx_cls:
        print("yeah gtg")
    else:
        print("kys")

if __name__ == "__main__":
    main()
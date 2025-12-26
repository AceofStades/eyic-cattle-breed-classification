import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

INPUT_MODELS = {

    "EfficientNet": "../experiments/exp_efficientnet_b3_1766733994/breed_classifier.onnx",
    "MobileNet":    "../experiments/exp_mobile_1766733609/breed_classifier_mobile.onnx"
}

def get_size(path):
    if not os.path.exists(path):
        return 0
    size_in_bytes = os.path.getsize(path)
    return size_in_bytes / (1024 * 1024)

def quantize_model(model_name, input_path):
    if not os.path.exists(input_path):
        return


    base_name, ext = os.path.splitext(input_path)
    preprocessed_path = f"{base_name}_preprocessed{ext}"
    output_path = f"{base_name}_int8{ext}"

    try:
        quant_pre_process(
            input_model_path=input_path,
            output_model_path=preprocessed_path,
            skip_symbolic_shape=False
        )
    except Exception as e:
        print(f"   how {e}")

    try:
        quantize_dynamic(
            model_input=preprocessed_path,  

            model_output=output_path,
            weight_type=QuantType.QUInt8    

        )
        print(f"   size: {get_size(output_path):.2f} MB (Reduction: {get_size(input_path) / get_size(output_path):.1f}x)")
    except Exception as e:
        print(f"   wtf {e}")

    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

if __name__ == "__main__":

    for name, path in INPUT_MODELS.items():
        if not os.path.exists(path):
             print(f"wrng pth {path}")

    for name, path in INPUT_MODELS.items():
        quantize_model(name, path)
import os
import cv2
import onnxruntime as ort
from rembg import remove, new_session

# Function to check ONNX Runtime execution provider
def get_onnx_runtime_device():
    try:
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            print("✅ Using GPU (CUDA) for ONNX Runtime.")
            return "CUDAExecutionProvider"
        else:
            print("⚠️ GPU not available. Falling back to CPU.")
            return "CPUExecutionProvider"
    except Exception as e:
        print(f"Error checking ONNX Runtime: {e}")
        return "CPUExecutionProvider"

# Get ONNX execution provider (GPU or CPU)
execution_provider = get_onnx_runtime_device()

import os
import cv2
import numpy as np
import torch

try:
    from realesrgan import RealESRGANer
except ImportError:
    raise ImportError("Please install the realesrgan package (pip install realesrgan) and basicsr-fixed (pip install basicsr-fixed)")

def upscale_image(cv_image: np.ndarray) -> np.ndarray:
    """
    Upscales the given image using RealESRGAN_x4plus.
    Expects a BGR OpenCV image.
    Returns the upscaled image (BGR).

    This function loads the x4+ model (scale=4) and applies the model's default scaling.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct the path to the weights file for the x4+ model.
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    weights_path = os.path.join(base_dir, "RealESRGAN_x4plus.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"RealESRGAN weights not found at {weights_path}")

    # Initialize the model with scale=4.
    model = RealESRGANer(
        scale=4,
        model_path=weights_path,
        dni_weight=None,
        device=device
    )

    # Convert image from BGR (OpenCV) to RGB as expected by the model.
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Enhance/upscale the image; using outscale=1 yields the model's default (4× upscale).
    sr_image, _ = model.enhance(image_rgb, outscale=1)

    # Convert back from RGB to BGR.
    sr_image_bgr = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    return sr_image_bgr

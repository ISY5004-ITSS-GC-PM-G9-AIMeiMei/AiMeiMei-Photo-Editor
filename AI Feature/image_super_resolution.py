import os
import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGAN
import requests


def download_model(model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "RealESRGAN_x4.pth")

    if not os.path.exists(model_path):
        print("Downloading Real-ESRGAN model weights...")
        url = "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model weights downloaded successfully!")
    else:
        print("Model weights already exist.")

    return model_path


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = download_model()
    model = RealESRGAN(device, scale=4)
    model.load_weights(model_path)
    print("Real-ESRGAN model loaded successfully!")
    return model


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def enhance_resolution(model, image_path, output_dir):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")

    print(f"Processing image: {image_path}")
    input_file_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create output directory
    output_root = "upscaled_result"
    output_folder = os.path.join(output_root, input_file_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory created: {output_folder}")

    # Load and upscale image
    image = load_image(image_path)
    upscaled_image = model.predict(np.array(image))
    upscaled_image = Image.fromarray(upscaled_image)

    # Save upscaled image
    output_path = os.path.join(output_folder, f"{input_file_name}_upscaled.png")
    upscaled_image.save(output_path)
    print(f"Upscaled image saved to: {output_path}")


if __name__ == "__main__":
    model = load_model()
    input_image_path = "test/group.jpg"  # Change this to your input file
    enhance_resolution(model, input_image_path, "upscaled_result")

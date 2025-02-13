import os
import torch
import numpy as np
from PIL import Image
import cv2  # Required for OpenCV processing


checkpoint = torch.load("models/BSRGANx4.pth")
print(type(checkpoint))


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("models", "BSRGANx4.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights '{model_path}' not found. Please upload the model file.")

    model = torch.load(model_path, map_location=device)
    model.eval()

    print("BSRGAN model loaded successfully!")
    return model, device


def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image '{image_path}': {e}")


def enhance_resolution(model, device, image_path, output_dir):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")

    print(f"Processing image: {image_path}")
    input_file_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create output directory
    output_folder = os.path.join(output_dir, input_file_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory created: {output_folder}")

    # Load and process image
    image = load_image(image_path)
    img_np = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(img_tensor)

    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)

    # Convert back to PIL and save
    upscaled_image = Image.fromarray(output_np)
    output_path = os.path.join(output_folder, f"{input_file_name}_upscaled.png")
    upscaled_image.save(output_path)
    print(f"Upscaled image saved to: {output_path}")


if __name__ == "__main__":
    model, device = load_model()


    input_image_path = "test/group.jpg"  # Change this to your input file
    enhance_resolution(model, device, input_image_path, "upscaled_result")

import os
import cv2
import numpy as np
import onnxruntime as ort
from rembg import remove, new_session
import requests


def check_onnx_runtime():
    """Check and print the ONNX Runtime device."""
    try:
        device = ort.get_device()
        print(f"ONNX Runtime is using: {device}")
    except Exception as e:
        print(f"Error checking ONNX Runtime: {e}")


def load_image(image_path):
    """Load an image from a file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Failed to load the image. Please check the file path and format.")
    return image


def preprocess_image(image):
    """Apply preprocessing techniques to improve segmentation."""
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    image_sharpened = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharpened


def create_output_directory(output_root, filename):
    """Create an output directory for saving results."""
    output_dir = os.path.join(output_root, filename)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def download_u2net(model_dir):
    """Download the U²-Net model to the specified directory if not already present."""
    model_path = os.path.join(model_dir, "u2net.onnx")
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        print("Downloading U²-Net model...")
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("U²-Net model downloaded successfully!")
    return model_path


def initialize_u2net(model_dir="models/"):
    """Initialize the U²-Net model and store it in the specified directory."""
    model_path = download_u2net(model_dir)
    return new_session(model_path)


def segment_image(image, session):
    """Perform human segmentation using U²-Net."""
    return remove(image, session=session)


def save_images(output_dir, filename, human_rgba, original_image):
    """Save segmented human image and transparent background."""
    human_filename = os.path.join(output_dir, f"{filename}_human.png")
    background_filename = os.path.join(output_dir, f"{filename}_background.png")

    # Extract alpha channel (mask)
    alpha_channel = human_rgba[:, :, 3]

    # Create transparent background
    background_rgba = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    background_rgba[alpha_channel > 0] = (0, 0, 0, 0)

    # Save images
    cv2.imwrite(human_filename, human_rgba)
    cv2.imwrite(background_filename, background_rgba)
    print(f"Saved: {human_filename}")
    print(f"Saved: {background_filename}")


def main():
    check_onnx_runtime()
    image_path = "test/group.jpg"  # Change to your image path
    image = load_image(image_path)
    processed_image = preprocess_image(image)

    input_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_root = "segresult"
    output_dir = create_output_directory(output_root, input_filename)

    print("Initializing U²-Net Best Accuracy Model...")
    session = initialize_u2net()

    print("Running U²-Net for segmentation...")
    human_rgba = segment_image(processed_image, session)
    print("Segmentation completed successfully.")

    save_images(output_dir, input_filename, human_rgba, image)
    print(f"✅ Segmentation completed! Files saved in: {output_dir}")


if __name__ == "__main__":
    main()

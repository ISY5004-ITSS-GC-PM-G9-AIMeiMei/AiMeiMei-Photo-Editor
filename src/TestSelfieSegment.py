import os
import cv2
import onnxruntime as ort
import numpy as np
import requests

def check_onnx_runtime():
    """Check ONNX Runtime device and fallback to CPU if GPU is unavailable."""
    try:
        device = ort.get_device()
        print(f"ONNX Runtime is using: {device}")
        return device
    except Exception as e:
        print(f"Error checking ONNX Runtime: {e}. Defaulting to CPU.")
        return "CPU"


def load_image(image_path):
    """Load an image from a file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Failed to load the image. Please check the file path and format.")
    return image


def preprocess_image(image, target_size=(320, 320)):
    """Resize and normalize image for U²-Net model."""
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    image_resized = cv2.resize(image, target_size)  # Resize to 320x320 for U²-Net
    image_normalized = image_resized.astype(np.float32) / 255.0  # Normalize to [0,1]
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # Convert to (C, H, W)
    image_input = np.expand_dims(image_transposed, axis=0)  # Add batch dimension
    return image_input, original_size


def download_u2net(model_dir="models/"):
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
    """Initialize the U²-Net model with ONNX Runtime."""
    model_path = download_u2net(model_dir)
    device = check_onnx_runtime()

    print("Initializing U²-Net model...")
    try:
        providers = ["CUDAExecutionProvider"] if device == "GPU" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"U²-Net initialized successfully on {device}")
    except Exception as e:
        print(f"Error initializing U²-Net: {e}. Using CPU instead.")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    return session


def run_inference(image, session):
    """Run ONNX model inference on the input image."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image})[0]  # Run inference
    return output.squeeze(0)  # Remove batch dimension


def postprocess_mask(mask, original_size):
    """Post-process model output mask to binary format."""
    mask_resized = cv2.resize(mask[0], original_size)  # Resize to original image size
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255  # Convert to binary mask
    return mask_binary


def apply_mask(image, mask):
    """Apply the mask to the original image for transparent background effect."""
    rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # Convert to RGBA
    rgba_image[:, :, 3] = mask  # Apply mask to alpha channel
    return rgba_image


def save_images(output_dir, filename, segmented_image, mask):
    """Save segmented images and mask."""
    os.makedirs(output_dir, exist_ok=True)

    human_filename = os.path.join(output_dir, f"{filename}_human.png")
    mask_filename = os.path.join(output_dir, f"{filename}_mask.png")

    cv2.imwrite(human_filename, segmented_image)
    cv2.imwrite(mask_filename, mask)

    print(f"Saved: {human_filename}")
    print(f"Saved: {mask_filename}")


def run_segmentation():
    image_path = "test/2_people_together.jpg"  # Change to your image path
    image = load_image(image_path)
    image_input, original_size = preprocess_image(image)

    input_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_root = "segresult"
    output_dir = os.path.join(output_root, input_filename)

    session = initialize_u2net()

    print("Running U²-Net for segmentation...")
    mask = run_inference(image_input, session)
    mask_binary = postprocess_mask(mask, original_size)

    segmented_image = apply_mask(image, mask_binary)

    save_images(output_dir, input_filename, segmented_image, mask_binary)
    print(f"✅ Segmentation completed! Files saved in: {output_dir}")


if __name__ == "__main__":
    run_segmentation()

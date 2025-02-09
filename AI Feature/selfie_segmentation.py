import os
import cv2
import onnxruntime as ort
import requests
from rembg import remove, new_session  # Using U^2-Net for better segmentation


def check_onnx_runtime():
    """Check if ONNX Runtime is properly initialized."""
    try:
        device = ort.get_device()
        print(f"ONNX Runtime is using: {device}")
    except Exception as e:
        print(f"Error checking ONNX Runtime: {e}")


def download_model(model_dir="models", model_name="u2net.onnx"):
    """Download U²-Net model if not already available."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        print("Downloading U²-Net model...")
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("U²-Net model downloaded successfully!")
    else:
        print("U²-Net model already exists.")

    return model_path


def process_image(image_path):
    """Load and preprocess image for segmentation."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")

    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Failed to load the image. Please check the file path and format.")

    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    image_sharpened = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_rgba, image_sharpened


def save_results(output_dir, input_file_name, human_rgba, background_rgba):
    """Save segmentation results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    background_filename = os.path.join(output_dir, f"{input_file_name}_background.png")
    human_filename = os.path.join(output_dir, f"{input_file_name}_human.png")

    cv2.imwrite(background_filename, background_rgba)
    print(f"Saved refined background image: {background_filename}")

    cv2.imwrite(human_filename, human_rgba)
    print(f"Saved full human segmentation: {human_filename}")


def segment_objects(image_path, output_root="segresult"):
    """Perform object segmentation using U²-Net."""
    check_onnx_runtime()
    model_path = download_model()
    session = new_session(model_path)

    input_file_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(output_root, input_file_name)

    image_rgba, image_sharpened = process_image(image_path)
    print("Running U²-Net for segmentation...")
    human_rgba = remove(image_sharpened, session=session).copy()
    print("Segmentation completed successfully.")

    alpha_channel = human_rgba[:, :, 3]
    background_rgba = image_rgba.copy()
    background_rgba[alpha_channel > 0] = (0, 0, 0, 0)

    save_results(output_dir, input_file_name, human_rgba, background_rgba)
    print(f"✅ Segmentation completed! Files saved in: {output_dir}")


if __name__ == "__main__":
    input_image_path = "test/group.jpg"  # Change this to your input file
    segment_objects(input_image_path)

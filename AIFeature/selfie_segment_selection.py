import os
import sys
import cv2
from rembg import remove, new_session

def initialize_u2net(model_dir="models/"):
    """Initialize the U²-Net model and store it in the specified directory."""
    model_path = os.path.join(model_dir, "u2net.onnx")
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        print("Downloading U²-Net model...")
        import requests
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("U²-Net model downloaded successfully!")

    return new_session(model_path)


def segment_image(image_path, output_path, session):
    """Perform human segmentation and save the mask."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Error: Could not load image {image_path}")

    # Perform segmentation
    human_rgba = remove(image, session=session)

    # Extract only the alpha channel as the selection mask
    alpha_channel = human_rgba[:, :, 3]

    # Save selection mask
    cv2.imwrite(output_path, alpha_channel)
    print(f"Saved segmentation mask: {output_path}")


def main():
    """Main function to segment an image from command-line arguments."""
    if len(sys.argv) < 3:
        print("Usage: selfie_segment_selection.py <input_image> <output_mask>")
        sys.exit(1)

    input_image = sys.argv[1]
    output_mask = sys.argv[2]

    print("Initializing U²-Net model...")
    session = initialize_u2net()

    print("Running segmentation...")
    segment_image(input_image, output_mask, session)


if __name__ == "__main__":
    main()

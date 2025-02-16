import cv2
import os
import torch
import numpy as np
from realesrgan import RealESRGANer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from scipy.ndimage import label
from PIL import Image
from anomalib.models import Patchcore

# ================================
# 🔹 Auto-Detect Device (GPU or CPU)
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 Using {device.upper()} for AI models...")

# 🔹 URL to download LaMa model (optional)
LAMA_MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"

# ================================
# 🔹 LaMa Inpainting Class
# ================================
class SimpleLama:
    def __init__(self, device=None):
        """Load LaMa model manually without external dependencies."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.environ.get("LAMA_MODEL", "models/big-lama.pt")  # Change path if needed

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ LaMa model not found at {model_path}. Please download it.")

        print(f"🔄 Loading LaMa model on {self.device.upper()}...")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval().to(self.device)
        print(f"✅ LaMa model loaded successfully!")

    def preprocess(self, image, mask):
        """Convert images to tensors for LaMa processing."""
        if isinstance(image, Image.Image):  # Ensure image is PIL format
            image = np.array(image.convert("RGB"))  # Convert to RGB to avoid Pillow issues
        if isinstance(mask, Image.Image):  # Ensure mask is grayscale
            mask = np.array(mask.convert("L"))

        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float() / 255.0

        return image.to(self.device), mask.to(self.device)

    def __call__(self, image, mask):
        """Perform LaMa inpainting."""
        image_tensor, mask_tensor = self.preprocess(image, mask)

        with torch.no_grad():
            inpainted = self.model(image_tensor, mask_tensor)

        # Convert back to image format
        result = inpainted[0].permute(1, 2, 0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(result)

# ================================
# 🔹 Load AI Models on Detected Device
# ================================

def load_duconet_model(model_path):
    """Loads DucoNet model on GPU if available, otherwise CPU."""
    print("🔄 Loading DucoNet model...")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    print(f"✅ DucoNet model loaded successfully on {device.upper()}")
    return model

def load_models():
    """Loads all models on the detected device (GPU or CPU)."""
    print("🔄 Loading all AI models...")

    # Load LaMa using SimpleLama class
    lama_model = SimpleLama(device=device)

    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint").to(device)
    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)

    # Load Real-ESRGAN
    esrgan = RealESRGANer(model_path="models/RealESRGAN_x8.pth")

    # Load PatchCore (Replaces FastFlow)
    patchcore = Patchcore(task="segmentation").to(device)

    # Load DucoNet
    duconet = load_duconet_model("models/DucoNet1024.pth")

    print("✅ All AI models loaded successfully!")
    return lama_model, controlnet_pipe, esrgan, patchcore, duconet

# ================================
# 🔹 Image Processing Functions
# ================================
def convert_to_png(image_path):
    """Converts any image format to PNG, adding an alpha channel if needed."""
    img = Image.open(image_path).convert("RGBA")
    png_path = os.path.splitext(image_path)[0] + ".png"
    img.save(png_path)
    return png_path

def extract_hole_mask(image_path):
    """Extracts hole mask from PNG transparency or detects artificial holes in JPGs."""
    if not image_path.lower().endswith(".png"):
        print("⚠ Converting image to PNG for transparency support...")
        image_path = convert_to_png(image_path)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 4:  # PNG with alpha channel
        print("✅ Transparent PNG detected, extracting hole mask from alpha channel.")
        hole_mask = 255 - image[:, :, 3]  # Invert alpha: Transparent (0) → Hole (255)
    else:
        print("⚠ No transparency found, applying smart hole detection for JPG conversion.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        binary_mask = cv2.inRange(gray, 240, 255) + cv2.inRange(gray, 0, 15)
        combined_mask = cv2.bitwise_or(binary_mask, edges)
        mask_image = np.zeros_like(image, dtype=np.uint8)
        mask_image[:, :, 3] = 255 - combined_mask
        png_hole_mask = image_path.replace(".png", "_mask.png")
        cv2.imwrite(png_hole_mask, mask_image)
        hole_mask = combined_mask

    return hole_mask, image_path

# ================================
# 🔹 Repair & Enhance Function
# ================================
def repair_and_enhance(image_path, reference_paths=[]):
    """Converts input to PNG, detects holes, fixes them, and enhances quality."""

    # Ensure PNG format before processing
    hole_mask, png_image_path = extract_hole_mask(image_path)
    labeled_mask, num_features = label(hole_mask > 0)
    hole_sizes = {i: np.sum(labeled_mask == i) for i in range(1, num_features + 1)}

    image = cv2.imread(png_image_path, cv2.IMREAD_UNCHANGED)

    # Process each hole separately
    for hole_label, hole_size in hole_sizes.items():
        hole_mask_single = (labeled_mask == hole_label).astype(np.uint8) * 255

        if hole_size < 0.1 * (hole_mask.shape[0] * hole_mask.shape[1]):  # Small hole
            print(f"🟢 Processing small hole {hole_label} with LaMa")
            image = np.array(lama_model(Image.fromarray(image), Image.fromarray(hole_mask_single)))
        else:  # Large hole
            print(f"🟢 Processing large hole {hole_label} with ControlNet first, then refining with LaMa")
            inpainted_image = controlnet_pipe(prompt="Fill missing area", image=image, mask_image=hole_mask_single).images[0]
            image = np.array(lama_model(Image.fromarray(np.array(inpainted_image)), Image.fromarray(hole_mask_single)))

    # Save the output
    output_file = png_image_path.replace(".png", "_final.png")
    cv2.imwrite(output_file, image)
    print(f"✅ Image repair and enhancement complete! Saved as {output_file}.")

# ================================
# 🔹 Load Models & Run Processing
# ================================
lama_model, controlnet_pipe, esrgan, patchcore, duconet = load_models()
repair_and_enhance("image_with_holes.jpg", reference_paths=["ref1.jpg", "ref2.jpg"])

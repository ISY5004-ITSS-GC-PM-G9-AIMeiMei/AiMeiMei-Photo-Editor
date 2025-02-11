import sys
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from controlnet_aux import OpenposeDetector  # ControlNet for reference-based inpainting
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from torchmetrics.image.fid import FrechetInceptionDistance


class InpaintingApp:
    def __init__(self):
        """
        Initialize the inpainting application.
        - Loads the ControlNet model for inpainting.
        - Initializes the FID metric for realism scoring.
        """
        self.image = None  # Holds the main image
        self.mask = None  # Holds the mask indicating areas to be inpainted
        self.reference_images = []  # List to store reference images
        self.fid_metric = FrechetInceptionDistance(feature=2048)  # Metric for realism assessment
        self.load_model()

    def load_model(self):
        """
        Load the Stable Diffusion inpainting pipeline with ControlNet for guided inpainting.
        """
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        print("ControlNet model loaded successfully!")

    def load_image(self, file_path):
        """
        Load an image file and convert it to RGBA format.
        """
        self.image = Image.open(file_path).convert("RGBA")
        self.mask = Image.new("L", self.image.size, 0)  # Create a blank mask
        print("Image loaded successfully!")

    def load_reference_images(self, file_paths):
        """
        Load multiple reference images for guiding the inpainting process.
        """
        self.reference_images = [Image.open(f).convert("RGB") for f in file_paths]
        print(f"{len(self.reference_images)} reference images loaded!")

    def compute_fid(self, generated_image):
        """
        Compute the FID (Fréchet Inception Distance) score for the generated image
        against the reference images.
        """
        if not self.reference_images:
            print("No reference images loaded for FID calculation.")
            return

        # Convert images to numpy arrays with shape (C, H, W)
        ref_images_np = [np.array(ref).astype(np.uint8).transpose(2, 0, 1) for ref in self.reference_images]
        generated_image_np = np.array(generated_image).astype(np.uint8).transpose(2, 0, 1)

        # Update FID metric with generated and reference images
        self.fid_metric.update(torch.tensor([generated_image_np]), real=False)
        self.fid_metric.update(torch.tensor(ref_images_np), real=True)
        fid_score = self.fid_metric.compute().item()
        print(f"FID Score: {fid_score}")

    def apply_inpainting(self):
        """
        Apply AI-powered inpainting to the loaded image using Stable Diffusion.
        - Uses reference images if available.
        - Computes a mask from transparent areas.
        """
        if self.image:
            image_np = np.array(self.image)
            mask_np = (image_np[:, :, 3] == 0).astype(np.uint8) * 255  # Detect transparent areas for inpainting
            self.mask = Image.fromarray(mask_np, mode='L')

            if self.reference_images:
                # Use ControlNet OpenPose for guiding inpainting with reference structures
                controlnet_processor = OpenposeDetector()
                combined_edges = [controlnet_processor(ref) for ref in self.reference_images]

                prompt = "Fill the missing parts of the image using the reference structures."
                image_input = self.image.convert("RGB")

                result = self.pipe(
                    prompt=prompt,
                    image=image_input,
                    mask_image=self.mask,
                    control_image=combined_edges[0]  # Use first reference edges (Can be improved to blend)
                ).images[0]
            else:
                prompt = "Fill the missing areas naturally."
                result = self.pipe(
                    prompt=prompt,
                    image=self.image.convert("RGB"),
                    mask_image=self.mask
                ).images[0]

            self.image = result.convert("RGBA")
            self.compute_fid(self.image)  # Compute realism score
            print("Inpainting complete!")


if __name__ == "__main__":
    inpainting_app = InpaintingApp()

    # Replace these paths with actual image file paths
    image_path = "path_to_your_image.png"  # Path to input image
    reference_paths = ["ref1.png", "ref2.png"]  # Paths to reference images

    inpainting_app.load_image(image_path)
    inpainting_app.load_reference_images(reference_paths)
    inpainting_app.apply_inpainting()

import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from torchmetrics.image.fid import FrechetInceptionDistance


class InpaintingApp:
    def __init__(self):
        """
        Initialize the inpainting application using ControlNet and Stable Diffusion.
        - Loads the ControlNet model for inpainting.
        - Initializes the FID metric for realism scoring.
        """
        self.image = None  # Holds the main image
        self.mask = None  # Holds the mask indicating areas to be inpainted
        self.fid_metric = FrechetInceptionDistance(feature=2048)  # Metric for realism assessment

        # Ensure required directories exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        self.load_model()

    def load_model(self):
        """
        Load the Stable Diffusion inpainting pipeline with ControlNet.
        """
        model_path = "runwayml/stable-diffusion-inpainting"
        controlnet_path = "lllyasviel/sd-controlnet-canny"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load ControlNet model
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

        # Load Stable Diffusion Inpainting Pipeline
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            model_path,
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        )

        # Optimize Pipeline
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.to(device)

        print(f"ControlNet inpainting model loaded successfully on {device}!")

    def load_image(self, file_path):
        """
        Load an image file and convert it to RGB format.
        """
        if not os.path.exists(file_path):
            print(f"Error: Image file not found at {file_path}")
            self.image = None
            return

        try:
            self.image = Image.open(file_path).convert("RGB")
            print(f"Image loaded successfully! Size: {self.image.size}, Mode: {self.image.mode}")
        except Exception as e:
            print(f"Error loading image: {e}")
            self.image = None

    def create_transparent_mask(self):
        """
        Create a mask from the transparent areas of the image.
        """
        if self.image is None:
            print("Error: No image loaded to create a mask.")
            return

        image_np = np.array(self.image)
        mask_np = (image_np[:, :, 3] == 0).astype(np.uint8) * 255  # Detect transparent areas
        self.mask = Image.fromarray(mask_np, mode='L')

        if np.count_nonzero(mask_np) == 0:
            print("Warning: No transparent areas detected. The mask might be incorrect.")

        print("Mask created successfully!")

    def generate_control_image(self):
        """
        Generate a control image using Canny edge detection.
        """
        if self.image is None:
            print("Error: No image loaded to create a control image.")
            return

        control_image = self.image.convert("L").filter(ImageFilter.FIND_EDGES)
        print("Control image generated successfully!")
        return control_image

    def apply_inpainting(self, output_filename="output.png"):
        """
        Apply AI-powered inpainting to the transparent masked areas only.
        """
        if self.image is None:
            print("Error: No image loaded for inpainting. Please check the image path and try again.")
            return

        self.create_transparent_mask()
        if self.mask is None:
            print("Error: Mask generation failed.")
            return

        control_image = self.generate_control_image()
        if control_image is None:
            print("Error: Control image generation failed.")
            return

        print(f"Debug: Image size before inpainting: {self.image.size}, Mode: {self.image.mode}")
        print(f"Debug: Mask size: {self.mask.size}, Mode: {self.mask.mode}")

        try:
            result = self.pipe(
                prompt="Fill the missing areas naturally in a realistic manner.",
                image=self.image,
                mask_image=self.mask,
                control_image=control_image,
                num_inference_steps=20
            ).images[0]

            if result is None:
                print("Error: Model output is None, something went wrong during inpainting.")
                return

            self.image = result.convert("RGBA")

            # Save the result
            output_path = os.path.join("results", output_filename)
            self.image.save(output_path)
            print(f"Inpainting complete! Output saved to {output_path}")
        except Exception as e:
            print(f"Error during inpainting: {e}")


if __name__ == "__main__":
    inpainting_app = InpaintingApp()

    # Replace these paths with actual image file paths
    image_path = "segresult/2_people_together/2_people_together_background.png"

    inpainting_app.load_image(image_path)
    inpainting_app.apply_inpainting("output.png")

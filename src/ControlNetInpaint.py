import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
import glob
import os

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def load_controlnet():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint").to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch_dtype).to(device)
    return pipe

if __name__ == "__main__":
    pipe = load_controlnet()
    test_image = Image.open("test_image.jpg").convert("RGB")
    test_mask = Image.open("test_mask.jpg").convert("L")
    reference_images = [Image.open(img).convert("RGB") for img in glob.glob("reference_*.jpg")]
    result = pipe(prompt="Fill missing area", image=test_image, mask_image=test_mask, conditioning_image=reference_images).images[0]
    result.save("controlnet_result.jpg")
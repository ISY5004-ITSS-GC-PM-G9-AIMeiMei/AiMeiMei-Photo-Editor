# lama_inpaint.py
import torch
import os
import numpy as np
from PIL import Image

class SimpleLama:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "models/big-lama.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LaMa model not found at {model_path}. Please download it.")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval().to(self.device)

    def __call__(self, image, mask):
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_tensor = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0).float() / 255.0
        image_tensor, mask_tensor = image_tensor.to(self.device), mask_tensor.to(self.device)
        with torch.no_grad():
            inpainted = self.model(image_tensor, mask_tensor)
        result = (inpainted[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)

if __name__ == "__main__":
    model = SimpleLama()
    test_image = Image.open("test_image.jpg").convert("RGB")
    test_mask = Image.open("test_mask.jpg").convert("L")
    result = model(test_image, test_mask)
    result.save("lama_result.jpg")
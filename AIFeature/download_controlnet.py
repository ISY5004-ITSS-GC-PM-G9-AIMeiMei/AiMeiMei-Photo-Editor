import os
from huggingface_hub import snapshot_download

# List of ControlNet models to download
controlnet_models = [
    "lllyasviel/sd-controlnet-canny",
    "lllyasviel/sd-controlnet-depth",
    "lllyasviel/sd-controlnet-hed",
    "lllyasviel/sd-controlnet-mlsd",
    "lllyasviel/sd-controlnet-openpose",
    "lllyasviel/sd-controlnet-scribble",
    "lllyasviel/sd-controlnet-seg"
]

# Directory to save models`
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Download each model
for model in controlnet_models:
    print(f"Downloading {model}...")
    snapshot_download(repo_id=model, cache_dir=model_dir)
    print(f"{model} downloaded successfully!")

print("All ControlNet models are downloaded.")

import requests
import os
import json
import base64
import time

# Function to Read API Key from a Text File
def get_api_key(filename="../../keys/googlevision.txt"):
    """Reads the API key from a text file."""
    with open(filename, "r") as file:
        return file.read().strip()

# Read API Key
GOOGLE_API_KEY = get_api_key()

def encode_image(image_path):
    """Convert an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_similar_images(image_path, num_images=100):  # Changed to 100
    """Uploads an image and retrieves visually similar images from Google Vision API."""

    # Convert image to Base64 (fixes URL access issues)
    base64_image = encode_image(image_path)

    # Prepare API Request
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "requests": [{
            "image": {"content": base64_image},
            "features": [{"type": "WEB_DETECTION"}]
        }]
    }

    # Send Request
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    # Print API Response for Debugging
    print("\n🔍 Full API Response (Debugging Mode):\n")
    print(json.dumps(data, indent=4))

    # Extract Visually Similar Images
    similar_images = []
    try:
        if "responses" in data:
            web_detection = data["responses"][0].get("webDetection", {})
            if "visuallySimilarImages" in web_detection:
                for img in web_detection["visuallySimilarImages"][:num_images]:  # Fetch up to 100 images
                    similar_images.append(img["url"])

        # Print URLs as clickable links
        if similar_images:
            print("\n✅ Found Similar Images (Click the URLs below to test manually):\n")
            for idx, img_url in enumerate(similar_images, start=1):
                print(f"{idx}. {img_url}")  # Clickable in most IDEs
        else:
            print("\n❌ No similar images found in API response.")

    except Exception as e:
        print("Error processing response:", e)

    return similar_images

def download_images(image_urls, uploaded_image_path, save_folder="similar_images"):
    """Downloads images from given URLs into a folder named after the uploaded file."""

    # Get uploaded image filename without extension
    uploaded_filename = os.path.splitext(os.path.basename(uploaded_image_path))[0]

    # Create new folder: similar_images/{uploaded_filename}/
    image_folder = os.path.join(save_folder, uploaded_filename)
    os.makedirs(image_folder, exist_ok=True)

    print("\n📥 Starting Download...\n")

    for idx, img_url in enumerate(image_urls, start=1):
        image_path = os.path.join(image_folder, f"{uploaded_filename}_{idx}.jpg")

        # Retry logic (tries 3 times before skipping)
        for attempt in range(3):
            try:
                print(f"🔄 Downloading ({attempt+1}/3): {img_url}")
                response = requests.get(img_url, stream=True, timeout=10)

                if response.status_code == 200:
                    with open(image_path, "wb") as file:
                        for chunk in response.iter_content(1024):
                            file.write(chunk)
                    print(f"✅ Downloaded: {image_path}")
                    break  # Exit retry loop if successful
                else:
                    print(f"⚠️ Failed to download (Status Code: {response.status_code})")

            except requests.exceptions.RequestException as e:
                print(f"❌ Error: {e} (Attempt {attempt+1}/3)")

            time.sleep(2)  # Wait 2 seconds before retrying

        else:
            print(f"❌ Skipping image {img_url} after 3 failed attempts.")

# Example Usage
uploaded_image = "test/2_people_together.jpg"
similar_images = get_similar_images(uploaded_image)

if similar_images:
    download_images(similar_images, uploaded_image)
else:
    print("\n❌ No similar images found.")
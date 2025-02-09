import requests
import os
import time


# Function to Read Bing API Key from a Text File
def get_bing_api_key(filename="bingsearch.txt"):
    """Reads the Bing API key from a text file."""
    with open(filename, "r") as file:
        return file.read().strip()


# Read Bing API Key
BING_API_KEY = get_bing_api_key()
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/images/search"


def get_similar_images_bing(query, num_images=100):
    """Search for visually similar images using Bing Image Search API."""

    params = {
        "q": query,  # Search query
        "count": num_images,  # Number of images to return
        "imageType": "Photo",
        "safeSearch": "Strict"
    }

    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}

    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    data = response.json()

    # Extract Image URLs
    similar_images = []
    if "value" in data:
        for img in data["value"]:
            similar_images.append(img["contentUrl"])

    # Print URLs
    if similar_images:
        print("\n✅ Found Similar Images from Bing (Click the URLs below to test manually):\n")
        for idx, img_url in enumerate(similar_images, start=1):
            print(f"{idx}. {img_url}")  # Clickable in most IDEs
    else:
        print("\n❌ No similar images found in Bing API response.")

    return similar_images


def download_images(image_urls, query, save_folder="bing_similar_images"):
    """Downloads images from Bing Image Search results."""

    # Create a folder for the query
    query_folder = os.path.join(save_folder, query.replace(" ", "_"))
    os.makedirs(query_folder, exist_ok=True)

    print("\n📥 Starting Download...\n")

    for idx, img_url in enumerate(image_urls, start=1):
        image_path = os.path.join(query_folder, f"{query.replace(' ', '_')}_{idx}.jpg")

        # Retry logic (tries 3 times before skipping)
        for attempt in range(3):
            try:
                print(f"🔄 Downloading ({attempt + 1}/3): {img_url}")
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
                print(f"❌ Error: {e} (Attempt {attempt + 1}/3)")

            time.sleep(2)  # Wait 2 seconds before retrying

        else:
            print(f"❌ Skipping image {img_url} after 3 failed attempts.")


# Example Usage
search_query = "Marina Bay Sands Singapore"  # Change this to your search term
similar_images = get_similar_images_bing(search_query)

if similar_images:
    download_images(similar_images, search_query)
else:
    print("\n❌ No similar images found.")

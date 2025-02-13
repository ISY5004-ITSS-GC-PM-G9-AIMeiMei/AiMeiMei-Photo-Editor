import requests
import exifread
import base64

def get_api_key(filename="../../keys/googlevision.txt"):
    """Reads the API key from a text file."""
    try:
        with open(filename, "r") as file:
            api_key = file.read().strip()
            print(f"[DEBUG] Successfully read API key from {filename}")
            return api_key
    except Exception as e:
        print(f"[ERROR] Unable to read API key file: {e}")
        return None


# Read API Key
GOOGLE_API_KEY = get_api_key()


def encode_image(image_path):
    """Encodes image as base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            print(f"[DEBUG] Successfully encoded image {image_path} as base64.")
            return encoded_image
    except Exception as e:
        print(f"[ERROR] Failed to encode image {image_path}: {e}")
        return None


def get_decimal_coordinates(tags):
    """Extracts and converts GPS coordinates from EXIF metadata."""

    def convert_to_degrees(value):
        d = float(value[0].num) / float(value[0].den)
        m = float(value[1].num) / float(value[1].den)
        s = float(value[2].num) / float(value[2].den)
        return d + (m / 60.0) + (s / 3600.0)

    if 'GPSInfo' not in tags:
        print("[DEBUG] No GPSInfo found in EXIF metadata.")
        return None, None

    gps_info = tags['GPSInfo']
    print(f"[DEBUG] Extracted GPSInfo: {gps_info}")

    lat = convert_to_degrees(gps_info[2]) if 2 in gps_info else None
    lng = convert_to_degrees(gps_info[4]) if 4 in gps_info else None

    if lat and lng:
        if gps_info[1] == 'S':
            lat = -lat
        if gps_info[3] == 'W':
            lng = -lng
        print(f"[DEBUG] Converted GPS Coordinates: Latitude = {lat}, Longitude = {lng}")
        return lat, lng

    print("[DEBUG] Unable to extract valid GPS coordinates.")
    return None, None


def extract_gps_from_image(image_path):
    """Extracts GPS data from image metadata."""
    try:
        with open(image_path, 'rb') as img_file:
            tags = exifread.process_file(img_file, details=False)

        print(f"[DEBUG] Extracted EXIF tags: {tags}")

        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat, lng = get_decimal_coordinates(tags)
            return lat, lng
        else:
            print("[DEBUG] No GPS data found in image EXIF metadata.")
    except Exception as e:
        print(f"[ERROR] Unable to read image metadata: {e}")

    return None, None


# 1. Landmark Recognition (using base64 image)
def recognize_landmark(image_path):
    base64_image = encode_image(image_path)

    if not base64_image:
        print("[ERROR] Failed to encode image. Skipping landmark recognition.")
        return None, None

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    request_body = {
        "requests": [
            {
                "image": {"content": base64_image},
                "features": [{"type": "LANDMARK_DETECTION"}]
            }
        ]
    }

    print(f"[DEBUG] Sending landmark recognition request to Google Vision API.")

    try:
        response = requests.post(url, json=request_body)
        result = response.json()
        print(f"[DEBUG] Google Vision API Response: {result}")

        if "landmarkAnnotations" in result["responses"][0]:
            landmark = result["responses"][0]["landmarkAnnotations"][0]
            print(f"[DEBUG] Detected Landmark: {landmark['description']} at {landmark['locations'][0]['latLng']}")
            lat_lng = landmark["locations"][0]["latLng"]
            return landmark["description"], (lat_lng["latitude"], lat_lng["longitude"])

    except Exception as e:
        print(f"[ERROR] Landmark recognition failed: {e}")

    return None, None


# 2. Get Place from GPS
def get_place_from_gps(lat, lng):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_API_KEY}"

    print(f"[DEBUG] Sending request to Google Geocoding API for coordinates ({lat}, {lng})")

    try:
        response = requests.get(url)
        result = response.json()
        print(f"[DEBUG] Google Geocoding API Response: {result}")

        if result["status"] == "OK":
            place = result["results"][0]["formatted_address"]
            print(f"[DEBUG] Retrieved Place: {place}")
            return place
    except Exception as e:
        print(f"[ERROR] Failed to retrieve place from GPS coordinates: {e}")

    return "Unknown Location"


# 3. Full Place Recognition
def recognize_place(image_path):
    print(f"[INFO] Recognizing place for image: {image_path}")

    # Extract GPS from image metadata
    lat, lng = extract_gps_from_image(image_path)

    if lat and lng:
        print(f"[INFO] GPS Coordinates extracted: Latitude = {lat}, Longitude = {lng}")
    else:
        print("[INFO] No GPS coordinates found in image.")

    # Try landmark detection first
    landmark, landmark_coords = recognize_landmark(image_path)

    if landmark:
        print(f"[INFO] Detected Landmark: {landmark}")
        final_lat, final_lng = landmark_coords
        place_name = landmark
    elif lat and lng:
        print("[INFO] Trying to fetch place details using GPS coordinates...")
        place_name = get_place_from_gps(lat, lng)
        final_lat, final_lng = lat, lng
    else:
        print("[INFO] No recognizable place found.")
        return {
            "place": "No place recognized",
            "latitude": None,
            "longitude": None
        }

    return {
        "place": place_name,
        "latitude": final_lat,
        "longitude": final_lng
    }


# Example usage
result = recognize_place("test/800px-Singapore_Merlion_BCT.jpg")

# Final output
print("\n=== FINAL OUTPUT ===")
print(f"Recognized Place: {result['place']}")
print(f"Latitude: {result['latitude']}")
print(f"Longitude: {result['longitude']}")

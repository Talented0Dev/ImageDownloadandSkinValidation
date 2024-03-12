import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

def download_images_from_csv(csv_file, output_directory):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each image URL in the DataFrame
    for index, row in df.iterrows():
        image_url = row['Image_URL']

        # Download the image from the URL
        response = requests.get(image_url)
        
        if response.status_code == 200:
            # Open the image using PIL
            image = Image.open(BytesIO(response.content))

            # Convert image to RGB mode
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            # Generate a unique image name (you can modify this if needed)
            image_name = f"image_{index}"

            # Save the image to the output directory
            image_path = os.path.join(output_directory, f"{image_name}.jpg")
            image.save(image_path)
            print(f"Image downloaded successfully: {image_path}")

        else:
            print(f"Failed to download image from URL: {image_url}. Status code: {response.status_code}")

if __name__ == "__main__":
    # Specify the path to the CSV file and output directory
    csv_file = "images.csv"  # Update with the correct path to your CSV file
    output_directory = "downloaded_images"

    # Download images from the CSV file
    download_images_from_csv(csv_file, output_directory)
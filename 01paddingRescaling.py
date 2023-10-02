"""
As SEResNeXt only accepts 224 by 224 pixels RGB colour 
image input, each ROI is firstly preprocessed with a 
padding and-rescaling operation for distortion-free size 
normalization (Li et al., 2022).

  Author: dvdsosa
  Date: 2023-09-21
"""

from PIL import Image # pip install Pillow
import os
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

# Input and output folders
input_folder = "/home/dsosatr/tesis/DYB-PlanktonNet"  # input folder
output_folder = "/home/dsosatr/tesis/DYB-PlanktonNew"  # output folder

# Target size for the images
target_size = (224, 224)

# Function to preprocess and save an image
def preprocess_image(input_path, output_path):
    try:
        # Open the image
        img = Image.open(input_path)

        # Resize the image while maintaining aspect ratio
        img.thumbnail(target_size)

        # Create a new image with the target size and paste the resized image onto it
        new_img = Image.new("RGB", target_size)
        new_img.paste(img, ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2))

        # Save the processed image
        new_img.save(output_path)

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

# Get a list of all image files to process
image_files = []
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

# Create the output folder structure if it doesn't exist
for input_path in image_files:
    relative_path = os.path.relpath(input_path, input_folder)
    output_path = os.path.join(output_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Process images with a progress bar
with tqdm(total=len(image_files), unit='img') as pbar:
    for input_path in image_files:
        relative_path = os.path.relpath(input_path, input_folder)
        output_path = os.path.join(output_folder, relative_path)

        preprocess_image(input_path, output_path)

        pbar.update(1)

print("Image preprocessing complete.")
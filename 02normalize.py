"""
After that, the pixel value of all the ROIs in the 
training dataset are converted to [0, 1]

  Author: dvdsosa
  Date: 2023-09-25
"""
import cv2
import numpy as np
import os
from tqdm import tqdm  # Import tqdm for progress bar

# Define the input and output folder paths
input_folder_path = "/Users/dvdsosa/Projects/ARGO/pCabildo/tesis/articulos/02procesado/DYB-Test"
output_folder_path = "/Users/dvdsosa/Projects/ARGO/pCabildo/tesis/articulos/02procesado/DYB-Test2"

# Function to preprocess and save an image
def preprocess_image(input_path, output_path):
    try:
        original_image = cv2.imread(input_path)

        # Normalize the image using min-max normalization
        normalized_image = cv2.normalize(original_image.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        # If both images are the same data type, then
        # calculate the absolute difference between the images
        #difference_image = cv2.absdiff(original_image.astype('float32'), normalized_image)

        # Restore for an UINT8_T format
        restoreToSave = cv2.convertScaleAbs(normalized_image * 255)

        # Display the original and normalized images
        #cv2.imshow('Original Image', original_image)
        #cv2.imshow('Normalized Image', normalized_image)
        #cv2.imshow('Normalized Image UINT8', restoreToSave)
        #cv2.imshow('Difference Image', difference_image)
        #cv2.waitKey(0)

        # Save the normalized image to the output folder with the same filename and path as the original image
        cv2.imwrite(output_path, restoreToSave)

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        cv2.destroyAllWindows()

# Get a list of all image files to process
image_files = []
for root, _, files in os.walk(input_folder_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

# Create the output folder structure if it doesn't exist
for input_path in image_files:
    relative_path = os.path.relpath(input_path, input_folder_path)
    output_path = os.path.join(output_folder_path, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Process images with a progress bar
with tqdm(total=len(image_files), unit='img') as pbar:
    for input_path in image_files:
        relative_path = os.path.relpath(input_path, input_folder_path)
        output_path = os.path.join(output_folder_path, relative_path)

        preprocess_image(input_path, output_path)

        pbar.update(1)

print("Normalization completed!")
cv2.destroyAllWindows()
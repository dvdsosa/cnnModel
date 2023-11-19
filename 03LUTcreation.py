import torch
import os
import numpy as np
import sqlite3

from torchvision import datasets, transforms
from SupContrast.networks.resnet_big import FeatureExtractor
from PIL import Image
from tqdm import tqdm
from io import BytesIO

# Instantiate the feature extractor
feature_extractor = FeatureExtractor(name='resnet50')

# Load the pre-trained weights
state = torch.load('/home/dsosatr/tesis/cnnmodel/SupContrast/save/SupCon/path_models/SupCon_path_resnet50_lr_0.2_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/ckpt_epoch_1000.pth')

# Filter out the keys for the projection head
state_dict = {k: v for k, v in state['model'].items() if 'head' not in k}

# Load the pre-trained weights into the feature extractor
feature_extractor.load_state_dict(state_dict, strict=False)

# Define the transformation.
transform = transforms.Compose([
    transforms.ToTensor(), # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Perform the same normalization as the one used while traiing
])

# Connect to the SQLite database
conn = sqlite3.connect('image_features.db')
# Create a cursor object
c = conn.cursor()
# Create table
# https://sqlbolt.com/lesson/creating_tables
c.execute('''CREATE TABLE IF NOT EXISTS features (
          image_id INTEGER PRIMARY KEY,
          class TEXT NOT NULL, 
          image_name TEXT NOT NULL,
          tensor BLOB NOT NULL
)''')

def get_file_info(root_folder):
    for dirpath, dirnames, filenames in tqdm(os.walk(root_folder)):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            subfolder_name = os.path.basename(dirpath)

            # Load each training set image
            image = Image.open(full_path)
            # Transform the image according to the defined transformation
            to_tensor = transform(image)
            # Add a new dimension at the 0th position in the tensor’s shape
            to_tensor = to_tensor.unsqueeze(0)  
            # Use the feature extractor
            feature_vector = feature_extractor(to_tensor)

            # Convert tensor to bytes
            buffer = BytesIO()
            torch.save(feature_vector, buffer)
            feature_vector_bytes = buffer.getvalue()

            # Insert a row of data via parameterized query to prevent SQL injection
            c.execute("INSERT INTO features (class, image_name, tensor) VALUES (?, ?, ?)", (subfolder_name, filename, feature_vector_bytes))

            print(f"Full path: {full_path}")
            print(f"File name: {filename}")
            print(f"Subfolder name: {subfolder_name}")
            print("-------------------")

# Replace 'root_folder' with the path to your root folder
get_file_info('/home/dsosatr/tesis/DYBtrainCropped/')

# Save (commit) the changes
conn.commit()

# Close the connection
conn.close()
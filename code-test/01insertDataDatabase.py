import torch
import sqlite3

from torchvision import datasets, transforms
from SupContrast.networks.resnet_big import FeatureExtractor
from PIL import Image
from io import BytesIO

# Define the transformation.
transform = transforms.Compose([
    transforms.ToTensor(), # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Perform the same normalization as the one used while traiing
])

# Instantiate the feature extractor
feature_extractor = FeatureExtractor(name='resnet50')
# Load the pre-trained weights
state = torch.load('/home/dsosatr/tesis/cnnmodel/SupContrast/save/SupCon/path_models/SupCon_path_resnet50_lr_0.2_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/ckpt_epoch_1000.pth')
# Filter out the keys for the projection head
state_dict = {k: v for k, v in state['model'].items() if 'head' not in k}
# Load the pre-trained weights into the feature extractor
feature_extractor.load_state_dict(state_dict, strict=False)

# Load each training set image
image = Image.open("/home/dsosatr/tesis/DYBtrainCropped/028_Harpacticoid/DYB-28935127-16.jpg")
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

# Connect to the SQLite database
conn = sqlite3.connect('test_database.db')

# Create a cursor object
c = conn.cursor()

# Create table
# https://sqlbolt.com/lesson/creating_tables
c.execute('''CREATE TABLE IF NOT EXISTS features (
          image_id INTEGER PRIMARY KEY,
          image_name TEXT NOT NULL,
          class TEXT NOT NULL, 
          vector BLOB NOT NULL
)''')

# Insert a row of data via parameterized query to prevent SQL injection
c.execute("INSERT INTO features (image_name, class, vector) VALUES (?, ?, ?)", ('filename-test', 'subfolder_name-test', feature_vector_bytes))

# Save (commit) the changes
conn.commit()

# Close the connection
conn.close()

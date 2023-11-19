import torch
import sqlite3
from io import BytesIO

# Connect to the SQLite database
conn = sqlite3.connect('image_features.db')

# Create a cursor object
c = conn.cursor()

# Retrieve data
c.execute("SELECT tensor FROM features WHERE image_id = ?", (1,))
row = c.fetchone()

if row is not None:
    # Get the byte string from the row
    feature_vector_bytes = row[0]

    # Convert bytes back to tensor
    buffer = BytesIO(feature_vector_bytes)
    feature_vector = torch.load(buffer)

    print(feature_vector)
else:
    print("No data found for the specified image name.")

# Close the connection
conn.close()
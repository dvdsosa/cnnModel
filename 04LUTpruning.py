from io import BytesIO

from tqdm import tqdm
import torch
import sqlite3
import torch.nn as nn

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a CosineSimilarity object
cos = nn.CosineSimilarity(dim=0, eps=1e-6)

def lookup_table_pruning(lookup_table, classes, id, similarity_threshold):
    # Connect to the SQLite database
    # conn = sqlite3.connect('trainingSetPruned.db')
    # Create a cursor object
    # c = conn.cursor()

    # Iterate over each pair of entries in the lookup table
    for i in range(lookup_table.size(0)):
        indices_to_remove = []
        for j in range(i+1, lookup_table.size(0)):
            # Calculate the similarity between the two entries
            similarity = cos(lookup_table[i], lookup_table[j])

            # If the similarity is greater than the threshold and the classes of the two entries are the same
            if similarity > similarity_threshold and classes[i] == classes[j]:
                indices_to_remove.append(id[j])

        # If the size of the set is greater than 1 or exactly 1, save the index of the row to erase
        if len(indices_to_remove) > 1:
            # Retrieve data
            # c.execute("DELETE FROM features WHERE image_id = ?", (id[i],))
            print(f"Indices to remove: {id[i]}")
        elif len(indices_to_remove) == 1:
            j = indices_to_remove[0]
            print(f"Indices to remove: {id[j]}")
            # c.execute("DELETE FROM features WHERE image_id = ?", (id[j],))

    # Close the connection
    # conn.close()

# Connect to the SQLite database
conn = sqlite3.connect('trainingSetNew.db')
# Create a cursor object
c = conn.cursor()
# Retrieve data
c.execute("SELECT image_id, class, tensor FROM features")
# Fetch all the rows
rows = c.fetchall()
# Close the connection
conn.close()

# Initialize empty lists
images_id = []
classes = []
tensors = []

for row in tqdm(rows, desc="Processing rows"):
    image_id = row[0]
    class_value = row[1]
    tensor_bytes = row[2]
    
    # Convert bytes back to tensor
    buffer = BytesIO(tensor_bytes)
    tensor_value = torch.load(buffer).to(device)

    # Remove the dimension of size 1
    tensor_value = tensor_value.squeeze(0)

    # Append to the respective lists
    images_id.append(image_id)
    classes.append(class_value)
    tensors.append(tensor_value)

# Convert the list of tensors into a single tensor
tensors = torch.stack(tensors)

lookup_table_pruning(tensors, classes, images_id, 0.7)

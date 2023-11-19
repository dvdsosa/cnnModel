from io import BytesIO

from tqdm import tqdm
import sqlite3
import torch
import torch.nn as nn

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a CosineSimilarity object
cos = nn.CosineSimilarity(dim=0, eps=1e-6).to(device)

# Define a similarity function
def sim_func(x, y):
    # Compute the cosine similarity
    return cos(x, y)

# Define a function to determine the optimal threshold for a class
def determine_threshold(class_examples, test_examples, sim_func):
    # Initialize the threshold and the  best accuracy
    threshold = 0
    best_accuracy = 0

    # Try different thresholds
    for t in tqdm(torch.arange(0, 1, 0.01), desc='Determining threshold'):
        # Prune the lookup table
        pruned_lookup_table = []
        for x in class_examples:
            sim_scores = []
            for test_example in test_examples:
                sim_score = sim_func(x, test_example)
                sim_scores.append(sim_score)
            mean_sim_score = torch.stack(sim_scores).mean().item()
            if mean_sim_score > t:
                pruned_lookup_table.append(x)

        # Calculate the accuracy on the test examples
        total_sim_score = 0
        for x in pruned_lookup_table:
            sim_scores = []
            for test_example in test_examples:
                sim_score = sim_func(x, test_example)
                sim_scores.append(sim_score)
            mean_sim_score = torch.stack(sim_scores).mean().item()
            total_sim_score += mean_sim_score
        accuracy = total_sim_score / len(test_examples)

        # Update the threshold and the best accuracy if necessary
        if accuracy > best_accuracy:
            threshold = t
            best_accuracy = accuracy

    return threshold

# Connect to the SQLite database
conn = sqlite3.connect('image_features.db')
# Create a cursor object
c = conn.cursor()
# Retrieve data
c.execute("SELECT tensor FROM features")
# Fetch all the rows
rows = c.fetchall()
# Close the connection
conn.close()
# Convert the list of tuples into a list of tensors
tensor_list = [torch.load(BytesIO(row[0])).to(device) for row in rows]
# If you want to concatenate all tensors into one, make sure they all have the same shape
class_examples = torch.cat(tensor_list, dim=0)

# Connect to the SQLite database
conn = sqlite3.connect('test_features.db')
# Create a cursor object
c = conn.cursor()
# Retrieve data
c.execute("SELECT tensor FROM features")
# Fetch all the rows
rows = c.fetchall()
# Close the connection
conn.close()
# Convert the list of tuples into a list of tensors
tensor_list = [torch.load(BytesIO(row[0])).to(device) for row in rows]
# If you want to concatenate all tensors into one, make sure they all have the same shape
test_examples = torch.cat(tensor_list, dim=0)

# Determine the optimal threshold
optimal_threshold = determine_threshold(class_examples, test_examples, sim_func)

print(f"The optimal threshold is {optimal_threshold}")

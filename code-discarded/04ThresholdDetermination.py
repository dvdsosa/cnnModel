'''
Q: Explain me the threshold determination with words, step by step for a noobie

A: Sure, let's break it down:

1. **Start with a lookup table and a test dataset**: The lookup table contains examples 
from different classes. Each example is a pair of a feature vector and a class label. The 
test dataset contains examples that we want to classify.

2. **Define a similarity function**: This is a function that takes two feature vectors and 
returns a score that represents how similar they are. The higher the score, the more similar 
the vectors.

3. **Define a threshold**: This is a value that we use to decide whether two feature vectors 
are considered similar. If the similarity score between two vectors is greater than the threshold, 
we consider them similar.

4. **Iterate over different thresholds**: We start with a threshold of 0 and increment it by 
a small amount (e.g., 0.01) at each step. For each threshold:

    1. **Prune the lookup table**: For each example in the lookup table, we calculate the 
    similarity score with each example in the test dataset. We then calculate the mean of 
    these scores. If the mean score is greater than the threshold, we keep the example in the 
    lookup table. Otherwise, we remove it. This results in a pruned lookup table.

    2. **Calculate the accuracy**: We use the pruned lookup table to classify the examples in 
    the test dataset and calculate the accuracy of the classification.

5. **Choose the threshold that gives the best accuracy**: After iterating over all thresholds, 
we choose the one that gave the best accuracy on the test dataset. This is the optimal threshold.

This process is called threshold determination. It's a way to optimize the lookup table for both 
accuracy and efficiency. By removing examples that are not similar to the test examples (according 
to the threshold), we can make the lookup table smaller and the retrieval process faster, without 
sacrificing too much accuracy.


Q: When pruning the lookup table, why should I try different thresholds from 0 to 1 ?

A: The process of trying different thresholds from 0 to 1 is a form of hyperparameter tuning. 
The threshold is a hyperparameter that determines how similar two examples need to be in 
order to be considered the same. 

By iterating over different thresholds, you can find the value that gives the best balance 
between accuracy and efficiency for your specific task. 

Here's why:
- If the threshold is too low (close to 0), then even examples that are not very similar will
 be considered the same. This could lead to a large lookup table with many redundant examples, 
 which could slow down the retrieval process and decrease accuracy.
- If the threshold is too high (close to 1), then only very similar examples will be considered 
the same. This could lead to a small lookup table with few examples, which could speed up the 
retrieval process but also decrease accuracy because there might be fewer relevant examples to 
compare against.

So, by trying different thresholds, you can find the "sweet spot" that gives the best trade-off 
between accuracy and efficiency. The range of 0 to 1 is used because similarity scores 
(like cosine similarity or Jaccard index) typically fall within this range. However, the exact 
range you should use depends on the similarity metric you're using and the nature of your data.
'''
from io import BytesIO

from tqdm import tqdm
from memory_profiler import profile
import sqlite3
import torch
import torch.nn as nn
import sys
import gc
import numpy as np

# Check if CUDA is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a CosineSimilarity object
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

# Define a similarity function
@profile
def sim_func(x, y):
    # Compute the cosine similarity
    result = cos(x, y)
    #del x, y
    #gc.collect()
    return result

# Define a lookup table pruning function
def lookup_table_pruning(lookup_table, sim_func, threshold):
    pruned_lookup_table = lookup_table.copy()
    n = len(lookup_table)
    for i in range(n):
        xi, yi = lookup_table[i]
        X = []
        for j in range(i+1, n):
            xj, yj = lookup_table[j]
            sim = sim_func(xi, xj)
            if sim > threshold and yi == yj:
                X.append(xj)
        if len(X) > 1:
            pruned_lookup_table.remove((xi, yi))
        elif len(X) == 1:
            x = X[0]
            pruned_lookup_table.remove((xi, yi))
            pruned_lookup_table.append((x, yi))
    return pruned_lookup_table

def pruneLUT(pruned_lookup_table, c_training, c_testing, t, class_len, test_len):
    # Prune the lookup table
    # pruned_lookup_table = [x for x in class_examples if sim_func(x, test_examples).mean() > t]
    for ii in tqdm(range(1, class_len+1), desc="First nested for loop...", leave=False):
        # Query image_id ii tensor variable from the database
        # Execute the query
        c_training.execute("SELECT tensor FROM features WHERE image_id = ?", (ii,))
        # Fetch the results
        row = c_training.fetchone()
        # Get the byte string from the row
        feature_vector_bytes = row[0]
        # Convert bytes back to tensor
        buffer = BytesIO(feature_vector_bytes)
        x = torch.load(buffer).to(device)

        sim_score_sum = 0
        count = 0
        for jj in range(1, test_len+1):
            # Query image_id ii tensor variable from the database
            # Execute the query
            c_testing.execute("SELECT tensor FROM features WHERE image_id = ?", (jj,))
            # Fetch the results
            row = c_testing.fetchone()
            # Get the byte string from the row
            feature_vector_bytes = row[0]
            # Convert bytes back to tensor
            buffer = BytesIO(feature_vector_bytes)
            test_example = torch.load(buffer).to(device)

            sim_score = cos(x, test_example)
            sim_score_sum += sim_score.item()
            count += 1

        mean_sim_score = sim_score_sum / count
        if mean_sim_score > t:
            pruned_lookup_table = torch.cat((pruned_lookup_table, x.unsqueeze(0)), dim=0)

    return pruned_lookup_table

def getAccuracy(pruned_lookup_table, c_testing, test_len):
    # Calculate the accuracy on the test examples
    # accuracy = sum(sim_func(x, test_examples).mean() for x in pruned_lookup_table) / len(test_examples)
    total_sim_score = 0
    for x in tqdm(pruned_lookup_table, desc="Second nested for loop...", leave=False):
        # Move x to the GPU
        x = x.to(device)

        sim_score_sum = 0
        count = 0
        for jj in range(1, test_len+1):
            # Query image_id ii tensor variable from the database
            # Execute the query
            c_testing.execute("SELECT tensor FROM features WHERE image_id = ?", (jj,))
            # Fetch the results
            row = c_testing.fetchone()
            # Get the byte string from the row
            feature_vector_bytes = row[0]
            # Convert bytes back to tensor
            buffer = BytesIO(feature_vector_bytes)
            test_example = torch.load(buffer).to(device)

            sim_score = cos(x, test_example)
            sim_score_sum += sim_score.item()
            count += 1

        mean_sim_score = sim_score_sum / count
        total_sim_score += mean_sim_score

    accuracy = total_sim_score / test_len

    return accuracy

# Define a function to determine the optimal threshold for a class
def determine_threshold(c_training, c_testing):
    # Initialize the threshold and the best accuracy
    threshold = 0
    best_accuracy = 0

    # Query for total training-set table length
    c_training.execute("SELECT COUNT(*) FROM features")
    # Fetch the results
    class_examples_len = c_training.fetchone()
    class_len = class_examples_len[0]

    # Query for total test-set table length
    c_testing.execute("SELECT COUNT(*) FROM features")
    # Fetch the results
    test_examples_len = c_testing.fetchone()
    test_len = test_examples_len[0]

    # Try different thresholds
    for t in tqdm(torch.arange(0, 1, 0.01), desc='Determining threshold'):

        # Prune the lookup table
        pruned_lookup_table = torch.empty((0,), dtype=torch.float32, device=device)

        pruned_lookup_table = pruneLUT(pruned_lookup_table, c_training, c_testing, t, class_len, test_len)
        
        accuracy = getAccuracy(pruned_lookup_table, c_testing, test_len)

        # Update the threshold and the best accuracy if necessary
        if accuracy > best_accuracy:
            threshold = t
            best_accuracy = accuracy

    return threshold, best_accuracy

# Connect to the SQLite database
conn_training = sqlite3.connect('image_features.db')
# Create a cursor object
c_training = conn_training.cursor()

# Connect to the SQLite database
conn_testing = sqlite3.connect('test_features.db')
# Create a cursor object
c_testing = conn_testing.cursor()

# Determine the optimal threshold
optimal_threshold, best_accuracy = determine_threshold(c_training, c_testing)

print(f"The optimal threshold is {optimal_threshold} with an accuracy of {best_accuracy}")

# Close the connection
conn_training.close()

# Close the connection
conn_testing.close()


"""
The provided code seems to be loading tensors from a database inside a nested loop, which can be memory-intensive. Here are some suggestions to improve the memory usage:

1. **Batch Processing**: Instead of loading all the tensors into memory, you can load them in batches. This will significantly reduce the memory footprint.

2. **Database Optimization**: Instead of querying the database for each tensor, you can fetch all the tensors at once and store them in a list. This will reduce the number of database queries, which can be expensive in terms of time and memory.

3. **In-place Operations**: Try to perform operations in-place whenever possible. This means that you modify the original variable instead of creating a new one.

4. **Garbage Collection**: You are already using `gc.collect()` to clear unused memory, which is good. However, you can also use `del` to delete variables that are no longer needed.

5. **Use NumPy Arrays**: If possible, use NumPy arrays instead of lists for storing your tensors. NumPy arrays are more memory efficient and faster for numerical computations.

6. **Use Sparse Matrices**: If your tensors are mostly zeros, consider using sparse matrices. Sparse matrices only store non-zero elements, which can save a lot of memory.

7. **Use Lower Precision**: If high precision is not necessary for your computations, consider using lower precision data types. For example, you can use `torch.float32` instead of `torch.float64`.

Remember, optimizing memory usage often involves trade-offs with computational speed and code complexity. It's important to find a balance that works best for your specific use case. Happy coding! 😊
"""
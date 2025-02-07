from io import BytesIO

from tqdm import tqdm
import torch
import numpy as np
import sqlite3
import threading
import torch.nn as nn

# Create a CosineSimilarity object
cos = nn.CosineSimilarity(dim=0, eps=1e-6)

def load_features(cursor):
    """
    Loads all feature vectors and their associated labels from the database into memory.

    Args:
        cursor (sqlite3.Cursor): The database cursor for executing SQL commands.

    Returns:
        list: A list of tuples where each tuple contains a label (str) and a feature vector (numpy.ndarray).
    """
    cursor.execute('SELECT label, vector FROM plankton_features')
    rows = cursor.fetchall()
    features = [(label, torch.from_numpy(np.frombuffer(vector, dtype=np.float32).copy())) for label, vector in rows]
    return features

def prune_lookup_table(LUT, T_S):
    """
    Prune the lookup table by removing redundant vectors using cosine similarity.

    Parameters:
        LUT: List of tuples [(tensor, class_label), ...]
        T_S: Similarity threshold for pruning
    
    Returns:
        Pruned LUT'
    """
    n = len(LUT)
    LUT_prime = LUT.copy()  # Copy of LUT to modify
    cosine_sim = torch.nn.CosineSimilarity(dim=0)  # Cosine similarity function
    indices_to_remove = set()  # Store indices to remove later

    for i in tqdm(range(n), desc="Pruning LUT"):
        if i in indices_to_remove:  # Skip if already marked for removal
            continue

        x_i, y_i = LUT[i]
        S_i = set()

        for j in range(i + 1, n):  # Compare with later vectors
            if j in indices_to_remove:  # Skip if already marked for removal
                continue

            x_j, y_j = LUT[j]
            S_ij = cosine_sim(y_i, y_j)  # Compute cosine similarity

            if S_ij > T_S and x_i == x_j:  # Only check same-class vectors
                S_i.add(j)

        if len(S_i) > 1:  # If multiple similar vectors exist, remove the current one
            indices_to_remove.add(i)
        elif len(S_i) == 1:  # If exactly one match, remove the matched one
            j = list(S_i)[0]
            indices_to_remove.add(j)

    # Remove all accumulated indices in reverse order to avoid shifting issues
    LUT_prime = [LUT_prime[i] for i in range(len(LUT_prime)) if i not in indices_to_remove]

    return LUT_prime

def main():
    
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('plankton_db.sqlite')
    cursor = conn.cursor()
    LUT = load_features(cursor)

    T_S = 0.9  # Similarity threshold

    prune_thread = threading.Thread(target=lambda: prune_lookup_table(LUT, T_S))
    prune_thread.start()
    prune_thread.join()
    
    print(f"Original LUT size: {len(LUT)}")
    print(f"Pruned LUT size: {len(pruned_LUT)}")
    

if __name__ == '__main__':
    main()
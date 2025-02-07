from io import BytesIO

from tqdm import tqdm
import torch
import numpy as np
import sqlite3
import threading
import torch.nn as nn
import time  # Import time module to measure time

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

def prune_lookup_table_cuda(LUT, T_S):
    """
    Prunes a lookup table (LUT) using cosine similarity on a CUDA-enabled GPU.
    Args:
        LUT (list of tuples): The lookup table to be pruned, where each entry is a tuple (label, feature_vector).
        T_S (float): The similarity threshold for pruning. Pairs with cosine similarity above this threshold are considered similar.
    Returns:
        list of tuples: The pruned lookup table with redundant entries removed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    vectors = torch.stack([x[1] for x in LUT]).to(device)  # Stack feature vectors into a tensor and move to GPU
    labels = torch.tensor([hash(x[0]) for x in LUT], dtype=torch.long, device=device)  # Convert labels to unique integers

    # Normalize vectors for cosine similarity
    normalized_vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)

    # Compute cosine similarity matrix (size: num_samples x num_samples)
    similarity_matrix = torch.mm(normalized_vectors, normalized_vectors.T)  # Fast batch computation

    # Create a mask to ignore self-comparisons
    mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=device)

    # Apply thresholding: keep only pairs above T_S
    similar_pairs = (similarity_matrix > T_S) & ~mask

    indices_to_remove = set()
    for i in tqdm(range(len(LUT)), desc="Pruning LUT"):
        if i in indices_to_remove:
            continue  # Skip already marked indices

        # Find similar indices
        similar_indices = torch.where(similar_pairs[i])[0].tolist()

        # Filter only same-class vectors
        similar_indices = [j for j in similar_indices if labels[j] == labels[i]]

        if len(similar_indices) > 1:  # Multiple similar vectors → remove current
            indices_to_remove.add(i)
        elif len(similar_indices) == 1:  # One match → remove the other
            indices_to_remove.add(similar_indices[0])

    # Create pruned LUT by filtering out marked indices
    LUT_prime = [LUT[i] for i in range(len(LUT)) if i not in indices_to_remove]

    return LUT_prime

def store_feature(label, feature_vector, cursor, conn):
    """
    Stores a feature vector in the database with the associated label.

    Args:
        label (str): The label associated with the feature vector.
        feature_vector (numpy.ndarray): The feature vector to be stored.
        cursor (sqlite3.Cursor): The database cursor for executing SQL commands.
        conn (sqlite3.Connection): The database connection object.

    Returns:
        None
    """
    cursor.execute('''
    INSERT INTO plankton_features (label, vector)
    VALUES (?, ?)
    ''', (label, feature_vector.tobytes()))
    conn.commit()

def init_db(conn, cursor):
    """
    Initializes the database by creating the 'plankton_features' table if it does not already exist.

    Args:
        conn (sqlite3.Connection): The connection object to the SQLite database.
        cursor (sqlite3.Cursor): The cursor object to execute SQL commands.

    The 'plankton_features' table has the following columns:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - label: TEXT NOT NULL
        - vector: BLOB NOT NULL

    Commits the transaction after executing the SQL command.
    """
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plankton_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT NOT NULL,
        vector BLOB NOT NULL
    )
    ''')
    conn.commit()

def main():
    
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('plankton_db.sqlite')
    cursor = conn.cursor()
    LUT  = load_features(cursor)

    T_S = 0.9  # Similarity threshold
   
    # pruned_LUT = prune_lookup_table(LUT, T_S)    
    pruned_LUT = prune_lookup_table_cuda(LUT, T_S)

    print(f"Original LUT size: {len(LUT)}")
    print(f"Pruned LUT size: {len(pruned_LUT)}")

    # Connect or create to SQLite database and save the pruned LUT
    conn = sqlite3.connect('plankton_pruned_db.sqlite')
    cursor = conn.cursor()
    init_db(conn, cursor)
    for label, feature_vector in tqdm(pruned_LUT, desc="Saving pruned LUT..."):
        store_feature(label, feature_vector.cpu().numpy(), cursor, conn)
    conn.close()

if __name__ == '__main__':
    main()
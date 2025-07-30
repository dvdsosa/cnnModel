import faiss
import sqlite3
import numpy as np
import os
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

FAISS_INDEX_PATH = 'faiss_index_stage2.bin'
SQLITE_DB_PATH = 'plankton_db_stage2.sqlite'
SIMILARITY_THRESHOLD = 0.93

def load_faiss_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    return faiss.read_index(index_path)

def load_feature_mappings(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT faiss_id, label FROM feature_mappings")
    mappings = cursor.fetchall()
    conn.close()
    return mappings

def remove_from_db(db_path, faiss_ids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany("DELETE FROM feature_mappings WHERE faiss_id = ?", [(int(fid),) for fid in faiss_ids])
    conn.commit()
    conn.close()

def prune_lut(faiss_index, mappings, threshold):
    cos = nn.CosineSimilarity(dim=0)
    label_to_ids = defaultdict(list)
    for faiss_id, label in mappings:
        label_to_ids[label].append(faiss_id)

    ids_to_remove = set()
    for label, ids in tqdm(label_to_ids.items(), desc="Pruning labels"):
        ids = sorted(ids)
        n = len(ids)
        # For each i, compare with j > i
        for i in range(n):
            if ids[i] in ids_to_remove:
                continue
            if not (0 <= ids[i] < faiss_index.ntotal):  # Ensure the ID is within bounds
                print(f"ID {ids[i]} out of bounds, skipping.")
                input("Press Enter to continue...")
                continue
            # Get feature vector for i
            xi = torch.tensor(faiss_index.reconstruct(int(ids[i])))
            Si = []
            for j in range(i+1, n):
                if ids[j] in ids_to_remove:
                    continue
                if not (0 <= ids[j] < faiss_index.ntotal):  # Ensure the ID is within bounds
                    print(f"ID {ids[j]} out of bounds, skipping.")
                    input("Press Enter to continue...")
                    continue
                xj = torch.tensor(faiss_index.reconstruct(int(ids[j])))
                #print(f"xi length: {len(xi)}, xj length: {len(xj)}, i: {i}, j: {j}")
                # Cosine similarity using PyTorch
                sij = float(cos(xi, xj))
                if sij > threshold:
                    Si.append(j)
            # Remove redundant examples
            if len(Si) > 0:
                # Remove all j in Si (redundant examples)
                for j in Si:
                    ids_to_remove.add(ids[j])
    return ids_to_remove

def save_pruned_db(db_path, mappings, ids_to_remove, output_db_path, keep_ids):
    """
    Create a new SQLite database with the same schema and only the non-pruned mappings.
    Reassign faiss_id values to be sequential (0, 1, ..., N-1) matching the order of keep_ids.
    Returns the new mappings as a list of (new_faiss_id, label).
    """
    import shutil
    # Copy the original DB to the new location
    shutil.copy(db_path, output_db_path)
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    # Remove redundant mappings from the new DB
    cursor.executemany(
        "DELETE FROM feature_mappings WHERE faiss_id = ?",
        [(int(fid),) for fid in ids_to_remove]
    )
    # Reassign faiss_id sequentially
    # Get the label for each keep_id in order
    placeholders = ",".join("?" for _ in keep_ids)
    cursor.execute(f"SELECT faiss_id, label FROM feature_mappings WHERE faiss_id IN ({placeholders})", keep_ids)
    id_label_map = {int(row[0]): row[1] for row in cursor.fetchall()}
    new_mappings = []
    for new_id, old_id in enumerate(keep_ids):
        label = id_label_map[old_id]
        cursor.execute("UPDATE feature_mappings SET faiss_id = ? WHERE faiss_id = ?", (new_id, old_id))
        new_mappings.append((new_id, label))
    conn.commit()
    conn.close()
    print(f"Pruned SQLite DB saved at {output_db_path}")
    return new_mappings

def save_pruned_faiss_index(faiss_index, keep_ids, output_path):
    # Reconstruct vectors and build new index in the order of keep_ids
    vectors = np.stack([faiss_index.reconstruct(fid) for fid in keep_ids]).astype(np.float32)
    dim = vectors.shape[1]
    new_index = faiss.IndexFlatIP(dim)
    new_index.add(vectors)
    faiss.write_index(new_index, output_path)
    print(f"Pruned FAISS index saved with {new_index.ntotal} vectors at {output_path}")

def main():
    """
    Main function to perform LUT pruning on a FAISS index and a feature mapping database.

    This function:
    1. Loads a FAISS index from a specified file path.
    2. Loads feature mappings from a SQLite database.
    3. Identifies redundant examples based on a similarity threshold.
    4. Creates a new pruned SQLite database.
    5. Saves the pruned FAISS index to a new file.

    Constants:
        FAISS_INDEX_PATH (str): Path to the FAISS index file.
        SQLITE_DB_PATH (str): Path to the SQLite database file.
SIMILARITY_THRESHOLD = 1.0

    Outputs:
        Prints the number of feature mappings loaded, the number of redundant examples pruned,
        and confirmation messages for database and FAISS index updates.
    """

    faiss_index = load_faiss_index(FAISS_INDEX_PATH)
    mappings = load_feature_mappings(SQLITE_DB_PATH)
    print(f"Loaded {len(mappings)} feature mappings.")

    print(f"Using similarity threshold: {SIMILARITY_THRESHOLD}")
    ids_to_remove = prune_lut(faiss_index, mappings, SIMILARITY_THRESHOLD)
    print(f"Pruning {len(ids_to_remove)} redundant examples.")

    # Determine keep_ids in original order
    keep_ids = [int(faiss_id) for faiss_id, _ in mappings if int(faiss_id) not in ids_to_remove]

    # Save pruned database to a new file and get new mappings with sequential faiss_id
    new_mappings = save_pruned_db(SQLITE_DB_PATH, mappings, ids_to_remove, 'plankton_db_stage2_pruned.sqlite', keep_ids)

    # Save pruned FAISS index in the same order as new_mappings (sequential IDs)
    save_pruned_faiss_index(faiss_index, keep_ids, 'faiss_index_stage2_pruned.bin')

if __name__ == '__main__':
    main()

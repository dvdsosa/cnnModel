import subprocess
import re
import json
import os
import sqlite3
import numpy as np

from collections import Counter
from scipy.stats import entropy

PRUNING_SCRIPT = "11_lut_pruning_sam2.py"
INFERENCE_SCRIPT = "12_inference_sam2.py"
RESULTS_FILE = "threshold_results_stage2.json"

def update_threshold_in_file(filepath, threshold):
    with open(filepath, "r") as f:
        lines = f.readlines()
    with open(filepath, "w") as f:
        for line in lines:
            if line.strip().startswith("SIMILARITY_THRESHOLD"):
                f.write(f"SIMILARITY_THRESHOLD = {threshold}\n")
            else:
                f.write(line)

def run_script(script):
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    return result.stdout

def extract_metrics(output):
    # Adjusted regex to match the input string format
    pattern = (
        r"Multiclass Accuracy:\s*([\d.]+)\n"
        r"Multiclass Precision \(macro\):\s*([\d.]+)\n"
        r"Multiclass Recall \(macro\):\s*([\d.]+)\n"
        r"Multiclass F1 Score \(macro\):\s*([\d.]+)\n"
        r"Total elapsed time:\s*([\d.]+)\s*seconds\n"
        r"Mean processing time per image:\s*([\d.]+)\s*seconds\n"
        r"FAISS index file size:\s*([\d.]+)\s*MB\n"
    )
    match = re.search(pattern, output)
    if match:
        return {
            "Multiclass Accuracy": float(match.group(1)),
            "Multiclass Precision (macro)": float(match.group(2)),
            "Multiclass Recall (macro)": float(match.group(3)),
            "Multiclass F1 Score (macro)": float(match.group(4)),
            "Total elapsed time (seconds)": float(match.group(5)),
            "Mean processing time per image (seconds)": float(match.group(6)),
            "FAISS index file size (MB)": float(match.group(7)),
        }
    return None

def get_class_counts_from_db(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} does not exist")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feature_mappings'")
        if not cursor.fetchone():
            # List all tables for debugging
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Error: Table 'feature_mappings' not found in {db_path}")
            print(f"Available tables: {[table[0] for table in tables]}")
            conn.close()
            return []
        
        cursor.execute("SELECT label FROM feature_mappings")
        labels = [row[0] for row in cursor.fetchall()]
        conn.close()
        counts = Counter(labels)
        return list(counts.values())
    except Exception as e:
        print(f"Error reading database {db_path}: {e}")
        return []

def gini_coefficient(counts):
    counts = np.array(counts)
    counts = counts[counts > 0]
    n = len(counts)
    if n == 0:
        return float('nan')
    sorted_counts = np.sort(counts)
    index = np.arange(1, n+1)
    return (np.sum((2 * index - n - 1) * sorted_counts)) / (n * np.sum(sorted_counts))

def imbalance_ratio(counts):
    counts = np.array(counts)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return float('nan')
    return float(np.max(counts)) / float(np.min(counts))

def class_entropy(counts):
    counts = np.array(counts)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return float('nan')
    p = counts / counts.sum()
    return float(entropy(p))

def main():
    results = {}
    # Load previous results if file exists
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    thresholds = [round(x, 2) for x in list(frange(0.8, 1.01, 0.01))]
    for threshold in thresholds:
        threshold_str = str(threshold)
        if threshold_str in results:
            print(f"Skipping threshold {threshold} (already done)")
            continue
        print(f"Testing SIMILARITY_THRESHOLD = {threshold}")
        update_threshold_in_file(PRUNING_SCRIPT, threshold)
        subprocess.run(["python3", PRUNING_SCRIPT])
        output = run_script(INFERENCE_SCRIPT)
        metrics = extract_metrics(output)
        # --- Compute Gini coefficient, imbalance ratio, and entropy from pruned DB ---
        class_counts = get_class_counts_from_db('plankton_db_stage2_pruned.sqlite')
        gini = gini_coefficient(class_counts)
        ir = imbalance_ratio(class_counts)
        ent = class_entropy(class_counts)
        if metrics:
            metrics["Gini Coefficient"] = round(gini, 3)
            metrics["Imbalance Ratio"] = round(ir, 3)
            metrics["Entropy"] = round(ent, 3)
            results[threshold_str] = metrics
            print(f"Results for {threshold}: {metrics}")
        else:
            print(f"Metrics not found for threshold {threshold}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

def frange(start, stop, step):
    while start < stop:
        yield round(start, 2)
        start += step

if __name__ == "__main__":
    main()

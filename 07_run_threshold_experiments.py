import subprocess
import re
import json
import os

PRUNING_SCRIPT = "05LUTpruning.py"
INFERENCE_SCRIPT = "06Inference.py"
RESULTS_FILE = "threshold_results.json"

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
        if metrics:
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

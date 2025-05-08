import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('threshold_results.json', 'r') as f:
    data = json.load(f)

# Prepare thresholds and metrics
thresholds = np.arange(0.8, 1.01, 0.01)
thresholds_str = [f"{t:.2f}" for t in thresholds]

accuracy = []
mean_time = []
faiss_size = []

for t in thresholds:
    t_str = f"{t:.2f}"
    entry = data.get(t_str)
    if entry is None:
        # Try matching JSON key with float-cast string (removes trailing zeros)
        t_str_alt = str(float(t_str))
        entry = data.get(t_str_alt)
    if entry:
        accuracy.append(entry["Multiclass Accuracy"])
        mean_time.append(entry["Mean processing time per image (seconds)"])
        faiss_size.append(entry["FAISS index file size (MB)"])
    else:
        accuracy.append(np.nan)
        mean_time.append(np.nan)
        faiss_size.append(np.nan)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",
    "axes.grid": True,
    "grid.linestyle": ":"
})

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Similarity Threshold')
ax1.set_ylabel('Multiclass Accuracy (%)', color=color1)
l1, = ax1.plot(thresholds, accuracy, color=color1, marker='o', linestyle='-', label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color1)

# Second y-axis for FAISS size
ax2 = ax1.twinx()
color3 = 'tab:green'
ax2.set_ylabel('FAISS index file size (MB)', color=color3)
l3, = ax2.plot(thresholds, faiss_size, color=color3, marker='^', linestyle='-.', label='FAISS index file size')
ax2.tick_params(axis='y', labelcolor=color3)

# Third y-axis for mean processing time
ax3 = ax1.twinx()
color2 = 'tab:red'
ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
ax3.set_ylabel('Mean processing time per image (s)', color=color2)
l2, = ax3.plot(thresholds, mean_time, color=color2, marker='s', linestyle='--', label='Mean processing time per image')
ax3.tick_params(axis='y', labelcolor=color2)

# Unified legend
lines = [l1, l2, l3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=11)

plt.title('Similarity Threshold vs Accuracy, Processing Time, and FAISS Size', fontsize=15, fontweight='bold', family='serif')
plt.tight_layout()
plt.savefig('08_thresholds_comparison.png', dpi=300)
plt.show()
plt.close()

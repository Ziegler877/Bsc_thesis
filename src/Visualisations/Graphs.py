import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# --- SETUP: Academic Light Theme ---
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Generate dummy data
np.random.seed(42)
auth_a = np.random.randn(20, 2) * 0.5 + [2, 5]
auth_b = np.random.randn(20, 2) * 0.6+ [6, 5]
auth_c = np.random.randn(20, 2) * 0.45 + [4, 2]
new_text = np.array([[4.5, 4.5]])

colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
labels = ['Author A', 'Author B', 'Author C']

# --- PANEL 1: Closed-Set Classification ---
ax1.scatter(auth_a[:,0], auth_a[:,1], c=colors[0], label='Author A', alpha=0.6)
ax1.scatter(auth_b[:,0], auth_b[:,1], c=colors[1], label='Author B', alpha=0.6)
ax1.scatter(auth_c[:,0], auth_c[:,1], c=colors[2], label='Author C', alpha=0.6)
ax1.scatter(new_text[:,0], new_text[:,1], c='red', marker='*', s=200, label='New Text')

# Draw fake SVM boundaries
ax1.plot([0, 4.5], [0, 4.5], 'k--', alpha=0.7)
ax1.plot([3.5, 5.7], [8, 0], 'k--', alpha=0.7)


ax1.set_xlim(0, 8); ax1.set_ylim(0, 8)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.legend(loc='upper left')

# --- PANEL 2: Open-Set Retrieval ---
for i, data in enumerate([auth_a, auth_b, auth_c]):
    # Lighter dots (shorter training/history)
    ax1.scatter(data[:,0], data[:,1], c=colors[i], alpha=0.2, s=30)
    ax2.scatter(data[:,0], data[:,1], c=colors[i], alpha=0.2, s=30)

# Centroids and Thresholds
centroids = [np.mean(auth_a, axis=0), np.mean(auth_b, axis=0), np.mean(auth_c, axis=0)]
for i, c in enumerate(centroids):
    ax2.scatter(c[0], c[1], c=colors[i], marker='X', s=150, edgecolor='black', zorder=5)
    # Threshold boundary (Author B is the match)
    radius = 2.5 if i == 1 else 1.7
    circle = Circle(c, radius, color=colors[i], fill=False, linestyle=':', linewidth=2, label=f'{labels[i]} Threshold')
    ax2.add_patch(circle)
    ax2.plot([new_text[0,0], c[0]], [new_text[0,1], c[1]], color='gray', alpha=0.3, linestyle='-')

ax2.scatter(new_text[:,0], new_text[:,1], c='red', marker='*', s=250, label='New Text', edgecolors='black')
ax2.set_xlim(0, 8); ax2.set_ylim(0, 8)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.grid(False) # Ensure no grid lines

# Custom Legend for Panel 2
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Texts (Light)'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='black', label='Author Centroid'),
    Line2D([0], [0], color='black', linestyle=':', label='Threshold Boundary'),
    Line2D([0], [0], marker='*', color='w',  markerfacecolor='red',markersize=20, label='New Text')
]
ax2.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig('comparison_diagram.pdf', format='pdf')
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Data from your thesis
datasets = ['C50 (Reuters)', 'DarkReddit']
sota = [73.08, 84.60]        # SVM Baseline & VeriDark
zero_shot = [61.36, 61.51]   # E5 Small (Chunk) & Llama 3 (Dyn)
lora = [65.82, 93.58]        # Llama 3 (LoRA) & Llama 3 (LoRA)

x = np.arange(len(datasets))
width = 0.25  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create grouped bars
rects1 = ax.bar(x - width, sota, width, label='SOTA (Closed-Set)', color='#7f7f7f')
rects2 = ax.bar(x, zero_shot, width, label='Best Zero-Shot', color='#aec7e8')
rects3 = ax.bar(x + width, lora, width, label='Best LoRA', color='#1f77b4')

# Add formatting and labels
ax.set_ylabel('C - Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax.set_ylim(0, 105)

# Function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('performance_barchart.pdf', format='pdf', dpi=300)
plt.show()
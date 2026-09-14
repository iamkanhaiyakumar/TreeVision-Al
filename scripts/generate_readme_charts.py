import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('docs/images', exist_ok=True)

# 1. Bar Chart: Model Performance Comparison
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

metrics = ['Precision', 'Recall', 'F1-Score']
deepforest_vals = [72.73, 48.75, 58.37]
yolo_vals = [23.52, 44.09, 30.67]

x = np.arange(len(metrics))
width = 0.35

rects1 = ax1.bar(x - width/2, deepforest_vals, width, label='DeepForest Baseline (Production)', color='#1b4d3e')
rects2 = ax1.bar(x + width/2, yolo_vals, width, label='Custom YOLOv8s (Option B Cloud GPU)', color='#2b6cb0')

ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax1.set_title('Detection Metrics on Held-Out Test Set (Yellowstone, 279 Trees)', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.legend(loc='upper right', frameon=True)

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontweight='bold', color='#1b4d3e')
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontweight='bold', color='#2b6cb0')

# Runtime Chart
models = ['DeepForest\n(RetinaNet)', 'Custom YOLOv8s\n(Option B)']
speeds = [85.0, 14.2]
colors_list = ['#718096', '#319795']

rects3 = ax2.bar(models, speeds, width=0.45, color=colors_list)
ax2.set_ylabel('Latency per 640px Tile (ms)', fontsize=11, fontweight='bold')
ax2.set_title('Inference Speed Comparison (CPU/Edge Hardware)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 100)

for rect in rects3:
    h = rect.get_height()
    ax2.annotate(f'{h:.1f} ms', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontweight='bold')

ax2.annotate('⚡ ~6x Faster!', xy=(1, 14.2), xytext=(0.7, 35),
             arrowprops=dict(facecolor='#319795', shrink=0.08, width=2, headwidth=6),
             fontweight='bold', color='#319795', fontsize=11)

plt.tight_layout()
plt.savefig('docs/images/benchmark_comparison.png', dpi=300)
plt.close()
print('Generated docs/images/benchmark_comparison.png')

# 2. Precision-Recall Curve across Thresholds
with open('evaluation/threshold_optimization.json', 'r') as f:
    thresh_data = json.load(f)

df_thresh = thresh_data['deepforest']
confs = [d['confidence'] for d in df_thresh]
precs = [d['precision'] * 100 for d in df_thresh]
recs = [d['recall'] * 100 for d in df_thresh]
f1s = [d['f1'] * 100 for d in df_thresh]

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
ax.plot(confs, precs, marker='o', linewidth=2.5, color='#2f855a', label='Precision (%)')
ax.plot(confs, recs, marker='s', linewidth=2.5, color='#c53030', label='Recall (%)')
ax.plot(confs, f1s, marker='^', linewidth=2, linestyle='--', color='#2b6cb0', label='F1-Score (%)')

# Highlight peak precision
ax.annotate('Peak Precision: 85.3%\n(Only 10 False Alarms)', xy=(0.55, 85.29), xytext=(0.43, 90),
            arrowprops=dict(facecolor='#2f855a', shrink=0.08, width=2, headwidth=6),
            fontweight='bold', color='#2f855a', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6fffa', edgecolor='#2f855a'))

# Highlight balanced operating point
ax.annotate('Balanced Operating Point (0.30)\nPrecision: 72.7% | Recall: 48.8%', xy=(0.30, 58.37), xytext=(0.20, 68),
            arrowprops=dict(facecolor='#2b6cb0', shrink=0.08, width=2, headwidth=6),
            fontweight='bold', color='#2b6cb0', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ebf8ff', edgecolor='#2b6cb0'))

ax.set_xlabel('Confidence Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=11, fontweight='bold')
ax.set_title('DeepForest Precision-Recall Operating Curve (Option A Optimization)', fontsize=12, fontweight='bold')
ax.set_xlim(0.22, 0.58)
ax.set_ylim(15, 100)
ax.legend(loc='lower left', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('docs/images/threshold_operating_curve.png', dpi=300)
plt.close()
print('Generated docs/images/threshold_operating_curve.png')

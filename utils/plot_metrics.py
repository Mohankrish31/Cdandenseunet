import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# === Load the metrics CSV === #
df = pd.read_csv("metrics_results_with_ebcm.csv")
# === Compute summary statistics === #
metrics = ["PSNR", "SSIM", "LPIPS", "EBCM"]
means = [df[m].mean() for m in metrics]
stds  = [df[m].std() for m in metrics]
# === Plot bar chart with error bars === #
plt.figure(figsize=(8,6))
x_pos = np.arange(len(metrics))
colors = ['blue', 'green', 'red', 'purple']
plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.8, color=colors, capsize=10)
plt.xticks(x_pos, metrics)
plt.ylabel("Metric Value")
plt.title("Summary Metrics (Mean ± Std)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Annotate values
for i, (mean, std) in enumerate(zip(means, stds)):
    plt.text(i, mean + std + 0.01, f"{mean:.3f}±{std:.3f}", ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

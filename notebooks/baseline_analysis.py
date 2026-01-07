import json
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from pathlib import Path

# -------- load metrics --------
metrics_path = Path("runs/kmers_lr_mf_k3/metrics.json")
with open(metrics_path) as f:
    metrics = json.load(f)

thr_info = metrics["threshold_info"]
curve = thr_info["curve_sampled"]
best_thr = thr_info["threshold"]
best_f1 = thr_info["best_micro_f1"]

thresholds = [t for t, _ in curve]
f1_scores = [f for _, f in curve]

# -------- plot --------
plt.figure(figsize=(6, 4))
plt.plot(thresholds, f1_scores, marker="o", linewidth=2)
plt.axvline(
    best_thr,
    linestyle="--",
    linewidth=2,
    label=f"Best threshold = {best_thr:.2f}"
)

plt.xlabel("Decision threshold")
plt.ylabel("Micro-F1 score")
plt.title("Threshold vs Micro-F1 (Validation)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig("./visualization/baseline_threshold_vs_f1.png", dpi=300)

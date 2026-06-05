import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"]
THRESHOLD = 0.3


def plot_drift_over_time(results):
    batch_indices = [r["batch"] for r in results]
    drift_matrix = np.array([r["drift_scores"] for r in results])
    alerts = [r["alert"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for f in range(len(FEATURE_NAMES)):
        ax.plot(batch_indices, drift_matrix[:, f],
                label=FEATURE_NAMES[f], color=colors[f], linewidth=2)

    for i, alert in enumerate(alerts):
        if alert:
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.08)
            for f in range(len(FEATURE_NAMES)):
                ax.plot(i, drift_matrix[i, f], "ro", markersize=6)

    ax.axhline(y=THRESHOLD, color="red", linestyle="--",
               linewidth=1.2, label=f"Threshold ({THRESHOLD})")
    ax.set_xlabel("Batch Number", fontsize=12)
    ax.set_ylabel("Drift Score", fontsize=12)
    ax.set_title("Feature Drift Over Time", fontsize=14, fontweight="bold")
    ax.set_xticks(batch_indices)
    ax.grid(True, linestyle="--", alpha=0.4)

    alert_patch = mpatches.Patch(color="red", alpha=0.2, label="Alert Zone")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [alert_patch], labels + ["Alert Zone"],
              loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("drift_over_time.png", dpi=150)
    plt.show()
    print("Saved: drift_over_time.png")


def plot_feature_drift_bar(results):
    drift_matrix = np.array([r["drift_scores"] for r in results])
    avg_drift = drift_matrix.mean(axis=0)

    colors = ["#2ca02c" if v < THRESHOLD else "#d62728" for v in avg_drift]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(FEATURE_NAMES, avg_drift, color=colors,
                  edgecolor="black", linewidth=0.7)

    ax.axhline(y=THRESHOLD, color="red", linestyle="--",
               linewidth=1.2, label=f"Threshold ({THRESHOLD})")

    for bar, val in zip(bars, avg_drift):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Average Drift Score", fontsize=12)
    ax.set_title("Average Drift Per Feature (All Batches)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("feature_drift_bar.png", dpi=150)
    plt.show()
    print("Saved: feature_drift_bar.png")
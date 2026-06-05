import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.data_simulator import generate_batch_sequence
from src.windowing import run_windowing
from src.visualizer import plot_drift_over_time, plot_feature_drift_bar
from src.baseline import compute_baseline, save_baseline

THRESHOLD = 0.3
N_BATCHES = 20
DRIFT_START = 12
N_SAMPLES = 100


def train_model():
    print("Step 1: Training model on Iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target

    X = df.drop(columns=["species"])
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    baseline = compute_baseline(X_train)
    save_baseline(baseline)

    print(f"Model trained. ({len(X_train)} training rows)\n")
    return X_train.values


def simulate_batches(X_train):
    print("Step 2: Simulating data batches...")
    batches = generate_batch_sequence(
        X_train,
        n_batches=N_BATCHES,
        drift_start=DRIFT_START,
        n_samples=N_SAMPLES
    )
    print(f"Generated {N_BATCHES} batches — "
          f"{DRIFT_START} clean, {N_BATCHES - DRIFT_START} drifted.\n")
    return batches


def detect_drift(X_train, batches):
    print("Step 3: Running drift detection...")
    results = run_windowing(X_train, batches, threshold=THRESHOLD)
    total_alerts = sum(1 for r in results if r["alert"])
    print(f"Drift detection complete. "
          f"{total_alerts}/{N_BATCHES} batches triggered alerts.\n")
    return results


def visualize(results):
    print("Step 4: Generating visualizations...")
    plot_drift_over_time(results)
    plot_feature_drift_bar(results)
    print("Visualizations saved.\n")


if __name__ == "__main__":
    print("=" * 50)
    print("       ML MODEL DRIFT MONITOR")
    print("=" * 50 + "\n")

    X_train = train_model()
    batches = simulate_batches(X_train)
    results = detect_drift(X_train, batches)
    visualize(results)

    print("=" * 50)
    print("Pipeline complete.")
    print("=" * 50)
import pickle
import pandas as pd
import numpy as np
import os

def compute_baseline(df: pd.DataFrame) -> dict:
    baseline = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        values = df[col].dropna().values
        baseline[col] = {
            "mean":   float(np.mean(values)),
            "std":    float(np.std(values)),
            "min":    float(np.min(values)),
            "max":    float(np.max(values)),
            "values": values  # keep as numpy array, not list
        }
    return baseline

def save_baseline(baseline: dict, path: str = "data/baseline.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(baseline, f)
    print(f"[Baseline] Saved to {path}")

def load_baseline(path: str = "data/baseline.pkl") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline file not found: {path}")
    with open(path, "rb") as f:
        baseline = pickle.load(f)
    print(f"[Baseline] Loaded from {path}")
    return baseline


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target

    baseline = compute_baseline(df.drop(columns=["species"]))
    save_baseline(baseline)
    loaded = load_baseline()

    print("\nSample baseline entry for 'sepal length (cm)':")
    for k, v in loaded["sepal length (cm)"].items():
        if k != "values":
            print(f"  {k}: {v:.4f}")
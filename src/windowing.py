import numpy as np
import pandas as pd
from src.drift_calculator import calculate_drift
from src.alerting import check_alerts

FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"]

def build_baseline_dict(X_train):
    df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    baseline = {}
    for feature in FEATURE_NAMES:
        baseline[feature] = {"values": df[feature].values}
    return baseline

def run_windowing(X_train, batches, threshold=0.5):
    results = []
    baseline = build_baseline_dict(X_train)

    for i, batch in enumerate(batches):
        batch_df = pd.DataFrame(batch, columns=FEATURE_NAMES)
        drift_result = calculate_drift(baseline, batch_df)

        # Extract KS scores as a simple list for visualization
        drift_scores = [drift_result[f]["ks"] for f in FEATURE_NAMES]

        # check_alerts returns list of drifted features — alert if non-empty
        drifted = check_alerts(drift_result, ks_threshold=threshold)
        alert = len(drifted) > 0

        results.append({
            "batch": i,
            "drift_scores": drift_scores,
            "alert": alert
        })

    return results
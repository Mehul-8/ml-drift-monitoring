import pandas as pd
import numpy as np
import scipy.stats as stats

def ks_drift_score(baseline_array:np.ndarray ,new_array:np.ndarray)->dict:
    statistic,p_value=stats.ks_2samp(baseline_array,new_array)
    return {
        "statistic": round(float(statistic),4),
        "p_value": round(float(p_value),4)
    }

def  psi_score(baseline_array:np.ndarray, new_array:np.ndarray, bins: int=10)-> float:
    epsilon = 1e-4
    _, bin_edges = np.histogram(baseline_array, bins=bins)
    baseline_counts, _ = np.histogram(baseline_array, bins=bin_edges)
    new_counts, _ = np.histogram(new_array, bins=bin_edges)
    baseline_pct = baseline_counts / len(baseline_array)
    new_pct = new_counts / len(new_array)
 
    baseline_pct = np.where(baseline_pct == 0, epsilon, baseline_pct)
    new_pct = np.where(new_pct == 0, epsilon, new_pct)
    psi_values = (new_pct - baseline_pct) * np.log(new_pct / baseline_pct)
    psi = float(np.sum(psi_values))
    """
    PSI < 0.1 → stable
    PSI 0.1 - 0.2 → moderate drift
    PSI > 0.2 → significant drift
    """
    return round(psi, 4)

def calculate_drift(baseline: dict,new_batch: pd.DataFrame) -> dict:
    drift_results = {}

    for feature, stats_dict in baseline.items():
        if feature not in new_batch.columns:
            print(f"[Warning] Feature '{feature}' not found in new batch. Skipping.")
            continue
        baseline_values = stats_dict["values"]
        new_values      = new_batch[feature].dropna().values
        if len(new_values) == 0:
            print(f"[Warning] Feature '{feature}' has no valid values in batch. Skipping.")
            continue

        ks_result = ks_drift_score(baseline_values, new_values)
        psi       = psi_score(baseline_values, new_values)
        drift_results[feature] = {
            "ks":      ks_result["statistic"],
            "p_value": ks_result["p_value"],
            "psi":     psi
        }
    return drift_results


if __name__ == "__main__":
    from baseline import compute_baseline
    from sklearn.datasets import make_classification
    import pandas as pd

    
    X_train, _ = make_classification(n_samples=1000, n_features=5, random_state=42)
    df_train   = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(5)])
    baseline   = compute_baseline(df_train)

    
    X_clean, _ = make_classification(n_samples=200, n_features=5, random_state=99)
    df_clean   = pd.DataFrame(X_clean, columns=[f"feature_{i}" for i in range(5)])

    
    X_drift    = X_clean.copy()
    X_drift   += 2.5  # artificial drift
    df_drift   = pd.DataFrame(X_drift, columns=[f"feature_{i}" for i in range(5)])

    print("\n--- Clean batch ---")
    clean_results = calculate_drift(baseline, df_clean)
    for f, r in clean_results.items():
        print(f"  {f}: KS={r['ks']}, PSI={r['psi']}")

    print("\n--- Drifted batch ---")
    drift_results = calculate_drift(baseline, df_drift)
    for f, r in drift_results.items():
        print(f"  {f}: KS={r['ks']}, PSI={r['psi']}")
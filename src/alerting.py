import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("drift_monitor")


def check_alerts(drift_results: dict,
                 ks_threshold: float = 0.3,
                 psi_threshold: float = 0.2) -> list:
    drifted_features = []

    print("\n" + "=" * 55)
    print(f"  Drift Alert Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    for feature, scores in drift_results.items():
        ks  = scores["ks"]
        psi = scores["psi"]
        p   = scores["p_value"]

        ks_drifted  = ks  > ks_threshold
        psi_drifted = psi > psi_threshold

        if ks_drifted or psi_drifted:
            drifted_features.append(feature)
            logger.warning(
                f"[DRIFT DETECTED] '{feature}' — "
                f"KS={ks:.3f} (thresh={ks_threshold}), "
                f"PSI={psi:.3f} (thresh={psi_threshold}), "
                f"p-value={p:.4f}"
            )
        else:
            logger.info(
                f"[STABLE]         '{feature}' — "
                f"KS={ks:.3f}, PSI={psi:.3f}"
            )

    print("=" * 55)

    if drifted_features:
        print(f"\n  {len(drifted_features)} feature(s) drifted: {drifted_features}")
    else:
        print("\n  All features stable. No action needed.")

    print()
    return drifted_features


def summarize_drift(drift_results: dict) -> dict:
    summary = {}
    for feature, scores in drift_results.items():
        summary[feature] = {
            "ks":      scores["ks"],
            "psi":     scores["psi"],
            "drifted": scores["ks"] > 0.3 or scores["psi"] > 0.2
        }
    return summary


if __name__ == "__main__":
    fake_results = {
        "feature_0": {"ks": 0.08, "psi": 0.05, "p_value": 0.81},
        "feature_1": {"ks": 0.55, "psi": 0.38, "p_value": 0.001},
        "feature_2": {"ks": 0.12, "psi": 0.09, "p_value": 0.44},
        "feature_3": {"ks": 0.41, "psi": 0.22, "p_value": 0.003},
    }

    drifted = check_alerts(fake_results)
    print("Drifted features:", drifted)

    summary = summarize_drift(fake_results)
    print("\nSummary:")
    for f, s in summary.items():
        print(f"  {f}: {s}")
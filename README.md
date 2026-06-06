# ML Model Drift Monitor

A lightweight Python tool that monitors machine learning models for data drift in production. It detects when incoming data starts diverging from the training distribution using statistical tests, logs alerts, and visualizes drift over time.

---

## What It Does

- Trains a baseline model on the Iris dataset
- Simulates incoming data batches — some clean, some drifted
- Detects feature drift using **KS Test** and **PSI (Population Stability Index)**
- Logs alerts when drift crosses a configurable threshold
- Generates visualizations showing drift scores over time and per feature

---

## Project Structure

```
ML Model drift/
│
├── main.py                   # Entry point — runs the full pipeline
│
├── src/
│   ├── __init__.py
│   ├── baseline.py           # Computes and saves baseline statistics
│   ├── data_simulation.py    # Generates clean and drifted data batches
│   ├── drift_calculator.py   # KS Test and PSI drift scoring
│   ├── windowing.py          # Runs drift detection across all batches
│   ├── alerting.py           # Logs and reports drift alerts
│   └── visualization.py      # Plots drift scores over time
│
└── data/
    └── baseline.pkl          # Saved baseline stats (auto-generated on first run)
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ml-model-drift-monitor.git
cd ml-model-drift-monitor
```

### 2. Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

### 3. Run the Pipeline

```bash
python main.py
```

---

## How It Works

### Pipeline Steps

```
Step 1 → Train model on Iris dataset, save baseline statistics
Step 2 → Simulate 20 data batches (12 clean + 8 drifted)
Step 3 → Run drift detection on each batch using KS Test + PSI
Step 4 → Generate and save visualizations
```

### Drift Detection Methods

| Method | What It Measures | Alert Threshold |
|---|---|---|
| KS Test | Distribution shape difference | > 0.3 |
| PSI | Population stability shift | > 0.2 |

A batch triggers an alert if **any feature** exceeds either threshold.

### Drift Simulation

Clean batches are sampled from the same distribution as training data. Drifted batches have their mean shifted by `1.5 × std` to simulate real-world distribution shift.

---

## Output

After running `main.py` you will see:

**Terminal output:**
```
==================================================
       ML MODEL DRIFT MONITOR
==================================================

Step 1: Training model on Iris dataset...
Step 2: Simulating data batches...
Step 3: Running drift detection...
Step 4: Generating visualizations...
==================================================
Pipeline complete.
==================================================
```

**Saved files:**
- `drift_over_time.png` — line chart of drift scores per feature across all batches
- `feature_drift_bar.png` — bar chart of average drift per feature

---

## Configuration

All key parameters are set at the top of `main.py`:

```python
THRESHOLD   = 0.3   # KS score threshold for drift alert
N_BATCHES   = 20    # Total number of batches to simulate
DRIFT_START = 12    # Batch index where drift begins
N_SAMPLES   = 100   # Number of samples per batch
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| numpy | any | Array operations |
| pandas | any | DataFrame handling |
| scipy | any | KS Test |
| scikit-learn | any | Model training, Iris dataset |
| matplotlib | any | Visualizations |

Install all at once:
```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

---

## Team

Built as a 2-week college project by a 2-person team.

| Member | Responsibilities |
|---|---|
| Person A | Drift engine — baseline, KS test, PSI, alerting |
| Person B | Data pipeline — simulation, windowing, visualization, main.py |

---

## Concepts Used

- **Data Drift** — when incoming data distribution shifts away from training data
- **KS Test** — non-parametric test comparing two distributions
- **PSI** — measures how much a distribution has shifted between two samples
- **Windowing** — processing data in time-based batches to track drift over time

---

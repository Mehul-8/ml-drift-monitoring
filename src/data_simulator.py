import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

N_BATCHES = 20
DRIFT_START = 12
N_SAMPLES = 100


def generate_batch_sequence(X_train, n_batches, drift_start, n_samples):
    batches = []
    n_features = X_train.shape[1]
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    for i in range(n_batches):
        if i < drift_start:
            batch = np.random.normal(loc=mean, scale=std, size=(n_samples, n_features))
        else:
            drifted_mean = mean + 1.5 * std
            batch = np.random.normal(loc=drifted_mean, scale=std, size=(n_samples, n_features))
        batches.append(batch)

    return batches


def train_model():
    print("Step 1: Training model...")
    df = pd.read_csv("iris.csv")        # ← fix sep if needed
    df.columns = df.columns.str.strip()

    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    np.save("X_train.npy", X_train.values)
    print(f"Model trained. X_train saved. ({len(X_train)} rows)\n")
    return X_train.values


def simulate(X_train):
    print("Step 2: Simulating data batches...")
    batches = generate_batch_sequence(
        X_train,
        n_batches=N_BATCHES,
        drift_start=DRIFT_START,
        n_samples=N_SAMPLES
    )
    print(f"Generated {N_BATCHES} batches — {DRIFT_START} clean, {N_BATCHES - DRIFT_START} drifted.\n")
    return batches
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from tqdm import tqdm

from app.model.DataPreparation import (
    generate_cluster, preprocess, isClusterDead, isSplitBrain, isSingleType
)
from app.model.metrics import compute_metrics, save_metrics, print_metrics

MODEL_PATH = "split_brain_model_rf.pkl"
METRICS_PATH = "metrics_rf.json"
BUFFER_PATH = "rf_teach_buffer.pkl"

_cached_model = None


def _get_model() -> RandomForestClassifier:
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model()
    return _cached_model


def generate_dataset_balanced(n_normal: int, n_splitbrain: int):
    X, y = [], []
    for _ in tqdm(range(n_normal), desc="[RF] Normal data"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    for _ in tqdm(range(n_splitbrain), desc="[RF] Split-brain data"):
        nodes, matrix = generate_cluster()
        while not isSplitBrain(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(1)
    return np.array(X), np.array(y)


def generate_dataset_natural(n_samples: int):
    X, y = [], []
    for _ in tqdm(range(n_samples), desc="[RF] Natural eval data"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    return np.array(X), np.array(y)


def train_model(
    n_normal: int = 500_000,
    n_splitbrain: int = 100_000,
    eval_size: int = 20_000,
) -> RandomForestClassifier:

    global _cached_model

    print("[RF] Генерація тренувальних даних...")
    X_train, y_train = generate_dataset_balanced(n_normal, n_splitbrain)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    print(f"[RF] Train: {len(y_train)} (neg={neg}, pos={pos})")

    print("[RF] Генерація eval-вибірки (природній розподіл)...")
    X_eval, y_eval = generate_dataset_natural(eval_size)
    print(f"[RF] Eval SB rate = {y_eval.mean():.3f}")

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[RF] Готово за {train_time:.1f}с")

    idx = np.random.choice(len(X_train), min(10_000, len(X_train)), replace=False)
    t0 = time.time()
    y_proba_tr = model.predict_proba(X_train[idx])[:, 1]
    inf_ms_tr = (time.time() - t0) / len(idx) * 1000

    tr_metrics = compute_metrics(
        "RandomForest", y_train[idx], y_proba_tr,
        label="train_sample", inference_ms=inf_ms_tr
    )
    tr_metrics["train_time_seconds"] = round(train_time, 2)
    tr_metrics["n_estimators"] = model.n_estimators
    print_metrics(tr_metrics)
    save_metrics(tr_metrics, METRICS_PATH)

    t0 = time.time()
    y_proba_ev = model.predict_proba(X_eval)[:, 1]
    inf_ms_ev = (time.time() - t0) / len(X_eval) * 1000

    ev_metrics = compute_metrics(
        "RandomForest", y_eval, y_proba_ev,
        label="eval_natural", inference_ms=inf_ms_ev
    )
    ev_metrics["train_time_seconds"] = round(train_time, 2)
    print_metrics(ev_metrics)
    save_metrics(ev_metrics, METRICS_PATH)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    _cached_model = model
    return model

def load_model() -> RandomForestClassifier:
    if Path(MODEL_PATH).exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_model()


def predict_rf(nodes, matrix) -> float:
    if isSingleType(nodes):
        return 0.0
    model = _get_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    t0 = time.time()
    proba = model.predict_proba(x_input)[0, 1]
    elapsed_ms = (time.time() - t0) * 1000
    print(f"RF predict: {proba:.4f} ({elapsed_ms:.2f} ms)")
    return float(proba)

def teach_rf(nodes, matrix) -> float:
    BUFFER_SIZE = 500
    ADDITIONAL_TREES = 50

    if Path(BUFFER_PATH).exists():
        with open(BUFFER_PATH, "rb") as f:
            buffer = pickle.load(f)
    else:
        buffer = {"X": [], "y": []}

    x_input = preprocess(nodes, matrix)
    label = int(isSplitBrain(nodes, matrix))
    buffer["X"].append(x_input)
    buffer["y"].append(label)

    model = _get_model()
    proba = model.predict_proba(x_input.reshape(1, -1))
    current_loss = log_loss([label], proba, labels=[0, 1])

    if len(buffer["X"]) >= BUFFER_SIZE:
        print(f"[RF] Буфер {BUFFER_SIZE}. +{ADDITIONAL_TREES} дерев...")
        X_buf = np.array(buffer["X"])
        y_buf = np.array(buffer["y"])

        small_model = RandomForestClassifier(
            n_estimators=ADDITIONAL_TREES,
            max_depth=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=None,
            n_jobs=-1,
        )
        small_model.fit(X_buf, y_buf)

        model.estimators_ += small_model.estimators_
        model.n_estimators += small_model.n_estimators

        global _cached_model
        _cached_model = model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        buffer = {"X": [], "y": []}

    with open(BUFFER_PATH, "wb") as f:
        pickle.dump(buffer, f)
    return current_loss * 1_000_000


def benchmark_rf(n_samples: int = 10_000):
    print("[RF] Бенчмарк...")
    model = _get_model()
    X, y = generate_dataset_natural(n_samples)
    t0 = time.time()
    y_proba = model.predict_proba(X)[:, 1]
    inf_ms = (time.time() - t0) / n_samples * 1000
    metrics = compute_metrics("RandomForest", y, y_proba,
                              label="benchmark_natural", inference_ms=inf_ms)
    print_metrics(metrics)
    save_metrics(metrics, METRICS_PATH)
    return metrics
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from tqdm import tqdm

from app.model.DataPreparation import (
    generate_cluster, preprocess, isClusterDead, isSplitBrain, isSingleType
)
from app.model.metrics import compute_metrics, save_metrics, print_metrics

MODEL_PATH = "models/split_brain_model_gb.pkl"
METRICS_PATH = "metrics_gb.json"
BUFFER_PATH = "gb_teach_buffer.pkl"

_cached_model = None


def _get_model() -> GradientBoostingClassifier:
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model()
    return _cached_model


def generate_dataset_balanced(n_normal: int, n_splitbrain: int):
    X, y = [], []
    for _ in tqdm(range(n_normal), desc="[GB] Normal data"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    for _ in tqdm(range(n_splitbrain), desc="[GB] Split-brain data"):
        nodes, matrix = generate_cluster()
        while not isSplitBrain(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(1)
    return np.array(X), np.array(y)


def generate_dataset_natural(n_samples: int):
    X, y = [], []
    for _ in tqdm(range(n_samples), desc="[GB] Natural eval data"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    return np.array(X), np.array(y)


def train_model(
        n_normal: int = 900_000,
        n_splitbrain: int = 500_000,
        eval_size: int = 20_000,
) -> GradientBoostingClassifier:
    global _cached_model

    print("[GB] Генерація тренувальних даних...")
    X_train, y_train = generate_dataset_balanced(n_normal, n_splitbrain)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    ratio = neg / pos
    sample_weights = np.where(y_train == 1, ratio, 1.0)
    print(f"[GB] Train: {len(y_train)} (neg={neg}, pos={pos}, ratio={ratio:.2f})")

    print("[GB] Генерація eval-вибірки (природній розподіл)...")
    X_eval, y_eval = generate_dataset_natural(eval_size)
    print(f"[GB] Eval SB rate = {y_eval.mean():.3f}")

    model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=9,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        max_features=0.8,
        random_state=42,
        warm_start=True,
        verbose=1,
    )

    t0 = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    train_time = time.time() - t0
    print(f"[GB] Готово за {train_time:.1f}с")

    idx = np.random.choice(len(X_train), min(10_000, len(X_train)), replace=False)
    t0 = time.time()
    y_proba_tr = model.predict_proba(X_train[idx])[:, 1]
    inf_ms_tr = (time.time() - t0) / len(idx) * 1000

    tr_metrics = compute_metrics(
        "GradientBoosting", y_train[idx], y_proba_tr,
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
        "GradientBoosting", y_eval, y_proba_ev,
        label="eval_natural", inference_ms=inf_ms_ev
    )
    ev_metrics["train_time_seconds"] = round(train_time, 2)
    print_metrics(ev_metrics)
    save_metrics(ev_metrics, METRICS_PATH)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    _cached_model = model
    return model


def load_model() -> GradientBoostingClassifier:
    if Path(MODEL_PATH).exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_model()


def predict_gb(nodes, matrix) -> float:
    if isSingleType(nodes):
        return 0.0
    model = _get_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    t0 = time.time()
    proba = model.predict_proba(x_input)[0, 1]
    elapsed_ms = (time.time() - t0) * 1000
    print(f"GB predict: {proba:.4f} ({elapsed_ms:.2f} ms)")
    return float(proba)


def teach_gb(nodes, matrix) -> float:
    BUFFER_SIZE = 300
    ADDITIONAL_TREES = 20

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
        print(f"[GB] Буфер {BUFFER_SIZE}. +{ADDITIONAL_TREES} дерев...")
        X_buf = np.array(buffer["X"])
        y_buf = np.array(buffer["y"])
        pos_buf = (y_buf == 1).sum()
        neg_buf = (y_buf == 0).sum()
        sw = np.where(y_buf == 1, neg_buf / max(pos_buf, 1), 1.0)

        model.n_estimators += ADDITIONAL_TREES
        model.fit(X_buf, y_buf, sample_weight=sw)

        global _cached_model
        _cached_model = model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        buffer = {"X": [], "y": []}

    with open(BUFFER_PATH, "wb") as f:
        pickle.dump(buffer, f)
    return current_loss * 1


def benchmark_gb(n_samples: int = 10_000):
    print("[GB] Бенчмарк...")
    model = _get_model()
    X, y = generate_dataset_natural(n_samples)
    t0 = time.time()
    y_proba = model.predict_proba(X)[:, 1]
    inf_ms = (time.time() - t0) / n_samples * 1000
    metrics = compute_metrics("GradientBoosting", y, y_proba,
                              label="benchmark_natural", inference_ms=inf_ms)
    print_metrics(metrics)
    save_metrics(metrics, METRICS_PATH)
    return metrics

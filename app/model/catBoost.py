import os
import pickle
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from tqdm import tqdm

from app.model.DataPreparation import (
    generate_cluster, preprocess, isClusterDead, isSplitBrain, isSingleType
)
from app.model.metrics import compute_metrics, save_metrics, print_metrics

MODEL_PATH = "split_brain_model_cb.pkl"
METRICS_PATH = "metrics_cb.json"
BUFFER_PATH = "cb_teach_buffer.pkl"

_cached_model = None


def _get_model() -> CatBoostClassifier:
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model()
    return _cached_model


def generate_dataset_balanced(n_normal: int, n_splitbrain: int):
    X, y = [], []
    for _ in tqdm(range(n_normal), desc="[CB] Normal data"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    for _ in tqdm(range(n_splitbrain), desc="[CB] Split-brain data"):
        nodes, matrix = generate_cluster()
        while not isSplitBrain(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(1)
    return np.array(X), np.array(y)


def generate_dataset_natural(n_samples: int):
    X, y = [], []
    for _ in tqdm(range(n_samples), desc="[CB] Natural eval data"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    return np.array(X), np.array(y)



def train_model(
    n_normal: int = 900_000,
    n_splitbrain: int = 300_000,
    eval_size: int = 20_000,
) -> CatBoostClassifier:
    global _cached_model

    print("[CB] Генерація тренувальних даних...")
    X_train, y_train = generate_dataset_balanced(n_normal, n_splitbrain)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos
    print(f"[CB] Train: {len(y_train)} (neg={neg}, pos={pos}, "
          f"ratio={scale_pos_weight:.2f})")

    print("[CB] Генерація eval-вибірки (природній розподіл)...")
    X_eval, y_eval = generate_dataset_natural(eval_size)
    print(f"[CB] Eval SB rate = {y_eval.mean():.3f}")

    model = CatBoostClassifier(
        iterations=3000,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=100,
        grow_policy="Lossguide",
        min_data_in_leaf=10,
        l2_leaf_reg=5.0,
        verbose=200,
        thread_count=-1,
        random_seed=42,
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=(X_eval, y_eval), use_best_model=True)
    train_time = time.time() - t0
    print(f"[CB] Готово за {train_time:.1f}с, "
          f"найкраща ітерація: {model.best_iteration_}")

    idx = np.random.choice(len(X_train), min(10_000, len(X_train)), replace=False)
    t0 = time.time()
    y_proba_tr = model.predict_proba(X_train[idx])[:, 1]
    inf_ms_tr = (time.time() - t0) / len(idx) * 1000

    tr_metrics = compute_metrics(
        "CatBoost", y_train[idx], y_proba_tr,
        label="train_sample", inference_ms=inf_ms_tr
    )
    tr_metrics["train_time_seconds"] = round(train_time, 2)
    tr_metrics["best_iteration"] = model.best_iteration_
    tr_metrics["scale_pos_weight"] = round(scale_pos_weight, 4)
    print_metrics(tr_metrics)
    save_metrics(tr_metrics, METRICS_PATH)

    t0 = time.time()
    y_proba_ev = model.predict_proba(X_eval)[:, 1]
    inf_ms_ev = (time.time() - t0) / len(X_eval) * 1000

    ev_metrics = compute_metrics(
        "CatBoost", y_eval, y_proba_ev,
        label="eval_natural", inference_ms=inf_ms_ev
    )
    ev_metrics["train_time_seconds"] = round(train_time, 2)
    ev_metrics["scale_pos_weight"] = round(scale_pos_weight, 4)
    print_metrics(ev_metrics)
    save_metrics(ev_metrics, METRICS_PATH)

    print("[CB] Генерація порівняльного eval 50/50...")
    X_cmp, y_cmp = generate_dataset_balanced(5000, 5000)
    t0 = time.time()
    y_proba_cmp = model.predict_proba(X_cmp)[:, 1]
    inf_ms_cmp = (time.time() - t0) / len(X_cmp) * 1000

    cmp_metrics = compute_metrics(
        "CatBoost", y_cmp, y_proba_cmp,
        label="eval_balanced_5050", inference_ms=inf_ms_cmp
    )
    cmp_metrics["train_time_seconds"] = round(train_time, 2)
    print_metrics(cmp_metrics)
    save_metrics(cmp_metrics, METRICS_PATH)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    _cached_model = model
    return model


def load_model() -> CatBoostClassifier:
    if Path(MODEL_PATH).exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_model()


def predict_cb(nodes, matrix) -> float:
    if isSingleType(nodes):
        return 0.0
    model = _get_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    t0 = time.time()
    proba = model.predict_proba(x_input)[0, 1]
    elapsed_ms = (time.time() - t0) * 1000
    print(f"CB predict: {proba:.4f} ({elapsed_ms:.2f} ms)")
    return float(proba)


def teach_cb(nodes, matrix) -> float:
    BUFFER_SIZE = 300
    ADDITIONAL_ITERATIONS = 50

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
        print(f"[CB] Буфер {BUFFER_SIZE}. +{ADDITIONAL_ITERATIONS} ітерацій...")
        X_buf = np.array(buffer["X"])
        y_buf = np.array(buffer["y"])
        pos_buf = (y_buf == 1).sum()
        neg_buf = (y_buf == 0).sum()
        spw = neg_buf / max(pos_buf, 1)

        new_model = CatBoostClassifier(
            iterations=ADDITIONAL_ITERATIONS,
            depth=7,
            learning_rate=0.02,
            loss_function="Logloss",
            scale_pos_weight=spw,
            grow_policy="Lossguide",
            min_data_in_leaf=5,
            verbose=False,
            thread_count=-1,
            random_seed=None,
        )
        new_model.fit(X_buf, y_buf, init_model=model)

        global _cached_model
        _cached_model = new_model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)
        buffer = {"X": [], "y": []}

    with open(BUFFER_PATH, "wb") as f:
        pickle.dump(buffer, f)
    return current_loss


def benchmark_cb(n_samples: int = 10_000):
    print("[CB] Бенчмарк...")
    model = _get_model()
    X, y = generate_dataset_natural(n_samples)
    t0 = time.time()
    y_proba = model.predict_proba(X)[:, 1]
    inf_ms = (time.time() - t0) / n_samples * 1000
    metrics = compute_metrics("CatBoost", y, y_proba,
                              label="benchmark_natural", inference_ms=inf_ms)
    print_metrics(metrics)
    save_metrics(metrics, METRICS_PATH)
    return metrics
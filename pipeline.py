"""
pipeline.py

Навчання → Калібрування → Оцінка → Threshold → Графіки

Зміни:
  - n_splitbrain > 0: тренувальна вибірка збагачена ~20% SB
  - scale_pos_weight від природнього розподілу (не від збагаченого)
  - Оптимальний threshold через Youden's J після калібрування
  - Threshold зберігається у JSON поруч з моделлю
  - compute_metrics рахує метрики при оптимальному threshold
  - R² прибрано (не інформативний для цієї задачі)
  - Додано PR-AUC, Brier Skill, Spearman r
  - Новий графік threshold_analysis.png

Запуск:
  python pipeline.py
  python pipeline.py --skip_training
  python pipeline.py --quick
  python pipeline.py --only base
  python pipeline.py --only ensembles
"""

import argparse
import os
import pickle
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SplitBrainDetector.settings")
django.setup()

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    brier_score_loss, log_loss, mean_absolute_error,
    precision_recall_curve, average_precision_score, roc_curve,
)
from tqdm import tqdm

from app.model.DataPreparation import (
    generate_cluster, preprocess, isClusterDead, isSplitBrain,
)
from ensemble_model import EnsembleModel
from ensemble import get_ensemble

CONFIGS = {
    "full": {
        "cb": {"n_normal": 720_000, "n_splitbrain": 380_000},
        "gb": {"n_normal": 720_000, "n_splitbrain": 380_000},
        "rf": {"n_normal": 400_000, "n_splitbrain": 100_000},
        "eval_size": 20_000,
        "cal_size": 8_000,
        "val_size": 10_000,
    },
    "quick": {
        "cb": {"n_normal": 4_000, "n_splitbrain": 1_000},
        "gb": {"n_normal": 4_000, "n_splitbrain": 1_000},
        "rf": {"n_normal": 4_000, "n_splitbrain": 1_000},
        "eval_size": 2_000,
        "cal_size": 500,
        "val_size": 1_000,
    },
}

NATURAL_SPW = (1 - 0.09) / 0.09

BASE_COLORS = {
    "CatBoost": "#0A84FF",
    "GradientBoosting": "#FF6B35",
    "RandomForest": "#34C759",
}
ENS_COLORS = {
    "E1_Mixed": "#9B59B6",
    "E2_CB_Bias": "#E74C3C",
    "E3_CB_Hyper": "#1ABC9C",
}
ALL_COLORS = {**BASE_COLORS, **ENS_COLORS}
BASE_SHORT = {"CatBoost": "CB", "GradientBoosting": "GB", "RandomForest": "RF"}
ENS_SHORT = {"E1_Mixed": "E1", "E2_CB_Bias": "E2", "E3_CB_Hyper": "E3"}
ALL_SHORT = {**BASE_SHORT, **ENS_SHORT}

MODEL_PATHS = {
    "cb": "models/split_brain_model_cb.pkl",
    "gb": "models/split_brain_model_gb.pkl",
    "rf": "models/split_brain_model_rf.pkl",
}
CAL_PATHS = {
    "cb": "models/split_brain_model_cb_cal.pkl",
    "gb": "models/split_brain_model_gb_cal.pkl",
    "rf": "models/split_brain_model_rf_cal.pkl",
}
THRESHOLD_PATHS = {
    "cb": "models/threshold_cb.json",
    "gb": "models/threshold_gb.json",
    "rf": "models/threshold_rf.json",
}
MODEL_KEYS = {"cb": "CatBoost", "gb": "GradientBoosting", "rf": "RandomForest"}
OUTPUT_DIR = Path("models/pipeline_output")


def generate_balanced(n_normal, n_sb, tag=""):
    X, y = [], []
    for _ in tqdm(range(n_normal), desc=f"{tag} Normal"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    for _ in tqdm(range(n_sb), desc=f"{tag} SB forced"):
        nodes, matrix = generate_cluster()
        while not isSplitBrain(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(1)
    idx = np.random.permutation(len(X))
    return np.array(X)[idx], np.array(y)[idx]


def generate_natural(n, tag=""):
    nodes_list, matrices_list, y = [], [], []
    for _ in tqdm(range(n), desc=f"{tag} Natural"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        nodes_list.append(nodes)
        matrices_list.append(matrix)
        y.append(int(isSplitBrain(nodes, matrix)))
    return nodes_list, matrices_list, np.array(y)


def generate_natural_xy(n, tag=""):
    nl, ml, y = generate_natural(n, tag)
    X = np.array([preprocess(n_, m) for n_, m in zip(nl, ml)])
    return X, np.array(y)


def find_optimal_threshold(y_true, y_proba):
    fpr, tpr, thr_roc = roc_curve(y_true, y_proba)
    j = tpr - fpr
    idx_j = np.argmax(j)

    prec, rec, thr_pr = precision_recall_curve(y_true, y_proba)
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-8)
    idx_f1 = np.argmax(f1s)

    hr_mask = rec[:-1] >= 0.80
    if hr_mask.any():
        idx_hr = np.where(hr_mask)[0][-1]
        hr_t = float(thr_pr[idx_hr])
        hr_p = float(prec[idx_hr])
        hr_r = float(rec[idx_hr])
    else:
        idx_hr = idx_f1
        hr_t = float(thr_pr[idx_f1])
        hr_p = float(prec[idx_f1])
        hr_r = float(rec[idx_f1])

    return {
        "recommended": "youden",
        "youden": {
            "threshold": round(float(thr_roc[idx_j]), 4),
            "tpr": round(float(tpr[idx_j]), 4),
            "fpr": round(float(fpr[idx_j]), 4),
            "j_score": round(float(j[idx_j]), 4),
        },
        "max_f1": {
            "threshold": round(float(thr_pr[idx_f1]), 4),
            "f1": round(float(f1s[idx_f1]), 4),
        },
        "high_recall": {
            "threshold": round(hr_t, 4),
            "precision": round(hr_p, 4),
            "recall": round(hr_r, 4),
        },
    }


def save_threshold(info, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    t = info["youden"]["threshold"]
    print(f"  Threshold: {path}  Youden t={t:.4f}")


def load_threshold(path):
    with open(path) as f:
        return json.load(f)


def compute_metrics(model_name, y_true, y_proba,
                    inference_ms=0.0, label="eval", threshold=None):
    y_true = np.array(y_true, dtype=float)
    y_proba = np.clip(np.array(y_proba, dtype=float), 1e-7, 1 - 1e-7)
    t = threshold if threshold is not None else 0.5
    y_pred = (y_proba >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    climatology = y_true.mean()
    uncertainty = climatology * (1 - climatology)
    brier = brier_score_loss(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    sp_r, _ = spearmanr(y_proba, y_true)

    bins, cal_bins = np.linspace(0, 1, 11), []
    for i in range(10):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if not mask.any():
            continue
        n_k = int(mask.sum())
        o_k = float(y_true[mask].mean())
        f_k = float(y_proba[mask].mean())
        cal_bins.append({"range": f"[{bins[i]:.1f}-{bins[i + 1]:.1f})",
                         "n": n_k, "pred": round(f_k, 4),
                         "actual": round(o_k, 4), "gap": round(f_k - o_k, 4)})

    ece = sum(b["n"] * abs(b["gap"]) for b in cal_bins) / max(len(y_true), 1)
    mce = max((abs(b["gap"]) for b in cal_bins), default=0.0)

    return {
        "model": model_name, "label": label,
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(y_true)),
        "sb_base_rate": round(float(climatology), 4),
        "threshold_used": round(float(t), 4),
        "class_distribution": {
            "negative": int((y_true == 0).sum()),
            "positive": int((y_true == 1).sum()),
        },
        "inference_time_per_sample_ms": round(inference_ms, 4),
        "classification": {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
            "pr_auc": round(float(pr_auc), 4),
            "false_negative_rate": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
            "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
            "confusion_matrix": {"TP": int(tp), "TN": int(tn),
                                 "FP": int(fp), "FN": int(fn)},
        },
        "probability": {
            "brier_score": round(float(brier), 4),
            "brier_skill": round(float(1.0 - brier / max(uncertainty, 1e-10)), 4),
            "log_loss": round(float(log_loss(y_true, y_proba)), 4),
            "mae": round(float(mean_absolute_error(y_true, y_proba)), 4),
            "mae_baseline": round(float(mean_absolute_error(
                y_true, np.full_like(y_proba, climatology))), 4),
            "spearman_r": round(float(sp_r), 4),
        },
        "calibration": {
            "ece": round(float(ece), 4),
            "mce": round(float(mce), 4),
            "uncertain_zone_pct": round(float(
                ((y_proba >= 0.4) & (y_proba <= 0.6)).mean() * 100), 2),
            "calibration_bins": cal_bins,
        },
    }


def train_cb(cfg, val_size):
    from catboost import CatBoostClassifier
    print(f"\n{'─' * 55}\n  [CB] Training (~20% SB enriched)\n{'─' * 55}")
    X_train, y_train = generate_balanced(
        cfg["n_normal"], cfg["n_splitbrain"], "[CB]")
    pos = int((y_train == 1).sum());
    neg = int((y_train == 0).sum())
    X_ev, y_ev = generate_natural_xy(val_size, "[CB-val]")
    print(f"[CB] neg={neg} pos={pos} ({pos / len(y_train) * 100:.1f}% SB)  "
          f"spw={NATURAL_SPW:.2f}  val SB={y_ev.mean():.3f}")
    model = CatBoostClassifier(
        iterations=5000, depth=7, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=NATURAL_SPW,
        early_stopping_rounds=100,
        grow_policy="Lossguide", min_data_in_leaf=10,
        l2_leaf_reg=5.0, verbose=200, thread_count=-1, random_seed=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=(X_ev, y_ev), use_best_model=True)
    elapsed = time.time() - t0
    print(f"[CB] {elapsed:.1f}s  best_iter={model.best_iteration_}")
    return model, elapsed


def train_gb(cfg, val_size):
    from sklearn.ensemble import GradientBoostingClassifier
    print(f"\n{'─' * 55}\n  [GB] Training (~20% SB enriched)\n{'─' * 55}")
    X_train, y_train = generate_balanced(
        cfg["n_normal"], cfg["n_splitbrain"], "[GB]")
    pos = int((y_train == 1).sum());
    neg = int((y_train == 0).sum())
    sw = np.where(y_train == 1, NATURAL_SPW, 1.0)
    print(f"[GB] neg={neg} pos={pos} ({pos / len(y_train) * 100:.1f}% SB)  "
          f"spw={NATURAL_SPW:.2f}")
    model = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=9,
        subsample=0.8, max_features=0.8,
        n_iter_no_change=20, validation_fraction=0.1,
        random_state=42, verbose=1,
    )
    t0 = time.time()
    model.fit(X_train, y_train, sample_weight=sw)
    elapsed = time.time() - t0
    print(f"[GB] {elapsed:.1f}s  est={model.n_estimators_}")
    return model, elapsed


def train_rf(cfg, val_size):
    from sklearn.ensemble import RandomForestClassifier
    print(f"\n{'─' * 55}\n  [RF] Training (~20% SB enriched)\n{'─' * 55}")
    X_train, y_train = generate_balanced(
        cfg["n_normal"], cfg["n_splitbrain"], "[RF]")
    pos = int((y_train == 1).sum());
    neg = int((y_train == 0).sum())
    print(f"[RF] neg={neg} pos={pos} ({pos / len(y_train) * 100:.1f}% SB)")
    model = RandomForestClassifier(
        n_estimators=600, max_depth=15, min_samples_leaf=4,
        max_features="sqrt", max_samples=0.8,
        class_weight="balanced_subsample",
        oob_score=True, random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[RF] {elapsed:.1f}s  OOB={model.oob_score_:.4f}")
    return model, elapsed


def calibrate(model, X_cal, y_cal):
    cal = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def eval_model(model_name, model, nodes_list, matrices_list,
               y_true, label="eval", threshold=None):
    X = np.array([preprocess(n, m) for n, m in zip(nodes_list, matrices_list)])
    t0 = time.time()
    y_proba = model.predict_proba(X)[:, 1]
    inf_ms = (time.time() - t0) / len(X) * 1000
    return (compute_metrics(model_name, y_true, y_proba,
                            inf_ms, label, threshold=threshold),
            y_proba)


def eval_ensemble(ens: EnsembleModel, nodes_list, matrices_list,
                  y_true, label="eval", calibrated=True, threshold=None):
    X = np.array([preprocess(n, m) for n, m in zip(nodes_list, matrices_list)])
    t0 = time.time()
    y_proba = (ens.predict_proba_calibrated(X)
               if calibrated and ens._cal_models
               else ens.predict_proba_raw(X))
    inf_ms = (time.time() - t0) / len(X) * 1000
    return (compute_metrics(ens.name, y_true, y_proba,
                            inf_ms, label, threshold=threshold),
            y_proba)


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#E0E0E0", "axes.labelcolor": "#333333",
        "xtick.color": "#666666", "ytick.color": "#666666",
        "text.color": "#1A1A1A", "grid.color": "#F0F0F0",
        "grid.linewidth": 0.8, "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })


def _short(n): return ALL_SHORT.get(n, n[:4])


def _color(n): return ALL_COLORS.get(n, "#999999")


def plot_threshold_analysis(all_metrics, all_probas, all_thresholds,
                            y_true, out_dir):
    _style()
    names = list(all_metrics.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor="white")
    if n == 1:
        axes = [axes]
    fig.suptitle("Threshold Analysis  ·  Recall / FPR / F1 vs Threshold",
                 fontsize=14, fontweight="bold", y=1.02)

    ts = np.linspace(0.01, 0.99, 300)

    for ax, name in zip(axes, names):
        color = _color(name)
        y_prob = all_probas[name]
        tpr_l, fpr_l, f1_l = [], [], []

        for th in ts:
            yp = (y_prob >= th).astype(int)
            tn_, fp_, fn_, tp_ = confusion_matrix(y_true, yp).ravel()
            tpr_l.append(tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0)
            fpr_l.append(fp_ / (fp_ + tn_) if (fp_ + tn_) > 0 else 0)
            p_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
            r_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
            f1_l.append(2 * p_ * r_ / max(p_ + r_, 1e-8))

        ax.plot(ts, tpr_l, color="#2ECC71", lw=2, label="Recall (TPR)")
        ax.plot(ts, fpr_l, color="#E74C3C", lw=2, label="FPR")
        ax.plot(ts, f1_l, color="#3498DB", lw=2, label="F1-score")

        if name in all_thresholds:
            info = all_thresholds[name]
            t_y = info["youden"]["threshold"]
            t_f1 = info["max_f1"]["threshold"]
            ax.axvline(t_y, color=color, lw=1.5, linestyle="--",
                       label=f"Youden  t={t_y:.3f}")
            ax.axvline(t_f1, color="#888888", lw=1, linestyle=":",
                       label=f"MaxF1   t={t_f1:.3f}")

        ax.set_title(_short(name), color=color, fontsize=12, fontweight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(zorder=0);
        ax.set_axisbelow(True)

    plt.tight_layout(pad=2)
    fname = out_dir / "threshold_analysis.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname.name}")


def plot_comparison(base_m, ens_m, base_p, ens_p, y_true, out_dir, suffix=""):
    _style()
    all_m = {**base_m, **ens_m}
    all_p = {**base_p, **ens_p}
    names = list(all_m.keys())
    colors = [_color(n) for n in names]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="white")
    fig.suptitle(f"Split-Brain Detector  ·  Base vs Ensemble{suffix}",
                 fontsize=17, fontweight="bold", y=0.99)

    ax = axes[0, 0]
    mk = ["roc_auc", "pr_auc", "recall", "precision", "f1_score"]
    ml = ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]
    x = np.arange(len(ml));
    w = 0.12
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [all_m[name]["classification"][k] for k in mk]
        off = (i - len(names) / 2) * w + w / 2
        ax.bar(x + off, vals, w, label=_short(name),
               color=color, alpha=0.85, zorder=3, linewidth=0)
    ax.set_xticks(x);
    ax.set_xticklabels(ml, fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.axhline(0.5, color="#AAAAAA", lw=1, linestyle="--")
    ax.set_title("Classification (at optimal threshold)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(axis="y");
    ax.set_axisbelow(True)

    ax = axes[0, 1]
    y_pos = np.arange(2)
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [all_m[name]["classification"][k]
                for k in ["false_negative_rate", "false_positive_rate"]]
        off = (i - len(names) / 2) * 0.12
        ax.scatter(vals, y_pos + off, color=color, s=90, zorder=3,
                   label=_short(name))
        for j, v in enumerate(vals):
            ax.text(v + 0.01, y_pos[j] + off, f"{v:.2f}",
                    va="center", fontsize=7, color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["FN rate (missed SB)", "FP rate (false alarm)"], fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("Error Rates  (lower = better)", fontweight="bold")
    ax.grid(axis="x");
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)

    ax = axes[0, 2]
    pk = ["brier_score", "log_loss", "ece"]
    pl = ["Brier ↓", "LogLoss ↓", "ECE ↓"]
    x = np.arange(len(pl));
    w = 0.12
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [all_m[name]["probability" if k in ("brier_score", "log_loss")
        else "calibration"][k] for k in pk]
        off = (i - len(names) / 2) * w + w / 2
        ax.bar(x + off, vals, w, label=_short(name),
               color=color, alpha=0.85, zorder=3, linewidth=0)
    ax.set_xticks(x);
    ax.set_xticklabels(pl, fontsize=9)
    ax.set_title("Probability Quality (lower = better)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(axis="y");
    ax.set_axisbelow(True)

    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], "--", color="#CCCCCC", lw=1.5, label="Perfect")
    for name, color in zip(names, colors):
        bd = all_m[name]["calibration"]["calibration_bins"]
        ece = all_m[name]["calibration"]["ece"]
        ax.plot([b["pred"] for b in bd], [b["actual"] for b in bd],
                "o-", color=color, lw=2, ms=5,
                label=f"{_short(name)} ECE={ece:.3f}")
    ax.set_xlim(0, 1);
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed SB frequency")
    ax.set_title("Reliability Diagram", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid();
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    for name, color in zip(names, colors):
        prec, rec, _ = precision_recall_curve(y_true, all_p[name])
        ap = average_precision_score(y_true, all_p[name])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{_short(name)} AP={ap:.3f}")
    ax.axhline(float(y_true.mean()), color="#CCCCCC", lw=1, linestyle="--",
               label=f"Base rate {y_true.mean():.3f}")
    ax.set_xlabel("Recall");
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02);
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid();
    ax.set_axisbelow(True)

    ax = axes[1, 2]
    for name, color in zip(names, colors):
        fpr, tpr, _ = roc_curve(y_true, all_p[name])
        auc = all_m[name]["classification"]["roc_auc"]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{_short(name)} {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#CCCCCC", lw=1.5)
    ax.set_xlabel("False Positive Rate");
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1.01);
    ax.set_ylim(0, 1.01)
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid();
    ax.set_axisbelow(True)

    plt.tight_layout(pad=2)
    fname = out_dir / f"comparison{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname.name}")


def plot_metrics_table(all_metrics, out_dir, title_suffix=""):
    _style()
    names = list(all_metrics.keys())
    rows = [
        ("CLASSIFICATION", "ROC-AUC", "classification", "roc_auc", True),
        ("", "PR-AUC", "classification", "pr_auc", True),
        ("", "Recall", "classification", "recall", True),
        ("", "Precision", "classification", "precision", True),
        ("", "F1-score", "classification", "f1_score", True),
        ("", "FN rate ↓", "classification", "false_negative_rate", False),
        ("", "FP rate ↓", "classification", "false_positive_rate", False),
        ("PROBABILITY", "Brier ↓", "probability", "brier_score", False),
        ("", "BrierSkill", "probability", "brier_skill", True),
        ("", "LogLoss ↓", "probability", "log_loss", False),
        ("", "Spearman r", "probability", "spearman_r", True),
        ("CALIBRATION", "ECE ↓", "calibration", "ece", False),
        ("", "MCE ↓", "calibration", "mce", False),
        ("TECHNICAL", "Inference ↓", None, "inference_time_per_sample_ms", False),
    ]
    n = len(names)
    fig, ax = plt.subplots(figsize=(5 + 3.5 * n, 10), facecolor="white")
    ax.axis("off")
    fig.suptitle(f"Metrics Table  ·  {title_suffix}",
                 fontsize=14, fontweight="bold", y=0.99)

    col_x = [0.01, 0.22] + [0.42 + i * 0.16 for i in range(n)]
    headers = ["", "Metric"] + [_short(m) for m in names]
    hcolors = ["#999", "#333"] + [_color(m) for m in names]

    for x, h, c in zip(col_x, headers, hcolors):
        ax.text(x, 0.97, h, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=c, va="top")
    ax.add_line(plt.Line2D([0.01, 0.99], [0.945, 0.945],
                           transform=ax.transAxes, color="#DDD", lw=1))

    cur_sec = None;
    y_pos = 0.895;
    rh = 0.048
    for idx, (sec, metric, cat, key, hb) in enumerate(rows):
        if sec and sec != cur_sec:
            cur_sec = sec
            ax.text(col_x[0], y_pos, sec, transform=ax.transAxes,
                    fontsize=8, fontweight="bold", color="#999",
                    va="center", style="italic")
        vals = [all_metrics[m][key] if cat is None
                else all_metrics[m][cat][key] for m in names]
        bi = vals.index(max(vals)) if hb else vals.index(min(vals))
        if idx % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (0, y_pos - rh * 0.48), 1, rh * 0.92,
                transform=ax.transAxes, facecolor="#F8F9FA", zorder=0))
        ax.text(col_x[1], y_pos, metric, transform=ax.transAxes,
                fontsize=10, color="#1A1A1A", va="center")
        for i, (v, x) in enumerate(zip(vals, col_x[2:])):
            is_b = (i == bi)
            ax.text(x, y_pos,
                    f"{'★ ' if is_b else '  '}{v:.4f}",
                    transform=ax.transAxes, fontsize=10,
                    color=_color(names[i]) if is_b else "#888",
                    fontweight="bold" if is_b else "normal", va="center")
        y_pos -= rh

    ax.add_line(plt.Line2D([0.01, 0.99], [y_pos + rh * 0.3, y_pos + rh * 0.3],
                           transform=ax.transAxes, color="#DDD", lw=0.8))
    ax.text(0.5, 0.01, "★ = best    ↓ = lower is better  |  threshold = Youden optimal",
            transform=ax.transAxes, ha="center", fontsize=9, color="#AAA")

    fname = out_dir / f"table_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname.name}")


def run(models, skip_training, quick, only):
    cfg = CONFIGS["quick" if quick else "full"]
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    train_times = {}

    run_base = only in ("all", "base")
    run_ens = only in ("all", "ensembles")

    if run_base and not skip_training:
        fns = {
            "cb": lambda: train_cb(cfg["cb"], cfg["val_size"]),
            "gb": lambda: train_gb(cfg["gb"], cfg["val_size"]),
            "rf": lambda: train_rf(cfg["rf"], cfg["val_size"]),
        }
        for key in models:
            m, el = fns[key]()
            save_model(m, MODEL_PATHS[key])
            train_times[MODEL_KEYS[key]] = round(el, 2)
    elif run_base:
        for key in models:
            if not Path(MODEL_PATHS[key]).exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATHS[key]}")
            print(f"[Load] {MODEL_KEYS[key]}")

    trained_base = {}
    if run_base:
        trained_base = {MODEL_KEYS[k]: load_model(MODEL_PATHS[k]) for k in models}

    print(f"\n[Pipeline] Generating eval ({cfg['eval_size']} samples)...")
    nodes_list, matrices_list, y_true = generate_natural(
        cfg["eval_size"], "[Eval]")
    print(f"[Pipeline] SB rate = {y_true.mean():.3f}  "
          f"({int(y_true.sum())} SB / {len(y_true)})")

    X_eval = np.array([preprocess(n, m)
                       for n, m in zip(nodes_list, matrices_list)])

    before_base, before_proba_base = {}, {}
    if run_base:
        print("\n[Pipeline] Evaluating base (before cal, t=0.5)...")
        for key in models:
            mn = MODEL_KEYS[key]
            m, p = eval_model(mn, trained_base[mn], nodes_list, matrices_list,
                              y_true, "before_cal", threshold=0.5)
            if mn in train_times:
                m["train_time_seconds"] = train_times[mn]
            before_base[mn] = m
            before_proba_base[mn] = p

    cal_base = {}
    thresholds_base = {}
    if run_base:
        print(f"\n[Pipeline] Calibrating ({cfg['cal_size']} samples, natural)...")
        X_cal, y_cal = generate_natural_xy(cfg["cal_size"], "[Cal]")
        print(f"  Cal SB rate = {y_cal.mean():.3f}")
        for key in models:
            mn = MODEL_KEYS[key]
            cal = calibrate(trained_base[mn], X_cal, y_cal)
            save_model(cal, CAL_PATHS[key])
            cal_base[mn] = cal

            y_prob_cal = cal.predict_proba(X_eval)[:, 1]
            t_info = find_optimal_threshold(y_true, y_prob_cal)
            thresholds_base[mn] = t_info
            save_threshold(t_info, THRESHOLD_PATHS[key])
            print(f"  [{mn}] Youden t={t_info['youden']['threshold']:.4f}  "
                  f"TPR={t_info['youden']['tpr']:.3f}  "
                  f"FPR={t_info['youden']['fpr']:.3f}  "
                  f"J={t_info['youden']['j_score']:.3f}")

    after_base, after_proba_base = {}, {}
    if run_base:
        print("\n[Pipeline] Evaluating base (after cal, Youden threshold)...")
        for key in models:
            mn = MODEL_KEYS[key]
            t_opt = thresholds_base[mn]["youden"]["threshold"]
            m, p = eval_model(mn, cal_base[mn], nodes_list, matrices_list,
                              y_true, "after_cal", threshold=t_opt)
            if mn in train_times:
                m["train_time_seconds"] = train_times[mn]
            after_base[mn] = m
            after_proba_base[mn] = p
            print(f"  [{mn}] AUC={m['classification']['roc_auc']:.4f}  "
                  f"Recall={m['classification']['recall']:.4f}  "
                  f"FNR={m['classification']['false_negative_rate']:.4f}  "
                  f"t={t_opt:.4f}")

    ens_raw_m, ens_raw_p = {}, {}
    ens_cal_m, ens_cal_p = {}, {}
    thresholds_ens = {}

    if run_ens:
        print("\n[Pipeline] Evaluating ensembles...")
        for key in ["e1", "e2", "e3"]:
            try:
                ens_raw = get_ensemble(key, calibrated=False)
                m, p = eval_ensemble(ens_raw, nodes_list, matrices_list,
                                     y_true, "before_cal",
                                     calibrated=False, threshold=0.5)
                ens_raw_m[ens_raw.name] = m
                ens_raw_p[ens_raw.name] = p
            except FileNotFoundError as e:
                print(f"  [{key}] not found: {e}")
                continue

            try:
                ens_cal = get_ensemble(key, calibrated=True)
                if ens_cal._cal_models is None:
                    continue
                y_prob_ens = ens_cal.predict_proba_calibrated(X_eval)
                t_info = find_optimal_threshold(y_true, y_prob_ens)
                thresholds_ens[ens_cal.name] = t_info
                t_opt = t_info["youden"]["threshold"]

                m, p = eval_ensemble(ens_cal, nodes_list, matrices_list,
                                     y_true, "after_cal",
                                     calibrated=True, threshold=t_opt)
                ens_cal_m[ens_cal.name] = m
                ens_cal_p[ens_cal.name] = p
                print(f"  [{key} cal] AUC={m['classification']['roc_auc']:.4f}  "
                      f"Recall={m['classification']['recall']:.4f}  "
                      f"t={t_opt:.4f}")
            except FileNotFoundError as e:
                print(f"  [{key} cal] not found: {e}")

    result = {}
    if run_base:
        result["base_before_cal"] = before_base
        result["base_after_cal"] = after_base
        result["thresholds_base"] = thresholds_base
    if run_ens:
        result["ens_before_cal"] = ens_raw_m
        result["ens_after_cal"] = ens_cal_m
        result["thresholds_ens"] = thresholds_ens

    with open(out_dir / "comparison.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "results": result}, f, indent=2, ensure_ascii=False)
    print("  Saved: comparison.json")

    print("\n[Pipeline] Generating charts...")
    if run_base and after_base:
        plot_metrics_table(after_base, out_dir, "Base After Cal")
        plot_threshold_analysis(after_base, after_proba_base,
                                thresholds_base, y_true, out_dir)
    if run_ens and ens_cal_m:
        plot_metrics_table(ens_cal_m, out_dir, "Ensembles After Cal")
    if run_base and run_ens and after_base and ens_cal_m:
        plot_comparison(after_base, ens_cal_m,
                        after_proba_base, ens_cal_p,
                        y_true, out_dir, "_after_cal")
    if run_base and before_base:
        plot_comparison(before_base, ens_raw_m if run_ens else {},
                        before_proba_base, ens_raw_p if run_ens else {},
                        y_true, out_dir, "_before_cal")

    elapsed = time.time() - t_total
    h, rem = divmod(int(elapsed), 3600)
    m_t, s = divmod(rem, 60)
    print(f"\n{'─' * 78}")
    print(f"  Pipeline complete  {h:02d}:{m_t:02d}:{s:02d}")
    print(f"{'─' * 78}")

    all_for_table = {}
    if run_base: all_for_table.update(after_base)
    if run_ens:  all_for_table.update(ens_cal_m)
    all_t = {**thresholds_base, **thresholds_ens}

    if all_for_table:
        print(f"\n  {'Model':<22} {'AUC':>7} {'PR-AUC':>8} "
              f"{'Recall':>8} {'FNR':>7} {'ECE':>8} {'t*':>7}")
        print(f"  {'─' * 65}")
        for name, md in all_for_table.items():
            cl = md["classification"]
            ca = md["calibration"]
            t_opt = all_t.get(name, {}).get("youden", {}).get("threshold", 0.5)
            print(f"  {name:<22} {cl['roc_auc']:>7.4f} {cl['pr_auc']:>8.4f} "
                  f"{cl['recall']:>8.4f} {cl['false_negative_rate']:>7.4f} "
                  f"{ca['ece']:>8.4f} {t_opt:>7.4f}")
        print(f"{'─' * 78}")
        print("  t* = Youden optimal threshold  |  FNR = False Negative Rate\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["cb", "gb", "rf"],
                        choices=["cb", "gb", "rf"])
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default="all",
                        choices=["all", "base", "ensembles"])
    args = parser.parse_args()

    run(models=args.models,
        skip_training=args.skip_training,
        quick=args.quick,
        only=args.only)

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
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    brier_score_loss, log_loss, mean_absolute_error, r2_score,
    precision_recall_curve, average_precision_score, roc_curve,
)
from tqdm import tqdm

from app.model.DataPreparation import (
    generate_cluster, preprocess, isClusterDead, isSplitBrain,
)

CONFIGS = {
    "full": {
        "cb": {"n_normal": 900_000, "n_splitbrain": 0},
        "gb": {"n_normal": 900_000, "n_splitbrain": 0},
        "rf": {"n_normal": 500_000, "n_splitbrain": 0},
        "eval_size":    20_000,
        "cal_size":      8_000,
        "val_size":     10_000,
    },
    "quick": {
        "cb": {"n_normal": 8_000, "n_splitbrain": 0},
        "gb": {"n_normal": 8_000, "n_splitbrain": 0},
        "rf": {"n_normal": 8_000, "n_splitbrain": 0},
        "eval_size":   2_000,
        "cal_size":    1_000,
        "val_size":    2_000,
    },
}

COLORS = {
    "CatBoost":         "#0A84FF",
    "GradientBoosting": "#FF6B35",
    "RandomForest":     "#34C759",
}
MODEL_KEYS  = {"cb": "CatBoost", "gb": "GradientBoosting", "rf": "RandomForest"}
SHORT       = {"CatBoost": "CB", "GradientBoosting": "GB", "RandomForest": "RF"}
MODEL_PATHS = {
    "cb": "split_brain_model_cb.pkl",
    "gb": "split_brain_model_gb.pkl",
    "rf": "split_brain_model_rf.pkl",
}
CAL_PATHS = {
    "cb": "split_brain_model_cb_calibrated.pkl",
    "gb": "split_brain_model_gb_calibrated.pkl",
    "rf": "split_brain_model_rf_calibrated.pkl",
}
OUTPUT_DIR = Path("models/pipeline_output")


def _style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#E0E0E0",
        "axes.labelcolor":   "#333333",
        "xtick.color":       "#666666",
        "ytick.color":       "#666666",
        "text.color":        "#1A1A1A",
        "grid.color":        "#F0F0F0",
        "grid.linewidth":    0.8,
        "font.family":       "serif",
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  True,
        "axes.spines.bottom":True,
        "axes.linewidth":    0.8,
    })


def generate_balanced(n_normal, n_splitbrain, tag=""):
    X, y = [], []
    for _ in tqdm(range(n_normal), desc=f"{tag} Normal"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    for _ in tqdm(range(n_splitbrain), desc=f"{tag} Split-brain"):
        nodes, matrix = generate_cluster()
        while not isSplitBrain(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(1)
    return np.array(X), np.array(y)


def generate_natural(n_samples, tag=""):
    nodes_list, matrices_list, y = [], [], []
    for _ in tqdm(range(n_samples), desc=f"{tag} Natural"):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        nodes_list.append(nodes)
        matrices_list.append(matrix)
        y.append(int(isSplitBrain(nodes, matrix)))
    return nodes_list, matrices_list, np.array(y)


def generate_natural_xy(n_samples, tag=""):
    nodes_list, matrices_list, y = generate_natural(n_samples, tag)
    X = np.array([preprocess(n, m) for n, m in zip(nodes_list, matrices_list)])
    return X, np.array(y)


def compute_metrics(model_name, y_true, y_proba, inference_ms=0.0, label="eval"):
    y_true  = np.array(y_true,  dtype=float)
    y_proba = np.array(y_proba, dtype=float)
    y_pred  = (y_proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    climatology = y_true.mean()
    uncertainty = climatology * (1 - climatology)
    brier       = brier_score_loss(y_true, y_proba)

    bins     = np.linspace(0, 1, 11)
    cal_bins = []
    for i in range(10):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        n_k = int(mask.sum())
        o_k = float(y_true[mask].mean())
        f_k = float(y_proba[mask].mean())
        cal_bins.append({
            "range":  f"[{bins[i]:.1f}-{bins[i+1]:.1f})",
            "n":      n_k,
            "pred":   round(f_k, 4),
            "actual": round(o_k, 4),
            "gap":    round(f_k - o_k, 4),
        })

    ece     = sum(b["n"] * abs(b["gap"]) for b in cal_bins) / max(len(y_true), 1)
    mce     = max((abs(b["gap"]) for b in cal_bins), default=0.0)
    sp_r, _ = spearmanr(y_proba, y_true)

    return {
        "model":        model_name,
        "label":        label,
        "timestamp":    datetime.now().isoformat(),
        "n_samples":    int(len(y_true)),
        "sb_base_rate": round(float(climatology), 4),
        "class_distribution": {
            "negative": int((y_true == 0).sum()),
            "positive": int((y_true == 1).sum()),
        },
        "inference_time_per_sample_ms": round(inference_ms, 4),
        "classification": {
            "accuracy":            round(accuracy_score(y_true, y_pred), 4),
            "precision":           round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":              round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score":            round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc":             round(roc_auc_score(y_true, y_proba), 4),
            "false_negative_rate": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
            "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
            "confusion_matrix":    {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        },
        "regression": {
            "mae":          round(float(mean_absolute_error(y_true, y_proba)), 4),
            "mae_baseline": round(float(mean_absolute_error(y_true, np.full_like(y_proba, climatology))), 4),
            "r2":           round(float(r2_score(y_true, y_proba)), 4),
            "brier_score":  round(float(brier), 4),
            "brier_skill":  round(float(1.0 - brier / max(uncertainty, 1e-10)), 4),
        },
        "calibration": {
            "log_loss":           round(float(log_loss(y_true, y_proba)), 4),
            "ece":                round(float(ece), 4),
            "mce":                round(float(mce), 4),
            "uncertain_zone_pct": round(float(((y_proba >= 0.4) & (y_proba <= 0.6)).mean() * 100), 2),
            "calibration_bins":   cal_bins,
        },
        "correlation": {
            "spearman_r": round(float(sp_r), 4),
        },
    }


def train_cb(cfg, val_size):
    from catboost import CatBoostClassifier
    print(f"\n{'─'*55}\n  [CB] Training CatBoost\n{'─'*55}")

    X_train, y_train = generate_balanced(cfg["n_normal"], cfg["n_splitbrain"], "[CB]")
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = neg / pos
    print(f"[CB] neg={neg}, pos={pos}, scale_pos_weight={spw:.2f}")

    print(f"[CB] Generating natural val set ({val_size} samples)...")
    X_ev, y_ev = generate_natural_xy(val_size, "[CB-val]")
    print(f"[CB] Val SB rate = {y_ev.mean():.3f}  ({int(y_ev.sum())} SB / {len(y_ev)})")

    model = CatBoostClassifier(
        iterations=3000, depth=7, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=spw, early_stopping_rounds=100,
        grow_policy="Lossguide", min_data_in_leaf=10,
        l2_leaf_reg=5.0, verbose=200, thread_count=-1, random_seed=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=(X_ev, y_ev), use_best_model=True)
    elapsed = time.time() - t0
    print(f"[CB] Done in {elapsed:.1f}s, best_iter={model.best_iteration_}")
    return model, elapsed


def train_gb(cfg, val_size):
    from sklearn.ensemble import GradientBoostingClassifier
    print(f"\n{'─'*55}\n  [GB] Training GradientBoosting\n{'─'*55}")

    X_train, y_train = generate_balanced(cfg["n_normal"], cfg["n_splitbrain"], "[GB]")
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    sw  = np.where(y_train == 1, neg / pos, 1.0)
    print(f"[GB] neg={neg}, pos={pos}, ratio={neg/pos:.2f}")

    model = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=9,
        min_samples_split=4, min_samples_leaf=2,
        subsample=0.8, max_features=0.8,
        n_iter_no_change=20, validation_fraction=0.1, tol=1e-4,
        random_state=42, verbose=1,
    )
    t0 = time.time()
    model.fit(X_train, y_train, sample_weight=sw)
    elapsed = time.time() - t0
    print(f"[GB] Done in {elapsed:.1f}s, estimators used={model.n_estimators_}")
    return model, elapsed


def train_rf(cfg, val_size):
    from sklearn.ensemble import RandomForestClassifier
    print(f"\n{'─'*55}\n  [RF] Training RandomForest (improved)\n{'─'*55}")

    X_train, y_train = generate_balanced(cfg["n_normal"], cfg["n_splitbrain"], "[RF]")
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    print(f"[RF] neg={neg}, pos={pos}")

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=15,
        min_samples_split=6,
        min_samples_leaf=4,
        max_features="sqrt",
        max_samples=0.8,
        bootstrap=True,
        class_weight="balanced_subsample",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[RF] Done in {elapsed:.1f}s, OOB score={model.oob_score_:.4f}")
    return model, elapsed


def calibrate_model(model, X_cal, y_cal, method="isotonic"):
    cal = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  Saved: {path}  ({size_mb:.1f} MB)")


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate(model_name, model, nodes_list, matrices_list, y_true, label="eval"):
    X       = np.array([preprocess(n, m) for n, m in zip(nodes_list, matrices_list)])
    t0      = time.time()
    y_proba = model.predict_proba(X)[:, 1]
    inf_ms  = (time.time() - t0) / len(X) * 1000
    return compute_metrics(model_name, y_true, y_proba, inf_ms, label), y_proba


def _bar_group(ax, model_names, metric_labels, values_per_model,
               title, colors, ylim=None, baseline=None, baseline_label=""):
    x = np.arange(len(metric_labels))
    w = 0.22
    for i, (mn, color) in enumerate(zip(model_names, colors)):
        vals = values_per_model[i]
        bars = ax.bar(x + i * w - w, vals, w,
                      label=SHORT[mn], color=color,
                      alpha=0.85, zorder=3, linewidth=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ylim[1] * 0.015 if ylim else 0.012),
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#1A1A1A")
    if ylim:
        ax.set_ylim(*ylim)
    if baseline is not None:
        ax.axhline(baseline, color="#AAAAAA", lw=1, linestyle="--", zorder=1)
        ax.text(len(metric_labels) - 0.5, baseline + 0.01,
                baseline_label, fontsize=8, color="#AAAAAA")
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="#E0E0E0")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def plot_dashboard(all_metrics, out_dir):
    _style()
    model_names = list(all_metrics.keys())
    colors      = [COLORS[m] for m in model_names]

    fig = plt.figure(figsize=(18, 13), facecolor="white")
    fig.suptitle("Split-Brain Cluster Detector  ·  Model Comparison",
                 fontsize=18, fontweight="bold", color="#1A1A1A", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.50, wspace=0.38,
                           left=0.06, right=0.97, top=0.94, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, :2])
    _bar_group(
        ax1, model_names,
        ["ROC-AUC", "F1-score", "Recall", "Precision"],
        [[all_metrics[m]["classification"][k]
          for k in ["roc_auc", "f1_score", "recall", "precision"]]
         for m in model_names],
        "Classification Metrics", colors,
        ylim=(0, 1.18), baseline=0.5, baseline_label="random baseline",
    )

    ax2 = fig.add_subplot(gs[0, 2])
    err_labels = ["FN rate\n(missed SB)", "FP rate\n(false alarms)"]
    err_keys   = ["false_negative_rate", "false_positive_rate"]
    y_pos = np.arange(2)
    for i, (mn, color) in enumerate(zip(model_names, colors)):
        vals = [all_metrics[mn]["classification"][k] for k in err_keys]
        off  = (i - 1) * 0.22
        ax2.scatter(vals, y_pos + off, color=color, s=110, zorder=3)
        for j, v in enumerate(vals):
            ax2.text(v + 0.015, y_pos[j] + off, f"{v:.3f}",
                     va="center", fontsize=8, color=color)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(err_labels, fontsize=9)
    ax2.set_xlim(-0.05, 1.1)
    ax2.set_title("Error Rates  (lower = better)", fontsize=11,
                  fontweight="bold", pad=10)
    ax2.grid(axis="x", zorder=0)
    ax2.set_axisbelow(True)

    ax3 = fig.add_subplot(gs[1, :2])
    _bar_group(
        ax3, model_names,
        ["MAE ↓", "Brier ↓", "Log-loss ↓", "ECE ↓"],
        [[all_metrics[m]["regression"]["mae"],
          all_metrics[m]["regression"]["brier_score"],
          all_metrics[m]["calibration"]["log_loss"],
          all_metrics[m]["calibration"]["ece"]]
         for m in model_names],
        "Regression & Calibration Metrics  (lower = better)", colors,
    )

    ax4 = fig.add_subplot(gs[1, 2])
    bss = [all_metrics[m]["regression"]["brier_skill"] for m in model_names]
    r2s = [all_metrics[m]["regression"]["r2"]          for m in model_names]
    y   = np.arange(len(model_names))
    ax4.barh(y - 0.2, bss, 0.35, color=colors, alpha=0.85,
             label="Brier Skill", linewidth=0)
    ax4.barh(y + 0.2, r2s, 0.35, color=colors, alpha=0.45, label="R²",
             hatch="///", edgecolor=colors, linewidth=0.5)
    for i, (b, r) in enumerate(zip(bss, r2s)):
        ax4.text(max(b, 0) + 0.01, i - 0.2, f"{b:.3f}",
                 va="center", fontsize=8, color=colors[i], fontweight="bold")
        ax4.text(max(r, 0) + 0.01, i + 0.2, f"{r:.3f}",
                 va="center", fontsize=8, color=colors[i])
    ax4.set_yticks(y)
    ax4.set_yticklabels([SHORT[m] for m in model_names], fontsize=10)
    ax4.axvline(0, color="#AAAAAA", lw=1)
    ax4.set_title("Skill Scores  (higher = better)", fontsize=11,
                  fontweight="bold", pad=10)
    ax4.legend(fontsize=8, framealpha=0.9, edgecolor="#E0E0E0")
    ax4.grid(axis="x", zorder=0)
    ax4.set_axisbelow(True)

    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot([0, 1], [0, 1], "--", color="#BBBBBB", lw=1.5,
             label="Perfect calibration", zorder=2)
    for mn, color in zip(model_names, colors):
        bins_data = all_metrics[mn]["calibration"]["calibration_bins"]
        pred = [b["pred"]   for b in bins_data]
        act  = [b["actual"] for b in bins_data]
        ece  = all_metrics[mn]["calibration"]["ece"]
        ax5.plot(pred, act, "o-", color=color, lw=2, ms=6,
                 label=f"{SHORT[mn]}  ECE={ece:.3f}", zorder=3)
        ax5.fill_between(pred, pred, act, alpha=0.06, color=color)
    ax5.set_xlabel("Predicted probability", fontsize=10)
    ax5.set_ylabel("Observed SB frequency", fontsize=10)
    ax5.set_title("Reliability Diagram  ·  Probability Calibration",
                  fontsize=11, fontweight="bold", pad=10)
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
    ax5.legend(fontsize=9, framealpha=0.9, edgecolor="#E0E0E0")
    ax5.grid(zorder=0)
    ax5.set_axisbelow(True)

    ax6 = fig.add_subplot(gs[2, 2])
    for mn, color in zip(model_names, colors):
        t_min = all_metrics[mn].get("train_time_seconds", 0) / 60
        auc   = all_metrics[mn]["classification"]["roc_auc"]
        ms    = all_metrics[mn]["inference_time_per_sample_ms"]
        ax6.scatter(t_min, auc, s=ms * 2000 + 60, color=color,
                    alpha=0.85, zorder=3, edgecolors="#CCCCCC", linewidths=0.8)
        ax6.annotate(f"{SHORT[mn]}\n{ms:.3f} ms",
                     (t_min, auc), xytext=(8, 4),
                     textcoords="offset points", fontsize=8, color=color)
    ax6.set_xlabel("Training time (min)", fontsize=10)
    ax6.set_ylabel("ROC-AUC",             fontsize=10)
    ax6.set_title("Training Time vs Quality\n(circle size = inference latency)",
                  fontsize=10, fontweight="bold", pad=10)
    ax6.grid(zorder=0)
    ax6.set_axisbelow(True)

    plt.savefig(out_dir / "01_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: 01_dashboard.png")


def plot_confusion_matrices(all_metrics, out_dir):
    _style()
    model_names = list(all_metrics.keys())
    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(5 * len(model_names), 5),
                             facecolor="white")
    if len(model_names) == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices  ·  Decision Threshold = 0.5",
                 fontsize=14, fontweight="bold", y=1.02)

    labels      = ["Not SB  (0)", "Split-Brain  (1)"]
    cell_names  = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
    cell_colors = {
        True:  ("#EBF5FB", "#2E86C1"),
        False: ("#FDEDEC", "#E74C3C"),
    }

    for ax, mn in zip(axes, model_names):
        color = COLORS[mn]
        cm    = all_metrics[mn]["classification"]["confusion_matrix"]
        arr   = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        total = arr.sum()

        for i in range(2):
            for j in range(2):
                v          = arr[i, j]
                pct        = v / total * 100
                is_correct = (i == j)
                bg_c, bd_c = cell_colors[is_correct]
                rect = plt.Rectangle([j, 1 - i], 1, 1,
                                     facecolor=bg_c,
                                     edgecolor=bd_c, linewidth=1.5)
                ax.add_patch(rect)
                ax.text(j + 0.5, 1 - i + 0.68, cell_names[(i, j)],
                        ha="center", va="center", fontsize=10,
                        color=bd_c, fontweight="bold", alpha=0.8)
                ax.text(j + 0.5, 1 - i + 0.43, f"{v:,}",
                        ha="center", va="center",
                        fontsize=15, color="#1A1A1A", fontweight="bold")
                ax.text(j + 0.5, 1 - i + 0.18, f"{pct:.1f}%",
                        ha="center", va="center",
                        fontsize=10, color="#666666")

        ax.set_title(SHORT[mn], color=color, fontsize=13, fontweight="bold", pad=14)
        ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels[::-1], fontsize=9, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual",    fontsize=10)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_aspect("equal")

    plt.tight_layout(pad=2)
    plt.savefig(out_dir / "02_confusion_matrices.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: 02_confusion_matrices.png")


def plot_distributions(all_metrics, probas, y_true, out_dir):
    _style()
    model_names = list(all_metrics.keys())
    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(5 * len(model_names), 5),
                             facecolor="white")
    if len(model_names) == 1:
        axes = [axes]
    fig.suptitle("Predicted Probability Distributions  P(split-brain)",
                 fontsize=14, fontweight="bold", y=1.02)

    bins = np.linspace(0, 1, 30)
    for ax, mn in zip(axes, model_names):
        color  = COLORS[mn]
        y_prob = probas[mn]
        neg_p  = y_prob[y_true == 0]
        pos_p  = y_prob[y_true == 1]

        ax.hist(neg_p, bins=bins, alpha=0.55, color="#2ECC71",
                label="Not SB  (actual 0)", density=True, zorder=3)
        ax.hist(pos_p, bins=bins, alpha=0.55, color="#E74C3C",
                label="Split-Brain  (actual 1)", density=True, zorder=3)
        ax.axvline(0.5, color="#AAAAAA", lw=1.5, linestyle="--",
                   label="Threshold 0.5")

        unc = float(((y_prob >= 0.4) & (y_prob <= 0.6)).mean() * 100)
        ax.text(0.97, 0.96, f"Uncertain zone:\n{unc:.0f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#666666",
                bbox=dict(facecolor="white", edgecolor="#DDDDDD",
                          boxstyle="round,pad=0.3"))
        ax.set_title(SHORT[mn], color=color, fontsize=12, fontweight="bold")
        ax.set_xlabel("P(split-brain)", fontsize=10)
        ax.set_ylabel("Density",        fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9, edgecolor="#E0E0E0")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

    plt.tight_layout(pad=2)
    plt.savefig(out_dir / "03_probability_distributions.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: 03_probability_distributions.png")


def plot_metrics_table(all_metrics, out_dir):
    _style()
    model_names = list(all_metrics.keys())

    rows = [
        ("CLASSIFICATION", "ROC-AUC",        "classification", "roc_auc",                    True),
        ("",               "F1-score",        "classification", "f1_score",                    True),
        ("",               "Recall",          "classification", "recall",                      True),
        ("",               "Precision",       "classification", "precision",                   True),
        ("",               "FN rate ↓",       "classification", "false_negative_rate",         False),
        ("",               "FP rate ↓",       "classification", "false_positive_rate",         False),
        ("REGRESSION",     "MAE ↓",           "regression",     "mae",                         False),
        ("",               "R²",              "regression",     "r2",                          True),
        ("",               "Brier Score ↓",   "regression",     "brier_score",                 False),
        ("",               "Brier Skill",     "regression",     "brier_skill",                 True),
        ("CALIBRATION",    "Log-loss ↓",      "calibration",    "log_loss",                    False),
        ("",               "ECE ↓",           "calibration",    "ece",                         False),
        ("",               "MCE ↓",           "calibration",    "mce",                         False),
        ("CORRELATION",    "Spearman r",      "correlation",    "spearman_r",                  True),
        ("TECHNICAL",      "Inference ms ↓",  None,             "inference_time_per_sample_ms",False),
    ]

    n = len(model_names)
    fig, ax = plt.subplots(figsize=(5 + 4 * n, 10), facecolor="white")
    ax.set_facecolor("white"); ax.axis("off")
    fig.suptitle("Summary Metrics Table  ·  Split-Brain Cluster Detector",
                 fontsize=16, fontweight="bold", y=0.99)

    col_x   = [0.01, 0.22] + [0.42 + i * 0.19 for i in range(n)]
    headers = ["", "Metric"] + [SHORT[m] for m in model_names]
    hcolors = ["#999999", "#333333"] + [COLORS[m] for m in model_names]

    for x, h, c in zip(col_x, headers, hcolors):
        ax.text(x, 0.97, h, transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=c, va="top")

    ax.add_line(plt.Line2D([0.01, 0.99], [0.942, 0.942],
                           transform=ax.transAxes, color="#DDDDDD", lw=1))

    current_section = None
    y     = 0.895
    row_h = 0.052

    for idx, (section, metric, cat, key, higher_better) in enumerate(rows):
        if section and section != current_section:
            current_section = section
            ax.text(col_x[0], y, section, transform=ax.transAxes,
                    fontsize=8, fontweight="bold", color="#999999",
                    va="center", style="italic")

        vals = []
        for mn in model_names:
            vals.append(all_metrics[mn][key] if cat is None
                        else all_metrics[mn][cat][key])

        best_i = (vals.index(max(vals)) if higher_better
                  else vals.index(min(vals)))

        if idx % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (0.0, y - row_h * 0.48), 1.0, row_h * 0.92,
                transform=ax.transAxes, facecolor="#F8F9FA", zorder=0,
            ))

        ax.text(col_x[1], y, metric, transform=ax.transAxes,
                fontsize=10.5, color="#1A1A1A", va="center")

        for i, (v, x) in enumerate(zip(vals, col_x[2:])):
            is_best = (i == best_i)
            color   = COLORS[model_names[i]] if is_best else "#888888"
            fw      = "bold" if is_best else "normal"
            txt     = f"★  {v:.4f}" if is_best else f"    {v:.4f}"
            ax.text(x, y, txt, transform=ax.transAxes,
                    fontsize=10.5, color=color, va="center", fontweight=fw)
        y -= row_h

    ax.add_line(plt.Line2D([0.01, 0.99], [y + row_h * 0.3, y + row_h * 0.3],
                           transform=ax.transAxes, color="#DDDDDD", lw=0.8))
    ax.text(0.5, 0.01, "★ = best value in row     ↓ = lower is better",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#AAAAAA")

    plt.savefig(out_dir / "04_metrics_table.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: 04_metrics_table.png")


def plot_radar_pr(all_metrics, probas, y_true, out_dir):
    _style()
    model_names = list(all_metrics.keys())
    colors      = [COLORS[m] for m in model_names]

    fig = plt.figure(figsize=(15, 6), facecolor="white")
    fig.suptitle("Model Profile  ·  Radar Chart & Precision-Recall Curve",
                 fontsize=14, fontweight="bold", y=1.01)

    ax_r = fig.add_subplot(121, polar=True)
    ax_r.set_facecolor("#FAFAFA")

    r_metrics = ["ROC-AUC", "F1", "Recall", "1-FNrate", "BrierSkill", "1-ECE"]
    r_getters = [
        lambda m: all_metrics[m]["classification"]["roc_auc"],
        lambda m: all_metrics[m]["classification"]["f1_score"],
        lambda m: all_metrics[m]["classification"]["recall"],
        lambda m: 1 - all_metrics[m]["classification"]["false_negative_rate"],
        lambda m: max(all_metrics[m]["regression"]["brier_skill"], 0),
        lambda m: max(1 - all_metrics[m]["calibration"]["ece"], 0),
    ]
    N      = len(r_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for mn, color in zip(model_names, colors):
        vals = [g(mn) for g in r_getters] + [r_getters[0](mn)]
        ax_r.plot(angles, vals, "o-", color=color, lw=2, ms=5, label=SHORT[mn])
        ax_r.fill(angles, vals, color=color, alpha=0.08)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(r_metrics, fontsize=9, color="#333333")
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.25, 0.5, 0.75])
    ax_r.set_yticklabels(["0.25", "0.5", "0.75"], fontsize=7, color="#AAAAAA")
    ax_r.grid(color="#E8E8E8", lw=0.8)
    ax_r.spines["polar"].set_color("#E0E0E0")
    ax_r.set_title("Radar Profile  (higher = better)",
                   fontsize=11, fontweight="bold", pad=20)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15),
                fontsize=9, framealpha=0.9, edgecolor="#E0E0E0")

    ax_pr = fig.add_subplot(122)
    for mn, color in zip(model_names, colors):
        y_prob       = probas[mn]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap           = average_precision_score(y_true, y_prob)
        ax_pr.plot(rec, prec, color=color, lw=2.5,
                   label=f"{SHORT[mn]}  AP={ap:.3f}")
        r50 = all_metrics[mn]["classification"]["recall"]
        p50 = all_metrics[mn]["classification"]["precision"]
        ax_pr.scatter([r50], [p50], color=color, s=100, zorder=5,
                      edgecolors="#CCCCCC", lw=0.8)
        ax_pr.annotate("t=0.5", (r50, p50),
                       xytext=(6, -12), textcoords="offset points",
                       fontsize=8, color=color)

    ax_pr.axhline(float(y_true.mean()), color="#BBBBBB", lw=1, linestyle="--",
                  label=f"Baseline P={y_true.mean():.3f}")
    ax_pr.set_xlabel("Recall",    fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.set_title("Precision-Recall Curve\n(dots = threshold 0.5)",
                    fontsize=11, fontweight="bold", pad=10)
    ax_pr.set_xlim(0, 1.02); ax_pr.set_ylim(0, 1.02)
    ax_pr.legend(fontsize=9, framealpha=0.9, edgecolor="#E0E0E0")
    ax_pr.grid(zorder=0)
    ax_pr.set_axisbelow(True)

    plt.tight_layout(pad=2)
    plt.savefig(out_dir / "05_radar_pr.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: 05_radar_pr.png")


def plot_calibration_comparison(before_metrics, after_metrics,
                                before_probas, after_probas,
                                y_true, out_dir):
    _style()
    model_names = list(before_metrics.keys())

    fig, axes = plt.subplots(2, len(model_names),
                             figsize=(5.5 * len(model_names), 10),
                             facecolor="white")
    fig.suptitle("Probability Calibration  ·  Before vs After Isotonic Regression",
                 fontsize=14, fontweight="bold", y=1.01)

    for col, mn in enumerate(model_names):
        color = COLORS[mn]
        for row, (label, m_dict) in enumerate([
            ("Before calibration", before_metrics),
            ("After calibration",  after_metrics),
        ]):
            ax        = axes[row][col]
            bins_data = m_dict[mn]["calibration"]["calibration_bins"]
            pred  = [b["pred"]   for b in bins_data]
            act   = [b["actual"] for b in bins_data]
            gaps  = [b["gap"]    for b in bins_data]
            ece   = m_dict[mn]["calibration"]["ece"]
            r2    = m_dict[mn]["regression"]["r2"]
            brier = m_dict[mn]["regression"]["brier_score"]

            ax.plot([0, 1], [0, 1], "--", color="#CCCCCC", lw=1.5,
                    label="Perfect", zorder=2)
            ax.plot(pred, act, "o-", color=color, lw=2.5, ms=7,
                    zorder=4, label=SHORT[mn])
            ax.fill_between(pred, pred, act, alpha=0.08, color=color)

            for p, a, g in zip(pred, act, gaps):
                fc = "#FDEDEC" if g > 0.1 else ("#EAF7EF" if g < -0.05 else "none")
                ax.plot([p, p], [p, a], lw=5, color=fc,
                        solid_capstyle="round", zorder=1, alpha=0.7)

            ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.05)
            ax.set_xlabel("Predicted probability", fontsize=9)
            ax.set_ylabel("Observed frequency",    fontsize=9)
            ax.grid(zorder=0); ax.set_axisbelow(True)
            ax.legend(fontsize=8, framealpha=0.9, edgecolor="#E0E0E0")

            title_color = "#C0392B" if row == 0 else "#27AE60"
            ax.set_title(
                f"{SHORT[mn]}  ·  {label}\n"
                f"ECE={ece:.3f}   R²={r2:.3f}   Brier={brier:.3f}",
                fontsize=10, fontweight="bold", color=title_color, pad=8,
            )

    plt.tight_layout(pad=2.5)
    plt.savefig(out_dir / "06_calibration_before_after.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: 06_calibration_before_after.png")


def plot_roc_curves(all_metrics, probas, y_true, out_dir):
    _style()
    model_names = list(all_metrics.keys())
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")

    for mn, color in zip(model_names, [COLORS[m] for m in model_names]):
        fpr, tpr, _ = roc_curve(y_true, probas[mn])
        auc_val     = all_metrics[mn]["classification"]["roc_auc"]
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{SHORT[mn]}  AUC={auc_val:.3f}")

    ax.plot([0, 1], [0, 1], "--", color="#CCCCCC", lw=1.5, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.01)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#E0E0E0")
    ax.grid(zorder=0); ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_dir / "07_roc_curves.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: 07_roc_curves.png")


def save_comparison_json(all_metrics, out_dir):
    path = out_dir / "comparison.json"
    with open(path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "models": all_metrics}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def run(models, skip_training, quick):
    cfg     = CONFIGS["quick" if quick else "full"]
    out_dir = OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    total_start = time.time()
    train_times = {}

    if not skip_training:
        train_fns = {
            "cb": lambda c: train_cb(c, cfg["val_size"]),
            "gb": lambda c: train_gb(c, cfg["val_size"]),
            "rf": lambda c: train_rf(c, cfg["val_size"]),
        }
        for key in models:
            model, elapsed = train_fns[key](cfg[key])
            save_model(model, MODEL_PATHS[key])
            train_times[MODEL_KEYS[key]] = round(elapsed, 2)
    else:
        for key in models:
            if not Path(MODEL_PATHS[key]).exists():
                raise FileNotFoundError(
                    f"Model {MODEL_PATHS[key]} not found. "
                    f"Run without --skip_training first."
                )
            print(f"[Load] {MODEL_KEYS[key]} loaded from {MODEL_PATHS[key]}")

    trained = {MODEL_KEYS[k]: load_model(MODEL_PATHS[k]) for k in models}

    print(f"\n[Pipeline] Generating shared eval set ({cfg['eval_size']} samples)...")
    nodes_list, matrices_list, y_true = generate_natural(cfg["eval_size"], "[Eval]")
    print(f"[Pipeline] SB rate = {y_true.mean():.3f}")

    print("\n[Pipeline] Evaluating (before calibration)...")
    before_metrics = {}
    before_probas  = {}
    for key in models:
        mn = MODEL_KEYS[key]
        m, p = evaluate(mn, trained[mn], nodes_list, matrices_list, y_true,
                        label="before_calibration")
        if mn in train_times:
            m["train_time_seconds"] = train_times[mn]
        before_metrics[mn] = m
        before_probas[mn]  = p

    print(f"\n[Pipeline] Calibrating ({cfg['cal_size']} samples, isotonic)...")
    nodes_cal, matrices_cal, y_cal = generate_natural(cfg["cal_size"], "[Cal]")
    X_cal = np.array([preprocess(n, m_) for n, m_ in zip(nodes_cal, matrices_cal)])

    calibrated = {}
    for key in models:
        mn  = MODEL_KEYS[key]
        cal = calibrate_model(trained[mn], X_cal, y_cal)
        save_model(cal, CAL_PATHS[key])
        calibrated[mn] = cal
        print(f"  [Cal] {mn} calibrated and saved.")

    print("\n[Pipeline] Evaluating (after calibration)...")
    after_metrics = {}
    after_probas  = {}
    for key in models:
        mn = MODEL_KEYS[key]
        m, p = evaluate(mn, calibrated[mn], nodes_list, matrices_list, y_true,
                        label="after_calibration")
        if mn in train_times:
            m["train_time_seconds"] = train_times[mn]
        after_metrics[mn] = m
        after_probas[mn]  = p

    print("\n[Pipeline] Generating charts...")
    plot_dashboard(after_metrics, out_dir)
    plot_confusion_matrices(after_metrics, out_dir)
    plot_distributions(after_metrics, after_probas, y_true, out_dir)
    plot_metrics_table(after_metrics, out_dir)
    plot_radar_pr(after_metrics, after_probas, y_true, out_dir)
    plot_calibration_comparison(before_metrics, after_metrics,
                                before_probas, after_probas, y_true, out_dir)
    plot_roc_curves(after_metrics, after_probas, y_true, out_dir)
    save_comparison_json({"before_calibration": before_metrics,
                          "after_calibration":  after_metrics}, out_dir)

    elapsed = time.time() - total_start
    h, rem  = divmod(int(elapsed), 3600)
    m_t, s  = divmod(rem, 60)

    print(f"\n{'─'*65}")
    print(f"  Pipeline complete  {h:02d}:{m_t:02d}:{s:02d}")
    print(f"  Output: {out_dir.resolve()}/")
    print(f"{'─'*65}")
    print(f"\n  {'Model':<22} {'AUC':>7} {'F1':>7} {'ECE↓(after)':>12} "
          f"{'R²(after)':>10} {'ECE↓(before)':>13}")
    print(f"  {'─'*65}")
    for key in models:
        mn = MODEL_KEYS[key]
        bm = before_metrics[mn]
        am = after_metrics[mn]
        print(
            f"  {mn:<22}"
            f"  {am['classification']['roc_auc']:>6.4f}"
            f"  {am['classification']['f1_score']:>6.4f}"
            f"  {am['calibration']['ece']:>11.4f}"
            f"  {am['regression']['r2']:>9.4f}"
            f"  {bm['calibration']['ece']:>12.4f}"
        )
    print(f"{'─'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline: train → calibrate → evaluate → charts"
    )
    parser.add_argument("--models", nargs="+", default=["cb", "gb", "rf"],
                        choices=["cb", "gb", "rf"])
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--quick",         action="store_true")
    args = parser.parse_args()

    run(models=args.models,
        skip_training=args.skip_training,
        quick=args.quick)
"""
metrics.py  —  спільний модуль метрик для всіх моделей

Використання:
    from app.model.metrics import compute_metrics, save_metrics, print_metrics

compute_metrics(model_name, y_true, y_proba) -> dict
    Повертає словник з усіма метриками:
      Класифікаційні : accuracy, precision, recall, f1, roc_auc
      Регресійні     : MAE, R², Brier Score, Brier Skill Score
      Калібрувальні  : ECE, MCE, Log-loss, reliability bins
      Кореляційні    : Spearman r
      Технічні       : inference_ms, клас розподіл
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    brier_score_loss, log_loss,
    mean_absolute_error, r2_score,
)

def compute_metrics(model_name: str,
                    y_true: np.ndarray,
                    y_proba: np.ndarray,
                    label: str = "eval",
                    inference_ms: float = 0.0,
                    n_bins: int = 10) -> dict:

    y_true = np.array(y_true, dtype=float)
    y_proba = np.array(y_proba, dtype=float)
    y_pred = (y_proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    classification = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
        "confusion_matrix": {
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
        },
        "false_negative_rate": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
        "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
    }

    mae = mean_absolute_error(y_true, y_proba)
    r2 = r2_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    climatology = y_true.mean()
    uncertainty = climatology * (1 - climatology)
    brier_skill = 1.0 - brier / max(uncertainty, 1e-10)

    regression = {
        "mae": round(float(mae), 4),
        "mae_baseline": round(float(
            mean_absolute_error(y_true, np.full_like(y_proba, climatology))
        ), 4),

        "r2": round(float(r2), 4),


        "brier_score": round(float(brier), 4),
        "brier_skill": round(float(brier_skill), 4),
        "brier_random": 0.25,

        "brier_uncertainty": round(float(uncertainty), 4),
    }

    bins = np.linspace(0, 1, n_bins + 1)
    reliability_val = 0.0
    resolution_val = 0.0
    calibration_bins = []

    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        n_k = int(mask.sum())
        o_k = float(y_true[mask].mean())
        f_k = float(y_proba[mask].mean())

        reliability_val += n_k * (f_k - o_k) ** 2
        resolution_val += n_k * (o_k - climatology) ** 2

        calibration_bins.append({
            "range":          f"[{bins[i]:.1f}-{bins[i+1]:.1f})",
            "n_samples":      n_k,
            "mean_predicted": round(f_k, 4),
            "actual_freq":    round(o_k, 4),
            "gap":            round(f_k - o_k, 4),
        })

    reliability_val /= max(len(y_true), 1)
    resolution_val /= max(len(y_true), 1)

    ece = sum(b["n_samples"] * abs(b["gap"]) for b in calibration_bins) / max(len(y_true), 1)
    mce = max((abs(b["gap"]) for b in calibration_bins), default=0.0)

    uncertain_zone = float(((y_proba >= 0.4) & (y_proba <= 0.6)).mean())

    calibration = {
        "log_loss":           round(float(log_loss(y_true, y_proba)), 4),
        "ece":                round(float(ece), 4),
        "mce":                round(float(mce), 4),
        "brier_reliability":  round(float(reliability_val), 4),
        "brier_resolution":   round(float(resolution_val), 4),
        "uncertain_zone_pct": round(uncertain_zone * 100, 2),
        "calibration_bins":   calibration_bins,
    }

    sp_r, sp_p = spearmanr(y_proba, y_true)
    correlation = {
        "spearman_r": round(float(sp_r), 4),
        "spearman_p": round(float(sp_p), 6),
    }

    return {
        "model":     model_name,
        "label":     label,
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(y_true),
        "sb_base_rate": round(float(climatology), 4),
        "class_distribution": {
            "negative": int((y_true == 0).sum()),
            "positive": int((y_true == 1).sum()),
        },
        "inference_time_per_sample_ms": round(inference_ms, 4),

        "classification": classification,
        "regression":     regression,
        "calibration":    calibration,
        "correlation":    correlation,
    }


def save_metrics(metrics: dict, path: str):
    history = []
    if Path(path).exists():
        with open(path, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append(metrics)
    with open(path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def print_metrics(metrics: dict):
    m = metrics
    cl = m["classification"]
    rg = m["regression"]
    ca = m["calibration"]
    co = m["correlation"]
    cm = cl["confusion_matrix"]

    print(f"\n{'═' * 62}")
    print(f"  {m['model']} [{m['label']}]")
    print(f"{'═' * 62}")
    print(f"  Вибірка:    {m['n_samples']}  "
          f"(neg={m['class_distribution']['negative']}, "
          f"pos={m['class_distribution']['positive']}, "
          f"SB rate={m['sb_base_rate']:.3f})")

    print(f"\n  ── Класифікаційні ──────────────────────────────")
    print(f"  Accuracy:   {cl['accuracy']:.4f}")
    print(f"  Precision:  {cl['precision']:.4f}")
    print(f"  Recall:     {cl['recall']:.4f}")
    print(f"  F1:         {cl['f1_score']:.4f}")
    print(f"  ROC-AUC:    {cl['roc_auc']:.4f}")
    print(f"  FN rate:    {cl['false_negative_rate']:.4f}  <- пропущені SB")
    print(f"  FP rate:    {cl['false_positive_rate']:.4f}  <- хибні тривоги")
    print(f"  Confusion:  TP={cm['TP']} TN={cm['TN']} "
          f"FP={cm['FP']} FN={cm['FN']}")

    print(f"\n  ── Регресійні ──────────────────────────────────")
    print(f"  MAE:        {rg['mae']:.4f}  "
          f"(baseline={rg['mae_baseline']:.4f})")
    print(f"  R²:         {rg['r2']:.4f}  "
          f"(=Brier Skill Score при бінарних мітках)")
    print(f"  Brier:      {rg['brier_score']:.4f}  "
          f"(random=0.25, ідеал=0.0)")
    print(f"  BrierSkill: {rg['brier_skill']:.4f}  "
          f"(>0 краще за baseline)")

    print(f"\n  ── Калібрувальні ───────────────────────────────")
    print(f"  Log-loss:   {ca['log_loss']:.4f}  "
          f"(ідеал→0, random=0.693)")
    print(f"  ECE:        {ca['ece']:.4f}  "
          f"(ідеал=0, <0.05 добре)")
    print(f"  MCE:        {ca['mce']:.4f}  "
          f"(найгірший бакет)")
    print(f"  Uncertain%: {ca['uncertain_zone_pct']:.1f}%  "
          f"(прогнози в зоні 0.4-0.6)")

    print(f"\n  ── Reliability diagram ─────────────────────────")
    print(f"  {'Діапазон':<12} {'Predicted':>10} {'Actual':>8} "
          f"{'Gap':>8} {'N':>7}")
    for b in ca["calibration_bins"]:
        gap_str = f"{b['gap']:+.4f}"
        flag = " ←" if abs(b["gap"]) > 0.1 else ""
        print(f"  {b['range']:<12} {b['mean_predicted']:>10.4f} "
              f"{b['actual_freq']:>8.4f} {gap_str:>8} "
              f"{b['n_samples']:>7}{flag}")

    print(f"\n  ── Кореляційні ─────────────────────────────────")
    print(f"  Spearman r: {co['spearman_r']:.4f}  "
          f"(p={co['spearman_p']:.2e})")

    print(f"\n  ── Технічні ────────────────────────────────────")
    print(f"  Inference:  {m['inference_time_per_sample_ms']:.4f} ms/sample")
    print(f"{'═' * 62}\n")
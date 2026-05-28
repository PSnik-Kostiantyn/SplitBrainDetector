"""
ensemble.py  —  навчання трьох ансамблів

Запуск:
  python ensemble.py
  python ensemble.py --quick
  python ensemble.py --skip e2 e3
"""

import argparse
import os
import time
from pathlib import Path

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SplitBrainDetector.settings")
django.setup()

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from tqdm import tqdm

from ensemble_model import EnsembleModel
from app.model.DataPreparation import (
    generate_cluster, preprocess, isClusterDead, isSplitBrain,
)

ENSEMBLE_DIR = Path("models/ensembles")
PATHS = {
    "e1": ENSEMBLE_DIR / "ensemble1_mixed.pkl",
    "e2": ENSEMBLE_DIR / "ensemble2_cb_bias.pkl",
    "e3": ENSEMBLE_DIR / "ensemble3_cb_hyper.pkl",
    "e1_cal": ENSEMBLE_DIR / "ensemble1_mixed_cal.pkl",
    "e2_cal": ENSEMBLE_DIR / "ensemble2_cb_bias_cal.pkl",
    "e3_cal": ENSEMBLE_DIR / "ensemble3_cb_hyper_cal.pkl",
}
CONFIGS = {
    "full": {"n_base": 800_000, "val_size": 20_000, "cal_size": 16_000},
    "quick": {"n_base": 5_000, "val_size": 500, "cal_size": 500},
}

NATURAL_SPW = (1 - 0.09) / 0.09 + 25

SB_RATIO = 0.35

_cache: dict = {}


def _gen_balanced(n_normal, n_sb, tag=""):
    X, y = [], []
    for _ in tqdm(range(n_normal), desc=f"{tag} normal", leave=False):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    for _ in tqdm(range(n_sb), desc=f"{tag} SB forced", leave=False):
        nodes, matrix = generate_cluster()
        while not isSplitBrain(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(1)
    idx = np.random.permutation(len(X))
    return np.array(X)[idx], np.array(y)[idx]


def _gen_natural(n, tag=""):
    X, y = [], []
    for _ in tqdm(range(n), desc=f"{tag} natural", leave=False):
        nodes, matrix = generate_cluster()
        while isClusterDead(nodes, matrix):
            nodes, matrix = generate_cluster()
        X.append(preprocess(nodes, matrix))
        y.append(int(isSplitBrain(nodes, matrix)))
    return np.array(X), np.array(y)


def _sb_split(n_total, sb_ratio):
    if sb_ratio <= 0.09:
        return n_total, 0
    n_sb = int((sb_ratio - 0.09) / (1 - 0.09) * n_total)
    n_normal = n_total - n_sb
    return n_normal, n_sb


def train_e1(cfg) -> EnsembleModel:
    n, vs = cfg["n_base"], cfg["val_size"]
    n_ok, n_sb = _sb_split(n, SB_RATIO)
    print(f"\n{'═' * 60}")
    print(f"  E1: RF + GB + CB  (total={n:,}, forced_SB={n_sb:,}, "
          f"target ~{SB_RATIO * 100:.0f}% SB)")
    print(f"{'═' * 60}")

    X, y = _gen_balanced(n_ok, n_sb, "E1")
    pos = int((y == 1).sum());
    neg = int((y == 0).sum())
    actual_pct = pos / len(y) * 100
    sw = np.where(y == 1, NATURAL_SPW, 1.0)
    print(f"  actual: pos={pos} neg={neg} ({actual_pct:.1f}% SB)  "
          f"NATURAL_SPW={NATURAL_SPW:.2f}")

    print("[E1] RF...")
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=4,
        max_features="sqrt", max_samples=0.8,
        class_weight="balanced_subsample",
        oob_score=True, random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    print(f"  OOB={rf.oob_score_:.4f}")

    print("[E1] GB...")
    gb = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        subsample=0.8, max_features=0.8,
        n_iter_no_change=20, validation_fraction=0.1,
        random_state=42, verbose=0,
    )
    gb.fit(X, y, sample_weight=sw)
    print(f"  estimators={gb.n_estimators_}")

    print("[E1] CB...")
    X_val, y_val = _gen_natural(vs, "E1-val")
    cb = CatBoostClassifier(
        iterations=5000, depth=7, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=NATURAL_SPW,
        early_stopping_rounds=150,
        grow_policy="Lossguide", min_data_in_leaf=10,
        l2_leaf_reg=5.0, verbose=0, thread_count=-1, random_seed=42,
    )
    cb.fit(X, y, eval_set=(X_val, y_val), use_best_model=True)
    print(f"  best_iter={cb.best_iteration_}")

    return EnsembleModel([rf, gb, cb], name="E1_Mixed")


def _cb_ratio(n_total, sb_ratio, val_size, tag) -> CatBoostClassifier:
    n_ok, n_sb = _sb_split(n_total, sb_ratio)
    X, y = _gen_balanced(n_ok, n_sb, tag)
    pos = int((y == 1).sum());
    neg = int((y == 0).sum())
    spw = neg / max(pos, 1)
    actual_pct = pos / len(y) * 100
    print(f"  [{tag}] neg={neg} pos={pos} "
          f"({actual_pct:.1f}% SB, target {sb_ratio * 100:.0f}%) spw={spw:.2f}")
    X_val, y_val = _gen_natural(val_size, f"{tag}-val")
    m = CatBoostClassifier(
        iterations=5000, depth=7, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=spw,
        early_stopping_rounds=150,
        grow_policy="Lossguide", min_data_in_leaf=10,
        l2_leaf_reg=5.0, verbose=0, thread_count=-1,
        random_seed=int(sb_ratio * 100),
    )
    m.fit(X, y, eval_set=(X_val, y_val), use_best_model=True)
    print(f"  [{tag}] best_iter={m.best_iteration_}")
    return m


def train_e2(cfg) -> EnsembleModel:
    n, vs = cfg["n_base"], cfg["val_size"]
    print(f"\n{'═' * 60}")
    print(f"  E2: CB-10% + CB-50% + CB-90%  (n={n:,})")
    print(f"{'═' * 60}")
    cb10 = _cb_ratio(n, 0.10, vs, "CB-10%SB")
    cb50 = _cb_ratio(n, 0.50, vs, "CB-50%SB")
    cb90 = _cb_ratio(n, 0.90, vs, "CB-90%SB")
    return EnsembleModel([cb10, cb50, cb90], name="E2_CB_Bias")


def train_e3(cfg) -> EnsembleModel:
    n, vs = cfg["n_base"], cfg["val_size"]
    n_ok, n_sb = _sb_split(n, SB_RATIO)
    print(f"\n{'═' * 60}")
    print(f"  E3: CB × 3 hyper  (total={n:,}, forced_SB={n_sb:,}, "
          f"target ~{SB_RATIO * 100:.0f}% SB)")
    print(f"{'═' * 60}")

    X, y = _gen_balanced(n_ok, n_sb, "E3")
    pos = int((y == 1).sum());
    neg = int((y == 0).sum())
    print(f"  actual: pos={pos} neg={neg} ({pos / len(y) * 100:.1f}% SB)  "
          f"NATURAL_SPW={NATURAL_SPW:.2f}")
    X_val, y_val = _gen_natural(vs, "E3-val")

    kw = dict(
        loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=NATURAL_SPW,
        early_stopping_rounds=150,
        verbose=0, thread_count=-1,
    )

    print("[E3] CB-A shallow lr=0.10  depth=5...")
    ca = CatBoostClassifier(
        iterations=5000, depth=5, learning_rate=0.10,
        l2_leaf_reg=3.0, random_seed=10, **kw)
    ca.fit(X, y, eval_set=(X_val, y_val), use_best_model=True)
    print(f"  best_iter={ca.best_iteration_}")

    print("[E3] CB-B deep lr=0.03  depth=9...")
    cb = CatBoostClassifier(
        iterations=5000, depth=9, learning_rate=0.03,
        grow_policy="Lossguide", min_data_in_leaf=15,
        l2_leaf_reg=8.0, random_seed=20, **kw)
    cb.fit(X, y, eval_set=(X_val, y_val), use_best_model=True)
    print(f"  best_iter={cb.best_iteration_}")

    print("[E3] CB-C balanced high-reg  depth=7...")
    cc = CatBoostClassifier(
        iterations=5000, depth=7, learning_rate=0.05,
        grow_policy="SymmetricTree", subsample=0.7,
        l2_leaf_reg=12.0, random_seed=30, **kw)
    cc.fit(X, y, eval_set=(X_val, y_val), use_best_model=True)
    print(f"  best_iter={cc.best_iteration_}")

    return EnsembleModel([ca, cb, cc], name="E3_CB_Hyper")


def get_ensemble(key: str, calibrated: bool = True) -> EnsembleModel:
    cache_key = f"{key}_{'cal' if calibrated else 'raw'}"
    if cache_key in _cache:
        return _cache[cache_key]

    path = PATHS[f"{key}_cal" if calibrated else key]
    if not path.exists():
        fallback = PATHS[key]
        if fallback.exists():
            ens = EnsembleModel.load(fallback)
            _cache[cache_key] = ens
            return ens
        raise FileNotFoundError(
            f"Ensemble '{key}' not found at {path}.\n"
            f"Run: python ensemble.py"
        )

    ens = EnsembleModel.load(path)
    _cache[cache_key] = ens
    return ens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Small dataset for quick test")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["e1", "e2", "e3"],
                        help="Skip specific ensembles")
    args = parser.parse_args()

    cfg = CONFIGS["quick" if args.quick else "full"]
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"\nConfig: {cfg}")
    print(f"SB_RATIO={SB_RATIO * 100:.0f}%  NATURAL_SPW={NATURAL_SPW:.2f}\n")

    trainers = {"e1": train_e1, "e2": train_e2, "e3": train_e3}
    trained = {}

    for key in ["e1", "e2", "e3"]:
        if key in args.skip:
            print(f"[{key.upper()}] Skipped — loading existing...")
            trained[key] = EnsembleModel.load(PATHS[key])
        else:
            t0 = time.time()
            trained[key] = trainers[key](cfg)
            trained[key].save(PATHS[key])
            print(f"[{key.upper()}] Done in {time.time() - t0:.1f}s")

    print(f"\n[Cal] Generating calibration set "
          f"({cfg['cal_size']} samples, natural distribution)...")
    X_cal, y_cal = _gen_natural(cfg["cal_size"], "[Cal]")
    print(f"[Cal] SB rate={y_cal.mean():.3f}  "
          f"({int(y_cal.sum())} SB / {len(y_cal)})")

    for key, ens in trained.items():
        print(f"\n[Cal] Calibrating {key.upper()}...")
        ens.calibrate(X_cal, y_cal)
        ens.save(PATHS[f"{key}_cal"])
        print(f"  [{key.upper()}] calibrated and saved.")

    elapsed = time.time() - t_total
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n{'─' * 60}")
    print(f"  Complete in {h:02d}:{m:02d}:{s:02d}")
    print(f"  Files: {ENSEMBLE_DIR.resolve()}/")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()

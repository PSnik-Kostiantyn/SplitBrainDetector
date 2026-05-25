import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from app.model.DataPreparation import preprocess


class EnsembleModel:

    def __init__(self, models: list, weights=None, name: str = "Ensemble"):
        self.models = models
        self.weights = np.array(weights or [1.0] * len(models), dtype=float)
        self.weights /= self.weights.sum()
        self.name = name
        self._cal_models = None

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        p = np.zeros(len(X))
        for m, w in zip(self.models, self.weights):
            p += w * m.predict_proba(X)[:, 1]
        return p

    def predict_proba_calibrated(self, X: np.ndarray) -> np.ndarray:
        if self._cal_models is None:
            raise RuntimeError("Not calibrated. Call calibrate() first.")
        p = np.zeros(len(X))
        for m, w in zip(self._cal_models, self.weights):
            p += w * m.predict_proba(X)[:, 1]
        return np.clip(p, 0.0, 1.0)

    def predict(self, nodes: list, matrix, use_calibrated: bool = True) -> float:
        x = preprocess(nodes, matrix).reshape(1, -1)
        if use_calibrated and self._cal_models is not None:
            return float(self.predict_proba_calibrated(x)[0])
        return float(self.predict_proba_raw(x)[0])

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                  method: str = "isotonic"):
        self._cal_models = []
        for i, m in enumerate(self.models):
            cal = CalibratedClassifierCV(estimator=m, method=method, cv="prefit")
            cal.fit(X_cal, y_cal)
            self._cal_models.append(cal)
            print(f"    [{self.name}] model {i + 1}/{len(self.models)} calibrated")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Saved: {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    @staticmethod
    def load(path) -> "EnsembleModel":
        with open(Path(path), "rb") as f:
            return _FixedUnpickler(f).load()


class _FixedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == "EnsembleModel":
            return EnsembleModel
        return super().find_class(module, name)

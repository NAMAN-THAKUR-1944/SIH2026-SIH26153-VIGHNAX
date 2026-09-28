"""Logistic-regression baseline on the *same* per-window state features.

The PS asks for a static classifier baseline "trained on the same features".
Two stateless models are fitted on S_t alone (no temporal memory):

  * ``detect``   : S_t -> attack at t                  (nowcast)
  * ``forecast`` : S_t -> attack anywhere in (t, t+K]  (early warning)

so every comparison with the world model uses identical inputs and targets;
the only difference is the world model's learned temporal dynamics.

Training uses scikit-learn; the fitted model is stored as plain JSON
(coefficients, intercept, thresholds) and scored with NumPy, so the shipped
file works with any scikit-learn version (or none) - a pickled estimator only
loads reliably in the exact version that wrote it.
"""

from __future__ import annotations

import json
import logging
from typing import Dict

import numpy as np

LOGGER = logging.getLogger(__name__)
FORMAT = "vighnax-baseline-lr/1"
TASKS = ("detect", "forecast")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


class BaselineLR:
    def __init__(self, random_state: int = 42, max_iter: int = 2000) -> None:
        self.fit_kwargs = dict(random_state=random_state, max_iter=max_iter, class_weight="balanced", C=1.0)
        self.params: Dict[str, Dict[str, np.ndarray]] = {}     # task -> {"coef": (D,), "intercept": float}
        self.thresholds = {"detect": 0.5, "forecast": 0.5}
        self.thresholds_by_dataset: Dict[str, Dict[str, float]] = {}

    def fit(self, x_scaled: np.ndarray, y_now: np.ndarray, y_future: np.ndarray) -> "BaselineLR":
        return self.fit_task("detect", x_scaled, y_now).fit_task("forecast", x_scaled, y_future)

    def fit_task(self, task: str, x_scaled: np.ndarray, y: np.ndarray) -> "BaselineLR":
        from sklearn.linear_model import LogisticRegression   # training only

        if len(np.unique(y)) < 2:
            raise ValueError(f"BaselineLR[{task}]: training labels contain a single class")
        m = LogisticRegression(**self.fit_kwargs).fit(x_scaled, y)
        self._set(task, m.coef_.ravel(), float(m.intercept_[0]))
        LOGGER.info("BaselineLR[%s] fitted on %d windows", task, len(y))
        return self

    def _set(self, task: str, coef, intercept: float) -> None:
        self.params[task] = {"coef": np.asarray(coef, dtype=np.float64), "intercept": float(intercept)}

    def predict_proba(self, x_scaled: np.ndarray, task: str) -> np.ndarray:
        """P(attack) for each row: the logistic function of the linear score, as in scikit-learn."""
        p = self.params[task]
        return _sigmoid(np.asarray(x_scaled, dtype=np.float64) @ p["coef"] + p["intercept"])

    def save(self, path: str) -> None:
        blob = {"format": FORMAT,
                "models": {t: {"coef": p["coef"].tolist(), "intercept": p["intercept"]} for t, p in self.params.items()},
                "thresholds": self.thresholds, "thresholds_by_dataset": self.thresholds_by_dataset}
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(blob, fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "BaselineLR":
        obj = cls()
        if path.endswith(".joblib"):                            # legacy pickled scikit-learn estimators
            import joblib
            blob = joblib.load(path)
            for task, m in blob["models"].items():
                obj._set(task, m.coef_.ravel(), float(m.intercept_[0]))
        else:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
            if blob.get("format") != FORMAT:
                raise ValueError(f"{path}: not a {FORMAT} file")
            for task, m in blob["models"].items():
                obj._set(task, m["coef"], m["intercept"])
        obj.thresholds = blob["thresholds"]
        obj.thresholds_by_dataset = blob.get("thresholds_by_dataset", {})
        return obj


def _selftest() -> None:
    import os
    import tempfile

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y_now = (x[:, 0] > 0.5).astype(int)
    y_fut = (x[:, 1] > 0.0).astype(int)
    m = BaselineLR().fit(x, y_now, y_fut)
    acc = ((m.predict_proba(x, "detect") > 0.5) == y_now).mean()
    assert acc > 0.9, acc
    path = os.path.join(tempfile.mkdtemp(), "lr.json")
    m.save(path)
    m2 = BaselineLR.load(path)
    assert np.allclose(m2.predict_proba(x, "forecast"), m.predict_proba(x, "forecast"), atol=1e-12)
    print("baseline_lr selftest: OK")


if __name__ == "__main__":
    _selftest()

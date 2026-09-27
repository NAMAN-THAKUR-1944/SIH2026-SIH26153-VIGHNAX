"""Logistic-regression baseline on the *same* per-window state features.

The PS asks for a static classifier baseline "trained on the same features".
Two stateless models are fitted on S_t alone (no temporal memory):

  * ``detect``   : S_t -> attack at t                  (nowcast)
  * ``forecast`` : S_t -> attack anywhere in (t, t+K]  (early warning)

so every comparison with the world model uses identical inputs and targets;
the only difference is the world model's learned temporal dynamics.
"""

from __future__ import annotations

import logging
from typing import Dict

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)


class BaselineLR:
    def __init__(self, random_state: int = 42, max_iter: int = 2000) -> None:
        kw = dict(random_state=random_state, max_iter=max_iter, class_weight="balanced", C=1.0)
        self.models: Dict[str, LogisticRegression] = {"detect": LogisticRegression(**kw),
                                                      "forecast": LogisticRegression(**kw)}
        self.thresholds = {"detect": 0.5, "forecast": 0.5}
        self.thresholds_by_dataset: Dict[str, Dict[str, float]] = {}

    def fit(self, x_scaled: np.ndarray, y_now: np.ndarray, y_future: np.ndarray) -> "BaselineLR":
        for task, y in (("detect", y_now), ("forecast", y_future)):
            if len(np.unique(y)) < 2:
                raise ValueError(f"BaselineLR[{task}]: training labels contain a single class")
            self.models[task].fit(x_scaled, y)
            LOGGER.info("BaselineLR[%s] fitted on %d windows", task, len(y))
        return self

    def predict_proba(self, x_scaled: np.ndarray, task: str) -> np.ndarray:
        return self.models[task].predict_proba(x_scaled)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump({"models": self.models, "thresholds": self.thresholds,
                     "thresholds_by_dataset": self.thresholds_by_dataset}, path)

    @classmethod
    def load(cls, path: str) -> "BaselineLR":
        obj = cls()
        blob = joblib.load(path)
        obj.models, obj.thresholds = blob["models"], blob["thresholds"]
        obj.thresholds_by_dataset = blob.get("thresholds_by_dataset", {})
        return obj


def _selftest() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y_now = (x[:, 0] > 0.5).astype(int)
    y_fut = (x[:, 1] > 0.0).astype(int)
    m = BaselineLR().fit(x, y_now, y_fut)
    acc = ((m.predict_proba(x, "detect") > 0.5) == y_now).mean()
    assert acc > 0.9, acc
    print("baseline_lr selftest: OK")


if __name__ == "__main__":
    _selftest()

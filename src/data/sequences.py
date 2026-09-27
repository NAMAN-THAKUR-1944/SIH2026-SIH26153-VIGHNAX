"""Scaling and sequence windowing shared by training, evaluation and inference."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.state_builder import STATE_FEATURES


class StateScaler:
    """Per-feature standardisation fitted on training windows only."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> None:
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray) -> "StateScaler":
        self.mean = x.mean(0)
        std = x.std(0)
        self.std = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "StateScaler":
        return cls(np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64))


def state_matrix(states: pd.DataFrame) -> np.ndarray:
    return states[list(STATE_FEATURES)].to_numpy(dtype=np.float64)


def split_chronological(states: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(states) * (1 - val_fraction))
    return states.iloc[:cut], states.iloc[cut:]


def val_mask(n: int, block: int, every: int) -> np.ndarray:
    """Blocked validation: every ``every``-th contiguous block of ``block`` windows is validation.

    Attacks are not spread uniformly over a capture, so a single tail split can
    be all-benign or all-attack; interleaved blocks cover the whole timeline.
    """
    return (np.arange(n) // block) % every == every - 1


def contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append((start, i)); start = None
    if start is not None:
        segs.append((start, len(mask)))
    return segs


def make_sequences(
    x: np.ndarray, y: np.ndarray, stage: np.ndarray, length: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows of ``length`` over one contiguous scenario (never across scenarios)."""
    n = len(x) - length + 1
    if n <= 0:
        return (np.empty((0, length, x.shape[1]), np.float32), np.empty((0, length), np.int64),
                np.empty((0, length), np.int64))
    idx = np.arange(0, n, stride)[:, None] + np.arange(length)[None, :]
    return x[idx].astype(np.float32), y[idx].astype(np.int64), stage[idx].astype(np.int64)


def context_windows(x: np.ndarray, context: int) -> np.ndarray:
    """One context window ending at every time step t (left-padded by repeating the first state)."""
    pad = np.repeat(x[:1], context - 1, axis=0)
    xp = np.concatenate([pad, x], axis=0)
    idx = np.arange(len(x))[:, None] + np.arange(context)[None, :]
    return xp[idx].astype(np.float32)


def future_targets(y: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """For each t: y_any = any attack in (t, t+K], and a validity mask (full horizon available)."""
    n = len(y)
    y_any = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    for t in range(n):
        seg = y[t + 1: t + 1 + horizon]
        valid[t] = len(seg) == horizon
        y_any[t] = int(seg.max()) if len(seg) else 0
    return y_any, valid


def future_matrix(y: np.ndarray, horizon: int) -> np.ndarray:
    """(n, K) matrix of y_{t+k}; -1 where beyond the end."""
    n = len(y)
    out = -np.ones((n, horizon), dtype=np.int64)
    for k in range(1, horizon + 1):
        out[: n - k, k - 1] = y[k:]
    return out

"""End-to-end checks on the metric helpers and sequence utilities."""

import numpy as np

from src.data.sequences import context_windows, future_targets
from src.evaluation.metrics import binary_metrics, lead_time, onsets


def test_future_targets():
    y = np.array([0, 0, 0, 1, 0, 0])
    y_any, valid = future_targets(y, 2)
    assert y_any.tolist() == [0, 1, 1, 0, 0, 0]
    assert valid.tolist() == [True, True, True, True, False, False]


def test_context_windows_left_pad():
    x = np.arange(4, dtype=float)[:, None]
    ctx = context_windows(x, 3)
    assert ctx.shape == (4, 3, 1)
    assert ctx[0, :, 0].tolist() == [0, 0, 0] and ctx[3, :, 0].tolist() == [1, 2, 3]


def test_fpr_and_lead_time():
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    m = binary_metrics(y, np.array([0, 0, .9, 0, .9, .9, 0, 0, 0, 0]), 0.5)
    assert m["fpr"] == 1 / 8 and m["recall"] == 1.0
    score = np.array([0, 0, .8, .8, .8, 0, 0, 0, 0, 0])
    assert onsets(y, quiet=3) == [4]
    lt = lead_time(y, score, 0.5, horizon=3, window_s=10)
    assert lt["warned_before_onset"] == 1 and lt["mean_lead_s"] == 20.0

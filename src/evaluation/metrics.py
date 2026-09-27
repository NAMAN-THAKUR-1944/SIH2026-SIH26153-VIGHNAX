"""Benchmark metrics: F1 / precision / recall / FPR / AUROC and early-warning lead time."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        nan = float("nan")
        return {"tpr_at_fpr_01": nan, "tpr_at_fpr_05": nan, "tpr_at_fpr_10": nan, "f1": nan, "precision": nan, "recall": nan, "fpr": nan, "auroc": nan, "auprc": nan,
                "threshold": float(threshold), "n": 0, "positives": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    both = len(np.unique(y)) > 1
    tpr_at = {}
    if both:
        from sklearn.metrics import roc_curve
        fpr_c, tpr_c, _ = roc_curve(y, p)
        for target in (0.01, 0.05, 0.10):
            tpr_at[f"tpr_at_fpr_{int(target * 100):02d}"] = float(np.interp(target, fpr_c, tpr_c))
    else:
        tpr_at = {k: float("nan") for k in ("tpr_at_fpr_01", "tpr_at_fpr_05", "tpr_at_fpr_10")}
    return tpr_at | {
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if both else float("nan"),
        "threshold": float(threshold), "n": int(len(y)), "positives": int(y.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 (chosen on validation data only, for every model alike)."""
    grid = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 201)))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def best_youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising Youden's J = TPR - FPR.

    Unlike max-F1, J does not depend on class prevalence, so a validation set
    dominated by one long, mostly-malicious capture cannot drag the operating
    point to "flag everything".
    """
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(min(max(thr[best], 0.0), 1.0))


def fpr_budget_threshold(y: np.ndarray, p: np.ndarray, max_fpr: float = 0.05) -> float:
    """Lowest threshold whose false-positive rate on (validation) negatives is <= max_fpr.

    The usual SOC operating point: detect as much as possible within a fixed
    false-alarm budget. Falls back to Youden when there are no negatives.
    """
    neg = np.sort(np.asarray(p, dtype=float)[np.asarray(y) == 0])
    if len(neg) == 0:
        return best_youden_threshold(y, p)
    k = int(np.floor((1.0 - max_fpr) * len(neg)))
    return float(neg[min(k, len(neg) - 1)] + 1e-9)


def select_threshold(y: np.ndarray, p: np.ndarray, rule: str = "fpr05") -> float:
    if rule.startswith("fpr"):
        return fpr_budget_threshold(y, p, int(rule[3:] or 5) / 100.0)
    return best_youden_threshold(y, p) if rule == "youden" else best_f1_threshold(y, p)


def onsets(y: np.ndarray, quiet: int = 3) -> List[int]:
    """Indices where an attack starts after at least ``quiet`` benign windows."""
    out, run = [], quiet
    for t, v in enumerate(np.asarray(y).astype(int)):
        if v == 1 and run >= quiet:
            out.append(t)
        run = 0 if v == 1 else run + 1
    return out


def lead_time(y: np.ndarray, score: np.ndarray, threshold: float, horizon: int, window_s: float,
              quiet: int = 3) -> Dict[str, float]:
    """Early warning before each attack onset.

    ``score[t]`` is a forecast made at time t for (t, t+K]. For an onset at t0,
    the warning time is t0 - t_first where t_first is the earliest t in
    [t0-K, t0-1] such that the score stays >= threshold from t_first to t0-1.
    """
    y = np.asarray(y).astype(int)
    leads, warned = [], 0
    ons = onsets(y, quiet)
    for t0 in ons:
        first = None
        for t in range(t0 - 1, max(t0 - horizon, 0) - 1, -1):
            if score[t] >= threshold:
                first = t
            else:
                break
        if first is not None:
            warned += 1
            leads.append((t0 - first) * window_s)
        else:
            leads.append(0.0)
    return {
        "onsets": len(ons),
        "warned_before_onset": warned,
        "warned_frac": warned / len(ons) if ons else float("nan"),
        "mean_lead_s": float(np.mean(leads)) if leads else float("nan"),
        "mean_lead_s_when_warned": float(np.mean([l for l in leads if l > 0])) if warned else 0.0,
    }


def stage_metrics(y_stage: np.ndarray, p_stage: np.ndarray, present_only: bool = True) -> Dict[str, float]:
    from sklearn.metrics import f1_score as f1
    if len(y_stage) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    pred = p_stage.argmax(-1)
    labels = sorted(set(np.unique(y_stage)) | set(np.unique(pred))) if present_only else None
    return {"accuracy": float((pred == y_stage).mean()),
            "macro_f1": float(f1(y_stage, pred, labels=labels, average="macro", zero_division=0))}

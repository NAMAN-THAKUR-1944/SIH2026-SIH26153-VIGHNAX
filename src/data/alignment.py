"""Clock alignment between flow records and packet captures."""

from __future__ import annotations

import numpy as np


def best_offset(flows, pk, window_s: float, candidates_s=range(-4 * 3600, 4 * 3600 + 1, 1800)) -> float:
    """Clock offset (s) that best aligns packet volume with flow volume per window.

    Flow files may be written in local time while captures are in UTC, and the
    two need not start at the same moment, so first-timestamp matching is
    unreliable. Instead correlate per-window packet counts with per-window flow
    counts for every candidate offset and keep the best.
    """
    fl = (flows["start_time"] // window_s).astype("int64").value_counts()
    pk_counts = pk.set_index("window_id")["pk_count_log"]
    best, best_r = 0.0, -2.0
    for off in candidates_s:
        shifted = pk_counts.copy()
        shifted.index = shifted.index + int(off // window_s)
        common = shifted.index.intersection(fl.index)
        if len(common) < 10:
            continue
        r = np.corrcoef(shifted.loc[common].to_numpy(), np.log1p(fl.loc[common].to_numpy()))[0, 1]
        if np.isfinite(r) and r > best_r:
            best, best_r = float(off), float(r)
    return best

"""Regenerate the results block in README.md from results/metrics.json.

Every sentence that compares the two models is computed from the numbers, so
re-running after a retrain can never leave a stale or over-claiming statement.

    python scripts/update_readme_results.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.sources import DATASET_NAMES, NON_TEMPORAL  # noqa: E402
from src.data.state_builder import STATE_FEATURES  # noqa: E402

R = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
BY, M = R["test_by_dataset"], R["test_macro"]


def f(v, d=2):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


TIE = 0.01  # differences below this are reported as ties


def pair(r, task, key="auroc"):
    w, l = r[task]["world_model"][key], r[task]["baseline_lr"][key]
    bold = (not np.isnan(w)) and (np.isnan(l) or w - l >= TIE)
    return (f"**{f(w)}**" if bold else f(w)) + " / " + f(l)


rows = []
for ds, r in BY.items():
    lw, ll = r["lead_time"]["world_model"], r["lead_time"]["baseline_lr"]
    if ds in NON_TEMPORAL:
        cells = ["n/a", "n/a", pair(r, "detect"), "n/a"]
    else:
        cells = [pair(r, "forecast"), pair(r, "forecast_from_benign"), pair(r, "detect"),
                 f"{lw['warned_before_onset']}/{lw['onsets']} vs {ll['warned_before_onset']}/{ll['onsets']}"]
    rows.append(f"| {DATASET_NAMES.get(ds, ds)} | {r['windows']:,} | " + " | ".join(cells) + " |")

mac = [("forecast", "auroc"), ("forecast_from_benign", "auroc"), ("detect", "auroc")]
mrow = " | ".join(("**" + f(M[f'{t}.{k}.world_model']) + "**" if M[f'{t}.{k}.world_model'] - M[f'{t}.{k}.baseline_lr'] >= TIE
                   else f(M[f'{t}.{k}.world_model'])) + " / " + f(M[f'{t}.{k}.baseline_lr']) for t, k in mac)

temporal = [d for d in BY if d not in NON_TEMPORAL]
def gap(d, t):
    return BY[d][t]["world_model"]["auroc"] - BY[d][t]["baseline_lr"]["auroc"]


wins = {t: [d for d in temporal if gap(d, t) >= TIE] for t, _ in mac}
losses = {t: [DATASET_NAMES.get(d, d) for d in temporal if gap(d, t) <= -TIE] for t, _ in mac}
ties = {t: [DATASET_NAMES.get(d, d) for d in temporal if abs(gap(d, t)) < TIE] for t, _ in mac}
det_all = [d for d in BY if gap(d, "detect") >= TIE]
det_loss = [DATASET_NAMES.get(d, d) for d in BY if gap(d, "detect") <= -TIE]
det_tie = [DATASET_NAMES.get(d, d) for d in BY if abs(gap(d, "detect")) < TIE]
lead_w = sum(BY[d]["lead_time"]["world_model"]["warned_before_onset"] for d in temporal)
lead_l = sum(BY[d]["lead_time"]["baseline_lr"]["warned_before_onset"] for d in temporal)
onsets = sum(BY[d]["lead_time"]["world_model"]["onsets"] for d in temporal)
ta = R["test_all"]

lines = [
    "<!-- RESULTS-START -->",
    f"### Results: one model, seven datasets, test data unseen in training",
    "",
    "AUROC, world model / logistic-regression baseline on the same features (bold = world model better). Every test "
    "source is a day, capture, time slice or botnet family that was not used for training.",
    "",
    "| Dataset | Test windows | Forecast (next 60 s) | Early warning (still benign) | Detection (now) | Warned before onset |",
    "|---|---|---|---|---|---|",
    *rows,
    f"| **Macro average** (6 temporal datasets) | | {mrow} | {lead_w}/{onsets} vs {lead_l}/{onsets} |",
    "",
    f"* **Forecasting (next 60 s):** world model ahead on {len(wins['forecast'])} of {len(temporal)} datasets"
    + (f", tied on {', '.join(ties['forecast'])}" if ties["forecast"] else "")
    + (f", LR ahead on {', '.join(losses['forecast'])}." if losses["forecast"] else ".")
    + f" Early warning from still-benign windows: ahead on {len(wins['forecast_from_benign'])}"
    + (f", tied on {', '.join(ties['forecast_from_benign'])}" if ties["forecast_from_benign"] else "")
    + (f", LR ahead on {', '.join(losses['forecast_from_benign'])}." if losses["forecast_from_benign"] else "."),
    f"* **Detection of the current window:** world model ahead on {len(det_all)} of {len(BY)} datasets"
    + (f", tied on {', '.join(det_tie)}" if det_tie else "")
    + (f", LR ahead on {', '.join(det_loss)}." if det_loss else "."),
    f"* **Early warning before attack onsets:** {lead_w} of {onsets} onsets across the temporal datasets were forecast "
    f"before they started by the world model, {lead_l} by LR.",
    f"* **MITRE ATT&CK stage identification** on malicious test windows (all datasets): accuracy "
    f"{f(ta['stage_now_malicious'].get('accuracy', float('nan')))}, macro-F1 {f(ta['stage_now_malicious'].get('macro_f1', float('nan')))}.",
    "* Alert thresholds are calibrated per network on that dataset's own validation split (the dashboard uses the matching "
    "profile, or a global one for unknown networks). F1 / FPR per dataset are in `results/RESULTS.md`.",
    "* CICIoT2023 captures hold one activity each; its episodes are composed (benign capture, then attack capture), so only "
    "detection is scored there.",
    "",
    "![Forecast timeline on an unseen botnet family](results/timeline_s47.png)",
    "",
    "Full tables (pooled metrics, F1 / FPR, validation per dataset, packet-block ablation, runtime): "
    "[`results/RESULTS.md`](results/RESULTS.md). Batched inference costs " + f(R["performance"]["ms_per_window"], 2)
    + f" ms per 10 s window on {'the GPU' if 'cuda' in R['performance']['device'] else 'CPU'} (32 Monte-Carlo rollouts; "
    "about 8 ms per window when windows are streamed one at a time on a laptop CPU).",
    "<!-- RESULTS-END -->",
]
block = "\n".join(lines)
p = os.path.join(ROOT, "README.md")
s = open(p, encoding="utf-8").read()
a, b = s.index("<!-- RESULTS-START -->"), s.index("<!-- RESULTS-END -->") + len("<!-- RESULTS-END -->")
open(p, "w", encoding="utf-8").write(s[:a] + block + s[b:])
print(block)

"""Benchmark the world model against the logistic-regression baseline.

Splits
  * validation : last 20 % (chronological) of each *training* scenario
  * test       : whole scenarios of botnet families never seen in training

Tasks (identical inputs S_t and targets for both models; thresholds chosen on validation)
  * detect   : is the network compromised in window t?
  * forecast : will malicious activity occur in the next K windows (t, t+K]?
Also reported for the world model: per-horizon AUROC, MITRE stage accuracy,
early-warning lead time before attack onsets, unsupervised "surprise" AUROC,
and a CSV-only ablation (packet-level block removed).

    python evaluate.py   ->  results/metrics.json, results/RESULTS.md, results/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data import stages as st
from src.data.sequences import StateScaler, context_windows, future_matrix, future_targets, state_matrix, val_mask
from src.data.sources import DATASET_NAMES, NON_TEMPORAL, load_sources, site_index
from src.evaluation.metrics import binary_metrics, lead_time, stage_metrics
from src.models.baseline_lr import BaselineLR
from src.models.world_model import CyberWorldModel

FAMILY = {46: "Virut", 48: "Sogou", 51: "Rbot", 52: "Rbot", 53: "NSIS.ay", 47: "Menti", 43: "Neris",
          42: "Neris", 44: "Rbot", 45: "Rbot", 49: "Murlo", 50: "Neris", 54: "Virut"}


def load_model(path: str, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = ck["config"]["model"]
    model = CyberWorldModel(len(ck["features"]), mc["latent_dim"], mc["hidden_dim"], st.N_STAGES, mc["dropout"],
                            mc.get("obs_skip", False), ck.get("n_sites", 0))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def run_world_model(model, x: np.ndarray, context: int, horizon: int, n_samples: int, device, bs: int = 256,
                    site: int = -1):
    ctx = context_windows(x, context)
    outs: Dict[str, List[np.ndarray]] = {}
    for i in range(0, len(ctx), bs):
        c = torch.as_tensor(ctx[i:i + bs], device=device)
        s = torch.full((len(c),), site, dtype=torch.long, device=device) if model.n_sites else None
        fc = model.forecast(c, horizon, n_samples, site=s)
        for k, v in fc.items():
            outs.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def evaluate_part(model, lr, df: pd.DataFrame, scaler: StateScaler, cfg, device, drop_packets=None, keep=None,
                  unknown_site: bool = False):
    context, horizon = cfg["model"]["context"], cfg["model"]["horizon"]
    x = scaler.transform(state_matrix(df))
    if drop_packets is not None:
        idx, vals = drop_packets
        x[:, idx] = vals
    y = df["is_attack"].to_numpy().astype(int)
    stage = df["stage"].to_numpy().astype(int)
    torch.manual_seed(cfg["seed"])
    site = -1 if unknown_site else site_index(str(df["dataset"].iloc[0])) if "dataset" in df else -1
    wm = run_world_model(model, x, context, horizon, cfg["eval"]["mc_samples"], device, site=site)
    y_any, valid = future_targets(y, horizon)
    out = {
        "y": y, "stage": stage, "y_any": y_any, "valid": valid, "y_future": future_matrix(y, horizon),
        "stage_future": future_matrix(stage, horizon),
        "wm_now": wm["nowcast"], "wm_any": wm["p_any"], "wm_step": wm["p_step"],
        "wm_stage_now": wm["stage_now"], "wm_stage_future": wm["stage_future"], "wm_surprise": wm["surprise"],
        "wm_any_lo": wm["p_any_lo"], "wm_any_hi": wm["p_any_hi"],
        "lr_now": lr.predict_proba(x, "detect"), "lr_any": lr.predict_proba(x, "forecast"),
    }
    if keep is not None:  # score only these windows (history still comes from the full capture)
        out = {k: v[keep] for k, v in out.items()}
    return out


def concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summarize(r: Dict[str, np.ndarray], ck, lr, cfg, dataset=None) -> Dict:
    horizon, win = cfg["model"]["horizon"], cfg["data"]["window_s"]
    thr = ck.get("thresholds_by_dataset", {}).get(dataset, ck["thresholds"])
    lr_thr = lr.thresholds_by_dataset.get(dataset, lr.thresholds)
    v = r["valid"]
    q = v & (r["y"] == 0)  # early warning: network currently benign - will an attack start within K?
    out = {
        "detect": {"world_model": binary_metrics(r["y"], r["wm_now"], thr["nowcast"]),
                   "baseline_lr": binary_metrics(r["y"], r["lr_now"], lr_thr["detect"])},
        "forecast": {"world_model": binary_metrics(r["y_any"][v], r["wm_any"][v], thr["forecast"]),
                     "baseline_lr": binary_metrics(r["y_any"][v], r["lr_any"][v], lr_thr["forecast"])},
        "forecast_from_benign": {"world_model": binary_metrics(r["y_any"][q], r["wm_any"][q], thr["forecast"]),
                                 "baseline_lr": binary_metrics(r["y_any"][q], r["lr_any"][q], lr_thr["forecast"])},
    }
    per_h = []
    for k in range(horizon):
        m = r["y_future"][:, k] >= 0
        yk = r["y_future"][m, k]
        per_h.append({"k": k + 1, "seconds_ahead": (k + 1) * win,
                      "auroc": binary_metrics(yk, r["wm_step"][m, k], 0.5)["auroc"]})
    out["forecast_per_horizon"] = per_h
    # Stage identification on malicious windows: which ATT&CK stage (argmax over attack stages only).
    mal = r["stage"] != st.BENIGN
    ident = lambda y, p: stage_metrics(y, np.concatenate([np.full(p.shape[:-1] + (1,), -1.0), p[..., 1:]], -1))
    out["stage_now_all"] = stage_metrics(r["stage"], r["wm_stage_now"])
    out["stage_now_malicious"] = ident(r["stage"][mal], r["wm_stage_now"][mal]) if mal.any() else {}
    fk = r["stage_future"][:, -1]
    m = (fk >= 0) & (fk != st.BENIGN)
    out[f"stage_t+{horizon}_malicious"] = ident(fk[m], r["wm_stage_future"][m, -1]) if m.any() else {}
    out["stage_majority_baseline_malicious"] = (float(np.bincount(r["stage"][mal]).max() / mal.sum())
                                                if mal.any() else float("nan"))
    out["lead_time"] = {
        "world_model": lead_time(r["y"], r["wm_any"], thr["forecast"], horizon, win),
        "baseline_lr": lead_time(r["y"], r["lr_any"], lr_thr["forecast"], horizon, win),
    }
    out["surprise_auroc_unsupervised"] = binary_metrics(r["y"], r["wm_surprise"], np.median(r["wm_surprise"]))["auroc"]
    out["windows"], out["malicious_windows"] = int(len(r["y"])), int(r["y"].sum())
    return out


def plot_timeline(path: str, title: str, df: pd.DataFrame, r: Dict[str, np.ndarray], ck, win: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = (np.arange(len(r["y"])) * win) / 60.0
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].fill_between(t, 0, r["y"], step="post", color="#e76f51", alpha=0.18, label="ground truth: malicious window")
    ax[0].plot(t, r["lr_any"], color="#8d99ae", lw=1, label="LR baseline: P(attack in next K)")
    ax[0].fill_between(t, r["wm_any_lo"], r["wm_any_hi"], color="#2a9d8f", alpha=0.2, lw=0)
    ax[0].plot(t, r["wm_any"], color="#264653", lw=1.4, label="World model: P(attack in next K) (10-90% band)")
    ax[0].axhline(ck["thresholds"]["forecast"], color="#264653", ls=":", lw=0.8)
    ax[0].set_ylim(0, 1.02); ax[0].set_ylabel("probability"); ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title(title, fontsize=10)
    colors = ["#e9ecef", "#f4a261", "#e9c46a", "#a8dadc", "#457b9d", "#6d597a", "#e63946"]
    ax[1].scatter(t, r["stage"], c=[colors[s] for s in r["stage"]], s=6, label="true stage")
    ax[1].plot(t, r["wm_stage_now"].argmax(1), color="#264653", lw=0.8, alpha=0.7, label="predicted stage")
    ax[1].set_yticks(range(st.N_STAGES)); ax[1].set_yticklabels([s.key for s in st.STAGES], fontsize=7)
    ax[1].set_xlabel("minutes"); ax[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fmt(m: Dict) -> str:
    return (f"{m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['fpr']:.3f} | {m['auroc']:.3f} | "
            f"{m['tpr_at_fpr_05']:.3f} | {m['tpr_at_fpr_10']:.3f}")


def _f(v, d=3) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


MACRO_KEYS = [("forecast", "auroc"), ("forecast_from_benign", "auroc"), ("detect", "auroc"),
              ("forecast", "f1"), ("forecast", "fpr"), ("detect", "f1"), ("detect", "fpr")]


def macro(by_ds: Dict[str, Dict], datasets: List[str]) -> Dict:
    """Unweighted mean over datasets, so one large dataset cannot dominate."""
    out = {}
    for task, key in MACRO_KEYS:
        for model in ("world_model", "baseline_lr"):
            vals = [by_ds[d][task][model][key] for d in datasets if not np.isnan(by_ds[d][task][model][key])]
            out[f"{task}.{key}.{model}"] = float(np.mean(vals)) if vals else float("nan")
    return out


def write_markdown(path: str, res: Dict, cfg) -> None:
    K, win = cfg["model"]["horizon"], cfg["data"]["window_s"]
    L = ["# Benchmark results", "",
         f"Generated by `python evaluate.py` from `{cfg['artifacts']['model']}`. Window = {win:g} s, context = "
         f"{cfg['model']['context']} windows, forecast horizon K = {K} windows ({K * win:g} s). One model is trained on all "
         "datasets; every test source (day, capture, time slice or botnet family) is unseen in training.",
         "F1 / FPR use each model's operating threshold, chosen per dataset on that dataset's validation split only "
         "(highest detection with at most 5% false alarms on validation, same rule for both models; i.e. per-site calibration). "
         "AUROC is threshold-free. WM = world model, LR = logistic-regression baseline on the same features.", "",
         "## Test results per dataset", "",
         "| Dataset | Windows (malicious) | Forecast AUROC WM / LR | Early-warning AUROC WM / LR | Detection AUROC WM / LR | "
         "Forecast F1 WM / LR | Forecast FPR WM / LR | Warned before onset WM / LR |",
         "|---|---|---|---|---|---|---|---|"]
    for ds, r in res["test_by_dataset"].items():
        def pair(t, k, r=r):
            return f"{_f(r[t]['world_model'][k])} / {_f(r[t]['baseline_lr'][k])}"
        lw, ll = r["lead_time"]["world_model"], r["lead_time"]["baseline_lr"]
        if ds in NON_TEMPORAL:
            cells = ["n/a (composed episodes)"] * 2 + [pair("detect", "auroc")] + ["n/a"] * 3
        else:
            cells = [pair("forecast", "auroc"), pair("forecast_from_benign", "auroc"), pair("detect", "auroc"),
                     pair("forecast", "f1"), pair("forecast", "fpr"),
                     f"{lw['warned_before_onset']}/{lw['onsets']} / {ll['warned_before_onset']}/{ll['onsets']}"]
        L.append(f"| {DATASET_NAMES.get(ds, ds)} | {r['windows']:,} ({r['malicious_windows']:,}) | " + " | ".join(cells) + " |")
    m = res["test_macro"]
    L.append("| **Macro average** (temporal datasets) | | "
             + " | ".join(f"**{_f(m[t + '.' + k + '.world_model'])}** / {_f(m[t + '.' + k + '.baseline_lr'])}"
                          for t, k in [("forecast", "auroc"), ("forecast_from_benign", "auroc"), ("detect", "auroc"),
                                       ("forecast", "f1"), ("forecast", "fpr")]) + " | |")
    if res.get("test_unknown_site_macro"):
        u = res["test_unknown_site_macro"]
        L.append("| Macro average, **unknown network** (no site profile) | | "
                 + " | ".join(f"{_f(u[t + '.' + k + '.world_model'])} / {_f(u[t + '.' + k + '.baseline_lr'])}"
                              for t, k in [("forecast", "auroc"), ("forecast_from_benign", "auroc"), ("detect", "auroc"),
                                           ("forecast", "f1"), ("forecast", "fpr")]) + " | |")
    L += ["", "CICIoT2023 captures contain one activity each; its test episodes are a benign capture followed by an "
          "attack capture, so the transition is artificial and only detection is reported for it.", ""]

    r = res["test"]
    L += ["## Pooled test windows (temporal datasets)", "",
          f"{r['windows']:,} windows, {r['malicious_windows']:,} malicious. Pooling weights large datasets more; see the "
          "per-dataset table and its macro average above.", "",
          "| Task | Model | F1 | Precision | Recall | FPR | AUROC | TPR @ 5% FPR | TPR @ 10% FPR |", "|---|---|---|---|---|---|---|---|---|"]
    for task, name in (("detect", "Detection (window t)"), ("forecast", f"Forecast (next {K} windows)"),
                       ("forecast_from_benign", "Early warning (from benign windows)")):
        L.append(f"| {name} | World model | {fmt(r[task]['world_model'])} |")
        L.append(f"| {name} | LR baseline | {fmt(r[task]['baseline_lr'])} |")
    ta = res["test_all"]
    L += ["", "**MITRE stage identification** (malicious windows, all test datasets): accuracy "
          + _f(ta["stage_now_malicious"].get("accuracy", float("nan"))) + ", macro-F1 "
          + _f(ta["stage_now_malicious"].get("macro_f1", float("nan"))) + " (majority-stage rate "
          + _f(ta["stage_majority_baseline_malicious"]) + ").", ""]

    L += ["## Validation (interleaved 5-min blocks of the training sources)", "",
          "| Dataset | Windows | Forecast AUROC WM / LR | Detection AUROC WM / LR | Early-warning AUROC WM / LR |", "|---|---|---|---|---|"]
    for ds, r in res["validation_by_dataset"].items():
        def vp(t, r=r):
            return f"{_f(r[t]['world_model']['auroc'])} / {_f(r[t]['baseline_lr']['auroc'])}"
        L.append(f"| {DATASET_NAMES.get(ds, ds)} | {r['windows']:,} | {vp('forecast')} | {vp('detect')} | {vp('forecast_from_benign')} |")
    if res.get("test_flow_only"):
        L += ["", "## Ablation - packet block removed (test, datasets with packet captures)", "",
              "| Dataset | Input | Forecast AUROC | Early-warning AUROC | Detection AUROC |", "|---|---|---|---|---|"]
        for ds, ab in res["test_flow_only"].items():
            full = res["test_by_dataset"][ds]
            for name, r in (("flow + packet", full), ("flow only", ab)):
                L.append(f"| {DATASET_NAMES.get(ds, ds)} | {name} | {_f(r['forecast']['world_model']['auroc'])} | "
                         f"{_f(r['forecast_from_benign']['world_model']['auroc'])} | {_f(r['detect']['world_model']['auroc'])} |")
    perf = res.get("performance", {})
    if perf:
        L += ["", "## Runtime", "", f"World-model inference: {perf['ms_per_window']:.2f} ms per window "
              f"({perf['mc_samples']} Monte-Carlo rollouts x K={K}) on {perf['device']}.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(cfg["artifacts"]["model"], device)
    lr = BaselineLR.load(cfg["artifacts"]["baseline"])
    scaler = StateScaler.from_dict(ck["scaler"])
    d = cfg["data"]
    out_dir = cfg["artifacts"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    drop = (np.array(ck["packet_missing"]["idx"]), np.array(ck["packet_missing"]["values"], dtype=np.float32))

    test_parts: Dict[str, List] = {}
    unknown_parts: Dict[str, List] = {}
    flow_only: Dict[str, List] = {}
    plotted = set()
    for key, df in load_sources(cfg, "test").items():
        ds = str(df["dataset"].iloc[0])
        r = evaluate_part(model, lr, df, scaler, cfg, device)
        test_parts.setdefault(ds, []).append(r)
        if model.n_sites:
            unknown_parts.setdefault(ds, []).append(evaluate_part(model, lr, df, scaler, cfg, device, unknown_site=True))
        if df["pk_mask"].sum() > 0:
            flow_only.setdefault(ds, []).append(evaluate_part(model, lr, df, scaler, cfg, device, drop_packets=drop))
        name = key.split("__")[1]
        if ds == "ctu13" or (ds not in plotted and ds not in NON_TEMPORAL and r["y"].sum() > 0):
            plotted.add(ds)
            fn = f"timeline_{name}.png" if ds == "ctu13" else f"timeline_{ds}.png"
            label = (f"CTU-13 capture {name[1:]} ({FAMILY.get(int(name[1:]), '')}, unseen family)" if ds == "ctu13"
                     else f"{DATASET_NAMES.get(ds, ds)} - {name} (unseen in training)")
            plot_timeline(os.path.join(out_dir, fn), f"{label} - forecast of malicious activity in the next "
                          f"{cfg['model']['horizon']} windows", df, r, ck, d["window_s"])

    val_parts: Dict[str, List] = {}
    for key, df in load_sources(cfg, "train").items():
        keep = val_mask(len(df), d["val_block"], d["val_every"])
        if not keep.any():
            continue
        r = evaluate_part(model, lr, df, scaler, cfg, device, keep=keep)
        if len(r["y"]):
            val_parts.setdefault(str(df["dataset"].iloc[0]), []).append(r)

    by_ds = {ds: summarize(concat(p), ck, lr, cfg, ds) for ds, p in test_parts.items()}
    temporal = [ds for ds in by_ds if ds not in NON_TEMPORAL]
    res = {
        "test_by_dataset": by_ds,
        "test_macro": macro(by_ds, temporal),
        "test": summarize(concat([r for ds in temporal for r in test_parts[ds]]), ck, lr, cfg),
        "test_all": summarize(concat([r for p in test_parts.values() for r in p]), ck, lr, cfg),
        "validation_by_dataset": {ds: summarize(concat(p), ck, lr, cfg, ds) for ds, p in val_parts.items()},
        "validation": summarize(concat([r for p in val_parts.values() for r in p]), ck, lr, cfg),
        "test_flow_only": {ds: summarize(concat(p), ck, lr, cfg, ds) for ds, p in flow_only.items()},
        "test_unknown_site_by_dataset": {ds: summarize(concat(p), ck, lr, cfg, ds) for ds, p in unknown_parts.items()},
        "config": cfg,
    }
    x = torch.randn(512, cfg["model"]["context"], len(ck["features"]), device=device)
    t0 = time.time()
    model.forecast(x, cfg["model"]["horizon"], cfg["eval"]["mc_samples"])
    if res["test_unknown_site_by_dataset"]:
        res["test_unknown_site_macro"] = macro(res["test_unknown_site_by_dataset"], temporal)
    res["performance"] = {"ms_per_window": (time.time() - t0) * 1000 / 512, "device": str(device),
                          "mc_samples": cfg["eval"]["mc_samples"]}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    write_markdown(os.path.join(out_dir, "RESULTS.md"), res, cfg)
    print(open(os.path.join(out_dir, "RESULTS.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()

"""Train the network world model and the logistic-regression baseline on every dataset.

    python scripts/prepare_data.py      # CTU-13 raw -> datasets/processed/s*.pkl
    python scripts/prepare_all.py       # CIC-IDS2017/2018, UNSW-NB15, LANL, DARPA 1999, CICIoT2023
    python train.py                     # -> models/world_model.pt, models/baseline_lr.joblib
    python evaluate.py                  # -> results/metrics.json, results/RESULTS.md

Everything is driven by configs/default.yaml and seeded for reproducibility. Each
epoch draws the same number of training sequences from every dataset, so large
datasets (DARPA, CIC) cannot drown out small ones (LANL, CICIoT).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.data import stages as st
from src.data.packet_features import PACKET_FEATURES
from src.data.sequences import StateScaler, context_windows, contiguous_segments, future_targets, state_matrix, val_mask
from src.data.sources import SITES, load_sources, site_index
from src.data.state_builder import FEATURE_LABELS, STATE_FEATURES
from src.evaluation.metrics import select_threshold
from src.integrity import write_checksums
from src.models.baseline_lr import BaselineLR
from src.models.world_model import CyberWorldModel

LOGGER = logging.getLogger("train")

# log1p-scaled counts that grow with the size of the monitored network.
VOLUME_FEATURES = ("fl_count_log", "fl_bytes_sum_log", "fl_pkts_sum_log", "fl_src_ips_log", "fl_dst_ips_log",
                   "fl_dst_ports_log", "fl_host_max_flows_log", "fl_host_max_bytes_log", "fl_host_max_dns_log",
                   "fl_max_dports_per_src_log", "fl_max_dsts_per_src_log", "pk_count_log",
                   "pk_scan_max_ports_log", "pk_scan_max_hosts_log")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def missing_packet_vector(scaler: StateScaler) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled values the packet block takes when no capture is available."""
    idx = np.array([STATE_FEATURES.index(f) for f in PACKET_FEATURES] + [STATE_FEATURES.index("pk_mask")])
    raw_full = scaler.mean.copy()
    raw_full[idx] = 0.0
    return idx, scaler.transform(raw_full[None])[0][idx]


@torch.no_grad()
def val_scores(model, val: List[Dict], context: int, horizon: int, device, bs: int = 1024):
    """Per-dataset (nowcast, forecast) labels and scores on validation windows (full-history context)."""
    per: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for v in val:
        m = v["mask"]
        if not m.any():
            continue
        ctx = context_windows(v["x"], context)[m]
        y_any, valid = future_targets(v["y"], horizon)
        now, fut = [], []
        for i in range(0, len(ctx), bs):
            c = torch.as_tensor(ctx[i:i + bs], device=device)
            site = torch.full((len(c),), v.get("site", -1), dtype=torch.long, device=device)
            now.append(torch.sigmoid(model.observe(c, sample=False, site=site)["attack_logit"][:, -1]).cpu().numpy())
            fut.append(model.forecast_score(c, horizon, site=site).cpu().numpy())
        d = per.setdefault(v["dataset"], {"yn": [], "pn": [], "yf": [], "pf": []})
        d["yn"].append(v["y"][m]); d["pn"].append(np.concatenate(now))
        keep = valid[m]
        d["yf"].append(y_any[m][keep]); d["pf"].append(np.concatenate(fut)[keep])
    return {k: {kk: np.concatenate(vv) for kk, vv in d.items()} for k, d in per.items()}


def macro_auroc(scores) -> Tuple[float, Dict[str, float]]:
    per = {}
    for ds, d in scores.items():
        vals = [roc_auc_score(d[y], d[p]) for y, p in (("yn", "pn"), ("yf", "pf")) if len(np.unique(d[y])) > 1]
        if vals:
            per[ds] = float(np.mean(vals))
    return (float(np.mean(list(per.values()))) if per else 0.0), per


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--recalibrate", action="store_true",
                    help="keep the trained weights; only recompute thresholds and refit the LR baseline")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc, tc, dc = cfg["model"], cfg["train"], cfg["data"]
    context, horizon = mc["context"], mc["horizon"]
    T = context + horizon

    # ---------------------------------------------------------------- data
    src = load_sources(cfg, "train")
    n_sites = len(SITES) if mc.get("site_conditioning") else 0
    masks = {k: val_mask(len(df), dc["val_block"], dc["val_every"]) for k, df in src.items()}
    x_train_raw = np.concatenate([state_matrix(df[~masks[k]]) for k, df in src.items()])
    scaler = StateScaler().fit(x_train_raw)

    arrays, index_by_ds, val = [], {}, []
    for j, (k, df) in enumerate(src.items()):
        xs = scaler.transform(state_matrix(df))
        y = df["is_attack"].to_numpy().astype(np.int64)
        s = df["stage"].to_numpy().astype(np.int64)
        ds = str(df["dataset"].iloc[0])
        arrays.append((xs, y, s, site_index(ds)))
        for a, b in contiguous_segments(~masks[k]):       # sequences never touch validation blocks
            starts = np.arange(a, b - T + 1)
            if len(starts):
                index_by_ds.setdefault(ds, []).append(np.stack([np.full(len(starts), j), starts], 1))
        val.append({"x": xs, "y": y, "mask": masks[k], "dataset": ds, "site": site_index(ds) if n_sites else -1})
    index_by_ds = {k: np.concatenate(v) for k, v in index_by_ds.items()}
    n_ds = len(index_by_ds)
    per_epoch = int(tc.get("samples_per_epoch", 12000))
    LOGGER.info("train windows=%d sources=%d datasets=%s | sequences per dataset=%s | device=%s",
                len(x_train_raw), len(src), n_ds, {k: len(v) for k, v in index_by_ds.items()}, device)

    y_all = np.concatenate([df["is_attack"].to_numpy()[~masks[k]] for k, df in src.items()])
    s_all = np.concatenate([df["stage"].to_numpy()[~masks[k]] for k, df in src.items()])
    pos_weight = torch.tensor(float(np.clip((y_all == 0).sum() / max((y_all == 1).sum(), 1), 0.2, 20.0)), device=device)
    freq = np.bincount(s_all, minlength=st.N_STAGES).astype(float)
    stage_w = np.where(freq > 0, 1.0 / np.sqrt(np.maximum(freq, 1)), 0.0)
    stage_w = torch.tensor(stage_w / stage_w[freq > 0].mean(), dtype=torch.float32, device=device)
    LOGGER.info("attack prevalence=%.3f pos_weight=%.2f stage counts=%s", y_all.mean(), pos_weight.item(),
                dict(zip(st.STAGE_KEYS, freq.astype(int).tolist())))

    pk_idx, pk_missing = missing_packet_vector(scaler)
    pk_idx_t = torch.as_tensor(pk_idx, device=device)
    pk_missing_t = torch.as_tensor(pk_missing, device=device)
    vol = torch.zeros(len(STATE_FEATURES), device=device)
    for name in VOLUME_FEATURES:
        i = STATE_FEATURES.index(name)
        vol[i] = 1.0 / float(scaler.std[i])

    def batch(rows: np.ndarray):
        xb = np.stack([arrays[j][0][s:s + T] for j, s in rows])
        yb = np.stack([arrays[j][1][s:s + T] for j, s in rows])
        sb = np.stack([arrays[j][2][s:s + T] for j, s in rows])
        site = np.array([arrays[j][3] for j, _ in rows], dtype=np.int64)
        return (torch.as_tensor(xb, device=device), torch.as_tensor(yb, device=device), torch.as_tensor(sb, device=device),
                torch.as_tensor(site, device=device))

    # --------------------------------------------------------------- model
    model = CyberWorldModel(len(STATE_FEATURES), mc["latent_dim"], mc["hidden_dim"], st.N_STAGES,
                            mc["dropout"], mc.get("obs_skip", False), n_sites).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    best, best_state, bad, history = -1.0, None, 0, []
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    prev_train_time = 0.0
    if args.recalibrate:
        prev = torch.load(cfg["artifacts"]["model"], map_location=device, weights_only=False)
        best_state, history, prev_train_time = prev["state_dict"], prev["history"], prev.get("train_seconds", 0.0)
    for epoch in range(1, (0 if args.recalibrate else tc["epochs"]) + 1):
        model.train()
        take = per_epoch // n_ds
        rows = np.concatenate([v[rng.integers(0, len(v), take)] for v in index_by_ds.values()])
        rows = rows[rng.permutation(len(rows))]
        agg: Dict[str, float] = {}
        for i in range(0, len(rows), tc["batch_size"]):
            xb, yb, sb, site = batch(rows[i:i + tc["batch_size"]])
            if n_sites and mc.get("site_dropout", 0) > 0:   # teach the unknown-network mode too
                site = torch.where(torch.rand(len(site), device=device) < mc["site_dropout"], torch.full_like(site, -1), site)
            if tc.get("volume_aug", 0) > 0:
                shift = (torch.rand(len(xb), 1, 1, device=device) * 2 - 1) * tc["volume_aug"]
                xb = xb + shift * vol
            drop = torch.rand(len(xb), device=device) < tc["modality_dropout"]
            if drop.any():  # train to cope with CSV-only inputs (no packet capture)
                xb = xb.clone()
                sub = xb[drop]
                sub[:, :, pk_idx_t] = pk_missing_t
                xb[drop] = sub
            loss, parts = model.compute_loss(xb, yb, sb, horizon, tc, pos_weight, stage_w,
                                             site=site if n_sites else None)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v * len(xb)
        sched.step()
        model.eval()
        score, per = macro_auroc(val_scores(model, val, context, horizon, device))
        row = {k: v / len(rows) for k, v in agg.items()} | {"epoch": epoch, "val_macro_auroc": score, "val_per_dataset": per}
        history.append(row)
        LOGGER.info("epoch %02d loss=%.3f bce_now=%.3f bce_fut=%.3f | val macro AUROC=%.3f %s", epoch, row["loss"],
                    row["bce_now"], row["bce_fut"], score, {k: round(v, 3) for k, v in per.items()})
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                LOGGER.info("early stop at epoch %d (best val macro AUROC %.3f)", epoch, best)
                break
    model.load_state_dict(best_state)
    model.eval()
    train_time = prev_train_time if args.recalibrate else time.time() - t0

    # ------------------------------------------ thresholds (validation only)
    rule = cfg["eval"].get("threshold_rule", "youden")
    vs = val_scores(model, val, context, horizon, device)
    thr_now = select_threshold(np.concatenate([d["yn"] for d in vs.values()]), np.concatenate([d["pn"] for d in vs.values()]), rule)
    thr_fut = select_threshold(np.concatenate([d["yf"] for d in vs.values()]), np.concatenate([d["pf"] for d in vs.values()]), rule)
    # Per-network calibration: each dataset's own validation split picks its operating point
    # (the global one is the fallback for unknown networks).
    per_thr = {}
    for ds_name, dv in vs.items():
        if len(np.unique(dv["yn"])) > 1 and len(np.unique(dv["yf"])) > 1:
            per_thr[ds_name] = {"nowcast": select_threshold(dv["yn"], dv["pn"], rule),
                                "forecast": select_threshold(dv["yf"], dv["pf"], rule)}

    xs_train = scaler.transform(x_train_raw)
    benign_ref = xs_train[y_all == 0].mean(0)
    os.makedirs(os.path.dirname(cfg["artifacts"]["model"]), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "config": cfg, "features": list(STATE_FEATURES),
        "feature_labels": FEATURE_LABELS, "scaler": scaler.to_dict(),
        "thresholds": {"nowcast": thr_now, "forecast": thr_fut}, "thresholds_by_dataset": per_thr,
        "benign_reference": benign_ref.tolist(),
        "packet_missing": {"idx": pk_idx.tolist(), "values": pk_missing.tolist()},
        "stages": [s.__dict__ for s in st.STAGES], "history": history, "train_seconds": train_time,
        "datasets": sorted(index_by_ds), "n_sites": n_sites, "sites": SITES if n_sites else [],
    }, cfg["artifacts"]["model"])
    LOGGER.info("saved %s (thresholds now=%.3f forecast=%.3f, %.0fs)", cfg["artifacts"]["model"], thr_now, thr_fut, train_time)

    # ------------------------------------------------------- LR baseline
    # Same windows, same features, same targets as the world model - just no temporal model.
    y_fut_all, valid_all = [], []
    for k, df in src.items():
        ya, va = future_targets(df["is_attack"].to_numpy(), horizon)
        y_fut_all.append(ya[~masks[k]]); valid_all.append(va[~masks[k]])
    y_fut_all, valid_all = np.concatenate(y_fut_all), np.concatenate(valid_all)
    lr = BaselineLR(random_state=cfg["seed"])
    lr.models["detect"].fit(xs_train, y_all)
    lr.models["forecast"].fit(xs_train[valid_all], y_fut_all[valid_all])
    xv = np.concatenate([v["x"][v["mask"]] for v in val])
    yv = np.concatenate([v["y"][v["mask"]] for v in val])
    lr.thresholds["detect"] = select_threshold(yv, lr.predict_proba(xv, "detect"), rule)
    yf, vf = [], []
    for v in val:
        a, b = future_targets(v["y"], horizon)
        yf.append(a[v["mask"]]); vf.append(b[v["mask"]])
    yf, vf = np.concatenate(yf), np.concatenate(vf)
    lr.thresholds["forecast"] = select_threshold(yf[vf], lr.predict_proba(xv[vf], "forecast"), rule)
    lr.thresholds_by_dataset = {}
    for v in val:
        m = v["mask"]
        if not m.any():
            continue
        a, b = future_targets(v["y"], horizon)
        e = lr.thresholds_by_dataset.setdefault(v["dataset"], {"x": [], "y": [], "yf": [], "vf": []})
        e["x"].append(v["x"][m]); e["y"].append(v["y"][m]); e["yf"].append(a[m]); e["vf"].append(b[m])
    for ds_name, e in list(lr.thresholds_by_dataset.items()):
        x_, y_, yf_, vf_ = (np.concatenate(e[k]) for k in ("x", "y", "yf", "vf"))
        if len(np.unique(y_)) > 1 and len(np.unique(yf_[vf_])) > 1:
            lr.thresholds_by_dataset[ds_name] = {"detect": select_threshold(y_, lr.predict_proba(x_, "detect"), rule),
                                                 "forecast": select_threshold(yf_[vf_], lr.predict_proba(x_[vf_], "forecast"), rule)}
        else:
            del lr.thresholds_by_dataset[ds_name]
    lr.save(cfg["artifacts"]["baseline"])
    LOGGER.info("saved %s (thresholds %s)", cfg["artifacts"]["baseline"], lr.thresholds)
    sums = os.path.join(os.path.dirname(cfg["artifacts"]["model"]) or ".", "SHA256SUMS")
    write_checksums([cfg["artifacts"]["model"], cfg["artifacts"]["baseline"]], sums)
    LOGGER.info("wrote %s", sums)
    os.makedirs(cfg["artifacts"]["results_dir"], exist_ok=True)
    with open(os.path.join(cfg["artifacts"]["results_dir"], "training_history.json"), "w") as fh:
        json.dump(history, fh, indent=1)


if __name__ == "__main__":
    main()

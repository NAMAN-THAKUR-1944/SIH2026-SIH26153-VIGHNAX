"""Offline inference: CSV / binetflow and/or PCAP in -> forecasts, stages, explanations out.

Runs entirely locally (no network access, no cloud APIs).

    engine = InferenceEngine("models/world_model.pt", "models/baseline_lr.joblib")
    analysis = engine.analyze(flow_path="capture.binetflow", pcap_path="capture.pcap")
    detail = engine.explain_window(analysis_id, window_index)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.data import stages as st
from src.data.alignment import best_offset
from src.data.datasets import detect_flow_format, load_flows_any
from src.data.packet_features import extract_packet_windows
from src.data.sequences import StateScaler, context_windows, state_matrix
from src.data.state_builder import FLOW_FEATURES, STATE_FEATURES, build_states, flow_window_features
from src.evaluation.explainer import ForecastExplainer
from src.models.baseline_lr import BaselineLR
from src.models.world_model import CyberWorldModel

LOGGER = logging.getLogger(__name__)

PCAP_EXT = (".pcap", ".pcapng", ".cap", ".pcap.bz2", ".pcap.gz", ".pcapng.gz", ".pcapng.bz2")


def is_pcap(path: str) -> bool:
    return path.lower().endswith(PCAP_EXT)


class WindowNotReady(Exception):
    """An explanation was requested for a window the stream has not processed yet."""


class InferenceEngine:
    def __init__(self, model_path: str, baseline_path: Optional[str] = None, device: str = "cpu",
                 torch_threads: Optional[int] = 1) -> None:
        # The model is small: per-window work is dominated by thread-pool overhead, so one intra-op thread is
        # faster than many (~6 vs ~8 ms per forecast, ~2x faster explanations) and leaves cores for the server.
        if torch_threads:
            torch.set_num_threads(torch_threads)
        ck = torch.load(model_path, map_location=device, weights_only=False)
        self.cfg = ck["config"]
        mc = self.cfg["model"]
        self.model = CyberWorldModel(len(ck["features"]), mc["latent_dim"], mc["hidden_dim"], st.N_STAGES, mc["dropout"],
                                     mc.get("obs_skip", False), ck.get("n_sites", 0))
        self.sites: List[str] = ck.get("sites", [])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.features: List[str] = ck["features"]
        self.datasets: List[str] = ck.get("datasets", ["ctu13"])
        assert self.features == list(STATE_FEATURES), "checkpoint feature schema differs from code"
        self.labels: Dict[str, str] = ck["feature_labels"]
        self.scaler = StateScaler.from_dict(ck["scaler"])
        self.thresholds = ck["thresholds"]
        self.thresholds_by_dataset: Dict[str, Dict[str, float]] = ck.get("thresholds_by_dataset", {})
        self.context, self.horizon = mc["context"], mc["horizon"]
        self.window_s = float(self.cfg["data"]["window_s"])
        self.mc_samples = int(self.cfg["eval"]["mc_samples"])
        self.explainer = ForecastExplainer(self.model, self.horizon, np.asarray(ck["benign_reference"]), steps=32)
        self.baseline = BaselineLR.load(baseline_path) if baseline_path and os.path.isfile(baseline_path) else None
        self._store: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        # Forecasts and explanations take turns (they would otherwise compete for the interpreter lock), so the
        # compute time reported per window is the forecast's own cost even while explanations run alongside.
        self._compute = threading.Lock()

    # ------------------------------------------------------------------ input
    def load(self, flow_path: Optional[str] = None, pcap_path: Optional[str] = None,
             time_offset_s: Optional[float] = None) -> Dict:
        if not flow_path and not pcap_path:
            raise ValueError("Provide a flow CSV/binetflow, a PCAP, or both")
        t0 = time.time()
        flows, pk, pcap_flows, meta = None, None, None, {}
        if flow_path:
            flows = load_flows_any(flow_path)  # CTU / CIC-IDS2017 / CIC-IDS2018 / UNSW-NB15 / LANL / own schema
            meta["flows"] = int(len(flows))
            meta["flow_format"] = detect_flow_format(flow_path)
        if pcap_path:
            pk, pcap_flows = extract_packet_windows(pcap_path, window_s=self.window_s, with_flows=flows is None)
            meta["packet_windows"] = int(len(pk))
            if flows is not None and len(pk):
                if time_offset_s is None:
                    time_offset_s = best_offset(flows, pk, self.window_s)
                pk["window_id"] = pk["window_id"] + int(time_offset_s // self.window_s)
                meta["pcap_time_offset_s"] = float(time_offset_s)
            if flows is None:
                flows = pcap_flows
                meta["flows"] = int(len(flows))
                meta["flows_source"] = "rebuilt from PCAP"
                meta["warning"] = ("Flow records were rebuilt from the packets. The model has seen rebuilt flows only in its "
                                   "DARPA 1999 and CICIoT2023 training data; for networks with a flow exporter, supplying its "
                                   "flow records together with the PCAP gives more reliable forecasts.")
        meta["parse_seconds"] = round(time.time() - t0, 2)
        return {"flows": flows, "packets": pk, "meta": meta}

    # --------------------------------------------------------------- analysis
    def analyze(self, flow_path: Optional[str] = None, pcap_path: Optional[str] = None,
                time_offset_s: Optional[float] = None, name: str = "upload") -> Dict:
        data = self.load(flow_path, pcap_path, time_offset_s)
        t0 = time.time()
        flows, pk = data["flows"], data["packets"]
        has_labels = flows is not None and "label" in flows and flows["label"].notna().any()
        states = build_states(flows, pk, self.window_s, with_labels=has_labels)
        x = self.scaler.transform(state_matrix(states))
        ctx = context_windows(x, self.context)
        torch.manual_seed(0)
        outs: Dict[str, List[np.ndarray]] = {}
        with torch.no_grad():
            for i in range(0, len(ctx), 256):
                fc = self.model.forecast(torch.as_tensor(ctx[i:i + 256]), self.horizon, self.mc_samples)
                for k, v in fc.items():
                    outs.setdefault(k, []).append(v.numpy())
        fc = {k: np.concatenate(v) for k, v in outs.items()}

        # Most likely *malicious* stage over the horizon (benign excluded), per window.
        fut = fc["stage_future"].max(axis=1)                       # (N, S) peak prob per stage over K
        mal = fut[:, 1:]
        stage_pred = mal.argmax(1) + 1
        thr_f, thr_n = self.thresholds["forecast"], self.thresholds["nowcast"]
        alerts = self._incidents(fc["p_any"], fc["nowcast"], thr_f, thr_n)

        result = {
            "id": uuid.uuid4().hex[:12], "name": name, "window_s": self.window_s,
            "horizon": self.horizon, "context": self.context,
            "thresholds": {"forecast": thr_f, "nowcast": thr_n},
            "meta": data["meta"] | {"windows": int(len(states)), "has_labels": bool(has_labels),
                                    "has_packets": bool(pk is not None and len(pk)),
                                    "infer_seconds": round(time.time() - t0, 2)},
            "t0": float(states["window_start"].iloc[0]),
            "nowcast": fc["nowcast"].round(4).tolist(),
            "p_any": fc["p_any"].round(4).tolist(),
            "p_step": fc["p_step"].round(4).tolist(),
            "p_lo": fc["p_lo"].round(4).tolist(), "p_hi": fc["p_hi"].round(4).tolist(),
            "p_any_lo": fc["p_any_lo"].round(4).tolist(), "p_any_hi": fc["p_any_hi"].round(4).tolist(),
            "surprise": fc["surprise"].round(3).tolist(),
            "stage_now": fc["stage_now"].round(4).tolist(),
            "stage_future": fc["stage_future"].round(4).tolist(),
            "stage_pred": stage_pred.tolist(),
            "stages": [st.describe(i) | {"idx": i} for i in range(st.N_STAGES)],
            "incidents": alerts,
            "flow_count": np.expm1(states["fl_count_log"].to_numpy()).round().astype(int).tolist(),
            "packet_count": np.expm1(states["pk_count_log"].to_numpy()).round().astype(int).tolist(),
        }
        if self.baseline is not None:
            result["lr_any"] = self.baseline.predict_proba(x, "forecast").round(4).tolist()
            result["lr_threshold"] = self.baseline.thresholds["forecast"]
        if has_labels:
            result["truth_attack"] = states["is_attack"].astype(int).tolist()
            result["truth_stage"] = states["stage"].astype(int).tolist()
        with self._lock:
            self._store = {result["id"]: {"x": x, "states": states, "flows": flows, "result": result}}
        return result

    def analyze_stream(self, flow_path: Optional[str] = None, pcap_path: Optional[str] = None,
                       time_offset_s: Optional[float] = None, name: str = "upload", speed: float = 60.0,
                       profile: Optional[str] = None):
        """Generator version of :meth:`analyze` for the live dashboard.

        Telemetry is parsed first; then the world model processes the windows
        one at a time in time order and each window's forecast is emitted as
        soon as it is computed. Emission is paced to ``speed`` x real time
        (1 = one 10 s window every 10 s, like a live sensor; 0 = no pacing).
        The reported compute time excludes the pacing.

        The analysis is registered before the first window is emitted, so
        :meth:`explain_window` can explain every window already processed while
        the stream is still running (never a window it has not reached yet).
        """
        kinds = " + ".join(k for k, v in (("flow records", flow_path), ("packet capture", pcap_path)) if v)
        yield {"type": "status", "msg": f"Parsing {kinds}…"}
        data = self.load(flow_path, pcap_path, time_offset_s)
        # Site profile -> operating thresholds calibrated on that network's validation data.
        fmt_profile = {"ctu": "ctu13", "cic2017": "cicids2017", "cic": "cicids2018", "unsw": "unswnb15", "lanl": "lanl"}
        profile = profile or fmt_profile.get(data["meta"].get("flow_format", ""))
        thr = self.thresholds_by_dataset.get(profile, self.thresholds)
        lr_thr = (self.baseline.thresholds_by_dataset.get(profile, self.baseline.thresholds)
                  if self.baseline is not None else None)
        data["meta"]["profile"] = profile if profile in self.thresholds_by_dataset else "global"
        site_id = self.sites.index(profile) if profile in self.sites else -1
        site_t = torch.tensor([site_id]) if self.model.n_sites else None
        flows, pk = data["flows"], data["packets"]
        has_labels = flows is not None and "label" in flows and flows["label"].notna().any()
        yield {"type": "status", "msg": "Building network states S_t (10 s windows)…"}
        states = build_states(flows, pk, self.window_s, with_labels=has_labels)
        x = self.scaler.transform(state_matrix(states))
        ctx = context_windows(x, self.context)
        n = len(ctx)
        aid = uuid.uuid4().hex[:12]
        head = {
            "type": "meta", "id": aid, "name": name, "n": n, "window_s": self.window_s, "horizon": self.horizon,
            "context": self.context, "thresholds": dict(thr), "t0": float(states["window_start"].iloc[0]),
            "stages": [st.describe(i) | {"idx": i} for i in range(st.N_STAGES)],
            "meta": data["meta"] | {"windows": int(n), "has_labels": bool(has_labels),
                                    "has_packets": bool(pk is not None and len(pk))},
            "flow_count": np.expm1(states["fl_count_log"].to_numpy()).round().astype(int).tolist(),
            "packet_count": np.expm1(states["pk_count_log"].to_numpy()).round().astype(int).tolist(),
        }
        lr_any = self.baseline.predict_proba(x, "forecast") if self.baseline is not None else None
        if lr_any is not None:
            head["lr_threshold"] = lr_thr["forecast"]
        if has_labels:
            head["truth_attack"] = states["is_attack"].astype(int).tolist()
            head["truth_stage"] = states["stage"].astype(int).tolist()
        entry = {"x": x, "states": states, "flows": flows, "result": None, "site": site_id, "processed": 0}
        with self._lock:
            self._store = {aid: entry}
        yield head

        pace = self.window_s / speed if speed and speed > 0 else 0.0
        torch.manual_seed(0)
        cols: Dict[str, List] = {}
        compute = 0.0
        t_start = time.perf_counter()
        for i in range(n):
            with self._compute:
                t0 = time.perf_counter()
                with torch.no_grad():
                    fc = self.model.forecast(torch.as_tensor(ctx[i:i + 1]), self.horizon, self.mc_samples, site=site_t)
                fc = {k: v[0].numpy() for k, v in fc.items()}
                stage_pred = int(fc["stage_future"].max(axis=0)[1:].argmax() + 1)
                compute += time.perf_counter() - t0
            ev = {"type": "window", "i": i, "nowcast": round(float(fc["nowcast"]), 4),
                  "p_any": round(float(fc["p_any"]), 4), "p_any_lo": round(float(fc["p_any_lo"]), 4),
                  "p_any_hi": round(float(fc["p_any_hi"]), 4), "surprise": round(float(fc["surprise"]), 3),
                  "p_step": fc["p_step"].round(4).tolist(), "p_lo": fc["p_lo"].round(4).tolist(),
                  "p_hi": fc["p_hi"].round(4).tolist(), "stage_now": fc["stage_now"].round(4).tolist(),
                  "stage_future": fc["stage_future"].round(4).tolist(), "stage_pred": stage_pred}
            if lr_any is not None:
                ev["lr_any"] = round(float(lr_any[i]), 4)
            for k, v in ev.items():
                if k not in ("type", "i"):
                    cols.setdefault(k, []).append(v)
            entry["processed"] = i + 1
            yield ev
            # pace against a fixed schedule, so a short wait for an explanation does not accumulate as drift
            time.sleep(max(0.0, t_start + (i + 1) * pace - time.perf_counter()))

        incidents = self._incidents(np.asarray(cols["p_any"]), np.asarray(cols["nowcast"]),
                                    thr["forecast"], thr["nowcast"])
        result = {k: v for k, v in head.items() if k != "type"} | cols | {"incidents": incidents}
        result["meta"]["infer_seconds"] = round(compute, 2)
        with self._lock:
            entry["result"] = result
        yield {"type": "done", "id": aid, "incidents": incidents, "infer_seconds": round(compute, 2),
               "ms_per_window": round(1000 * compute / max(n, 1), 2), "speed": speed}

    @staticmethod
    def _incidents(p_any: np.ndarray, now: np.ndarray, thr_f: float, thr_n: float, gap: int = 3) -> List[Dict]:
        """Group consecutive alerting windows into incidents (first warning / peak / end)."""
        alert = (p_any >= thr_f) | (now >= thr_n)
        out, start, last = [], None, None
        for t, a in enumerate(alert):
            if a:
                if start is None:
                    start = t
                last = t
            elif start is not None and t - last > gap:
                out.append((start, last)); start = None
        if start is not None:
            out.append((start, last))
        return [{"start": int(s), "end": int(e), "peak": int(s + np.argmax(p_any[s:e + 1])),
                 "max_p": float(p_any[s:e + 1].max())} for s, e in out]

    # ------------------------------------------------------------ drill-down
    def explain_window(self, analysis_id: str, idx: int, top_hosts: int = 10) -> Dict:
        with self._lock:
            entry = self._store.get(analysis_id)
        if entry is None:
            raise KeyError("Unknown or expired analysis id - run the analysis again")
        x, states, flows = entry["x"], entry["states"], entry["flows"]
        site_id = entry.get("site", -1)
        idx = int(np.clip(idx, 0, len(x) - 1))
        if idx >= entry.get("processed", len(x)):
            raise WindowNotReady(f"window {idx} has not been processed yet")
        ctx = context_windows(x, self.context)[idx]
        with self._compute:
            ex = self.explainer.explain(ctx, self.features, site=site_id)
            flagged = self._flagged_flows(x, states, flows, idx, ctx, ex["score"], top_hosts, site_id)
        feats = []
        for name, val in zip(ex["features"][:12], ex["attribution"][:12]):
            feats.append({"feature": name, "label": self.labels.get(name, name), "value": val,
                          "level": "packet" if name.startswith("pk_") else "flow",
                          "raw": float(states[name].iloc[idx])})
        return {"index": idx, "score": ex["score"], "baseline_score": ex["baseline_score"],
                "features": feats, "time_attribution": ex["time_attribution"],
                "flagged": flagged}

    def _flagged_flows(self, x, states, flows, idx, ctx, score, top_hosts, site_id: int = -1) -> Dict:
        """Leave-one-host-out occlusion: which hosts' flows drive the forecast in window idx."""
        if flows is None or not len(flows):
            return {"hosts": [], "flows": []}
        wid = int(states.index[idx])
        f = flows[(flows["start_time"] // self.window_s).astype("int64") == wid]
        if f.empty:
            return {"hosts": [], "flows": []}
        # Candidate hosts: the top few sources along each behaviour the state encodes
        # (volume alone would miss a quiet spam bot among busy benign servers).
        src = f["src_ip"].astype(str)
        dport = pd.to_numeric(f["dst_port"], errors="coerce").fillna(-1)
        proto = f["protocol"].astype(str).str.upper()
        failed = (pd.to_numeric(f.get("flag_ack"), errors="coerce").fillna(0) == 0) & (proto == "TCP")
        per = pd.DataFrame({
            "flows": src.value_counts(),
            "smtp": src[dport.isin([25, 465, 587])].value_counts(),
            "failed": src[failed].value_counts(),
            "ports": f.groupby(src)["dst_port"].nunique(),
            "hosts": f.groupby(src)["dst_ip"].nunique(),
            "icmp": src[proto.str.startswith("ICMP")].value_counts(),
            "dns": src[dport == 53].value_counts(),
            "bytes": pd.to_numeric(f["byte_count"], errors="coerce").groupby(src).sum(),
        }).fillna(0)
        cand = []
        for col in per.columns:
            for h in per[per[col] > 0][col].nlargest(3).index:
                if h not in cand:
                    cand.append(h)
        cand = cand[: max(top_hosts, 24)]
        flow_idx = [self.features.index(c) for c in FLOW_FEATURES]
        rows, ctxs = [], []
        for host in cand:
            rest = f[f["src_ip"].astype(str) != host]
            raw = state_matrix(states.iloc[[idx]]).copy()
            if len(rest):
                fw = flow_window_features(rest, self.window_s)
                raw[0, flow_idx] = fw.iloc[0][list(FLOW_FEATURES)].to_numpy()
            else:
                raw[0, flow_idx] = 0.0
            c = ctx.copy()
            c[-1] = self.scaler.transform(raw)[0]
            ctxs.append(c); rows.append(host)
        with torch.no_grad():
            site_t = torch.full((len(ctxs),), site_id, dtype=torch.long) if self.model.n_sites else None
            scores = self.model.forecast_score(torch.as_tensor(np.stack(ctxs)), self.horizon, site=site_t).numpy()
        hosts = sorted(({"host": h, "contribution": float(score - s),
                         "flows": int((f["src_ip"].astype(str) == h).sum())} for h, s in zip(rows, scores)),
                       key=lambda d: -d["contribution"])
        best = hosts[0]["contribution"]
        top = [h["host"] for h in hosts if h["contribution"] >= max(0.02, 0.2 * best)][:3] or [hosts[0]["host"]]
        cols = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "packet_count", "byte_count", "duration", "label"]
        sel = f[f["src_ip"].astype(str).isin(top)].sort_values("byte_count", ascending=False).head(25)
        flow_rows = []
        for r in sel[[c for c in cols if c in sel]].itertuples(index=False):
            d = r._asdict()
            lab = d.get("label")
            d["label"] = "" if lab is None or (isinstance(lab, float) and np.isnan(lab)) else str(lab)
            for k in ("src_port", "dst_port", "packet_count", "byte_count"):
                if k in d and d[k] is not None and not (isinstance(d[k], float) and np.isnan(d[k])):
                    d[k] = int(d[k])
            if "duration" in d:
                d["duration"] = None if d["duration"] is None or np.isnan(d["duration"]) else round(float(d["duration"]), 3)
            flow_rows.append(d)
        return {"hosts": hosts, "flows": flow_rows}

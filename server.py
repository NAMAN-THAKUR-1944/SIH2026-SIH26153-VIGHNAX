"""VIGHNAX demo server - runs fully offline (no cloud APIs, no CDN, no web fonts).

    python server.py            ->  http://127.0.0.1:5000

Accepts a flow CSV (CIC-IDS / CTU-13 binetflow) and/or a PCAP (.pcap/.pcapng,
optionally .bz2/.gz), runs the trained world model, and serves the infiltration
probability timeline, K-step forecasts, MITRE stages, explanations and flagged
flows. Every number shown in the UI comes from the model or from
results/metrics.json - nothing is simulated.
"""

from __future__ import annotations

import bz2
import gzip
import json
import logging
import os
import shutil
import tempfile
import zipfile

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from src.config import load_config
from src.inference.engine import InferenceEngine, WindowNotReady, is_pcap
from src.integrity import verify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = load_config(os.path.join(ROOT, "configs", "default.yaml"))
SAMPLES_DIR = os.path.join(ROOT, "samples")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3  # 2 GB uploads

_engine = None
_integrity = None


def integrity() -> dict:
    """SHA-256 of the shipped weights checked against models/SHA256SUMS (computed once per server start)."""
    global _integrity
    if _integrity is None:
        paths = [os.path.join(ROOT, CFG["artifacts"][k]) for k in ("model", "baseline")]
        _integrity = verify(paths, os.path.join(os.path.dirname(paths[0]), "SHA256SUMS"))
    return _integrity


def engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine(os.path.join(ROOT, CFG["artifacts"]["model"]),
                                  os.path.join(ROOT, CFG["artifacts"]["baseline"]))
    return _engine


def _samples():
    path = os.path.join(SAMPLES_DIR, "samples.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    e = engine()
    return jsonify({"model": CFG["artifacts"]["model"], "window_s": e.window_s, "context": e.context,
                    "horizon": e.horizon, "thresholds": e.thresholds, "features": len(e.features),
                    "mc_samples": e.mc_samples, "samples": _samples(), "datasets": e.datasets,
                    "integrity": integrity()})


@app.route("/api/benchmark")
def benchmark():
    path = os.path.join(ROOT, CFG["artifacts"]["results_dir"], "metrics.json")
    if not os.path.isfile(path):
        return jsonify({"error": "results/metrics.json not found - run python evaluate.py"}), 404
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    keep = ("detect", "forecast", "forecast_from_benign", "lead_time", "forecast_per_horizon", "windows", "malicious_windows",
            "stage_now_malicious", "surprise_auroc_unsupervised")
    by_ds = {ds: {k: r[k] for k in keep if k in r} for ds, r in res.get("test_by_dataset", {}).items()}
    return jsonify({"test": {k: res["test"][k] for k in keep if k in res["test"]},
                    "validation": {k: res["validation"][k] for k in keep if k in res["validation"]},
                    "test_by_dataset": by_ds, "test_macro": res.get("test_macro", {}),
                    "test_scenarios": CFG["data"]["test_scenarios"],
                    "horizon": CFG["model"]["horizon"], "window_s": CFG["data"]["window_s"]})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    tmp = tempfile.mkdtemp(prefix="vighnax_")
    try:
        flow_path = pcap_path = None
        offset = None
        sample = request.form.get("sample")
        if sample:
            spec = next((s for s in _samples() if s["id"] == sample), None)
            if spec is None:
                return jsonify({"error": f"unknown sample {sample!r}"}), 400
            flow_path = os.path.join(SAMPLES_DIR, spec["flows"]) if spec.get("flows") else None
            pcap_path = os.path.join(SAMPLES_DIR, spec["pcap"]) if spec.get("pcap") else None
            offset = spec.get("time_offset_s")
            name = spec["title"]
        else:
            names = []
            for field in ("flows", "pcap"):
                f = request.files.get(field)
                if f and f.filename:
                    fn = secure_filename(f.filename) or field
                    dest = os.path.join(tmp, fn)
                    f.save(dest)
                    names.append(fn)
                    if is_pcap(fn):
                        pcap_path = dest
                    else:
                        flow_path = dest
            if not names:
                return jsonify({"error": "upload a flow CSV/binetflow and/or a PCAP"}), 400
            name = " + ".join(names)
            if request.form.get("time_offset_s"):
                offset = float(request.form["time_offset_s"])
        result = engine().analyze(flow_path=flow_path, pcap_path=pcap_path, time_offset_s=offset, name=name)
        return jsonify(result)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


PCAP_MAGIC = (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d", b"\x0a\x0d\x0d\x0a")


def sniff(path: str) -> str:
    """Classify an uploaded file by content, not name: 'zip', 'pcap' or 'flows'.

    Looks through gzip / bz2 compression. Returns the kind; the caller renames
    the file with matching extensions so the parsers pick the right decoder.
    """
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head[:2] == b"PK":
        return "zip"
    opener = bz2.open if head[:3] == b"BZh" else gzip.open if head[:2] == b"\x1f\x8b" else open
    try:
        with opener(path, "rb") as fh:
            inner = fh.read(4)
    except OSError:
        inner = head
    return "pcap" if inner in PCAP_MAGIC else "flows"


def _classify_uploads(paths, tmp):
    """Return (flow_path, pcap_path) from any mix of files / zip archives."""
    flow_path = pcap_path = None
    queue = list(paths)
    while queue:
        path = queue.pop(0)
        kind = sniff(path)
        if kind == "zip":
            with zipfile.ZipFile(path) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    target = os.path.join(tmp, "unz_" + os.path.basename(member.filename))
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    queue.append(target)
            continue
        with open(path, "rb") as fh:
            head = fh.read(3)
        comp = ".bz2" if head == b"BZh" else ".gz" if head[:2] == b"\x1f\x8b" else ""
        new = path + (".pcap" if kind == "pcap" else ".csv") + comp
        os.replace(path, new)
        if kind == "pcap" and pcap_path is None:
            pcap_path = new
        elif kind == "flows" and flow_path is None:
            flow_path = new
    return flow_path, pcap_path


@app.route("/api/stream", methods=["POST"])
def stream():
    """Single entry point for the dashboard: upload any files (or pick a sample), get NDJSON events."""
    tmp = tempfile.mkdtemp(prefix="vighnax_")
    try:
        speed = max(0.0, float(request.form.get("speed", 60) or 0))  # x real time; 0 = no pacing
    except ValueError:
        speed = 60.0
    try:
        sample = request.form.get("sample")
        offset = None
        if sample:
            spec = next((s for s in _samples() if s["id"] == sample), None)
            if spec is None:
                shutil.rmtree(tmp, ignore_errors=True)
                return jsonify({"error": f"unknown sample {sample!r}"}), 400
            flow_path = os.path.join(SAMPLES_DIR, spec["flows"]) if spec.get("flows") else None
            pcap_path = os.path.join(SAMPLES_DIR, spec["pcap"]) if spec.get("pcap") else None
            offset, name = spec.get("time_offset_s"), spec["title"]
            profile = spec.get("dataset", "ctu13")
        else:
            saved = []
            for i, f in enumerate(request.files.getlist("files")):
                if f and f.filename:
                    dest = os.path.join(tmp, f"{i}_" + (secure_filename(f.filename) or "upload"))
                    f.save(dest)
                    saved.append(dest)
            if not saved:
                shutil.rmtree(tmp, ignore_errors=True)
                return jsonify({"error": "choose a file: flow records (CSV / binetflow) and/or a packet capture"}), 400
            name = ", ".join(os.path.basename(p).split("_", 1)[1] for p in saved)
            profile = None
            flow_path, pcap_path = _classify_uploads(saved, tmp)
    except Exception as exc:  # malformed zip etc.
        shutil.rmtree(tmp, ignore_errors=True)
        return jsonify({"error": f"could not read upload: {exc}"}), 400

    def events():
        try:
            for ev in engine().analyze_stream(flow_path=flow_path, pcap_path=pcap_path, time_offset_s=offset, name=name,
                                              speed=speed, profile=profile):
                yield json.dumps(ev) + "\n"
        except Exception as exc:  # report parse / format errors to the UI instead of dropping the stream
            yield json.dumps({"type": "error", "msg": str(exc)}) + "\n"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return Response(events(), mimetype="application/x-ndjson", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/api/explain/<analysis_id>/<int:idx>")
def explain(analysis_id: str, idx: int):
    try:
        return jsonify(engine().explain_window(analysis_id, idx))
    except KeyError as exc:
        return jsonify({"error": str(exc).strip("'\"")}), 404
    except WindowNotReady as exc:
        return jsonify({"error": str(exc)}), 409


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)

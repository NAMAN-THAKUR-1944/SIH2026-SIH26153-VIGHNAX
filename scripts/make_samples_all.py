"""Cut a small, real demo input from the *test* data of every non-CTU dataset.

For each dataset the first sustained attack onset in its test sources is found
and 12 minutes of labelled flow records around it are written to
samples/<dataset>.flows.csv.gz (common schema, readable by the dashboard's
universal loader). Datasets that come as packet captures (DARPA 1999, CICIoT2023)
also get the matching packets as samples/<dataset>.pcap.bz2.

    python scripts/make_samples_all.py
"""

from __future__ import annotations

import bz2
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import datasets as ds  # noqa: E402
from src.data.flow_parser import OUTPUT_COLUMNS  # noqa: E402
from src.data.sources import DATASET_NAMES  # noqa: E402

TITLES = {
    "cicids2017": "CIC-IDS2017 · Friday (unseen day)",
    "cicids2018": "CSE-CIC-IDS2018 · unseen day",
    "unswnb15": "UNSW-NB15 · 18 Feb 2015 (unseen day)",
    "lanl": "LANL cyber1 · red-team lateral movement (unseen slice)",
    "darpa1999": "DARPA 1999 · week 5 (unseen day)",
    "ciciot2023": "CICIoT2023 · unseen attack type",
}


def find_onset(y: np.ndarray, quiet: int = 18, min_after: int = 4, span: int = 18) -> int:
    """First onset of a *sustained* attack (>= 10 of the next 18 windows) after a quiet period;
    falls back to weaker onsets. Chosen from labels only, never from model scores."""
    for need in (10, min_after):
        for t in range(quiet, len(y) - span):
            if y[t] == 1 and y[t - quiet:t].sum() == 0 and y[t:t + span].sum() >= need:
                return t
    idx = np.where(y == 1)[0]
    idx = idx[idx >= quiet]
    return int(idx[0]) if len(idx) else -1


def write_pcap_slice(src_pcaps, dst: str, lo: float, hi: float, shift=None) -> int:
    """Copy packets with ts in [lo, hi) (after optional per-file time shift) into one classic pcap."""
    import dpkt
    from src.data.packet_features import _records
    n = 0
    with bz2.open(dst, "wb", compresslevel=9) as fh:
        wr = dpkt.pcap.Writer(fh, snaplen=65535, linktype=1)
        for path, delta in src_pcaps:
            for ts, buf, _lt in _records(path):
                ts += delta
                if ts < lo:
                    continue
                if ts >= hi:
                    break
                wr.writepkt(buf, ts=ts)
                n += 1
    return n


def main() -> None:
    cfg = load_config()
    d, spec_all = cfg["data"], cfg["datasets"]
    win = float(d["window_s"])
    reg_path = "samples/samples.json"
    reg = json.load(open(reg_path)) if os.path.isfile(reg_path) else []
    for dataset, spec in spec_all.items():
        best = None
        for need_sustained in (True, False):
            for path in sorted(glob.glob(os.path.join(d["processed_dir"], f"{dataset}__*.pkl"))):
                st = pd.read_pickle(path)
                if st["split"].iloc[0] != "test":
                    continue
                y = st["is_attack"].to_numpy()
                t = find_onset(y)
                if t >= 0 and (not need_sustained or y[t:t + 18].sum() >= 10):
                    best = (st, t, st["source"].iloc[0])
                    break
            if best is not None:
                break
        if best is None:
            print(f"{dataset}: no onset found in test data")
            continue
        st, t, source = best
        onset = float(st["window_start"].iloc[t])
        lo, hi = onset - 6 * 60, onset + 6 * 60
        pcap_out = None
        if dataset == "cicids2017":
            flows = ds.load_cic2017(spec["zip"], source)
        elif dataset == "cicids2018":
            flows = ds.load_cic2018(os.path.join(spec["dir"], f"{source}.csv"))
        elif dataset == "unswnb15":
            flows = ds.load_unsw(os.path.join(spec["dir"], f"UNSW-NB15_{source}.csv"), os.path.join(spec["dir"], "features.csv"))
        elif dataset == "lanl":
            a, b = (int(x) for x in source.split("-"))
            flows = ds.load_lanl(spec["flows"], spec["redteam"], a * 3600, b * 3600)
        elif dataset == "darpa1999":
            pcap = os.path.join(spec["dir"], f"{source}.tcpdump.gz")
            truth = ds.parse_darpa_truth(spec["truth"])
            flows, _ = ds.load_darpa(pcap, truth, win, spec["utc_offset_h"]["week4" if source.startswith("week4") else "week5"])
            pcap_out = [(pcap, 0.0)]
        elif dataset == "ciciot2023":
            benign = os.path.join(spec["dir"], f"{spec['benign']}.pcap")
            bf, _ = ds.load_ciciot_capture(benign, "BenignTraffic", win)
            mid = float(bf["start_time"].min() + (bf["start_time"].max() - bf["start_time"].min()) / 2)
            b_hi = mid + spec["benign_minutes"] * 60
            attack = os.path.join(spec["dir"], f"{source}.pcap")
            af0, _ = ds.load_ciciot_capture(attack, source, win)
            delta = b_hi - float(af0["start_time"].min())
            af, _ = ds.load_ciciot_capture(attack, source, win, shift_to=b_hi)
            flows = pd.concat([bf[(bf["start_time"] >= mid) & (bf["start_time"] < b_hi)], af], ignore_index=True)
            pcap_out = [(benign, 0.0), (attack, delta)]
        else:
            continue
        sl = flows[(flows["start_time"] >= lo) & (flows["start_time"] < hi)]
        sid = f"{dataset}_sample"
        sl[[c for c in OUTPUT_COLUMNS if c in sl]].to_csv(f"samples/{sid}.flows.csv.gz", index=False, compression="gzip")
        entry = {"id": sid, "flows": f"{sid}.flows.csv.gz", "time_offset_s": 0,
                 "title": f"{TITLES.get(dataset, DATASET_NAMES.get(dataset, dataset))} - 12 min around an attack onset",
                 "dataset": dataset}
        msg = f"{sid}: {len(sl):,} flows ({int(sl["is_attack"].fillna(0).sum()) if "is_attack" in sl else 0:,} labelled malicious)"
        if pcap_out:
            n = write_pcap_slice(pcap_out, f"samples/{sid}.pcap.bz2", lo, hi)
            entry["pcap"] = f"{sid}.pcap.bz2"
            msg += f", {n:,} packets"
        reg = [r for r in reg if r["id"] != sid] + [entry]
        print(msg, flush=True)
    for r in reg:
        r.setdefault("dataset", "ctu13")
    json.dump(reg, open(reg_path, "w"), indent=1)


if __name__ == "__main__":
    main()

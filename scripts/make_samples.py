"""Cut small, real demo inputs (flows + packets) from held-out CTU-13 captures.

For each scenario, finds the first attack onset preceded by >= ``--quiet`` minutes
of benign traffic, then writes the time slice [onset - before, onset + after] as

    samples/<id>.binetflow.gz   (labelled flow records, original CTU-13 format)
    samples/<id>.pcap.bz2       (header-truncated packets, classic pcap; may cover a shorter span)

and registers them in samples/samples.json for the dashboard.

    python scripts/make_samples.py --scenario 47 --before 6 --after 6 --pcap-before 3 --pcap-after 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluation.metrics import onsets  # noqa: E402

FAMILY = {47: "Menti", 43: "Neris", 46: "Virut", 48: "Sogou", 51: "Rbot", 52: "Rbot", 53: "NSIS.ay"}


def slice_flows(src: str, dst: str, t_lo: float, t_hi: float) -> int:
    """Copy binetflow rows whose StartTime (local, naive) lies in [t_lo, t_hi) epoch-as-UTC."""
    fmt = "%Y/%m/%d %H:%M:%S"
    lo = dt.datetime.fromtimestamp(t_lo, dt.UTC).strftime(fmt)
    hi = dt.datetime.fromtimestamp(t_hi, dt.UTC).strftime(fmt)
    n = 0
    with open(src, encoding="utf-8", errors="replace") as fin, gzip.open(dst, "wt", encoding="utf-8") as fout:
        fout.write(fin.readline())
        for line in fin:
            stamp = line[:19]
            if lo <= stamp < hi:
                fout.write(line); n += 1
            elif stamp >= hi:
                break
    return n


def slice_pcap(src: str, dst: str, t_lo_utc: float, t_hi_utc: float) -> int:
    import bz2
    import dpkt
    from src.data.packet_features import _records
    n = 0
    with bz2.open(dst, "wb", compresslevel=9) as fh:
        wr = dpkt.pcap.Writer(fh, snaplen=65535, linktype=1)
        for ts, buf, _lt in _records(src):
            if ts < t_lo_utc:
                continue
            if ts >= t_hi_utc:
                break
            wr.writepkt(buf, ts=ts); n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, required=True)
    ap.add_argument("--before", type=float, default=6.0, help="minutes before onset")
    ap.add_argument("--after", type=float, default=6.0, help="minutes after onset")
    ap.add_argument("--quiet", type=float, default=4.0, help="benign minutes required before onset")
    ap.add_argument("--onset", type=int, default=0, help="which qualifying onset to use (0 = first)")
    ap.add_argument("--pcap-before", type=float, default=None, help="minutes of packets before onset (default --before)")
    ap.add_argument("--pcap-after", type=float, default=None, help="minutes of packets after onset (default --after)")
    ap.add_argument("--at-index", type=int, default=None, help="centre the slice on this window index instead")
    ap.add_argument("--no-pcap", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    d = cfg["data"]
    win, off = d["window_s"], d["time_offset_s"]
    states = pd.read_pickle(os.path.join(d["processed_dir"], f"s{args.scenario}.pkl"))
    if args.at_index is not None:
        t0_idx = args.at_index
    else:
        ons = onsets(states["is_attack"].to_numpy(), quiet=int(args.quiet * 60 / win))
        if not ons:
            raise SystemExit("no qualifying onset")
        t0_idx = ons[min(args.onset, len(ons) - 1)]
    onset_t = float(states["window_start"].iloc[t0_idx])  # flow clock (local-as-UTC)
    t_lo, t_hi = onset_t - args.before * 60, onset_t + args.after * 60
    os.makedirs("samples", exist_ok=True)
    sid = f"ctu13_s{args.scenario}_{FAMILY.get(args.scenario, '').lower()}"
    fl = slice_flows(os.path.join(d["raw_dir"], f"s{args.scenario}.binetflow"), f"samples/{sid}.binetflow.gz", t_lo, t_hi)
    entry = {"id": sid, "flows": f"{sid}.binetflow.gz", "time_offset_s": off,
             "title": f"CTU-13 capture {args.scenario} ({FAMILY.get(args.scenario, '')} botnet, unseen in training) - "
                      f"{args.before + args.after:g} min around an attack onset"}
    msg = f"{sid}: {fl:,} flows"
    if not args.no_pcap:
        pb = args.before if args.pcap_before is None else args.pcap_before
        pa = args.after if args.pcap_after is None else args.pcap_after
        pk = slice_pcap(os.path.join(d["raw_dir"], f"s{args.scenario}.pcap.bz2"), f"samples/{sid}.pcap.bz2",
                        onset_t - pb * 60 - off, onset_t + pa * 60 - off)
        entry["pcap"] = f"{sid}.pcap.bz2"
        entry["title"] += f" (packets for the central {pb + pa:g} min)"
        msg += f", {pk:,} packets"
    reg_path = "samples/samples.json"
    reg = json.load(open(reg_path)) if os.path.isfile(reg_path) else []
    reg = [r for r in reg if r["id"] != sid] + [entry]
    json.dump(reg, open(reg_path, "w"), indent=1)
    print(msg, "| onset at", dt.datetime.fromtimestamp(onset_t, dt.UTC).strftime("%H:%M:%S"), "local")


if __name__ == "__main__":
    main()

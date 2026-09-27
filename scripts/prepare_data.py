"""Build per-window network-state tables from raw CTU-13 scenarios.

For every scenario listed in the config:
  1. parse the labelled bidirectional flows (.binetflow)   -> flow-level telemetry + labels
  2. stream the header-truncated full capture (.pcap.bz2)  -> packet-level telemetry
  3. align both on absolute time windows (auto-detected clock offset: the flow
     files are in Prague local time, the capture in UTC)
  4. save datasets/processed/s<ID>.pkl

Usage:
    python scripts/prepare_data.py                     # all scenarios in configs/default.yaml
    python scripts/prepare_data.py --scenarios 52 46   # a subset
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.data.alignment import best_offset  # noqa: E402


def prepare_scenario(sid: int, raw_dir: str, out_dir: str, window_s: float,
                     fixed_offset_s=None) -> str:
    import pandas as pd
    from src.data.flow_parser import parse_flow_csv
    from src.data.packet_features import extract_packet_windows
    from src.data.state_builder import build_states

    t0 = time.time()
    flow_path = os.path.join(raw_dir, f"s{sid}.binetflow")
    pcap_path = os.path.join(raw_dir, f"s{sid}.pcap.bz2")
    flows = parse_flow_csv(flow_path, source_format="ctu")
    pk = None
    offset = 0.0
    pk_cache = os.path.join(out_dir, f"s{sid}_packets.pkl")
    old_states = os.path.join(out_dir, f"s{sid}.pkl")
    if os.path.isfile(pk_cache):  # packet parsing is the slow part - reuse it
        pk, offset = pd.read_pickle(pk_cache), float("nan")
    elif os.path.isfile(old_states) and os.path.isfile(pcap_path):
        old = pd.read_pickle(old_states)
        if "pk_mask" in old and old["pk_mask"].sum() > 0:
            from src.data.packet_features import PACKET_FEATURES
            pk = old.loc[old["pk_mask"] > 0, list(PACKET_FEATURES)].reset_index()
            pk.to_pickle(pk_cache)
            offset = float("nan")
    if pk is None and os.path.isfile(pcap_path):
        pk, _ = extract_packet_windows(pcap_path, window_s=window_s, time_offset_s=0.0)
        offset = float(fixed_offset_s) if fixed_offset_s is not None else best_offset(flows, pk, window_s)
        pk["window_id"] = pk["window_id"] + int(offset // window_s)
        pk.to_pickle(pk_cache)
    states = build_states(flows, pk, window_s=window_s)
    states["scenario"] = sid
    out = os.path.join(out_dir, f"s{sid}.pkl")
    states.to_pickle(out)
    n_att = int(states["is_attack"].sum())
    return (f"s{sid}: {len(flows):,} flows, {len(states)} windows ({n_att} malicious), "
            f"pcap={'yes' if pk is not None else 'NO'} offset={'cached' if offset != offset else f'{offset:+.0f}s'}, "
            f"stages={states['stage'].value_counts().sort_index().to_dict()} [{time.time() - t0:.0f}s]")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scenarios", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    sids = args.scenarios or (d["train_scenarios"] + d["test_scenarios"])
    os.makedirs(d["processed_dir"], exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(prepare_scenario, s, d["raw_dir"], d["processed_dir"], d["window_s"],
                            d.get("time_offset_s")): s for s in sids}
        for fut in as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # report and keep going with the others
                print(f"s{futs[fut]}: FAILED {exc!r}", flush=True)


if __name__ == "__main__":
    main()

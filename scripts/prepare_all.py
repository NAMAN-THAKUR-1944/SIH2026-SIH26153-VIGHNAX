"""Build per-window network states for every non-CTU dataset in configs/default.yaml.

Each *source* (a day file, a capture, a time range) becomes one or more contiguous
segments (split wherever traffic stops for > gap_s), saved as
datasets/processed/<dataset>__<source>__<k>.pkl with a ``dataset`` and ``split``
column. CTU-13 is prepared by scripts/prepare_data.py.

    python scripts/prepare_all.py                 # everything
    python scripts/prepare_all.py --only unswnb15  # one dataset
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402


def split_on_gaps(flows, gap_s: float):
    import numpy as np
    f = flows.sort_values("start_time").reset_index(drop=True)
    t = f["start_time"].to_numpy()
    cuts = np.where(np.diff(t) > gap_s)[0] + 1
    bounds = [0, *cuts.tolist(), len(f)]
    return [f.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 0]


def job(dataset: str, source: str, split: str, spec: dict, d: dict, out_dir: str) -> str:
    import numpy as np
    import pandas as pd
    from src.data import datasets as ds
    from src.data.state_builder import build_states

    t0 = time.time()
    win = float(d["window_s"])
    pk = None
    if dataset == "cicids2017":
        flows = ds.load_cic2017(spec["zip"], source)
    elif dataset == "cicids2018":
        flows = ds.load_cic2018(os.path.join(spec["dir"], f"{source}.csv"))
    elif dataset == "unswnb15":
        flows = ds.load_unsw(os.path.join(spec["dir"], f"UNSW-NB15_{source}.csv"), os.path.join(spec["dir"], "features.csv"))
    elif dataset == "lanl":
        lo, hi = (int(x) for x in source.split("-"))
        flows = ds.load_lanl(spec["flows"], spec["redteam"], lo * 3600, hi * 3600)
    elif dataset == "darpa1999":
        truth = ds.parse_darpa_truth(spec["truth"])
        offset = spec["utc_offset_h"]["week4" if source.startswith("week4") else "week5"]
        flows, pk = ds.load_darpa(os.path.join(spec["dir"], f"{source}.tcpdump.gz"), truth, win, offset)
    elif dataset == "ciciot2023":
        # Episode = a slice of benign traffic followed by one attack capture (disclosed composition).
        benign_f, benign_pk = ds.load_ciciot_capture(os.path.join(spec["dir"], f"{spec['benign']}.pcap"), "BenignTraffic", win)
        bt = benign_f["start_time"]
        mid = float(bt.min() + (bt.max() - bt.min()) / 2)
        lo = float(bt.min()) if split == "train" else mid
        hi = lo + spec["benign_minutes"] * 60
        bf = benign_f[(benign_f["start_time"] >= lo) & (benign_f["start_time"] < hi)]
        bpk = benign_pk[(benign_pk["window_id"] >= lo // win) & (benign_pk["window_id"] < hi // win)]
        af, apk = ds.load_ciciot_capture(os.path.join(spec["dir"], f"{source}.pcap"), source, win, shift_to=hi)
        af = af[af["start_time"] < hi + spec["attack_minutes"] * 60]
        apk = apk[apk["window_id"] < (hi + spec["attack_minutes"] * 60) // win]
        flows = pd.concat([bf, af], ignore_index=True)
        pk = pd.concat([bpk, apk], ignore_index=True).groupby("window_id", as_index=False).first()
    else:
        raise ValueError(dataset)

    msgs = []
    for k, seg in enumerate(split_on_gaps(flows, d.get("gap_s", 600))):
        if len(seg) < 50:
            continue
        seg_pk = None
        if pk is not None and len(pk):
            w = (seg["start_time"] // win).astype("int64")
            seg_pk = pk[(pk["window_id"] >= w.min()) & (pk["window_id"] <= w.max())]
        st = build_states(seg, seg_pk, win)
        st["dataset"], st["source"], st["split"] = dataset, source, split
        name = f"{dataset}__{source}__{k}".replace("/", "_")
        st.to_pickle(os.path.join(out_dir, f"{name}.pkl"))
        msgs.append(f"{name}: {len(seg):,} flows, {len(st)} windows ({int(st['is_attack'].sum())} malicious), "
                    f"stages={st['stage'].value_counts().sort_index().to_dict()}")
    return f"[{time.time() - t0:.0f}s] " + " | ".join(msgs) if msgs else f"{dataset}/{source}: no usable segments"


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    out_dir = d["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for dataset, spec in cfg["datasets"].items():
        if args.only and dataset not in args.only:
            continue
        for old in glob.glob(os.path.join(out_dir, f"{dataset}__*.pkl")):
            os.remove(old)
        for split in ("train", "test"):
            for source in spec.get(split, []):
                jobs.append((dataset, str(source), split, spec))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(job, a, b, c, s, d, out_dir): (a, b) for a, b, c, s in jobs}
        for fut in as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:
                print(f"{futs[fut]}: FAILED {exc!r}", flush=True)


if __name__ == "__main__":
    main()

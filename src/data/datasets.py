"""Loaders for every dataset named in the PS, each producing the common flow schema.

Every loader returns ``(flows, packet_windows)`` where ``flows`` follows
:data:`src.data.flow_parser.OUTPUT_COLUMNS` (start_time in epoch seconds,
``label`` carrying the dataset's own attack name) and ``packet_windows`` is the
per-window packet block or ``None``. :func:`src.data.state_builder.build_states`
then turns any of them into network states S_t, and :mod:`src.data.stages`
maps the labels to MITRE ATT&CK stages.

Dataset-specific quirks handled here (documented, not hidden):
  * CIC-IDS2017 timestamps have minute resolution and a 12-hour clock without
    AM/PM. Hours < 8 are afternoon (+12 h); flows within a minute are spread
    evenly over its 60 s in file order.
  * CSE-CIC-IDS2018 "processed" CSVs carry no IP addresses (except one day), so
    host-level features are zeroed for them (see state_builder).
  * UNSW-NB15 CSVs have no header; column names come from NUSW-NB15_features.csv.
  * LANL flows use anonymised computer names and relative time (epoch 1); red-team
    authentication events label flows between the same computers as lateral movement.
  * DARPA 1999 provides only tcpdump + an attack truth list; flows are rebuilt from
    the packets and labelled by attacker/victim/time from the truth list.
  * CICIoT2023 captures are one activity per file; a benign capture and an attack
    capture are concatenated into an episode (disclosed; used for detection only).
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import re
import zipfile
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.flow_parser import OUTPUT_COLUMNS, FlowCsvNormalizer, parse_flow_csv
from src.data.packet_features import extract_packet_windows

LOGGER = logging.getLogger(__name__)


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    for c in OUTPUT_COLUMNS:
        if c not in frame:
            frame[c] = np.nan
    return frame


# ------------------------------------------------------------------ CIC-IDS2017
def load_cic2017(zip_path: str, member_pattern: str) -> pd.DataFrame:
    """One CIC-IDS2017 day file from GeneratedLabelledFlows.zip (matched by substring)."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if member_pattern.lower() in n.lower() and n.endswith(".csv"))
        with zf.open(name) as fh:
            raw = pd.read_csv(io.TextIOWrapper(fh, encoding="latin-1"), skipinitialspace=True, low_memory=False,
                              on_bad_lines="skip")
    return cic2017_frame(raw)


def cic2017_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """CIC-IDS2017 rows (minute-resolution, 12-hour timestamps) -> common flow schema."""
    raw = raw.dropna(how="all")
    ts = raw.pop("Timestamp").astype(str).str.strip()
    parts = ts.str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?")
    ok = parts[0].notna()
    raw, parts = raw[ok].copy(), parts[ok].astype(float)
    hour = parts[3].where(parts[3] >= 8, parts[3] + 12)          # 12-hour clock, working hours 08-17
    base = pd.to_datetime(dict(year=parts[2], month=parts[1], day=parts[0]))
    secs = (base - pd.Timestamp("1970-01-01")).dt.total_seconds() + hour * 3600 + parts[4] * 60
    has_sec = parts[5].notna()
    key = pd.Series(secs.round().to_numpy(), index=raw.index)
    rank = key.groupby(key).cumcount()
    size = key.groupby(key).transform("size")
    spread = np.where(has_sec, parts[5].fillna(0), rank / np.maximum(size, 1) * 60.0)
    start = pd.Series(secs.to_numpy() + spread, index=raw.index)
    flows = FlowCsvNormalizer("cic", dayfirst=True).normalize(raw.assign(Timestamp="2000-01-01"))
    flows["start_time"] = start.reindex(flows.index).to_numpy()  # normalize() keeps the original row index
    flows["timestamp"] = pd.to_datetime(flows["start_time"], unit="s").astype(str)
    return _finish(flows).sort_values("start_time").reset_index(drop=True)


# ------------------------------------------------------------------ CSE-CIC-IDS2018
def load_cic2018(csv_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    flows = parse_flow_csv(csv_path, source_format="cic", dayfirst=True, max_rows=max_rows)
    flows = flows[flows["start_time"] > 1.5e9].copy()  # drop the few 1970-dated junk rows
    # Like 2017, the 2018 files use a 12-hour clock without AM/PM: 01:00-06:59 means afternoon.
    hour = (flows["start_time"] % 86400) // 3600
    flows.loc[hour < 7, "start_time"] += 12 * 3600
    return _finish(flows).sort_values("start_time").reset_index(drop=True)


# ------------------------------------------------------------------ UNSW-NB15
UNSW_COLUMNS = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
                "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
                "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt",
                "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
                "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm",
                "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"]
UNSW_STATE_FLAGS = {"FIN": ("syn", "ack", "fin"), "CON": ("syn", "ack"), "RST": ("syn", "rst"),
                    "REQ": ("syn",), "INT": (), "CLO": ("fin",), "ACC": ("syn", "ack")}


def load_unsw(csv_path: str, features_csv: Optional[str] = None) -> pd.DataFrame:
    names = ([n.strip().lower() for n in pd.read_csv(features_csv, encoding="latin-1")["Name"]]
             if features_csv else UNSW_COLUMNS)
    raw = pd.read_csv(csv_path, header=None, names=names, low_memory=False, encoding="latin-1")
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")
    f = pd.DataFrame({
        "src_ip": raw["srcip"].astype(str), "src_port": num("sport").fillna(0).astype("int64"),
        "dst_ip": raw["dstip"].astype(str), "dst_port": num("dsport").fillna(0).astype("int64"),
        "protocol": raw["proto"].astype(str).str.upper(),
        "start_time": num("stime").astype(float), "duration": num("dur"),
        "fwd_packets": num("spkts"), "bwd_packets": num("dpkts"), "fwd_bytes": num("sbytes"), "bwd_bytes": num("dbytes"),
    })
    f["packet_count"] = f["fwd_packets"] + f["bwd_packets"]
    f["byte_count"] = f["fwd_bytes"] + f["bwd_bytes"]
    state = raw["state"].astype(str).str.upper()
    for fl in ("fin", "syn", "rst", "psh", "ack", "urg", "cwr", "ece"):
        f[f"flag_{fl}"] = np.where(f["protocol"] == "TCP",
                                   state.map(lambda s, fl=fl: float(fl in UNSW_STATE_FLAGS.get(s, ()))), np.nan)
    iat = (num("sintpkt").fillna(0) + num("dintpkt").fillna(0)) / 2000.0     # ms -> s, mean of both directions
    f["flow_iat_mean"] = iat
    f["flow_iat_max"] = np.maximum(num("sintpkt").fillna(0), num("dintpkt").fillna(0)) / 1000.0
    cat = raw["attack_cat"].astype(str).str.strip()
    lab = num("label").fillna(0)
    f["label"] = np.where(lab > 0, cat.where(~cat.isin(["", "nan"]), "attack"), "normal")
    f["is_attack"] = (lab > 0).astype(float)
    f["timestamp"] = pd.to_datetime(f["start_time"], unit="s").astype(str)
    return _finish(f).sort_values("start_time").reset_index(drop=True)


# ------------------------------------------------------------------ LANL
def load_lanl(flows_gz: str, redteam_gz: str, t_lo: int, t_hi: int, pair_slack_s: int = 300) -> pd.DataFrame:
    """LANL flows in [t_lo, t_hi) (relative seconds), labelled from red-team events.

    A flow is 'lanl-redteam-lateral-movement' when, within +-pair_slack_s of a red-team
    compromise event, it runs between the event's two computers, or from the attacking
    computer to any red-team victim computer. Red-team truth is authentication-based, so exact pair
    matches in router-level flows are rare; attacker-to-victim traffic is the network view.
    """
    rt = pd.read_csv(redteam_gz, header=None, names=["time", "user", "src", "dst"])
    rows: List[pd.DataFrame] = []
    cols = ["time", "duration", "src", "sport", "dst", "dport", "proto", "packets", "bytes"]
    with gzip.open(flows_gz, "rt") as fh:
        for chunk in pd.read_csv(fh, header=None, names=cols, chunksize=2_000_000):
            if chunk["time"].iloc[0] >= t_hi:
                break
            sel = chunk[(chunk["time"] >= t_lo) & (chunk["time"] < t_hi)]
            if len(sel):
                rows.append(sel)
    raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
    epoch0 = 1.6e9  # arbitrary anchor so windows have realistic absolute ids; LANL time is relative
    f = pd.DataFrame({
        "src_ip": raw["src"].astype(str), "dst_ip": raw["dst"].astype(str),
        "src_port": pd.to_numeric(raw["sport"], errors="coerce").fillna(0).astype("int64"),
        "dst_port": pd.to_numeric(raw["dport"], errors="coerce").fillna(0).astype("int64"),
        "protocol": raw["proto"].map({6: "TCP", 17: "UDP", 1: "ICMP"}).fillna("OTHER"),
        "start_time": raw["time"].astype(float) + epoch0, "duration": raw["duration"].astype(float),
        "packet_count": raw["packets"].astype(float), "byte_count": raw["bytes"].astype(float),
    })
    f["label"] = "normal"
    rts = rt[(rt["time"] >= t_lo - pair_slack_s) & (rt["time"] < t_hi + pair_slack_s)]
    if len(rts):
        t = raw["time"].to_numpy()
        src, dst = raw["src"].astype(str).to_numpy(), raw["dst"].astype(str).to_numpy()
        mal = np.zeros(len(f), dtype=bool)
        victim = np.isin(dst, rts["dst"].astype(str).unique())
        for _, ev in rts.iterrows():
            near = np.abs(t - ev["time"]) <= pair_slack_s
            # the compromise itself (either direction) or the attacking computer reaching any red-team victim
            mal |= near & (((src == ev["src"]) & (dst == ev["dst"])) | ((src == ev["dst"]) & (dst == ev["src"]))
                           | ((src == ev["src"]) & victim))
        f.loc[mal, "label"] = "lanl-redteam-lateral-movement"
    f["timestamp"] = f["start_time"].astype(str)
    return _finish(f).sort_values("start_time").reset_index(drop=True)


# ------------------------------------------------------------------ any format
LANL_COLUMNS = ["time", "duration", "src", "sport", "dst", "dport", "proto", "packets", "bytes"]


def detect_flow_format(path: str) -> str:
    """Identify a flow file by content: ctu | cic2017 | cic | unsw | lanl | vighnax."""
    opener = gzip.open if open(path, "rb").read(2) == b"\x1f\x8b" else open
    with opener(path, "rt", encoding="latin-1", errors="replace") as fh:
        first = fh.readline().strip()
        second = fh.readline().strip()
    low = first.lower()
    if "flow_key" in low and "is_attack" in low:
        return "vighnax"
    if "srcaddr" in low and "starttime" in low:
        return "ctu"
    if "timestamp" in low and ("flow duration" in low or "flow_duration" in low):
        ts = second.split(",")[[c.strip().lower() for c in first.split(",")].index("timestamp")] if second else ""
        return "cic2017" if re.match(r"^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}$", ts.strip()) or "flow id" in low else "cic"
    if low.startswith("time,duration,src") or (len(first.split(",")) in (9, 10) and re.match(r"^\d+,\d+,C\d+", first)):
        return "lanl"
    if len(first.split(",")) == 49 and re.match(r"^\d+\.\d+\.\d+\.\d+,", first):
        return "unsw"
    return "cic"


def load_flows_any(path: str) -> pd.DataFrame:
    """Load flow records in any PS dataset's native format into the common schema."""
    fmt = detect_flow_format(path)
    if fmt == "ctu":
        return parse_flow_csv(path, source_format="ctu")
    if fmt == "vighnax":
        f = pd.read_csv(path, low_memory=False)
        return _finish(f)
    if fmt == "cic2017":
        raw = pd.read_csv(path, skipinitialspace=True, low_memory=False, on_bad_lines="skip", encoding="latin-1")
        return cic2017_frame(raw)
    if fmt == "unsw":
        return load_unsw(path)
    if fmt == "lanl":
        raw = pd.read_csv(path, header=0 if open_first(path).lower().startswith("time,") else None,
                          names=None if open_first(path).lower().startswith("time,") else LANL_COLUMNS)
        f = pd.DataFrame({
            "src_ip": raw["src"].astype(str), "dst_ip": raw["dst"].astype(str),
            "src_port": pd.to_numeric(raw["sport"], errors="coerce").fillna(0).astype("int64"),
            "dst_port": pd.to_numeric(raw["dport"], errors="coerce").fillna(0).astype("int64"),
            "protocol": raw["proto"].map({6: "TCP", 17: "UDP", 1: "ICMP"}).fillna("OTHER"),
            "start_time": raw["time"].astype(float) + (1.6e9 if raw["time"].max() < 1e8 else 0.0),
            "duration": raw["duration"].astype(float), "packet_count": raw["packets"].astype(float),
            "byte_count": raw["bytes"].astype(float),
            "label": raw["label"] if "label" in raw else "normal",
        })
        return _finish(f).sort_values("start_time").reset_index(drop=True)
    return load_cic2018(path)


def open_first(path: str) -> str:
    opener = gzip.open if open(path, "rb").read(2) == b"\x1f\x8b" else open
    with opener(path, "rt", encoding="latin-1", errors="replace") as fh:
        return fh.readline().strip()


# ------------------------------------------------------------------ DARPA 1999
DARPA_CATEGORY = {"probe": "darpa-probe", "dos": "darpa-dos", "r2l": "darpa-r2l", "u2r": "darpa-u2r", "data": "darpa-data"}


def parse_darpa_truth(path: str) -> pd.DataFrame:
    recs, cur = [], {}
    for line in open(path, encoding="latin-1"):
        line = line.rstrip("\n")
        if line.startswith("ID:"):
            if cur:
                recs.append(cur)
            cur = {"id": line.split(":", 1)[1].strip()}
        elif ":" in line and not line.startswith("\t"):
            k, v = line.split(":", 1)
            cur[k.strip().lower()] = v.strip()
    if cur:
        recs.append(cur)
    df = pd.DataFrame(recs)
    norm_ip = lambda s: ".".join(str(int(p)) for p in s.split(".")) if re.match(r"^\d+\.\d+\.\d+\.\d+$", str(s)) else ""
    df["attacker_ip"] = df.get("attacker", "").map(lambda s: norm_ip(str(s).split(",")[0].strip()))
    df["victim_ip"] = df.get("victim", "").map(lambda s: norm_ip(str(s).split(",")[0].strip()))
    return df


def load_darpa(pcap_path: str, truth: pd.DataFrame, window_s: float, utc_offset_h: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild flows from a DARPA tcpdump and label them from the truth list.

    Truth times are US-Eastern local; ``utc_offset_h`` is -5 (EST) or -4 (EDT).
    """
    pk, flows = extract_packet_windows(pcap_path, window_s=window_s, with_flows=True)
    flows = flows.copy()
    flows["label"] = "normal"
    s = flows["start_time"].to_numpy()
    e = flows["end_time"].to_numpy()
    a, b = flows["src_ip"].astype(str).to_numpy(), flows["dst_ip"].astype(str).to_numpy()
    day0 = datetime.fromtimestamp(float(np.nanmin(s)) + utc_offset_h * 3600, timezone.utc).strftime("%m/%d/%Y")
    for _, ev in truth[truth.get("date", "") == day0].iterrows():
        try:
            hh, mm, ss = (int(x) for x in str(ev.get("start_time", "")).split(":"))
            dh, dm, ds = (int(x) for x in str(ev.get("duration", "00:00:00")).split(":"))
        except ValueError:
            continue
        mo, dd, yy = (int(x) for x in day0.split("/"))
        t0 = datetime(yy, mo, dd, hh, mm, ss, tzinfo=timezone.utc).timestamp() - utc_offset_h * 3600
        t1 = t0 + max(dh * 3600 + dm * 60 + ds, 1) + 60
        ips = {ev["attacker_ip"], ev["victim_ip"]} - {""}
        if not ips:
            continue
        hit = (s <= t1) & (e >= t0 - 60) & (np.isin(a, list(ips)) | np.isin(b, list(ips)))
        if ev["attacker_ip"] and ev["victim_ip"]:
            hit &= (np.isin(a, [ev["attacker_ip"], ev["victim_ip"]]) & np.isin(b, [ev["attacker_ip"], ev["victim_ip"]]))
        cat = str(ev.get("category", "")).strip().lower()
        flows.loc[hit, "label"] = f"{DARPA_CATEGORY.get(cat, 'darpa-attack')}-{str(ev.get('name', '')).split()[0].lower()}"
    flows["timestamp"] = flows["start_time"].astype(str)
    return _finish(flows).sort_values("start_time").reset_index(drop=True), pk


# ------------------------------------------------------------------ CICIoT2023
def load_ciciot_capture(pcap_path: str, label: str, window_s: float, shift_to: Optional[float] = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One CICIoT2023 capture (single activity); optionally time-shifted to start at ``shift_to``."""
    pk, flows = extract_packet_windows(pcap_path, window_s=window_s, with_flows=True)
    flows = flows.copy()
    if shift_to is not None and len(flows):
        delta = shift_to - float(flows["start_time"].min())
        flows["start_time"] += delta
        flows["end_time"] += delta
        pk = pk.copy()
        pk["window_id"] = pk["window_id"] + int(round(delta / window_s))
    flows["label"] = label
    flows["timestamp"] = flows["start_time"].astype(str)
    return _finish(flows), pk

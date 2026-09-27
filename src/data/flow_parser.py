"""Per-flow feature extraction from labelled CSV flow datasets.

Companion to :mod:`src.data.pcap_parser`. That module builds flows from raw
packets; this one ingests datasets that ship *already* aggregated per flow and
normalises their wildly inconsistent column names into one stable schema.

Supported source formats (auto-detected from the header):
  * ``cic``  - CIC-IDS2017 / CSE-CIC-IDS2018 (CICFlowMeter v3 and v4 headers)
  * ``ctu``  - CTU-13 ``.binetflow`` (Argus bidirectional netflow)

Features emitted per flow (5-tuple):
  * 5-tuple + canonical ``flow_key``      -> joinable against pcap_parser output
  * duration, seconds, unit-normalised    -> CIC ships microseconds, CTU seconds
  * packet / byte counts, per direction
  * TCP flag counts                       -> from CIC flag columns, or decoded
                                             from the Argus ``State`` field
  * inter-arrival-time statistics         -> CIC only; NaN for CTU
  * ``label`` / ``is_attack``

Usage:
    from src.data.flow_parser import parse_flow_csv
    df = parse_flow_csv("Friday-WorkingHours.pcap_ISCX.csv")

    python -m src.data.flow_parser flows.csv -o normalised.csv
    python -m src.data.flow_parser --selftest
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FlowKey = Tuple[str, int, str, int, str]

# ---------------------------------------------------------------------------
# Column vocabulary
# ---------------------------------------------------------------------------
# canonical name -> header spellings seen in the wild, already run through
# _norm(). CICFlowMeter v3 (IDS2017) and v4 (IDS2018) disagree on nearly every
# column; Argus/CTU-13 uses a different vocabulary again.
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "src_ip": ("source_ip", "src_ip", "srcaddr", "src_addr"),
    "src_port": ("source_port", "src_port", "sport"),
    "dst_ip": ("destination_ip", "dst_ip", "dstaddr", "dst_addr"),
    "dst_port": ("destination_port", "dst_port", "dport"),
    "protocol": ("protocol", "proto"),
    "timestamp": ("timestamp", "starttime", "start_time", "date_first_seen"),
    "duration": ("flow_duration", "dur", "duration"),
    "fwd_packets": ("total_fwd_packets", "tot_fwd_pkts", "total_fwd_packet", "srcpkts"),
    "bwd_packets": ("total_backward_packets", "tot_bwd_pkts", "total_bwd_packets", "dstpkts"),
    "packet_count": ("totpkts", "tot_pkts", "total_packets"),
    "fwd_bytes": ("total_length_of_fwd_packets", "totlen_fwd_pkts", "srcbytes",
                  "total_length_of_fwd_packet"),
    "bwd_bytes": ("total_length_of_bwd_packets", "totlen_bwd_pkts", "dstbytes"),
    "byte_count": ("totbytes", "tot_bytes", "total_bytes"),
    "flow_iat_mean": ("flow_iat_mean",),
    "flow_iat_std": ("flow_iat_std",),
    "flow_iat_min": ("flow_iat_min",),
    "flow_iat_max": ("flow_iat_max",),
    "fwd_iat_total": ("fwd_iat_total", "fwd_iat_tot"),
    "fwd_iat_mean": ("fwd_iat_mean",),
    "fwd_iat_std": ("fwd_iat_std",),
    "fwd_iat_min": ("fwd_iat_min",),
    "fwd_iat_max": ("fwd_iat_max",),
    "bwd_iat_total": ("bwd_iat_total", "bwd_iat_tot"),
    "bwd_iat_mean": ("bwd_iat_mean",),
    "bwd_iat_std": ("bwd_iat_std",),
    "bwd_iat_min": ("bwd_iat_min",),
    "bwd_iat_max": ("bwd_iat_max",),
    # "CWE Flag Count" is a typo for CWR that survived into both CIC releases.
    "flag_fin": ("fin_flag_count", "fin_flag_cnt"),
    "flag_syn": ("syn_flag_count", "syn_flag_cnt"),
    "flag_rst": ("rst_flag_count", "rst_flag_cnt"),
    "flag_psh": ("psh_flag_count", "psh_flag_cnt"),
    "flag_ack": ("ack_flag_count", "ack_flag_cnt"),
    "flag_urg": ("urg_flag_count", "urg_flag_cnt"),
    "flag_cwr": ("cwe_flag_count", "cwr_flag_count", "cwe_flag_cnt", "cwr_flag_cnt"),
    "flag_ece": ("ece_flag_count", "ece_flag_cnt"),
    "fwd_psh_flags": ("fwd_psh_flags",),
    "bwd_psh_flags": ("bwd_psh_flags",),
    "fwd_urg_flags": ("fwd_urg_flags",),
    "bwd_urg_flags": ("bwd_urg_flags",),
    "state": ("state",),
    "label": ("label", "class", "attack"),
}

_FLAG_NAMES = ("fin", "syn", "rst", "psh", "ack", "urg", "cwr", "ece")

# Argus encodes observed TCP flags as "<src flags>_<dst flags>", e.g. "FSPA_FSA".
_ARGUS_FLAG_LETTERS = {
    "F": "fin", "S": "syn", "R": "rst", "P": "psh",
    "A": "ack", "U": "urg", "C": "cwr", "E": "ece",
}
_ARGUS_STATE_RE = re.compile(r"^[FSRPAUCE]*_[FSRPAUCE]*$")

_IANA_PROTO = {0: "HOPOPT", 1: "ICMP", 6: "TCP", 17: "UDP", 47: "GRE", 58: "ICMPV6"}

# Anything not matched here is treated as an attack: CIC names its attacks
# directly ("DoS Hulk", "PortScan", ...) so "not benign" is the safer default.
_DEFAULT_BENIGN = frozenset({"benign", "normal", "background", "-", "0"})

# Columns carrying text; everything else in the output schema is numeric.
_TEXT_COLUMNS = ("src_ip", "dst_ip", "protocol", "label", "timestamp", "flow_key", "state")

# Emitted in this order, always, so downstream models see a fixed schema
# whatever the source file happened to contain.
OUTPUT_COLUMNS: Tuple[str, ...] = (
    "flow_key", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
    "timestamp", "start_time", "duration",
    "packet_count", "fwd_packets", "bwd_packets",
    "byte_count", "fwd_bytes", "bwd_bytes",
    "bytes_per_second", "packets_per_second", "bytes_per_packet",
    "fwd_bwd_packet_ratio", "fwd_bwd_byte_ratio",
    *(f"flag_{name}" for name in _FLAG_NAMES),
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "flow_iat_mean", "flow_iat_std", "flow_iat_min", "flow_iat_max",
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_min", "fwd_iat_max",
    "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_min", "bwd_iat_max",
    "label", "is_attack",
)

# Time-valued columns, rescaled to seconds by the per-format factor below.
_TIME_COLUMNS = ("duration",) + tuple(c for c in OUTPUT_COLUMNS if "_iat_" in c)

# CICFlowMeter reports microseconds; Argus reports seconds.
_TIME_SCALE = {"cic": 1e-6, "ctu": 1.0}


def _norm(name: object) -> str:
    """Fold a header cell to a comparable token: ' Fwd IAT Max' -> 'fwd_iat_max'."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def _canonical_key(src: str, sport: int, dst: str, dport: int, proto: str) -> FlowKey:
    """Endpoint-sorted 5-tuple, so both directions of a conversation match.

    ponytail: duplicated from pcap_parser rather than imported -- that module
    imports scapy at load time, and a CSV reader has no business requiring it.
    Six lines is cheaper than the dependency.
    """
    if (src, sport) <= (dst, dport):
        return (src, sport, dst, dport, proto)
    return (dst, dport, src, sport, proto)


def _detect_format(columns: Iterable[object]) -> str:
    """Sniff the source dataset from its header. Raises ValueError if unknown."""
    normalised = {_norm(c) for c in columns}
    if normalised & {"srcaddr", "totpkts", "dstaddr"}:
        return "ctu"
    if normalised & {"flow_duration", "tot_fwd_pkts", "total_fwd_packets", "flow_iat_mean"}:
        return "cic"
    raise ValueError(
        "Unrecognised flow CSV: expected CIC-IDS (CICFlowMeter) or CTU-13 "
        f"(Argus) headers, got {sorted(normalised)[:12]}..."
    )


def _rename_map(columns: Sequence[object]) -> Dict[object, str]:
    """Map original header cells to canonical names, first match wins.

    Duplicate aliases (CIC-IDS2017 ships two 'Fwd Header Length' columns, and
    some mirrors repeat flag columns) would collide on rename, so later hits for
    an already-claimed canonical name are dropped.
    """
    lookup = {alias: canon for canon, aliases in _ALIASES.items() for alias in aliases}
    mapping: Dict[object, str] = {}
    claimed: Set[str] = set()
    for column in columns:
        canon = lookup.get(_norm(column))
        if canon is not None and canon not in claimed:
            mapping[column] = canon
            claimed.add(canon)
    return mapping


def _proto_name(value: object) -> str:
    """'6' -> 'TCP', 'tcp' -> 'TCP', blank -> 'OTHER'."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return "OTHER"
    try:
        number = int(float(text))
    except ValueError:
        return text.upper()
    return _IANA_PROTO.get(number, f"IP{number}")


def _flags_from_state(state: object) -> Optional[Dict[str, int]]:
    """Decode an Argus ``State`` field into per-flag counts (0-2).

    Returns None for non-TCP states ('CON', 'INT', 'URP', ...), leaving the flag
    columns NaN rather than pretending a UDP flow had no SYN.
    """
    text = str(state).strip().upper()
    if not _ARGUS_STATE_RE.match(text):
        return None
    counts = {f"flag_{name}": 0 for name in _FLAG_NAMES}
    for side in text.split("_"):
        for letter in set(side):
            counts[f"flag_{_ARGUS_FLAG_LETTERS[letter]}"] += 1
    return counts


def _classify_label(label: object, benign_labels: Set[str]) -> float:
    """1 = attack, 0 = benign, NaN = unlabelled.

    Matching is substring-based because CTU-13 labels are sentences
    ('flow=From-Botnet-V42-TCP-Attempt') rather than class names.

    ponytail: a heuristic tuned for CIC + CTU. Pass ``benign_labels`` to
    recalibrate for another dataset instead of editing this function.
    """
    text = str(label).strip().lower()
    if not text or text in {"nan", "none"}:
        return np.nan
    if any(token in text for token in ("botnet", "attack", "malicious", "malware")):
        return 1.0
    if any(token in text for token in benign_labels):
        return 0.0
    return 1.0


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise divide with 0 and inf mapped to NaN rather than blowing up."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator.astype(float) / denominator.astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


class FlowCsvNormalizer:
    """Turns one chunk of raw dataset rows into the canonical feature schema.

    Split out from :func:`parse_flow_csv` so the same normalisation can be
    applied chunk-by-chunk to a multi-GB IDS2018 file without buffering it.
    """

    def __init__(self, source_format: str, benign_labels: Optional[Iterable[str]] = None,
                 dayfirst: bool = False) -> None:
        if source_format not in _TIME_SCALE:
            raise ValueError(f"Unknown source_format {source_format!r}")
        self.source_format = source_format
        self.benign_labels = set(benign_labels) if benign_labels else set(_DEFAULT_BENIGN)
        # CIC timestamps are ambiguous: IDS2017 writes 5/7/2017 (m/d), IDS2018
        # writes 26/02/2018 (d/m). Neither declares which. Hence the knob.
        self.dayfirst = dayfirst
        self.rows_in = 0
        self.rows_dropped = 0

    def normalize(self, chunk: pd.DataFrame) -> pd.DataFrame:
        self.rows_in += len(chunk)
        frame = chunk.rename(columns=_rename_map(list(chunk.columns)))
        frame = frame[[c for c in frame.columns if c in _ALIASES]].copy()

        for column in OUTPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan

        self._coerce_numeric(frame)
        frame = self._drop_junk_rows(frame)
        if frame.empty:
            return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

        frame["protocol"] = frame["protocol"].map(_proto_name)
        self._fill_direction_totals(frame)
        self._scale_times(frame)
        if self.source_format == "ctu":
            self._flags_from_argus(frame)
        self._derive_rates(frame)
        self._build_keys(frame)

        frame["is_attack"] = frame["label"].map(
            lambda value: _classify_label(value, self.benign_labels)
        )
        return frame[list(OUTPUT_COLUMNS)]

    def _coerce_numeric(self, frame: pd.DataFrame) -> None:
        """Numeric columns to float; CIC's literal 'Infinity' cells become NaN."""
        for column in OUTPUT_COLUMNS:
            if column in _TEXT_COLUMNS or column not in frame.columns:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    def _drop_junk_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove repeated header rows and blank lines.

        Concatenated CIC dumps embed the header again at each file boundary;
        those rows coerce to all-NaN across every numeric column, which no real
        flow ever does (a flow has at least a packet count or a duration).
        """
        numeric = [c for c in OUTPUT_COLUMNS
                   if c not in _TEXT_COLUMNS and c != "is_attack"]
        keep = frame[numeric].notna().any(axis=1)
        self.rows_dropped += int((~keep).sum())
        return frame[keep].copy()

    @staticmethod
    def _fill_direction_totals(frame: pd.DataFrame) -> None:
        """Reconcile totals and per-direction counts, whichever the source gave.

        CIC gives fwd/bwd and no total; CTU gives TotBytes plus SrcBytes only,
        so the backward half is the remainder.
        """
        for total, fwd, bwd in (("packet_count", "fwd_packets", "bwd_packets"),
                                ("byte_count", "fwd_bytes", "bwd_bytes")):
            frame[total] = frame[total].fillna(frame[fwd] + frame[bwd])
            frame[bwd] = frame[bwd].fillna(frame[total] - frame[fwd])
            frame[fwd] = frame[fwd].fillna(frame[total] - frame[bwd])

    def _scale_times(self, frame: pd.DataFrame) -> None:
        scale = _TIME_SCALE[self.source_format]
        if scale != 1.0:
            for column in _TIME_COLUMNS:
                frame[column] = frame[column] * scale
        frame["timestamp"] = frame["timestamp"].astype("string")
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce", dayfirst=self.dayfirst)
        # Epoch seconds, to line up with pcap_parser's start_time. Subtracting
        # the epoch keeps NaT as NaN; .astype("int64") would raise on it.
        frame["start_time"] = (parsed - pd.Timestamp("1970-01-01")).dt.total_seconds()

    @staticmethod
    def _flags_from_argus(frame: pd.DataFrame) -> None:
        decoded = frame["state"].map(_flags_from_state)
        mask = decoded.notna()
        if not mask.any():
            return
        expanded = pd.DataFrame(list(decoded[mask]), index=frame.index[mask])
        for column in expanded.columns:
            frame.loc[mask, column] = expanded[column]

    @staticmethod
    def _derive_rates(frame: pd.DataFrame) -> None:
        """Recomputed rather than read from CIC's own rate columns, which are
        riddled with Infinity for zero-duration flows."""
        frame["bytes_per_second"] = _safe_divide(frame["byte_count"], frame["duration"])
        frame["packets_per_second"] = _safe_divide(frame["packet_count"], frame["duration"])
        frame["bytes_per_packet"] = _safe_divide(frame["byte_count"], frame["packet_count"])
        frame["fwd_bwd_packet_ratio"] = _safe_divide(frame["fwd_packets"], frame["bwd_packets"])
        frame["fwd_bwd_byte_ratio"] = _safe_divide(frame["fwd_bytes"], frame["bwd_bytes"])

    @staticmethod
    def _build_keys(frame: pd.DataFrame) -> None:
        """Attach the direction-independent 5-tuple key.

        src_*/dst_* keep the dataset's own orientation (the connection
        initiator, which is what fwd/bwd counts are relative to); flow_key is
        the sorted form, so rows here join against pcap_parser output.
        """
        ports = {side: frame[f"{side}_port"].fillna(0).astype("int64")
                 for side in ("src", "dst")}
        frame["src_port"] = ports["src"]
        frame["dst_port"] = ports["dst"]
        frame["flow_key"] = [
            "|".join(str(part) for part in _canonical_key(
                str(src_ip), int(src_port), str(dst_ip), int(dst_port), str(proto)))
            for src_ip, src_port, dst_ip, dst_port, proto in zip(
                frame["src_ip"].fillna(""), ports["src"],
                frame["dst_ip"].fillna(""), ports["dst"], frame["protocol"])
        ]


def parse_flow_csv(
    path: str,
    source_format: Optional[str] = None,
    max_rows: Optional[int] = None,
    chunksize: int = 200_000,
    benign_labels: Optional[Iterable[str]] = None,
    dayfirst: bool = False,
    sep: str = ",",
) -> pd.DataFrame:
    """Read a CIC-IDS or CTU-13 flow file into the canonical feature schema.

    Args:
        path: CSV / .binetflow file.
        source_format: ``"cic"`` or ``"ctu"``; sniffed from the header if None.
        max_rows: stop after roughly this many input rows.
        chunksize: rows held in memory at once. IDS2018 day-files are ~7M rows.
        benign_labels: substrings that mark a label benign (see _classify_label).
        dayfirst: interpret ambiguous timestamps as d/m/y (IDS2018) instead of
            m/d/y (IDS2017).
        sep: field separator.

    Returns:
        One row per flow record, columns exactly :data:`OUTPUT_COLUMNS`.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file is empty, unreadable, or not a recognised format.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Flow CSV not found: {path}")

    reader_kwargs = dict(
        sep=sep, chunksize=chunksize, skipinitialspace=True, low_memory=False,
        encoding="utf-8", encoding_errors="replace", on_bad_lines="skip",
    )
    try:
        header = pd.read_csv(path, sep=sep, nrows=0, skipinitialspace=True,
                             encoding="utf-8", encoding_errors="replace")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Empty flow CSV: {path}") from exc
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Failed reading {path}: {exc}") from exc

    source_format = source_format or _detect_format(header.columns)
    normalizer = FlowCsvNormalizer(source_format, benign_labels, dayfirst)

    parts: List[pd.DataFrame] = []
    try:
        with pd.read_csv(path, **reader_kwargs) as chunks:
            for chunk in chunks:
                parts.append(normalizer.normalize(chunk))
                if max_rows is not None and normalizer.rows_in >= max_rows:
                    LOGGER.info("Stopped after %d rows (max_rows)", normalizer.rows_in)
                    break
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Failed reading {path}: {exc}") from exc

    frame = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=list(OUTPUT_COLUMNS)))
    LOGGER.info(
        "Parsed %s as %s: %d rows in, %d dropped, %d flows out",
        path, source_format, normalizer.rows_in, normalizer.rows_dropped, len(frame),
    )
    return frame


def _selftest() -> None:
    """Round-trip synthetic CIC and CTU files and assert the normalisation."""
    import tempfile
    import textwrap

    cic = textwrap.dedent("""\
        Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets,Total Length of Bwd Packets,Flow Bytes/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,FIN Flag Count,SYN Flag Count,RST Flag Count,PSH Flag Count,ACK Flag Count,URG Flag Count,CWE Flag Count,ECE Flag Count, Label
        10.0.0.1,1234,10.0.0.2,80,6,5/7/2017 3:30,2000000,10,4,1000,240,Infinity,500000,1200,900000,10,1,1,0,3,9,0,0,0,BENIGN
        10.0.0.9,4444,10.0.0.2,80,6,5/7/2017 3:31,0,2,0,0,0,Infinity,0,0,0,0,0,2,0,0,0,0,0,0,DoS Hulk
        Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets,Total Length of Bwd Packets,Flow Bytes/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,FIN Flag Count,SYN Flag Count,RST Flag Count,PSH Flag Count,ACK Flag Count,URG Flag Count,CWE Flag Count,ECE Flag Count, Label
        10.0.0.3,53,10.0.0.4,5353,17,5/7/2017 3:32,1000000,3,3,300,300,600,200000,50,300000,100,0,0,0,0,0,0,0,0,BENIGN
    """)

    ctu = textwrap.dedent("""\
        StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label
        2011/08/10 09:46:53.047277,4.0,tcp,147.32.84.165,1025,->,74.125.232.195,443,FSPA_FSPA,0,0,10,1500,900,flow=From-Botnet-V42-TCP
        2011/08/10 09:46:54.000000,2.0,udp,147.32.84.165,53,<->,8.8.8.8,53,CON,0,0,4,400,200,flow=Background-UDP
    """)

    with tempfile.TemporaryDirectory() as tmp:
        cic_path = os.path.join(tmp, "cic.csv")
        ctu_path = os.path.join(tmp, "ctu.binetflow")
        for target, text in ((cic_path, cic), (ctu_path, ctu)):
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(text)

        df = parse_flow_csv(cic_path)
        # The repeated header row in the middle is dropped, not parsed as a flow.
        assert len(df) == 3, f"expected 3 CIC flows, got {len(df)}"
        assert list(df.columns) == list(OUTPUT_COLUMNS)

        tcp = df.iloc[0]
        assert tcp.protocol == "TCP", tcp.protocol           # numeric 6 decoded
        assert tcp.duration == 2.0, tcp.duration             # microseconds -> seconds
        assert tcp.flow_iat_mean == 0.5, tcp.flow_iat_mean
        assert tcp.packet_count == 14 and tcp.byte_count == 1240
        assert tcp.bytes_per_second == 620.0, tcp.bytes_per_second
        assert tcp.flag_syn == 1 and tcp.flag_ack == 9
        assert tcp.is_attack == 0
        # Both directions sort into one key regardless of orientation.
        assert tcp.flow_key == "10.0.0.1|1234|10.0.0.2|80|TCP", tcp.flow_key

        zero = df.iloc[1]
        # Zero duration: CIC writes Infinity, we write NaN.
        assert np.isnan(zero.bytes_per_second), zero.bytes_per_second
        assert zero.is_attack == 1, "named CIC attack must classify as attack"
        assert np.isnan(zero.fwd_bwd_packet_ratio)           # no backward packets

        udp = df.iloc[2]
        assert udp.protocol == "UDP" and udp.is_attack == 0
        assert udp.flow_key == "10.0.0.3|53|10.0.0.4|5353|UDP", udp.flow_key

        ctu_df = parse_flow_csv(ctu_path)
        assert len(ctu_df) == 2, len(ctu_df)
        assert list(ctu_df.columns) == list(OUTPUT_COLUMNS)

        bot = ctu_df.iloc[0]
        assert bot.protocol == "TCP" and bot.duration == 4.0  # already seconds
        assert bot.fwd_bytes == 900 and bot.bwd_bytes == 600  # remainder derived
        assert bot.byte_count == 1500 and bot.packet_count == 10
        assert bot.flag_syn == 2 and bot.flag_psh == 2 and bot.flag_rst == 0
        assert bot.is_attack == 1, "Botnet label must classify as attack"
        assert np.isnan(bot.flow_iat_mean), "CTU has no IAT stats"

        dns = ctu_df.iloc[1]
        # 'CON' is not a TCP flag string -> flags stay unknown, not zero.
        assert np.isnan(dns.flag_syn), dns.flag_syn
        assert dns.is_attack == 0

    print("flow_parser selftest: OK")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalise CIC-IDS / CTU-13 flow CSVs into one feature schema.")
    parser.add_argument("csv", nargs="?", help="path to a CSV / .binetflow file")
    parser.add_argument("-o", "--output", help="write the feature table to this CSV")
    parser.add_argument("--format", choices=sorted(_TIME_SCALE), default=None,
                        help="override header auto-detection")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dayfirst", action="store_true",
                        help="timestamps are d/m/y (CSE-CIC-IDS2018) not m/d/y")
    parser.add_argument("--selftest", action="store_true", help="run built-in checks")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.selftest:
        _selftest()
        return 0
    if not args.csv:
        parser.error("a CSV path is required unless --selftest is given")

    try:
        df = parse_flow_csv(args.csv, source_format=args.format,
                            max_rows=args.max_rows, dayfirst=args.dayfirst)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} flows to {args.output}")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.head(20))
            print(f"\n{len(df)} flows, {len(df.columns)} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())

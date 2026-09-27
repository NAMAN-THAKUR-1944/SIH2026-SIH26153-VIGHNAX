"""Per-flow feature extraction from .pcap / .pcapng captures.

Flows are keyed by the *bidirectional* 5-tuple (ip_a, port_a, ip_b, port_b,
protocol): both directions of a conversation collapse into a single flow, with
per-direction counters kept separately.

Features extracted per flow:
  * TTL statistics (mean / variance / min / max)  -> OS + route fingerprint
  * TCP window size statistics                    -> stack + congestion fingerprint
  * payload size distribution (moments + histogram)
  * TCP retransmission counts                     -> loss / evasion signal

Usage:
    from src.data.pcap_parser import parse_pcap
    df = parse_pcap("capture.pcap")

    python -m src.data.pcap_parser capture.pcap -o flows.csv
    python -m src.data.pcap_parser --selftest
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from scapy.error import Scapy_Exception
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.utils import PcapReader
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "scapy is required by pcap_parser; run: pip install -r requirements.txt"
    ) from exc

LOGGER = logging.getLogger(__name__)

# Upper edges of the payload-size buckets. Last bucket is "MTU-sized or larger".
_PAYLOAD_BIN_EDGES = [0, 1, 64, 256, 512, 1024, 1460, np.inf]
_PAYLOAD_BIN_LABELS = [
    "zero", "1_63", "64_255", "256_511", "512_1023", "1024_1459", "1460_plus",
]

# TCP flag bits we care about: SYN and FIN each consume one sequence number,
# so a repeat of either is a retransmission even with an empty payload.
_TCP_SYN = 0x02
_TCP_FIN = 0x01

FlowKey = Tuple[str, int, str, int, str]


def _canonical_key(
    src: str, sport: int, dst: str, dport: int, proto: str
) -> Tuple[FlowKey, bool]:
    """Return (canonical flow key, is_forward).

    Endpoints are sorted so that packets travelling either way produce the same
    key. "Forward" means the packet ran src->dst in the canonical ordering,
    i.e. it came from whichever endpoint sorts first.
    """
    if (src, sport) <= (dst, dport):
        return (src, sport, dst, dport, proto), True
    return (dst, dport, src, sport, proto), False


@dataclass
class _FlowState:
    """Mutable accumulator for one flow while the capture is being read."""

    key: FlowKey
    start_time: float
    end_time: float
    ttls: List[int] = field(default_factory=list)
    windows: List[int] = field(default_factory=list)
    payloads: List[int] = field(default_factory=list)
    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    zero_window: int = 0
    fwd_retrans: int = 0
    bwd_retrans: int = 0
    # (seq, payload_len) signatures already seen, per direction.
    # ponytail: unbounded per flow; swap for a bounded LRU if you feed it
    # multi-GB captures. Fine for prototype-sized pcaps.
    seen_segments: Dict[bool, Set[Tuple[int, int]]] = field(
        default_factory=lambda: {True: set(), False: set()}
    )


def _payload_len(ip_layer, l4) -> int:
    """Bytes of L4 payload, computed from header fields where possible.

    ``len(layer.payload)`` is unreliable: Ethernet pads short frames, and the
    padding shows up as payload. Header arithmetic avoids that, with a fallback
    for crafted or truncated packets whose length fields are unset.
    """
    if isinstance(l4, TCP):
        if isinstance(ip_layer, IP):
            total = int(ip_layer.len or 0) - int(ip_layer.ihl or 5) * 4
        else:
            # ponytail: ignores IPv6 extension headers; they are rare in the
            # datasets we train on. Walk the header chain if that changes.
            total = int(ip_layer.plen or 0)
        if total > 0:
            return max(0, total - int(l4.dataofs or 5) * 4)
    elif isinstance(l4, UDP):
        length = int(l4.len or 0)
        if length > 0:
            return max(0, length - 8)  # UDP header is 8 bytes
    try:
        return len(bytes(l4.payload))
    except Exception:  # malformed layer
        return 0


def _moments(values: List[int], prefix: str) -> Dict[str, float]:
    """mean / variance / std / min / max / quartiles, NaN-filled when empty.

    Every flow gets the same keys so the resulting DataFrame has a stable
    schema regardless of which protocols appeared.
    """
    keys = ["mean", "variance", "std", "min", "max", "p25", "p50", "p75"]
    if not values:
        return {f"{prefix}_{k}": np.nan for k in keys}
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_variance": float(arr.var()),  # population variance
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
    }


def _payload_histogram(payloads: List[int]) -> Dict[str, float]:
    """Fraction of packets falling in each payload-size bucket."""
    if not payloads:
        return {f"payload_bin_{label}": np.nan for label in _PAYLOAD_BIN_LABELS}
    counts, _ = np.histogram(np.asarray(payloads, dtype=float), bins=_PAYLOAD_BIN_EDGES)
    total = counts.sum() or 1
    return {
        f"payload_bin_{label}": float(count) / float(total)
        for label, count in zip(_PAYLOAD_BIN_LABELS, counts)
    }


class PcapFlowExtractor:
    """Accumulates packets into flows and emits a feature table.

    Kept separate from :func:`parse_pcap` so the same accumulator can be fed
    from a live sniffer later without touching this file.
    """

    def __init__(self) -> None:
        self.flows: Dict[FlowKey, _FlowState] = {}
        self.packets_seen = 0
        self.packets_skipped = 0

    def process_packet(self, pkt) -> None:
        """Fold one scapy packet into its flow. Non-IP packets are skipped."""
        try:
            if IP in pkt:
                ip_layer, ttl = pkt[IP], int(pkt[IP].ttl)
            elif IPv6 in pkt:
                ip_layer, ttl = pkt[IPv6], int(pkt[IPv6].hlim)
            else:
                self.packets_skipped += 1
                return

            src, dst = str(ip_layer.src), str(ip_layer.dst)

            if TCP in pkt:
                l4, proto = pkt[TCP], "TCP"
                sport, dport = int(l4.sport), int(l4.dport)
            elif UDP in pkt:
                l4, proto = pkt[UDP], "UDP"
                sport, dport = int(l4.sport), int(l4.dport)
            else:
                # Non-TCP/UDP (ICMP, etc.): one flow per IP pair, ports 0.
                l4, proto, sport, dport = None, "OTHER", 0, 0

            key, is_fwd = _canonical_key(src, sport, dst, dport, proto)
            timestamp = float(pkt.time)
            payload = _payload_len(ip_layer, l4) if l4 is not None else len(bytes(ip_layer.payload))

            state = self.flows.get(key)
            if state is None:
                state = _FlowState(key=key, start_time=timestamp, end_time=timestamp)
                self.flows[key] = state
            # Captures are not guaranteed monotonic; clamp rather than assume.
            state.start_time = min(state.start_time, timestamp)
            state.end_time = max(state.end_time, timestamp)

            state.ttls.append(ttl)
            state.payloads.append(payload)
            if is_fwd:
                state.fwd_packets += 1
                state.fwd_bytes += payload
            else:
                state.bwd_packets += 1
                state.bwd_bytes += payload

            if proto == "TCP":
                window = int(l4.window)
                state.windows.append(window)
                if window == 0:
                    state.zero_window += 1
                self._count_retransmission(state, l4, payload, is_fwd)

            self.packets_seen += 1
        except Exception as exc:  # one bad packet must not kill the capture
            self.packets_skipped += 1
            LOGGER.debug("Skipping malformed packet: %s", exc)

    @staticmethod
    def _count_retransmission(state: _FlowState, tcp, payload: int, is_fwd: bool) -> None:
        """Flag a segment whose (seq, length) was already seen in this direction.

        ponytail: heuristic. It cannot tell a true retransmission from a
        reordered segment or a TCP keepalive. Upgrade to expected-seq tracking
        with a SACK-aware model if false positives hurt the classifier.
        """
        flags = int(tcp.flags)
        consumes_sequence = payload > 0 or (flags & (_TCP_SYN | _TCP_FIN))
        if not consumes_sequence:
            return
        signature = (int(tcp.seq), payload)
        seen = state.seen_segments[is_fwd]
        if signature in seen:
            if is_fwd:
                state.fwd_retrans += 1
            else:
                state.bwd_retrans += 1
        else:
            seen.add(signature)

    def _finalize(self, state: _FlowState) -> Dict[str, float]:
        packet_count = state.fwd_packets + state.bwd_packets
        retrans = state.fwd_retrans + state.bwd_retrans
        record: Dict[str, float] = {
            "src_ip": state.key[0],
            "src_port": state.key[1],
            "dst_ip": state.key[2],
            "dst_port": state.key[3],
            "protocol": state.key[4],
            "start_time": state.start_time,
            "end_time": state.end_time,
            "duration": state.end_time - state.start_time,
            "packet_count": packet_count,
            "fwd_packet_count": state.fwd_packets,
            "bwd_packet_count": state.bwd_packets,
            "fwd_payload_bytes": state.fwd_bytes,
            "bwd_payload_bytes": state.bwd_bytes,
            "payload_total_bytes": state.fwd_bytes + state.bwd_bytes,
        }
        record.update(_moments(state.ttls, "ttl"))
        record.update(_moments(state.windows, "tcp_window"))
        record["tcp_window_zero_count"] = state.zero_window
        record.update(_moments(state.payloads, "payload"))
        record.update(_payload_histogram(state.payloads))
        record["retransmission_count"] = retrans
        record["fwd_retransmission_count"] = state.fwd_retrans
        record["bwd_retransmission_count"] = state.bwd_retrans
        record["retransmission_ratio"] = retrans / packet_count if packet_count else np.nan
        return record

    def to_dataframe(self) -> pd.DataFrame:
        """One row per flow. Empty input yields an empty DataFrame, not an error."""
        if not self.flows:
            return pd.DataFrame()
        return pd.DataFrame([self._finalize(s) for s in self.flows.values()])


def parse_pcap(path: str, max_packets: Optional[int] = None) -> pd.DataFrame:
    """Parse ``path`` and return one feature row per bidirectional flow.

    Args:
        path: .pcap or .pcapng file.
        max_packets: stop after this many packets (useful on huge captures).

    Raises:
        FileNotFoundError: the capture does not exist.
        ValueError: the file is not a readable capture.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Capture file not found: {path}")

    extractor = PcapFlowExtractor()
    try:
        # PcapReader streams packet-by-packet; rdpcap() would load the whole
        # capture into RAM.
        with PcapReader(path) as reader:
            for packet in reader:
                extractor.process_packet(packet)
                if max_packets is not None and extractor.packets_seen >= max_packets:
                    LOGGER.info("Stopped after %d packets (max_packets)", max_packets)
                    break
    except Scapy_Exception as exc:
        raise ValueError(f"Not a readable capture file: {path} ({exc})") from exc
    except (OSError, EOFError) as exc:
        raise ValueError(f"Failed reading capture {path}: {exc}") from exc

    LOGGER.info(
        "Parsed %s: %d packets, %d skipped, %d flows",
        path, extractor.packets_seen, extractor.packets_skipped, len(extractor.flows),
    )
    return extractor.to_dataframe()


def _selftest() -> None:
    """Round-trip a synthetic capture and assert the features come out right."""
    import tempfile
    from scapy.packet import Raw
    from scapy.utils import wrpcap

    def seg(src, dst, sport, dport, seq, ttl, window, size):
        pkt = (
            IP(src=src, dst=dst, ttl=ttl)
            / TCP(sport=sport, dport=dport, seq=seq, flags="A", window=window)
            / Raw(b"x" * size)
        )
        return IP(bytes(pkt))  # rebuild so ip.len / dataofs are populated

    a, b = "10.0.0.1", "10.0.0.2"
    packets = [
        seg(a, b, 1234, 80, seq=1, ttl=64, window=8192, size=100),
        seg(a, b, 1234, 80, seq=101, ttl=64, window=8192, size=200),
        seg(a, b, 1234, 80, seq=101, ttl=60, window=0, size=200),  # retransmission
        seg(b, a, 80, 1234, seq=1, ttl=128, window=4096, size=0),  # reverse direction
        IP(bytes(IP(src=a, dst=b, ttl=64) / UDP(sport=53, dport=5353) / Raw(b"y" * 40))),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "selftest.pcap")
        wrpcap(path, packets)
        df = parse_pcap(path)

        assert len(df) == 2, f"expected 2 flows (TCP + UDP), got {len(df)}"

        tcp = df[df.protocol == "TCP"].iloc[0]
        # Both directions collapsed into one flow.
        assert tcp.packet_count == 4, tcp.packet_count
        assert tcp.fwd_packet_count == 3 and tcp.bwd_packet_count == 1
        # Duplicate (seq=101, len=200) caught exactly once.
        assert tcp.retransmission_count == 1, tcp.retransmission_count
        assert tcp.fwd_retransmission_count == 1 and tcp.bwd_retransmission_count == 0
        # TTLs 64/64/60/128 -> real spread.
        assert tcp.ttl_min == 60 and tcp.ttl_max == 128
        assert tcp.ttl_variance > 0, tcp.ttl_variance
        assert tcp.tcp_window_min == 0 and tcp.tcp_window_max == 8192
        assert tcp.tcp_window_zero_count == 1
        # Payloads 100/200/200/0 -> header arithmetic, not padded frame length.
        assert tcp.payload_total_bytes == 500, tcp.payload_total_bytes
        assert tcp.payload_bin_zero == 0.25, tcp.payload_bin_zero
        assert tcp.payload_bin_64_255 == 0.75

        udp = df[df.protocol == "UDP"].iloc[0]
        assert udp.packet_count == 1
        assert udp.payload_total_bytes == 40, udp.payload_total_bytes
        # No TCP packets in this flow -> window stats are NaN, not zero.
        assert np.isnan(udp.tcp_window_mean)

    print("pcap_parser selftest: OK")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract per-flow features from a pcap.")
    parser.add_argument("pcap", nargs="?", help="path to a .pcap / .pcapng file")
    parser.add_argument("-o", "--output", help="write the feature table to this CSV")
    parser.add_argument("--max-packets", type=int, default=None)
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
    if not args.pcap:
        parser.error("a pcap path is required unless --selftest is given")

    try:
        df = parse_pcap(args.pcap, max_packets=args.max_packets)
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

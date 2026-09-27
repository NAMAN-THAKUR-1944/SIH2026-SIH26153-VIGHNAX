"""Fast, streaming packet-level feature extraction per time window.

:mod:`src.data.pcap_parser` (scapy) is the readable reference implementation,
but scapy dissects ~5k packets/s - far too slow for the tens of millions of
packets in a single CTU-13 capture. This module reads records with ``dpkt`` and
decodes the few header fields we need with ``struct`` directly (~100k+ pkt/s),
aggregating straight into fixed time windows so memory stays bounded.

Packet-level features per window (the PS's packet-level list):
  * TTL mean / std and mean per-session TTL variance       -> OS / route fingerprint, spoofing
  * TCP window size mean / std, zero-window fraction          -> stack fingerprint, stalls
  * IP fragment fraction (MF flag or offset > 0), DF fraction  -> evasion, fragmentation attacks
  * payload-size distribution (7-bin histogram + mean / std)
  * port-scan signatures: max distinct dst ports / hosts probed by one source,
    sequential-port ratio of the top prober (sequential vs randomised scans)
  * retransmission / out-of-order fraction
  * per-flow inter-arrival time mean / std / max               -> timing, beaconing
  * SYN-only, RST, ICMP, ICMP-unreachable fractions

Optionally also emits one row per bidirectional flow (for PCAP-only inputs),
in the same column vocabulary as :mod:`src.data.flow_parser`.
"""

from __future__ import annotations

import bz2
import gzip
import logging
import math
import os
import struct
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PAYLOAD_BIN_EDGES = (0, 1, 64, 256, 512, 1024, 1460)  # left edges; last bin is 1460+
PAYLOAD_BIN_NAMES = ("zero", "1_63", "64_255", "256_511", "512_1023", "1024_1459", "1460_plus")

PACKET_FEATURES: Tuple[str, ...] = (
    "pk_count_log", "pk_ttl_mean", "pk_ttl_std", "pk_ttl_session_var",
    "pk_tcp_win_mean_log", "pk_tcp_win_std_log", "pk_tcp_zero_win_frac",
    "pk_frag_frac", "pk_df_frac",
    *(f"pk_payload_{n}" for n in PAYLOAD_BIN_NAMES), "pk_payload_mean_log", "pk_payload_std_log",
    "pk_retrans_frac", "pk_iat_mean_log", "pk_iat_std_log", "pk_iat_max_log",
    "pk_syn_only_frac", "pk_rst_frac", "pk_icmp_frac", "pk_icmp_unreach_frac", "pk_udp_frac",
    "pk_scan_max_ports_log", "pk_scan_max_hosts_log", "pk_scan_seq_ratio",
)

_ETH_IP4, _ETH_IP6, _ETH_VLAN, _ETH_QINQ = 0x0800, 0x86DD, 0x8100, 0x88A8
_DLT_EN10MB, _DLT_RAW, _DLT_LINUX_SLL, _DLT_RAW2 = 1, 101, 113, 12


def _open(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rb")
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _records(path: str) -> Iterator[Tuple[float, bytes, int]]:
    """Yield (timestamp, raw frame, linktype) from pcap or pcapng (optionally compressed)."""
    import dpkt  # local import: only needed for packet ingestion

    fh = _open(path)
    try:
        magic = fh.peek(4)[:4] if hasattr(fh, "peek") else b""
        if magic == b"\x0a\x0d\x0d\x0a":
            reader = dpkt.pcapng.Reader(fh)
        else:
            reader = dpkt.pcap.Reader(fh)
        linktype = reader.datalink()
        try:
            for ts, buf in reader:
                yield float(ts), buf, linktype
        except (dpkt.NeedData, struct.error, EOFError, ValueError) as exc:
            # Truncated tail (common with range-downloads or aborted captures).
            LOGGER.warning("Stopped reading %s early: %s", path, exc)
    finally:
        fh.close()


def _l3_offset(buf: bytes, linktype: int) -> Tuple[int, int]:
    """Return (offset of L3 header, ethertype) or (-1, 0) if not IP."""
    if linktype == _DLT_EN10MB:
        if len(buf) < 14:
            return -1, 0
        off, etype = 14, (buf[12] << 8) | buf[13]
        while etype in (_ETH_VLAN, _ETH_QINQ) and len(buf) >= off + 4:
            etype = (buf[off + 2] << 8) | buf[off + 3]
            off += 4
        return off, etype
    if linktype in (_DLT_RAW, _DLT_RAW2):
        if not buf:
            return -1, 0
        return 0, _ETH_IP4 if (buf[0] >> 4) == 4 else _ETH_IP6
    if linktype == _DLT_LINUX_SLL:
        if len(buf) < 16:
            return -1, 0
        return 16, (buf[14] << 8) | buf[15]
    return -1, 0


class _Window:
    __slots__ = ("n", "n_ip", "ttl_s", "ttl_ss", "flow_ttl", "n_tcp", "win_s", "win_ss", "zero_win",
                 "frag", "df", "pay_bins", "pay_s", "pay_ss", "retrans", "n_pay", "iat_s", "iat_ss",
                 "iat_max", "n_iat", "syn_only", "rst", "icmp", "icmp_unreach", "udp", "probe_ports",
                 "probe_hosts")

    def __init__(self) -> None:
        self.n = self.n_ip = self.n_tcp = self.zero_win = self.frag = self.df = 0
        self.ttl_s = self.ttl_ss = self.win_s = self.win_ss = self.pay_s = self.pay_ss = 0.0
        self.flow_ttl: Dict[tuple, List[float]] = {}
        self.pay_bins = [0] * len(PAYLOAD_BIN_EDGES)
        self.retrans = self.n_pay = self.n_iat = 0
        self.iat_s = self.iat_ss = self.iat_max = 0.0
        self.syn_only = self.rst = self.icmp = self.icmp_unreach = self.udp = 0
        self.probe_ports: Dict[str, set] = defaultdict(set)
        self.probe_hosts: Dict[str, set] = defaultdict(set)

    def finalize(self) -> Dict[str, float]:
        n_ip = max(self.n_ip, 1)
        ttl_mean = self.ttl_s / n_ip
        ttl_var = max(self.ttl_ss / n_ip - ttl_mean ** 2, 0.0)
        sess_vars = [max(ss / c - (s / c) ** 2, 0.0) for c, s, ss in self.flow_ttl.values() if c > 1]
        n_tcp = max(self.n_tcp, 1)
        win_mean = self.win_s / n_tcp
        win_var = max(self.win_ss / n_tcp - win_mean ** 2, 0.0)
        pay_mean = self.pay_s / n_ip
        pay_var = max(self.pay_ss / n_ip - pay_mean ** 2, 0.0)
        n_iat = max(self.n_iat, 1)
        iat_mean = self.iat_s / n_iat
        iat_var = max(self.iat_ss / n_iat - iat_mean ** 2, 0.0)
        max_ports, max_hosts, seq_ratio = 0, 0, 0.0
        if self.probe_ports:
            top = max(self.probe_ports, key=lambda s: len(self.probe_ports[s]))
            ports = sorted(self.probe_ports[top])
            max_ports = len(ports)
            if len(ports) > 2:
                diffs = np.diff(ports)
                seq_ratio = float(np.mean(diffs == 1))
            max_hosts = max(len(h) for h in self.probe_hosts.values())
        rec = {
            "pk_count_log": math.log1p(self.n),
            "pk_ttl_mean": ttl_mean / 255.0,
            "pk_ttl_std": math.sqrt(ttl_var) / 64.0,
            "pk_ttl_session_var": math.log1p(float(np.mean(sess_vars)) if sess_vars else 0.0),
            "pk_tcp_win_mean_log": math.log1p(win_mean),
            "pk_tcp_win_std_log": math.log1p(math.sqrt(win_var)),
            "pk_tcp_zero_win_frac": self.zero_win / n_tcp,
            "pk_frag_frac": self.frag / n_ip,
            "pk_df_frac": self.df / n_ip,
            "pk_payload_mean_log": math.log1p(pay_mean),
            "pk_payload_std_log": math.log1p(math.sqrt(pay_var)),
            "pk_retrans_frac": self.retrans / max(self.n_pay, 1),
            "pk_iat_mean_log": math.log1p(iat_mean * 1e3),
            "pk_iat_std_log": math.log1p(math.sqrt(iat_var) * 1e3),
            "pk_iat_max_log": math.log1p(self.iat_max * 1e3),
            "pk_syn_only_frac": self.syn_only / n_tcp,
            "pk_rst_frac": self.rst / n_tcp,
            "pk_icmp_frac": self.icmp / n_ip,
            "pk_icmp_unreach_frac": self.icmp_unreach / n_ip,
            "pk_udp_frac": self.udp / n_ip,
            "pk_scan_max_ports_log": math.log1p(max_ports),
            "pk_scan_max_hosts_log": math.log1p(max_hosts),
            "pk_scan_seq_ratio": seq_ratio,
        }
        for name, count in zip(PAYLOAD_BIN_NAMES, self.pay_bins):
            rec[f"pk_payload_{name}"] = count / n_ip
        return rec


def _payload_bin(size: int) -> int:
    for i in range(len(PAYLOAD_BIN_EDGES) - 1, -1, -1):
        if size >= PAYLOAD_BIN_EDGES[i]:
            return i
    return 0


def extract_packet_windows(
    path: str,
    window_s: float = 10.0,
    time_offset_s: float = 0.0,
    max_packets: Optional[int] = None,
    with_flows: bool = False,
    flow_idle_s: float = 60.0,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Stream a capture and return (per-window packet features, optional flows).

    Window ids are ``floor((ts + time_offset_s) / window_s)`` on absolute epoch
    time, so they line up with :mod:`src.data.state_builder` flow windows.
    ``time_offset_s`` corrects capture-vs-flow clock/timezone differences.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Capture file not found: {path}")

    unpack_ip4 = struct.Struct("!BBHHHBBH4s4s").unpack_from
    windows: Dict[int, _Window] = {}
    cur_id: Optional[int] = None
    cur: Optional[_Window] = None
    last_ts: Dict[tuple, float] = {}       # per-flow last packet time (IAT)
    max_seq: Dict[tuple, int] = {}         # per-direction highest seq end (retransmissions)
    flows: Dict[tuple, list] = {}          # optional flow table (active records)
    done_flows: List[list] = []            # records closed by the idle timeout
    seen = 0

    for ts, buf, linktype in _records(path):
        ts += time_offset_s
        wid = int(ts // window_s)
        if wid != cur_id:
            cur = windows.get(wid)
            if cur is None:
                cur = windows[wid] = _Window()
            cur_id = wid
            if len(last_ts) > 400_000:  # prune idle flow state to bound memory
                cutoff = ts - flow_idle_s
                for k in [k for k, v in last_ts.items() if v < cutoff]:
                    del last_ts[k]
            if len(max_seq) > 400_000:
                max_seq.clear()  # loses retransmission memory for long-idle flows only
        w = cur
        w.n += 1
        seen += 1
        if max_packets is not None and seen >= max_packets:
            break

        off, etype = _l3_offset(buf, linktype)
        if off < 0:
            continue
        if etype == _ETH_IP4:
            if len(buf) < off + 20:
                continue
            vihl, _tos, tot_len, _ident, frag_field, ttl, proto, _csum, src_b, dst_b = unpack_ip4(buf, off)
            ihl = (vihl & 0x0F) * 4
            if (frag_field & 0x2000) or (frag_field & 0x1FFF):
                w.frag += 1
            if frag_field & 0x4000:
                w.df += 1
            src = "%d.%d.%d.%d" % tuple(src_b)
            dst = "%d.%d.%d.%d" % tuple(dst_b)
            l4 = off + ihl
            l3_payload = tot_len - ihl
            ip_bytes = tot_len
            first_frag = (frag_field & 0x1FFF) == 0
        elif etype == _ETH_IP6:
            if len(buf) < off + 40:
                continue
            plen = (buf[off + 4] << 8) | buf[off + 5]
            proto, ttl = buf[off + 6], buf[off + 7]
            src, dst = buf[off + 8:off + 24].hex(), buf[off + 24:off + 40].hex()
            l4, l3_payload, first_frag = off + 40, plen, True
            ip_bytes = plen + 40
        else:
            continue

        w.n_ip += 1
        w.ttl_s += ttl
        w.ttl_ss += ttl * ttl

        sport = dport = 0
        payload = max(l3_payload, 0)
        flags = 0
        if proto == 6 and first_frag and len(buf) >= l4 + 16:
            sport = (buf[l4] << 8) | buf[l4 + 1]
            dport = (buf[l4 + 2] << 8) | buf[l4 + 3]
            seq = struct.unpack_from("!I", buf, l4 + 4)[0]
            data_off = (buf[l4 + 12] >> 4) * 4
            flags = buf[l4 + 13]
            win = (buf[l4 + 14] << 8) | buf[l4 + 15]
            payload = max(l3_payload - data_off, 0)
            w.n_tcp += 1
            w.win_s += win
            w.win_ss += win * win
            if win == 0 and not (flags & 0x04):
                w.zero_win += 1
            syn, ack = flags & 0x02, flags & 0x10
            if syn and not ack:
                w.syn_only += 1
                w.probe_ports[src].add(dport)
                w.probe_hosts[src].add(dst)
            if flags & 0x04:
                w.rst += 1
            if payload > 0:
                w.n_pay += 1
                dkey = (src, sport, dst, dport, True)
                end = seq + payload
                prev = max_seq.get(dkey)
                if prev is not None and end <= prev and prev - end < (1 << 30):
                    w.retrans += 1
                else:
                    max_seq[dkey] = end
        elif proto == 17 and first_frag and len(buf) >= l4 + 8:
            sport = (buf[l4] << 8) | buf[l4 + 1]
            dport = (buf[l4 + 2] << 8) | buf[l4 + 3]
            payload = max(l3_payload - 8, 0)
            w.udp += 1
            if payload <= 64 and dport < 1024:  # small UDP to service ports = probe
                w.probe_ports[src].add(dport)
                w.probe_hosts[src].add(dst)
        elif proto in (1, 58):
            w.icmp += 1
            if len(buf) > l4 and buf[l4] in (3, 1):  # dest-unreachable (v4 type 3 / v6 type 1)
                w.icmp_unreach += 1
            payload = max(l3_payload - 8, 0)

        w.pay_s += payload
        w.pay_ss += payload * payload
        w.pay_bins[_payload_bin(payload)] += 1

        # TTL variance is per direction (each sender has its own initial TTL);
        # IAT and the flow table use the direction-independent session key.
        tkey = (src, sport, dst, dport, proto)
        a, b = (src, sport), (dst, dport)
        skey = (a + b + (proto,)) if a <= b else (b + a + (proto,))
        st = w.flow_ttl.get(tkey)
        if st is None:
            w.flow_ttl[tkey] = [1, ttl, ttl * ttl]
        else:
            st[0] += 1; st[1] += ttl; st[2] += ttl * ttl
        prev_ts = last_ts.get(skey)
        if prev_ts is not None:
            gap = ts - prev_ts
            if gap >= 0:
                w.iat_s += gap; w.iat_ss += gap * gap; w.n_iat += 1
                if gap > w.iat_max:
                    w.iat_max = gap
        last_ts[skey] = ts

        if with_flows:
            rec = flows.get(skey)
            if rec is not None and ts - rec[6] > flow_idle_s:
                done_flows.append(rec)   # idle timeout: exporters start a new record, as Argus/NetFlow do
                rec = None
            if rec is None:
                # initiator = first packet's sender
                rec = flows[skey] = [src, sport, dst, dport, proto, ts, ts, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            rec[6] = ts
            fwd = src == rec[0] and sport == rec[1]
            if fwd:  # IP-layer bytes, matching NetFlow / Argus byte counters
                rec[7] += 1; rec[9] += ip_bytes
            else:
                rec[8] += 1; rec[10] += ip_bytes
            if proto == 6:
                rec[11] += bool(flags & 0x01); rec[12] += bool(flags & 0x02); rec[13] += bool(flags & 0x04)
                rec[14] += bool(flags & 0x08); rec[15] += bool(flags & 0x10); rec[16] += bool(flags & 0x20)

    LOGGER.info("Read %d packets from %s into %d windows", seen, path, len(windows))
    if not windows:
        return pd.DataFrame(columns=["window_id", *PACKET_FEATURES]), None
    ids = sorted(windows)
    pk = pd.DataFrame([windows[i].finalize() for i in ids], columns=list(PACKET_FEATURES))
    pk.insert(0, "window_id", np.array(ids, dtype=np.int64))

    flow_df = None
    if with_flows:
        cols = ["src_ip", "src_port", "dst_ip", "dst_port", "proto_num", "start_time", "end_time",
                "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
                "flag_fin", "flag_syn", "flag_rst", "flag_psh", "flag_ack", "flag_urg"]
        flow_df = pd.DataFrame(done_flows + list(flows.values()), columns=cols)
        flow_df["protocol"] = flow_df.pop("proto_num").map({6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPV6"}).fillna("OTHER")
        flow_df["duration"] = flow_df["end_time"] - flow_df["start_time"]
        flow_df["packet_count"] = flow_df["fwd_packets"] + flow_df["bwd_packets"]
        flow_df["byte_count"] = flow_df["fwd_bytes"] + flow_df["bwd_bytes"]
        flow_df["label"] = np.nan
        flow_df["is_attack"] = np.nan
    return pk, flow_df


def _selftest() -> None:
    """Write a tiny synthetic pcap with dpkt and check the aggregates."""
    import tempfile
    import dpkt

    def tcp_pkt(src, dst, sport, dport, flags, seq=1000, payload=b"", ttl=64, win=8192):
        tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, seq=seq, win=win, data=payload)
        ip = dpkt.ip.IP(src=bytes(map(int, src.split("."))), dst=bytes(map(int, dst.split("."))),
                        p=6, ttl=ttl, data=tcp)
        ip.len = 20 + len(bytes(tcp))
        return bytes(dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x11" * 6, type=0x0800, data=ip))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.pcap")
        with open(path, "wb") as fh:
            wr = dpkt.pcap.Writer(fh)
            # window 0: sequential SYN scan of ports 20..29 from 10.0.0.9
            for i, port in enumerate(range(20, 30)):
                wr.writepkt(tcp_pkt("10.0.0.9", "10.0.0.1", 40000, port, 0x02), ts=1.0 + i * 0.1)
            # window 1: data + one retransmission
            wr.writepkt(tcp_pkt("10.0.0.2", "10.0.0.3", 5555, 80, 0x18, seq=1, payload=b"x" * 100), ts=11.0)
            wr.writepkt(tcp_pkt("10.0.0.2", "10.0.0.3", 5555, 80, 0x18, seq=1, payload=b"x" * 100), ts=11.5)
        pk, flows = extract_packet_windows(path, window_s=10.0, with_flows=True)
    assert list(pk["window_id"]) == [0, 1], pk["window_id"].tolist()
    w0, w1 = pk.iloc[0], pk.iloc[1]
    assert abs(w0["pk_syn_only_frac"] - 1.0) < 1e-9
    assert abs(w0["pk_scan_max_ports_log"] - math.log1p(10)) < 1e-9
    assert abs(w0["pk_scan_seq_ratio"] - 1.0) < 1e-9
    assert abs(w1["pk_retrans_frac"] - 0.5) < 1e-9
    assert abs(w1["pk_payload_64_255"] - 1.0) < 1e-9
    assert len(flows) == 11 and int(flows["flag_syn"].sum()) == 10
    print("packet_features selftest: OK")


if __name__ == "__main__":
    _selftest()

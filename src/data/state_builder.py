"""Network state S_t: fuse flow-level and packet-level telemetry per time window.

Each row of the output is the state of the whole monitored network during one
window of ``window_s`` seconds (absolute epoch-aligned, so flow and packet
sources line up without joins on individual flows):

    S_t = [ flow block (FLOW_FEATURES) | packet block (PACKET_FEATURES) | pk_mask ]

* The flow block summarises flows *starting* in the window (NetFlow/IPFIX-style
  records from CIC-IDS CSVs, CTU-13 binetflow, or flows rebuilt from a PCAP).
* The packet block comes from :mod:`src.data.packet_features`. When no capture
  is available the block is zero and ``pk_mask`` = 0, so one model serves
  CSV-only, PCAP-only and CSV+PCAP inputs.

Labels per window (when the flows carry labels): the MITRE stage (see
:mod:`src.data.stages`), a binary ``is_attack`` (stage != benign) and the
fraction of malicious flows.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data import stages as st
from src.data.packet_features import PACKET_FEATURES

LOGGER = logging.getLogger(__name__)

FLOW_FEATURES = (
    "fl_count_log", "fl_bytes_sum_log", "fl_bytes_mean_log", "fl_bytes_max_log",
    "fl_pkts_sum_log", "fl_pkts_mean_log", "fl_pkts_max_log",
    "fl_dur_mean_log", "fl_dur_max_log", "fl_bytes_per_pkt_log",
    "fl_fwd_byte_share", "fl_unidirectional_frac",
    "fl_tcp_frac", "fl_udp_frac", "fl_icmp_frac",
    "fl_syn_frac", "fl_ack_frac", "fl_fin_frac", "fl_rst_frac", "fl_psh_frac", "fl_urg_frac",
    "fl_syn_no_ack_frac",
    "fl_iat_mean_log", "fl_iat_std_log", "fl_iat_max_log",
    "fl_src_ips_log", "fl_dst_ips_log", "fl_dst_ports_log",
    "fl_max_dports_per_src_log", "fl_max_dsts_per_src_log",
    "fl_port_dns", "fl_port_web", "fl_port_smtp", "fl_port_smb_rpc", "fl_port_irc",
    "fl_port_remote_admin", "fl_port_high", "fl_internal_frac",
    "fl_host_max_flows_log", "fl_host_max_smtp_log", "fl_host_max_failed_log", "fl_host_max_icmp_log",
    "fl_host_max_dns_log", "fl_host_max_bytes_log", "fl_host_max_smtp_dsts_log", "fl_host_max_web_dsts_log",
    "fl_smtp_sources_log",
)
STATE_FEATURES = FLOW_FEATURES + PACKET_FEATURES + ("pk_mask",)
HOST_FEATURES = ("fl_src_ips_log", "fl_dst_ips_log", "fl_max_dports_per_src_log", "fl_max_dsts_per_src_log",
                 "fl_host_max_flows_log", "fl_host_max_smtp_log", "fl_host_max_failed_log", "fl_host_max_icmp_log",
                 "fl_host_max_dns_log", "fl_host_max_bytes_log", "fl_host_max_smtp_dsts_log", "fl_host_max_web_dsts_log",
                 "fl_smtp_sources_log", "fl_internal_frac")

# Human-readable names for the explainability UI.
FEATURE_LABELS = {
    "fl_count_log": "flow count", "fl_bytes_sum_log": "bytes (total)", "fl_bytes_mean_log": "bytes / flow",
    "fl_bytes_max_log": "largest flow (bytes)", "fl_pkts_sum_log": "packets (total)",
    "fl_pkts_mean_log": "packets / flow", "fl_pkts_max_log": "largest flow (pkts)",
    "fl_dur_mean_log": "flow duration (mean)", "fl_dur_max_log": "flow duration (max)",
    "fl_bytes_per_pkt_log": "bytes / packet", "fl_fwd_byte_share": "fwd/bwd byte ratio",
    "fl_unidirectional_frac": "one-way flows", "fl_tcp_frac": "TCP share", "fl_udp_frac": "UDP share",
    "fl_icmp_frac": "ICMP share", "fl_syn_frac": "SYN flag", "fl_ack_frac": "ACK flag",
    "fl_fin_frac": "FIN flag", "fl_rst_frac": "RST flag", "fl_psh_frac": "PSH flag", "fl_urg_frac": "URG flag",
    "fl_syn_no_ack_frac": "SYN without ACK", "fl_iat_mean_log": "flow IAT mean",
    "fl_iat_std_log": "flow IAT std", "fl_iat_max_log": "flow IAT max",
    "fl_src_ips_log": "distinct source IPs", "fl_dst_ips_log": "distinct dest IPs",
    "fl_dst_ports_log": "distinct dest ports", "fl_max_dports_per_src_log": "ports probed by one host",
    "fl_max_dsts_per_src_log": "hosts contacted by one host", "fl_port_dns": "DNS (53)",
    "fl_port_web": "HTTP/S (80/443)", "fl_port_smtp": "SMTP (25/465/587)",
    "fl_port_smb_rpc": "SMB/RPC (135/139/445)", "fl_port_irc": "IRC (6660-6697)",
    "fl_port_remote_admin": "SSH/RDP/Telnet", "fl_port_high": "high ports (>=1024)",
    "fl_internal_frac": "internal-to-internal", "fl_host_max_flows_log": "top host: flows",
    "fl_host_max_smtp_log": "top host: SMTP flows", "fl_host_max_failed_log": "top host: failed connections",
    "fl_host_max_icmp_log": "top host: ICMP flows", "fl_host_max_dns_log": "top host: DNS queries",
    "fl_host_max_bytes_log": "top host: bytes sent", "fl_host_max_smtp_dsts_log": "top host: SMTP servers contacted",
    "fl_host_max_web_dsts_log": "top host: web servers contacted", "fl_smtp_sources_log": "hosts sending SMTP",
    "pk_count_log": "packet count",
    "pk_ttl_mean": "TTL mean", "pk_ttl_std": "TTL std", "pk_ttl_session_var": "TTL variance / session",
    "pk_tcp_win_mean_log": "TCP window mean", "pk_tcp_win_std_log": "TCP window std",
    "pk_tcp_zero_win_frac": "TCP zero window", "pk_frag_frac": "IP fragments", "pk_df_frac": "IP DF flag",
    "pk_payload_zero": "payload 0 B", "pk_payload_1_63": "payload 1-63 B", "pk_payload_64_255": "payload 64-255 B",
    "pk_payload_256_511": "payload 256-511 B", "pk_payload_512_1023": "payload 512-1023 B",
    "pk_payload_1024_1459": "payload 1024-1459 B", "pk_payload_1460_plus": "payload 1460+ B",
    "pk_payload_mean_log": "payload mean", "pk_payload_std_log": "payload std",
    "pk_retrans_frac": "retransmissions", "pk_iat_mean_log": "packet IAT mean", "pk_iat_std_log": "packet IAT std",
    "pk_iat_max_log": "packet IAT max", "pk_syn_only_frac": "SYN-only packets", "pk_rst_frac": "RST packets",
    "pk_icmp_frac": "ICMP packets", "pk_icmp_unreach_frac": "ICMP unreachable", "pk_udp_frac": "UDP packets",
    "pk_scan_max_ports_log": "port-scan: ports / host", "pk_scan_max_hosts_log": "port-scan: hosts / source",
    "pk_scan_seq_ratio": "sequential port scan", "pk_mask": "packet data present",
}

_INTERNAL = ("10.", "192.168.", "147.32.")


def _port_in(ports: pd.Series, values: Sequence[int]) -> pd.Series:
    return ports.isin(values).astype(float)


def _is_internal(ips: pd.Series) -> pd.Series:
    s = ips.astype(str)
    out = s.str.startswith(_INTERNAL)
    return out | s.str.match(r"^172\.(1[6-9]|2\d|3[01])\.")


def flow_window_features(flows: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Aggregate a flow table (flow_parser schema) into per-window flow features."""
    f = flows.dropna(subset=["start_time"]).copy()
    f["window_id"] = (f["start_time"] // window_s).astype(np.int64)
    num = lambda c: pd.to_numeric(f[c], errors="coerce") if c in f else pd.Series(np.nan, index=f.index)
    bytes_ = num("byte_count").fillna(0.0)
    pkts = num("packet_count").fillna(0.0)
    fwd_b, bwd_p = num("fwd_bytes"), num("bwd_packets")
    proto = f["protocol"].astype(str).str.upper()
    dport = num("dst_port").fillna(-1).astype(np.int64)
    is_tcp = proto == "TCP"
    flag = lambda n: (num(f"flag_{n}").fillna(0) > 0).astype(float)

    g = pd.DataFrame({
        "window_id": f["window_id"],
        "bytes": bytes_, "pkts": pkts, "dur": num("duration").fillna(0.0).clip(lower=0),
        "bpp": (bytes_ / pkts.replace(0, np.nan)).fillna(0.0),
        "fwd_share": (fwd_b / bytes_.replace(0, np.nan)).fillna(0.5).clip(0, 1),
        "unidir": (bwd_p.fillna(1) == 0).astype(float),
        "tcp": is_tcp.astype(float), "udp": (proto == "UDP").astype(float),
        "icmp": proto.str.startswith("ICMP").astype(float),
        "syn": flag("syn"), "ack": flag("ack"), "fin": flag("fin"), "rst": flag("rst"),
        "psh": flag("psh"), "urg": flag("urg"),
        "iat_mean": num("flow_iat_mean"), "iat_std": num("flow_iat_std"), "iat_max": num("flow_iat_max"),
        "p_dns": _port_in(dport, [53]), "p_web": _port_in(dport, [80, 443, 8080, 8443]),
        "p_smtp": _port_in(dport, [25, 465, 587]), "p_smb": _port_in(dport, [135, 137, 138, 139, 445]),
        "p_irc": ((dport >= 6660) & (dport <= 6697)).astype(float),
        "p_admin": _port_in(dport, [22, 23, 3389, 5900, 5985]),
        "p_high": (dport >= 1024).astype(float),
        "internal": (_is_internal(f["src_ip"]) & _is_internal(f["dst_ip"])).astype(float),
    })
    g["syn_no_ack"] = ((g["syn"] > 0) & (g["ack"] == 0) & (g["tcp"] > 0)).astype(float)
    grp = g.groupby("window_id", sort=True)
    tcp_n = grp["tcp"].sum().replace(0, np.nan)
    out = pd.DataFrame(index=grp.size().index)
    out["fl_count_log"] = np.log1p(grp.size())
    out["fl_bytes_sum_log"] = np.log1p(grp["bytes"].sum())
    out["fl_bytes_mean_log"] = np.log1p(grp["bytes"].mean())
    out["fl_bytes_max_log"] = np.log1p(grp["bytes"].max())
    out["fl_pkts_sum_log"] = np.log1p(grp["pkts"].sum())
    out["fl_pkts_mean_log"] = np.log1p(grp["pkts"].mean())
    out["fl_pkts_max_log"] = np.log1p(grp["pkts"].max())
    out["fl_dur_mean_log"] = np.log1p(grp["dur"].mean())
    out["fl_dur_max_log"] = np.log1p(grp["dur"].max())
    out["fl_bytes_per_pkt_log"] = np.log1p(grp["bpp"].mean())
    out["fl_fwd_byte_share"] = grp["fwd_share"].mean()
    out["fl_unidirectional_frac"] = grp["unidir"].mean()
    for col, name in (("tcp", "tcp"), ("udp", "udp"), ("icmp", "icmp")):
        out[f"fl_{name}_frac"] = grp[col].mean()
    for fl in ("syn", "ack", "fin", "rst", "psh", "urg"):
        out[f"fl_{fl}_frac"] = (g[fl] * g["tcp"]).groupby(g["window_id"]).sum() / tcp_n
    out["fl_syn_no_ack_frac"] = grp["syn_no_ack"].sum() / tcp_n
    # CICFlowMeter reports IAT; Argus does not -> 0 (missing), as agreed.
    out["fl_iat_mean_log"] = np.log1p(grp["iat_mean"].mean() * 1e3)
    out["fl_iat_std_log"] = np.log1p(grp["iat_std"].mean() * 1e3)
    out["fl_iat_max_log"] = np.log1p(grp["iat_max"].mean() * 1e3)
    keyed = pd.DataFrame({"window_id": f["window_id"], "src": f["src_ip"].astype(str),
                          "dst": f["dst_ip"].astype(str), "dport": dport})
    out["fl_src_ips_log"] = np.log1p(keyed.groupby("window_id")["src"].nunique())
    out["fl_dst_ips_log"] = np.log1p(keyed.groupby("window_id")["dst"].nunique())
    out["fl_dst_ports_log"] = np.log1p(keyed.groupby("window_id")["dport"].nunique())
    per_src = keyed.groupby(["window_id", "src"]).agg(p=("dport", "nunique"), h=("dst", "nunique"))
    out["fl_max_dports_per_src_log"] = np.log1p(per_src["p"].groupby(level=0).max())
    out["fl_max_dsts_per_src_log"] = np.log1p(per_src["h"].groupby(level=0).max())
    for col, name in (("p_dns", "dns"), ("p_web", "web"), ("p_smtp", "smtp"), ("p_smb", "smb_rpc"),
                      ("p_irc", "irc"), ("p_admin", "remote_admin"), ("p_high", "high")):
        out[f"fl_port_{name}"] = grp[col].mean()
    out["fl_internal_frac"] = grp["internal"].mean()

    # Host-level view: network-wide fractions dilute one compromised host among
    # thousands of benign flows, so also record the most extreme single source.
    failed = ((g["syn_no_ack"] > 0) | ((g["tcp"] > 0) & (g["unidir"] > 0))).astype(float)
    host = pd.DataFrame({"window_id": f["window_id"], "src": keyed["src"], "dst": keyed["dst"],
                         "smtp": g["p_smtp"], "failed": failed, "icmp": g["icmp"], "dns": g["p_dns"],
                         "web": g["p_web"], "fwd_bytes": fwd_b.fillna(0.0)})
    hg = host.groupby(["window_id", "src"])
    per_host = pd.DataFrame({"flows": hg.size(), "smtp": hg["smtp"].sum(), "failed": hg["failed"].sum(),
                             "icmp": hg["icmp"].sum(), "dns": hg["dns"].sum(), "bytes": hg["fwd_bytes"].sum()})
    mx = per_host.groupby(level=0).max()
    for col in ("flows", "smtp", "failed", "icmp", "dns", "bytes"):
        out[f"fl_host_max_{col}_log"] = np.log1p(mx[col])
    for col, name in (("smtp", "smtp_dsts"), ("web", "web_dsts")):
        sub = host[host[col] > 0]
        d = sub.groupby(["window_id", "src"])["dst"].nunique().groupby(level=0).max() if len(sub) else pd.Series(dtype=float)
        out[f"fl_host_max_{name}_log"] = np.log1p(d.reindex(out.index).fillna(0.0))
    out["fl_smtp_sources_log"] = np.log1p(host[host["smtp"] > 0].groupby("window_id")["src"].nunique()
                                          .reindex(out.index).fillna(0.0))
    if not f["src_ip"].notna().any() or set(f["src_ip"].astype(str).unique()) <= {"nan", "None", ""}:
        # Datasets without IP addresses (most CSE-CIC-IDS2018 files): host-level and
        # distinct-address features are unknown, not "one giant host".
        for c in HOST_FEATURES:
            out[c] = 0.0
    return out.fillna(0.0)


def window_labels(flows: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Per-window stage / is_attack / attack_frac from labelled flows."""
    f = flows.dropna(subset=["start_time"])
    wid = (f["start_time"] // window_s).astype(np.int64).to_numpy()
    stage = st.flow_stages(f["label"].astype(str).tolist(), f["dst_port"].tolist(), f["dst_ip"].astype(str).tolist())
    counts = pd.crosstab(wid, stage).reindex(columns=range(st.N_STAGES), fill_value=0)
    labels = pd.DataFrame(index=counts.index)
    labels["stage"] = [st.window_stage(row) for row in counts.to_numpy()]
    labels["is_attack"] = (labels["stage"] != st.BENIGN).astype(np.int8)
    labels["attack_frac"] = 1.0 - counts[st.BENIGN] / counts.sum(axis=1)
    for s in st.STAGES[1:]:
        labels[f"n_{s.key}"] = counts[s.idx]
    return labels


def build_states(
    flows: Optional[pd.DataFrame],
    packet_windows: Optional[pd.DataFrame] = None,
    window_s: float = 10.0,
    with_labels: bool = True,
) -> pd.DataFrame:
    """Return a dense, gap-free per-window state table (index = window_id)."""
    parts = []
    if flows is not None and len(flows):
        parts.append(flow_window_features(flows, window_s))
    pk = None
    if packet_windows is not None and len(packet_windows):
        pk = packet_windows.set_index("window_id")[list(PACKET_FEATURES)]
    ids = []
    for p in parts + ([pk] if pk is not None else []):
        ids.append((int(p.index.min()), int(p.index.max())))
    if not ids:
        raise ValueError("No flows or packets to build states from")
    lo, hi = min(i[0] for i in ids), max(i[1] for i in ids)
    index = pd.RangeIndex(lo, hi + 1, name="window_id")

    state = pd.DataFrame(0.0, index=index, columns=list(STATE_FEATURES))
    if parts:
        fl = parts[0].reindex(index)
        state.loc[:, list(FLOW_FEATURES)] = fl[list(FLOW_FEATURES)].fillna(0.0).to_numpy()
    if pk is not None:
        pkr = pk.reindex(index)
        present = pkr.notna().all(axis=1)
        state.loc[:, list(PACKET_FEATURES)] = pkr.fillna(0.0).to_numpy()
        state["pk_mask"] = present.astype(float).to_numpy()
    state["window_start"] = index.to_numpy() * window_s

    if with_labels and flows is not None and "label" in flows and flows["label"].notna().any():
        lab = window_labels(flows, window_s).reindex(index)
        state["stage"] = lab["stage"].fillna(st.BENIGN).astype(np.int64).to_numpy()
        state["is_attack"] = lab["is_attack"].fillna(0).astype(np.int64).to_numpy()
        state["attack_frac"] = lab["attack_frac"].fillna(0.0).to_numpy()
        for s in st.STAGES[1:]:
            state[f"n_{s.key}"] = lab[f"n_{s.key}"].fillna(0).astype(np.int64).to_numpy()
    return state


def _selftest() -> None:
    flows = pd.DataFrame({
        "start_time": [0.5, 1.0, 12.0, 13.0, 14.0, 15.0],
        "duration": [1.0, 2.0, 0.1, 0.1, 0.1, 0.1],
        "protocol": ["TCP", "UDP", "ICMP", "ICMP", "ICMP", "TCP"],
        "src_ip": ["10.0.0.1", "10.0.0.2", "147.32.84.165"] * 2,
        "dst_ip": ["8.8.8.8"] * 6, "src_port": [1, 2, 3, 4, 5, 6], "dst_port": [80, 53, 0, 0, 0, 6667],
        "packet_count": [10, 2, 100, 100, 100, 5], "byte_count": [1000, 120, 6000, 6000, 6000, 400],
        "fwd_bytes": [600, 60, 6000, 6000, 6000, 200], "bwd_packets": [4, 1, 0, 0, 0, 2],
        "flag_syn": [1, np.nan, np.nan, np.nan, np.nan, 1], "flag_ack": [1, np.nan, np.nan, np.nan, np.nan, 1],
        "label": ["flow=Background", "flow=From-Normal-X", "flow=From-Botnet-V1-ICMP",
                  "flow=From-Botnet-V1-ICMP", "flow=From-Botnet-V1-ICMP", "flow=From-Botnet-V1-TCP-CC1-IRC"],
    })
    s = build_states(flows, window_s=10.0)
    assert list(s.index) == [0, 1]
    assert s.loc[0, "stage"] == st.BENIGN and s.loc[1, "stage"] == st.IMPACT
    assert abs(s.loc[1, "fl_icmp_frac"] - 0.75) < 1e-9 and s.loc[1, "fl_port_irc"] == 0.25
    assert s["pk_mask"].sum() == 0
    print("state_builder selftest: OK")


if __name__ == "__main__":
    _selftest()

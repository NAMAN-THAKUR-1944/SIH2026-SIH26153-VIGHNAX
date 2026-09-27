"""Attack-stage taxonomy: dataset labels -> MITRE ATT&CK stages.

Every flow label (CTU-13 sentences such as ``flow=From-Botnet-V52-1-ICMP`` or
CIC-IDS class names such as ``DoS Hulk``) is mapped to one of the stages below.
A time window's stage is then the most advanced stage with enough supporting
flows in that window (see :func:`window_stage`).

The mapping is behaviour-based, not family-based: an ICMP flood is Impact
whichever botnet sends it. This is what lets the model generalise to botnet
families it never saw during training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class Stage:
    idx: int
    key: str
    name: str
    tactic_id: str
    techniques: str


# Order matters: idx is the class index used by the model's stage head.
STAGES: List[Stage] = [
    Stage(0, "benign", "Benign", "-", "-"),
    Stage(1, "recon", "Reconnaissance / Discovery", "TA0043 / TA0007",
          "T1595 Active Scanning, T1046 Network Service Discovery, T1016.001 Internet Connection Discovery"),
    Stage(2, "initial_access", "Initial Access", "TA0001",
          "T1190 Exploit Public-Facing Application, T1110 Brute Force"),
    Stage(3, "lateral", "Lateral Movement", "TA0008",
          "T1210 Exploitation of Remote Services"),
    Stage(4, "c2", "Command & Control", "TA0011",
          "T1071 Application Layer Protocol, T1071.004 DNS, T1573 Encrypted Channel, T1105 Ingress Tool Transfer"),
    Stage(5, "exfiltration", "Exfiltration", "TA0010",
          "T1041 Exfiltration Over C2 Channel, T1048 Exfiltration Over Alternative Protocol"),
    Stage(6, "impact", "Impact", "TA0040",
          "T1498 Network Denial of Service, T1496 Resource Hijacking (spam, click-fraud)"),
]
STAGE_KEYS = [s.key for s in STAGES]
N_STAGES = len(STAGES)
BENIGN, RECON, INITIAL, LATERAL, C2, EXFIL, IMPACT = range(N_STAGES)

# When several stages co-occur in one window, the most advanced wins.
_PRIORITY = [IMPACT, EXFIL, LATERAL, INITIAL, RECON, C2]

# Minimum supporting flows before a window is assigned a stage. A single failed
# connection is not a scan; a single C2 beacon *is* C2.
MIN_FLOWS = {IMPACT: 2, EXFIL: 1, LATERAL: 1, INITIAL: 2, RECON: 3, C2: 1}

_LATERAL_PORTS = {135, 137, 138, 139, 445, 3389, 5985, 5986, 22, 23}
_SMTP_PORTS = {25, 465, 587}
# Failed connections to these are C2 call-backs (HTTP/S, proxies, IRC), not scans.
_C2_PORTS = {80, 443, 8080, 8443, 3128, 1080} | set(range(6660, 6700))

# Dataset label prefixes checked first (they contain words like "dos" that must not be
# matched loosely): DARPA 1999 truth categories and the LANL red-team label.
_PREFIX: Dict[str, int] = {
    "darpa-probe": RECON, "darpa-dos": IMPACT, "darpa-r2l": INITIAL, "darpa-u2r": INITIAL,
    "darpa-data": EXFIL, "darpa-attack": INITIAL, "lanl-redteam": LATERAL,
}

# CIC-IDS2017 / CSE-CIC-IDS2018 / UNSW-NB15 / CICIoT2023 class names (substring match, in order).
_NAMED: Dict[str, int] = {
    "benign": BENIGN, "normal": BENIGN, "background": BENIGN,
    "portscan": RECON, "reconnaissance": RECON, "recon-": RECON, "vulnerabilityscan": RECON,
    "analysis": RECON, "fuzzers": RECON,
    "ftp-patator": INITIAL, "ssh-patator": INITIAL, "brute force": INITIAL,
    "bruteforce": INITIAL, "web attack": INITIAL, "sql injection": INITIAL, "sqlinjection": INITIAL, "xss": INITIAL,
    "commandinjection": INITIAL, "uploading_attack": INITIAL, "browserhijacking": INITIAL,
    "exploits": INITIAL, "generic": INITIAL, "shellcode": INITIAL,
    "heartbleed": EXFIL, "infiltration": LATERAL, "infilteration": LATERAL, "worms": LATERAL,
    "bot": C2, "backdoor": C2, "backdoors": C2,
    "mirai": IMPACT, "dos": IMPACT, "ddos": IMPACT, "loic": IMPACT, "hoic": IMPACT,
}


def _is_internal(ip: str) -> bool:
    return ip.startswith(("10.", "192.168.", "147.32.")) or re.match(r"^172\.(1[6-9]|2\d|3[01])\.", ip) is not None


def flow_stage(label: str, dst_port: float = np.nan, dst_ip: str = "") -> int:
    """Map one flow label to a stage index."""
    text = str(label).strip().lower()
    if not text or text in {"nan", "none"}:
        return BENIGN
    if "botnet" not in text:
        for key, stage in _PREFIX.items():
            if text.startswith(key):
                return stage
        for key, stage in _NAMED.items():
            if key in text:
                return stage
        return BENIGN if any(t in text for t in ("background", "normal", "benign")) else C2
    # CTU-13 botnet sentence.
    if "spam" in text or "smtp" in text or "http-ad" in text or "clickfraud" in text:
        return IMPACT
    if "icmp" in text or "flood" in text or "ddos" in text:
        return IMPACT
    if "attempt" in text:
        # Failed / unanswered connections: the destination port tells us why.
        try:
            port = int(dst_port)
        except (TypeError, ValueError):
            port = -1
        if port in _LATERAL_PORTS and _is_internal(str(dst_ip)):
            return LATERAL                      # internal SMB/RPC/RDP sweep
        if port in _SMTP_PORTS:
            return IMPACT                       # spam delivery attempts
        if port in _C2_PORTS or port >= 1024:
            return C2                           # failed C2 call-backs, P2P peer contacts
        return RECON                            # probing other service ports
    if "google-net" in text:
        return RECON  # connectivity check -> T1016.001 Internet Connection Discovery
    return C2  # CC*, IRC, P2P, DNS, HTTP/SSL beacons, custom encryption, downloads


def flow_stages(labels: Sequence[str], dst_ports: Sequence[float], dst_ips: Sequence[str]) -> np.ndarray:
    """Vectorised :func:`flow_stage` with a cache (labels repeat heavily)."""
    cache: Dict[tuple, int] = {}
    out = np.empty(len(labels), dtype=np.int8)
    for i, (lab, port, ip) in enumerate(zip(labels, dst_ports, dst_ips)):
        text = str(lab)
        # Only 'attempt' labels depend on port/ip; everything else caches on label.
        key = (text, port, str(ip)[:7]) if "ttempt" in text else (text,)
        stage = cache.get(key)
        if stage is None:
            stage = cache[key] = flow_stage(text, port, ip)
        out[i] = stage
    return out


def window_stage(counts: np.ndarray) -> int:
    """counts[stage] = supporting flows in the window -> window stage."""
    for stage in _PRIORITY:
        if counts[stage] >= MIN_FLOWS[stage]:
            return stage
    return BENIGN


def describe(stage_idx: int) -> Dict[str, str]:
    s = STAGES[int(stage_idx)]
    return {"key": s.key, "name": s.name, "tactic": s.tactic_id, "techniques": s.techniques}


def _selftest() -> None:
    assert flow_stage("flow=From-Botnet-V52-1-ICMP") == IMPACT
    assert flow_stage("flow=From-Botnet-V52-2-TCP-CC106-IRC-Not-Encrypted") == C2
    assert flow_stage("flow=From-Botnet-V46-TCP-Attempt-SPAM") == IMPACT
    assert flow_stage("flow=From-Botnet-V46-TCP-Attempt", 445, "147.32.84.2") == LATERAL
    assert flow_stage("flow=From-Botnet-V46-TCP-Attempt", 443, "8.8.8.8") == C2
    assert flow_stage("flow=From-Botnet-V46-TCP-Attempt", 22, "8.8.8.8") == RECON
    assert flow_stage("flow=From-Botnet-V47-TCP-Attempt", 25, "8.8.8.8") == IMPACT
    assert flow_stage("flow=From-Botnet-V53-UDP-Attempt", 41234, "8.8.8.8") == C2
    assert flow_stage("flow=From-Botnet-V52-1-UDP-DNS") == C2
    assert flow_stage("flow=Background-UDP-Established") == BENIGN
    assert flow_stage("flow=From-Normal-Grill") == BENIGN
    assert flow_stage("DoS Hulk") == IMPACT and flow_stage("PortScan") == RECON
    assert flow_stage("BENIGN") == BENIGN and flow_stage("SSH-Patator") == INITIAL
    assert flow_stage("darpa-dos-smurf") == IMPACT and flow_stage("darpa-probe-portsweep") == RECON
    assert flow_stage("darpa-data-secret") == EXFIL and flow_stage("lanl-redteam-lateral-movement") == LATERAL
    assert flow_stage("Backdoors") == C2 and flow_stage("Generic") == INITIAL and flow_stage("Worms") == LATERAL
    assert flow_stage("Recon-HostDiscovery") == RECON and flow_stage("Mirai-greeth_flood") == IMPACT
    assert flow_stage("DoS attacks-GoldenEye") == IMPACT and flow_stage("Infilteration") == LATERAL
    counts = np.zeros(N_STAGES); counts[C2] = 1; counts[RECON] = 2
    assert window_stage(counts) == C2          # 2 attempts is not yet a scan
    counts[IMPACT] = 40
    assert window_stage(counts) == IMPACT
    print("stages selftest: OK")


if __name__ == "__main__":
    _selftest()

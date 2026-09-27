"""Render docs/images/architecture.png (used by the README, the architecture document and slides)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED = "#1f2933", "#52606d"
FILL = {"in": "#e0f2fe", "proc": "#ecfdf5", "wm": "#eef2ff", "out": "#fff7ed", "ui": "#fdf2f8"}
EDGE = {"in": "#0284c7", "proc": "#059669", "wm": "#4f46e5", "out": "#ea580c", "ui": "#db2777"}


def box(ax, x, y, w, h, title, sub, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.2, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.52, sub, ha="center", va="top", fontsize=7.2, color=MUTED, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=MUTED))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, ha="center", fontsize=6.8, color=MUTED)


fig, ax = plt.subplots(figsize=(11, 4.1))
ax.set_xlim(0, 16.5); ax.set_ylim(0, 6); ax.axis("off")

box(ax, 0.1, 3.4, 2.6, 1.9, "Flow records", "CTU-13, CIC-IDS2017/18,\nUNSW-NB15, LANL flows\n(format auto-detected)", "in")
box(ax, 0.1, 0.6, 2.6, 1.9, "Packet capture", "pcap / pcapng\n(.bz2 / .gz, streamed)\n~125k packets/s", "in")
box(ax, 3.3, 3.4, 2.6, 1.9, "Flow loaders", "one schema for every\ndataset: flags, bytes,\nIAT, per-host behaviour", "proc")
box(ax, 3.3, 0.6, 2.6, 1.9, "packet_features", "TTL & session variance,\nTCP window, fragments,\npayload dist., scans, retrans.", "proc")
box(ax, 6.5, 1.8, 2.5, 2.4, "State S_t (10 s)", "47 flow-level\n(incl. per-host maxima)\n+ 30 packet-level\n+ packet-present mask", "proc")
box(ax, 9.6, 0.5, 3.3, 5.0, "World model", "", "wm")
for i, (t, s) in enumerate([("Encoder  q(z_t | S_t)", "Gaussian latent state"),
                            ("GRU memory  h_t", "temporal context"),
                            ("Prior  p(z_t+1 | h_t+1)", "learned transition dist."),
                            ("Decoder  p(S | z)", "reconstructs the state")]):
    y = 4.55 - i * 1.02
    ax.add_patch(FancyBboxPatch((9.85, y - 0.32), 2.8, 0.72, boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc="white", ec=EDGE["wm"], lw=0.8))
    ax.text(11.25, y + 0.16, t, ha="center", fontsize=7.8, weight="bold", color=INK)
    ax.text(11.25, y - 0.14, s, ha="center", fontsize=6.8, color=MUTED)
box(ax, 13.5, 3.35, 2.9, 2.15, "K-step forecast", "32 Monte-Carlo rollouts\nP(infiltration) per window\n+ 10-90 % band\nMITRE ATT&CK stage", "out")
box(ax, 13.5, 0.5, 2.9, 2.35, "Explain + alert", "Integrated Gradients\n(flow & packet features)\nhost occlusion ->\nflagged hosts & flows\nper-network thresholds", "out")
arrow(ax, 2.7, 4.35, 3.3, 4.35); arrow(ax, 2.7, 1.55, 3.3, 1.55)
arrow(ax, 5.9, 4.2, 6.5, 3.5, "window"); arrow(ax, 5.9, 1.7, 6.5, 2.5, "align")
arrow(ax, 9.0, 3.0, 9.6, 3.0)
arrow(ax, 12.9, 4.4, 13.5, 4.4, "imagine"); arrow(ax, 12.9, 1.7, 13.5, 1.7, "gradients")
ax.text(8.25, 0.05, "Offline dashboard: live timeline, forecast band, stage mix, attributions, flagged hosts & flows, incident report  |  no cloud APIs",
        ha="center", fontsize=7.6, color=EDGE["ui"], weight="bold")
fig.tight_layout(pad=0.2)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "images", "architecture.png")
fig.savefig(out, dpi=200)
print("wrote", out)

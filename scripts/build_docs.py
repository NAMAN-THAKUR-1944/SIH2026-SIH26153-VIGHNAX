"""Build the architecture document (<= 2 pages) and technical presentation (5 slides).

All numbers are read from results/metrics.json so the documents always match the benchmark.

    python scripts/make_diagram.py
    python scripts/build_docs.py   ->  docs/VIGHNAX_Architecture.docx, docs/VIGHNAX_Technical_Presentation.pptx
"""

import json
import os
import sys

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PRGB
from pptx.util import Inches
from pptx.util import Pt as PPt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "docs")
IMAGES = os.path.join(HERE, "images")
sys.path.insert(0, ROOT)
from src.data.sources import DATASET_NAMES, NON_TEMPORAL  # noqa: E402

R = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
BY, MAC = R["test_by_dataset"], R["test_macro"]
TEMPORAL = [d for d in BY if d not in NON_TEMPORAL]
LEAD_W = sum(BY[d]["lead_time"]["world_model"]["warned_before_onset"] for d in TEMPORAL)
LEAD_L = sum(BY[d]["lead_time"]["baseline_lr"]["warned_before_onset"] for d in TEMPORAL)
ONSETS = sum(BY[d]["lead_time"]["world_model"]["onsets"] for d in TEMPORAL)
ACCENT = RGBColor(0x0F, 0x76, 0x6E)
INK = RGBColor(0x1F, 0x29, 0x33)


def f(v, d=2):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def auc(ds, task, model):
    return BY[ds][task][model]["auroc"]


def mac(task, model):
    return MAC[f"{task}.auroc.{model}"]


TIE = 0.01  # AUROC differences below this are ties


def wins(task):
    return [d for d in TEMPORAL if auc(d, task, "world_model") - auc(d, task, "baseline_lr") >= TIE]


def losses(task, pool=None):
    return [DATASET_NAMES[d] for d in (pool or TEMPORAL) if auc(d, task, "baseline_lr") - auc(d, task, "world_model") >= TIE]


DET_LOSSES = losses("detect", list(BY))


def result_rows():
    rows = [("Dataset (unseen test data)", "Forecast WM / LR", "Early warning WM / LR", "Detection WM / LR")]
    for d in BY:
        fc = "n/a" if d in NON_TEMPORAL else f"{f(auc(d, 'forecast', 'world_model'))} / {f(auc(d, 'forecast', 'baseline_lr'))}"
        ew = "n/a" if d in NON_TEMPORAL else (f"{f(auc(d, 'forecast_from_benign', 'world_model'))} / "
                                               f"{f(auc(d, 'forecast_from_benign', 'baseline_lr'))}")
        rows.append((DATASET_NAMES[d], fc, ew, f"{f(auc(d, 'detect', 'world_model'))} / {f(auc(d, 'detect', 'baseline_lr'))}"))
    rows.append(("Macro average (temporal)", f"{f(mac('forecast', 'world_model'))} / {f(mac('forecast', 'baseline_lr'))}",
                 f"{f(mac('forecast_from_benign', 'world_model'))} / {f(mac('forecast_from_benign', 'baseline_lr'))}",
                 f"{f(mac('detect', 'world_model'))} / {f(mac('detect', 'baseline_lr'))}"))
    return rows


# ============================================================ architecture doc
def shade(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def build_doc(path):
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(1.6)
    sec.top_margin, sec.bottom_margin = Cm(1.3), Cm(1.2)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"; st.font.size = Pt(9.5)
    st.paragraph_format.space_after = Pt(2.5); st.paragraph_format.line_spacing = 1.05
    for lvl, size in ((1, 11.5), (2, 10)):
        h = doc.styles[f"Heading {lvl}"]
        h.font.name = "Calibri"; h.font.size = Pt(size); h.font.color.rgb = ACCENT; h.font.bold = True
        h.paragraph_format.space_before = Pt(5); h.paragraph_format.space_after = Pt(2)

    def para(text_runs, style=None, align=None, size=None, after=None):
        p = doc.add_paragraph(style=style)
        for txt, bold in text_runs:
            r = p.add_run(txt); r.bold = bold
            if size: r.font.size = Pt(size)
        if align: p.alignment = align
        if after is not None: p.paragraph_format.space_after = Pt(after)
        return p

    def bullet(runs):
        p = para(runs, style="List Bullet", after=1)
        p.paragraph_format.left_indent = Cm(0.5)
        return p

    t = para([("VIGHNAX — BhaviṣyAdvaktā: World-Model Forecasting of Network Attack Progression", True)], size=14, after=0)
    t.runs[0].font.color.rgb = INK
    para([("Architecture document · SIH 2026 · PS SIH26153 (NTRO) · AI-based Network Attack Forecasting from Network Traffic Data · "
           "Team VIGHNAX (Team ID 127364) · open source (MIT), fully offline", False)], size=8.5, after=4).runs[0].font.color.rgb = RGBColor(0x52, 0x60, 0x6D)

    doc.add_heading("1. Idea", level=1)
    para([("Instead of classifying flows one at a time, VIGHNAX learns ", False), ("how the state of a network evolves", True),
          (" — the transition distribution P(St+1 | S≤t) — from flow-level and packet-level telemetry, then rolls that model "
           "forward K steps to estimate whether the current trajectory leads to compromise, which MITRE ATT&CK stage comes next, "
           "and which traffic features drive the prediction. One model is trained on all seven datasets named in the PS.", False)])

    doc.add_heading("2. System architecture", level=1)
    doc.add_picture(os.path.join(IMAGES, "architecture.png"), width=Cm(17.6))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("3. Components", level=1)
    bullet([("Ingestion. ", True), ("A universal loader reads every PS dataset's native format by content: CTU-13 Argus "
            "binetflow, CIC-IDS2017/2018 CICFlowMeter CSV, UNSW-NB15 records, LANL flows; packet captures (pcap/pcapng, .gz/.bz2) "
            "are streamed at ~125k packets/s with headers decoded directly (a Scapy reference parser is included).", False)])
    bullet([("Network state S_t. ", True), ("Every 10 s window becomes a 78-dim vector: 47 flow-level features (5-tuple statistics, "
            "flags, bytes, packets, duration, IAT, fwd/bwd ratios, per-host maxima so one compromised host is not diluted) + 30 "
            "packet-level features (TTL mean and per-session variance, TCP window, IP fragment/DF flags, payload-size distribution, "
            "SYN-only/RST/ICMP shares, retransmissions, sequential-vs-randomised port-scan signatures) + a packet-present mask.", False)])
    bullet([("World model. ", True), ("Encoder q(z_t|S_t) → Gaussian latent; GRU memory h_t; prior p(z_t+1|h_t+1) = learned "
            "transition distribution; decoder p(S|z). Heads on (h, z) give P(infiltration) and a 7-way ATT&CK stage distribution. "
            "Loss = reconstruction + KL(posterior‖prior) + attack/stage losses on observed states and on imagined K-step rollouts "
            "against the true future labels (supervised dynamics learning from each dataset's attack timeline). Modality dropout "
            "(CSV-only inputs), volume augmentation (network size) and dataset-balanced sampling (each epoch draws equally from "
            "all seven datasets).", False)])
    bullet([("Forward simulation and explanations. ", True), ("32 Monte-Carlo latent trajectories rolled K = 6 windows (60 s) ahead: "
            "per-window P(infiltration) with a 10–90 % band and the predicted stage mix with technique IDs. Integrated Gradients "
            "(Aumann–Shapley / SHAP family) attributes each forecast to flow- and packet-level features; leave-one-host-out "
            "occlusion flags the responsible hosts and flows.", False)])
    bullet([("Interface. ", True), ("Offline Flask dashboard (bundled Chart.js and fonts). One drop zone for any mix of flow "
            "files, PCAPs or a .zip; the world model streams its forecast window by window; the timeline, forecast fan, attributions and "
            "flagged hosts update live at a selectable playback speed (1× = real time). Alert thresholds are calibrated per network (site profile).", False)])

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.add_heading("4. Data and evaluation protocol", level=1)
    para([("CTU-13 (flows + full packet captures), CIC-IDS2017 and CSE-CIC-IDS2018 (CICFlowMeter flows), UNSW-NB15 (flow "
           "records), LANL cyber1 (router flows + red-team events), DARPA 1999 (tcpdump + attack truth list) and CICIoT2023 "
           "(per-activity packet captures). ", False), ("Every test source is unseen in training", True),
          (": other botnet families (CTU-13), other days (CIC, UNSW, DARPA), another 12 h slice (LANL), other attack types "
           "(CICIoT2023). Validation = every 5th 5-minute block of the training sources. Labels map to ATT&CK stages by behaviour; "
           "all seven stages occur. The LR baseline uses the same state features and targets; operating thresholds are chosen per "
           "dataset on validation only (most detection with at most 5 % false alarms, same rule for both models).", False)])

    doc.add_heading("5. Results on unseen test data (AUROC, world model / LR baseline)", level=1)
    rows = result_rows()
    widths = [Cm(5.2), Cm(4.1), Cm(4.1), Cm(4.1)]
    tbl = doc.add_table(rows=len(rows), cols=4)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.width = widths[j]; c.text = ""
            r = c.paragraphs[0].add_run(val); r.font.size = Pt(8.5); r.bold = i == 0 or i == len(rows) - 1
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            if i == 0:
                shade(c, "E6F4F1")
    para([("", False)], after=1)
    bullet([("Forecasting: ", True), (f"world model ahead on {len(wins('forecast'))} of {len(TEMPORAL)} temporal datasets "
            f"(macro AUROC {f(mac('forecast', 'world_model'))} vs {f(mac('forecast', 'baseline_lr'))}; LR ahead on "
            f"{', '.join(losses('forecast')) or 'none'}); {LEAD_W} of {ONSETS} attack onsets forecast before they began, "
            f"vs {LEAD_L} for LR. On validation the world model leads on every dataset; unseen days generalise less well.", False)])
    bullet([("Stated plainly: ", True), ("detection of the current window: "
            + (f"LR is better on {', '.join(DET_LOSSES)}. " if DET_LOSSES else "world model ahead on every dataset. ")
            + "CICIoT2023 episodes are composed (benign capture then attack capture) and scored for detection only. Thresholds "
            "must be calibrated per site; PCAP-only input relies on flows rebuilt from packets.", False)])
    bullet([("Cost: ", True), (f"{f(R['performance']['ms_per_window'], 2)} ms per 10 s window batched on GPU, ~8 ms streamed on a laptop CPU "
            "(32 rollouts); ~125k packets/s "
            "ingestion. Reproduce: fetch_datasets.sh → prepare_data.py / prepare_all.py → train.py → evaluate.py (seeded config).", False)])
    doc.add_picture(os.path.join(ROOT, "results", "timeline_s47.png"), width=Cm(16.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = para([("Figure: unseen CTU-13 Menti capture. Shaded = ground-truth malicious windows; dark = world-model "
                 "P(attack within 60 s) with 10–90 % band; grey = LR baseline on the same features.", False)],
               size=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    cap.runs[0].italic = True
    doc.save(path)


# ============================================================ presentation
def build_ppt(path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    DARK, TEAL, WHITE = PRGB(0x0E, 0x11, 0x16), PRGB(0x2D, 0xD4, 0xBF), PRGB(0xE6, 0xED, 0xF3)

    def slide(title, kicker=None):
        s = prs.slides.add_slide(blank)
        bg = s.background.fill; bg.solid(); bg.fore_color.rgb = DARK
        tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9)).text_frame
        tb.text = title
        p = tb.paragraphs[0]; p.runs[0].font.size = PPt(30); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE
        if kicker:
            q = tb.add_paragraph(); q.text = kicker
            q.runs[0].font.size = PPt(14); q.runs[0].font.color.rgb = TEAL
        return s

    def bullets(s, items, x, y, w, h, size=16):
        tf = s.shapes.add_textbox(x, y, w, h).text_frame
        tf.word_wrap = True
        for i, (head, body) in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = PPt(8)
            r1 = p.add_run(); r1.text = head; r1.font.bold = True; r1.font.size = PPt(size); r1.font.color.rgb = TEAL
            r2 = p.add_run(); r2.text = " " + body; r2.font.size = PPt(size); r2.font.color.rgb = WHITE
        return tf

    s = slide("VIGHNAX — forecasting attacks before they complete",
              "SIH 2026 · SIH26153 (NTRO) · AI-based Network Attack Forecasting from Network Traffic Data · Team VIGHNAX (127364)")
    bullets(s, [
        ("Problem:", "IDS models label each flow in isolation and fire after the fact; an infiltration is a process unfolding over time."),
        ("Idea:", "learn a world model of network state dynamics P(S_t+1 | S_≤t) from flow + packet telemetry, and simulate K steps ahead."),
        ("Output:", "P(infiltration in next 60 s) with an uncertainty band, predicted MITRE ATT&CK stage, driving features, flagged hosts/flows."),
        ("Evidence:", f"one model, all 7 PS datasets, unseen test data: forecast macro AUROC {f(mac('forecast', 'world_model'))} vs "
                      f"{f(mac('forecast', 'baseline_lr'))} (LR); {LEAD_W}/{ONSETS} attack onsets forecast before they began (LR {LEAD_L})."),
        ("Practical:", "open source (MIT), fully offline: ~8 ms per 10 s window on a laptop CPU, "
                       f"{f(R['performance']['ms_per_window'], 2)} ms batched on GPU; ~125k packets/s ingestion."),
    ], Inches(0.6), Inches(1.7), Inches(12.1), Inches(5.3), 19)

    s = slide("Architecture: telemetry → network state → world model → forecast",
              "any PS dataset format or PCAP · flow-level + packet-level features fused into S_t every 10 s (78 dims)")
    s.shapes.add_picture(os.path.join(IMAGES, "architecture.png"), Inches(0.5), Inches(1.55), width=Inches(12.3))
    bullets(s, [
        ("Flow level:", "5-tuple, flags, bytes, packets, duration, IAT, fwd/bwd ratios + per-host maxima (a single bot is not diluted)."),
        ("Packet level:", "TTL & session variance, TCP window, fragments, payload distribution, port-scan signatures, retransmissions."),
    ], Inches(0.6), Inches(6.2), Inches(12.1), Inches(1.2), 14)

    s = slide("World model: learn the dynamics, then imagine the future",
              "RSSM-style latent state-space model (cf. World Models, Dreamer), trained on seven datasets' attack timelines")
    bullets(s, [
        ("Transition model:", "encoder q(z_t|S_t), GRU memory h_t, prior p(z_t+1|h_t+1) (learned transition distribution), decoder p(S|z)."),
        ("Supervised dynamics:", "attack + ATT&CK-stage heads trained on observed states AND on imagined K-step rollouts against the true future labels."),
        ("Forward simulation:", "32 Monte-Carlo trajectories, K = 6 x 10 s: probability per future window, 10–90 % band, stage mix."),
        ("Explainability:", "Integrated Gradients (Aumann–Shapley/SHAP family) vs a benign reference state; host-occlusion flags the responsible hosts."),
        ("Robustness:", "dataset-balanced training, modality dropout (CSV-only inputs), volume augmentation, per-site alert calibration."),
        ("Stages:", "Recon · Initial Access · Lateral Movement · C2 · Exfiltration · Impact — all present across the seven datasets."),
    ], Inches(0.6), Inches(1.75), Inches(12.1), Inches(5.5), 19)

    s = slide("Results: seven datasets, test data unseen in training",
              "AUROC world model / logistic regression on the same features; thresholds calibrated on validation only")
    rows = result_rows()
    tshape = s.shapes.add_table(len(rows), 4, Inches(0.6), Inches(1.7), Inches(7.8), Inches(4.4)).table
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tshape.cell(i, j); c.text = val
            para = c.text_frame.paragraphs[0]; para.runs[0].font.size = PPt(12 if i else 11)
            para.runs[0].font.bold = i == 0 or i == len(rows) - 1
            c.fill.solid(); c.fill.fore_color.rgb = PRGB(0x0F, 0x76, 0x6E) if i == 0 else PRGB(0x1C, 0x22, 0x2B)
            para.runs[0].font.color.rgb = WHITE
    tshape.columns[0].width = Inches(2.7)
    for j in range(1, 4):
        tshape.columns[j].width = Inches(1.7)
    bullets(s, [
        ("Forecasting:", f"ahead on {len(wins('forecast'))}/{len(TEMPORAL)} datasets; early warning ahead on "
                         f"{len(wins('forecast_from_benign'))}/{len(TEMPORAL)}."),
        ("Before onset:", f"{LEAD_W}/{ONSETS} attacks forecast before they began (LR {LEAD_L})."),
        ("Honest:", ("detection: LR better on " + ", ".join(DET_LOSSES) + ".") if DET_LOSSES else "detection: ahead everywhere."),
        ("Note:", "CICIoT2023 episodes are composed, so detection only."),
    ], Inches(8.7), Inches(1.7), Inches(4.3), Inches(5.3), 15)

    s = slide("Demo, deployment and next steps", "python server.py → http://127.0.0.1:5000 (no internet needed)")
    bullets(s, [
        ("Demo:", "drop any PS dataset file or PCAP (format auto-detected) or pick one of 8 bundled real captures → timeline, forecast "
                  "fan, ATT&CK stage mix, IG attributions and flagged hosts & flows all update live as each window is processed."),
        ("Reproducible:", "fetch_datasets.sh → prepare_data.py / prepare_all.py → train.py → evaluate.py; seeded YAML config; weights + results in the repo; pytest suite."),
        ("Enterprise / CII fit:", "streams captures in bounded memory, CPU or GPU, fully offline, explains every alert, per-site threshold profiles."),
        ("Limitations:", "CICIoT2023 episodes are composed; LANL labels come from authentication-based red-team events; alert thresholds need per-site calibration."),
        ("Next:", "graph (host-level GNN) state, online threshold calibration, streaming NetFlow/IPFIX collector, SOC integration (syslog/STIX)."),
    ], Inches(0.6), Inches(1.75), Inches(12.1), Inches(5.5), 19)
    prs.save(path)


if __name__ == "__main__":
    build_doc(os.path.join(HERE, "VIGHNAX_Architecture.docx"))
    build_ppt(os.path.join(HERE, "VIGHNAX_Technical_Presentation.pptx"))
    print("wrote docs/VIGHNAX_Architecture.docx and docs/VIGHNAX_Technical_Presentation.pptx")

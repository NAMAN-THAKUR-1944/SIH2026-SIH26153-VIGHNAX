"use strict";
// Incident report: a self-contained HTML file built in the browser from the current analysis (nothing is
// uploaded anywhere). It opens in any browser and prints to PDF for a case file. Every number in it comes
// from the same model outputs the dashboard shows.

const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function explanationAt(i) {
    if (explainCache.has(i)) return explainCache.get(i);
    const d = await (await fetch(`/api/explain/${A.id}/${i}`)).json();
    if (d.error) throw new Error(d.error);
    explainCache.set(d.index, d);
    return d;
}

// Where the first warning of an incident sits relative to the dataset's own labels (if the input has any).
function groundTruth(inc) {
    const t = A.truth_attack;
    if (!t) return "";
    let j = -1;
    if (t[inc.start]) { j = inc.start; while (j > 0 && t[j - 1]) j--; }
    else for (let k = inc.start; k <= inc.end; k++) if (t[k]) { j = k; break; }
    if (j < 0) return "Dataset labels: no malicious window during this incident (a false alarm with respect to the labels).";
    const d = (j - inc.start) * A.window_s;
    const when = d > 0 ? `${d} s before it` : d < 0 ? `${-d} s after it` : "in the same window";
    return `Dataset labels: malicious activity starts at ${clock(j)} (${minutes(j)} min); the first warning came ${when}.`;
}

function reportHtml(items, timelinePng, chartBg) {
    const integ = SERVER.integrity || {};
    const sum = (a) => a.reduce((x, y) => x + (y || 0), 0);
    const peak = Math.max(...A.p_any.map(v => v ?? 0));
    const weights = integ.files ? Object.entries(integ.files).map(([n, f]) =>
        `${esc(n)} <span class="mono">${esc(f.sha256 ? f.sha256.slice(0, 16) + "…" : "missing")}</span>`).join("<br>") : "–";
    const K = A.horizon * A.window_s;
    const incidents = items.map(({ inc, ex }, k) => {
        const st = A.stages[A.stage_pred[inc.peak]];
        const dur = (inc.end - inc.start + 1) * A.window_s;
        const feats = ex.features.slice(0, 6).map(f =>
            `<tr><td>${esc(f.label)}</td><td>${f.level === "packet" ? "packet" : "flow"}</td>` +
            `<td class="num ${f.value >= 0 ? "up" : "down"}">${f.value >= 0 ? "+" : ""}${(f.value * 100).toFixed(1)} pp</td></tr>`).join("");
        const hosts = ex.flagged.hosts.slice(0, 5).map(h =>
            `<tr><td class="mono">${esc(h.host)}</td><td class="num">${h.flows}</td>` +
            `<td class="num">${h.method ? (h.contribution * 100).toFixed(0) + "th pct" : (h.contribution >= 0 ? "+" : "") + (h.contribution * 100).toFixed(1) + " pp"}</td></tr>`).join("")
            || '<tr><td colspan="3" class="muted">No flows start in this window.</td></tr>';
        const flows = ex.flagged.flows.slice(0, 10).map(f =>
            `<tr><td class="mono">${esc(f.src_ip)}:${esc(f.src_port ?? "")}</td><td class="mono">${esc(f.dst_ip)}:${esc(f.dst_port ?? "")}</td>` +
            `<td>${esc(f.protocol)}</td><td class="num">${(f.packet_count ?? 0).toLocaleString()}</td><td class="num">${(f.byte_count ?? 0).toLocaleString()}</td>` +
            `<td>${esc((f.label || "").replace("flow=", "")) || "–"}</td></tr>`).join("");
        return `<section class="incident">
  <h2>Incident ${k + 1} <span class="muted">· ${clock(inc.start)} – ${clock(inc.end)} UTC</span></h2>
  <table class="kv">
    <tr><th>First warning</th><td>${clock(inc.start)} (${minutes(inc.start)} min into the capture), P(attack within ${K} s) = ${pct(A.p_any[inc.start])}</td></tr>
    <tr><th>Peak</th><td>${pct(inc.max_p)} at ${clock(inc.peak)} (${minutes(inc.peak)} min); 80 % of simulated futures ${pct(A.p_any_lo[inc.peak])} – ${pct(A.p_any_hi[inc.peak])}</td></tr>
    <tr><th>Duration</th><td>${dur >= 60 ? (dur / 60).toFixed(1) + " min" : dur + " s"} (${inc.end - inc.start + 1} windows)</td></tr>
    <tr><th>Predicted ATT&amp;CK stage</th><td><b>${esc(st.name)}</b> · ${esc(st.tactic)}${st.techniques ? " · " + esc(st.techniques) : ""}</td></tr>
    ${A.truth_attack ? `<tr><th>Ground truth</th><td>${esc(groundTruth(inc))}</td></tr>` : ""}
  </table>
  <div class="cols">
    <div><h3>Why the model raised it <span class="muted">· at the first warning</span></h3>
      <table><thead><tr><th>Feature</th><th>Level</th><th class="num">Effect on forecast</th></tr></thead><tbody>${feats}</tbody></table></div>
    <div><h3>Flagged hosts</h3>
      <table><thead><tr><th>Host</th><th class="num">Flows</th><th class="num">Contribution</th></tr></thead><tbody>${hosts}</tbody></table></div>
  </div>
  ${flows ? `<h3>Flows from the flagged hosts <span class="muted">· largest first</span></h3>
  <table><thead><tr><th>Source</th><th>Destination</th><th>Proto</th><th class="num">Packets</th><th class="num">Bytes</th><th>Dataset label</th></tr></thead><tbody>${flows}</tbody></table>` : ""}
</section>`;
    }).join("\n");

    const generated = new Date();
    return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VIGHNAX incident report · ${esc(A.name)}</title>
<style>
  :root { --ink: #231c15; --muted: #6f604f; --line: #e4dac6; --accent: #664c28; --up: #b43a39; --down: #2a6fc9; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #fbf8f1; color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 980px; margin: 0 auto; padding: 32px 24px 48px; }
  header { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; border-bottom: 2px solid var(--accent); padding-bottom: 12px; flex-wrap: wrap; }
  h1 { margin: 0; font-size: 22px; letter-spacing: .04em; } h1 span { color: var(--accent); }
  h2 { font-size: 17px; margin: 0 0 10px; } h3 { font-size: 13.5px; margin: 16px 0 6px; }
  .muted { color: var(--muted); font-weight: 400; }
  .mono { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12.5px; }
  .summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }
  .tile { border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; background: #fffdf7; }
  .tile b { display: block; font-size: 20px; } .tile span { color: var(--muted); font-size: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
  thead th { color: var(--muted); font-weight: 500; font-size: 12px; }
  .kv th { width: 190px; color: var(--muted); font-weight: 500; }
  .num { text-align: right; font-variant-numeric: tabular-nums; } .up { color: var(--up); } .down { color: var(--down); }
  .figure { border-radius: 10px; padding: 12px; margin: 8px 0 4px; }
  .figure img { width: 100%; display: block; }
  .incident { border: 1px solid var(--line); border-radius: 12px; padding: 18px; margin: 18px 0; background: #fffdf7; break-inside: avoid; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .note { color: var(--muted); font-size: 12px; margin-top: 24px; border-top: 1px solid var(--line); padding-top: 10px; }
  @media (max-width: 700px) { .summary { grid-template-columns: repeat(2, 1fr); } .cols { grid-template-columns: 1fr; } }
  @media print { body { background: #fff; } main { padding: 0; } .incident, .tile { background: #fff; } }
</style></head>
<body><main>
<header>
  <div><h1><span>VIGHNAX</span> incident report</h1><div class="muted">${esc(A.name)}</div></div>
  <div class="muted">Generated ${esc(generated.toLocaleString())} · offline, on this machine</div>
</header>
<div class="summary">
  <div class="tile"><b>${A.incidents.length}</b><span>incident${A.incidents.length === 1 ? "" : "s"} (alert episodes)</span></div>
  <div class="tile"><b>${pct(peak)}</b><span>peak P(attack within ${K} s)</span></div>
  <div class="tile"><b>${minutes(A.n)} min</b><span>of traffic · ${A.n} windows of ${A.window_s} s</span></div>
  <div class="tile"><b>${sum(A.flow_count).toLocaleString()}</b><span>flows · ${sum(A.packet_count).toLocaleString()} packets</span></div>
</div>
<table class="kv">
  <tr><th>Capture</th><td>${clock(0)} – ${clock(A.n - 1)} UTC</td></tr>
  <tr><th>Alert profile</th><td>${esc(A.meta.profile === "global" ? "global (unknown network)" : A.meta.profile)} · alert when P(attack within ${K} s) ≥ ${pct(A.thresholds.forecast, 1)}</td></tr>
  <tr><th>Model</th><td>World model trained on ${esc((SERVER.datasets || []).length)} datasets, ${esc(SERVER.mc_samples)} simulated futures per forecast, ${esc(SERVER.features)}-feature network state</td></tr>
  <tr><th>Model weights</th><td>${weights}<br><span class="muted">${integ.verified ? "Verified against models/SHA256SUMS" : "Not verified against models/SHA256SUMS"}</span></td></tr>
</table>
<h2 style="margin-top:22px">Risk over time</h2>
<div class="figure" style="background:${chartBg}"><img alt="Forecast timeline" src="${timelinePng}"></div>
<div class="muted" style="font-size:12px">Line: world model, P(attack within ${K} s) with its 80 % band · dashed: logistic-regression baseline · shaded: windows labelled malicious by the dataset (when labels exist).</div>
${incidents || `<p style="margin-top:20px"><b>No alerts.</b> The forecast stayed below the alert threshold for the whole capture.</p>`}
${A.incidents.length > items.length ? `<p class="muted">${A.incidents.length - items.length} further incidents are not detailed here; open the capture in VIGHNAX to inspect them.</p>` : ""}
<p class="note">Forecasts come from the VIGHNAX world model (Monte-Carlo simulation of future network states). Feature effects are
Integrated Gradients against a typical benign network state, in percentage points of the forecast; host contributions come from
removing each host's flows and re-running the forecast. Stages follow MITRE ATT&amp;CK. Produced offline; no data left this machine.</p>
</main></body></html>`;
}

async function exportReport() {
    if (!A || !A.id || streaming) return;
    const btn = $("export"), label = btn.textContent;
    btn.disabled = true; btn.textContent = "Preparing…";
    try {
        const incs = A.incidents.slice(0, 5);
        const items = [];
        for (const inc of incs) items.push({ inc, ex: await explanationAt(inc.start) });
        const html = reportHtml(items, charts.timeline.toBase64Image("image/png", 1), T("--surface-1"));
        const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
        const a = document.createElement("a");
        const d = new Date(), p = (n) => String(n).padStart(2, "0");
        const slug = A.name.split(" - ")[0].toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "capture";
        a.href = url;
        a.download = `vighnax-report-${slug}-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}.html`;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (e) {
        setStatus("Report failed: " + e.message, true);
    } finally {
        btn.textContent = label; btn.disabled = false;
    }
}
$("export").addEventListener("click", exportReport);

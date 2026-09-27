"use strict";
// VIGHNAX dashboard. All values come from the server: /api/stream (world-model inference, streamed
// window by window), /api/explain (Integrated Gradients + host occlusion), /api/benchmark (metrics.json).

const T = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
// Colours come from the CSS tokens of the active theme (style.css); re-read on every theme switch.
let COL = {};
// Stage colours: fixed categorical slots, stepped per theme (validated: adjacent CVD dE >= 8). Benign = neutral.
let STAGE_COL = [];
function readTheme() {
    COL = {
        text: T("--text"), text2: T("--text-2"), text3: T("--text-3"), line: T("--line"),
        model: T("--model"), band: T("--model-soft"), neutral: T("--neutral"), up: T("--up"), down: T("--down"),
        truth: T("--truth"), grid: T("--grid"), thr: T("--thr-line"), cur: T("--cursor-line"),
    };
    STAGE_COL = [0, 1, 2, 3, 4, 5, 6].map(k => T(`--st${k}`));
    Chart.defaults.color = COL.text3;
    Chart.defaults.borderColor = COL.grid;
    Object.assign(Chart.defaults.plugins.tooltip, { backgroundColor: T("--tip-bg"), borderColor: T("--tip-line"), titleColor: COL.text, bodyColor: COL.text2 });
}
const STATUS = {
    good: { label: "Low", icon: '<svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.5" fill="currentColor"/></svg>' },
    warning: { label: "Elevated", icon: '<svg viewBox="0 0 12 12"><path d="M6 1.5 11 10.5H1Z" fill="currentColor"/></svg>' },
    serious: { label: "High", icon: '<svg viewBox="0 0 12 12"><path d="M6 1 11 6 6 11 1 6Z" fill="currentColor"/></svg>' },
    critical: { label: "Critical", icon: '<svg viewBox="0 0 12 12"><rect x="1.5" y="1.5" width="9" height="9" rx="1.5" fill="currentColor"/></svg>' },
};

Chart.defaults.font.family = T("--sans");
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.display = false;
Object.assign(Chart.defaults.plugins.tooltip, {
    borderWidth: 1, padding: 10, cornerRadius: 8, boxPadding: 4, titleFont: { weight: "600" }, bodyFont: { family: T("--mono"), size: 11.5 },
});
readTheme();

const $ = (id) => document.getElementById(id);
const pct = (v, d = 0) => (v == null || isNaN(v)) ? "--" : (v * 100).toFixed(d) + "%";
let A = null;              // analysis state (filled as windows stream in)
let cursor = 0, received = 0, streaming = false, replayTimer = null, renderQueued = false;
const charts = {};
const explainCache = new Map();
let explainBusy = false;
let lastExplain = null;    // explanation currently on screen (re-drawn on theme switch)
let explainWanted = null;  // window the explanation panels should show next
let follow = true;         // while streaming / replaying, the panels follow the newest window
let runSpeed = 60;         // playback speed of the current run
let SERVER = {};           // /api/status (model, datasets, weight verification) for the incident report
let speed = 60;            // playback speed (x real time); 0 = as fast as possible
document.querySelectorAll("#speed button").forEach(b => b.addEventListener("click", () => {
    speed = +b.dataset.speed;
    document.querySelectorAll("#speed button").forEach(x => x.classList.toggle("on", x === b));
}));

// ----------------------------------------------------------------- theme
function syncThemeToggle() {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    $("theme-toggle").title = `Switch to ${next} theme`;
    $("theme-toggle").setAttribute("aria-label", `Switch to ${next} theme`);
}
function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("vighnax-theme", theme); } catch (e) { /* storage unavailable: theme lasts this visit */ }
    syncThemeToggle();
    readTheme();
    for (const k of Object.keys(charts)) { charts[k].destroy(); delete charts[k]; }
    if (!A) return;
    drawTimeline();
    drawRibbonLegend();
    $("ribbon").innerHTML = "";
    for (let i = 0; i < received; i++) appendRibbon(i);
    render();
    if (lastExplain) renderExplain(lastExplain);
}
$("theme-toggle").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
syncThemeToggle();

// ------------------------------------------------------------------ boot
async function init() {
    try {
        const st = await (await fetch("/api/status")).json();
        SERVER = st;
        const K = st.horizon * st.window_s;
        const DS = { ctu13: "CTU-13", cicids2017: "CIC-IDS2017", cicids2018: "CSE-CIC-IDS2018", unswnb15: "UNSW-NB15",
                     lanl: "LANL", darpa1999: "DARPA 1999", ciciot2023: "CICIoT2023" };
        const dsNames = (st.datasets || []).map(d => DS[d] || d);
        $("chips").innerHTML +=
            `<span class="chip" title="Trained on: ${dsNames.join(", ")}">World model &middot; ${dsNames.length} dataset${dsNames.length === 1 ? "" : "s"}</span>` +
            `<span class="chip" title="How far ahead the model forecasts: ${st.horizon} future windows of ${st.window_s} s each.">Forecasts ${K} s ahead</span>` +
            `<span class="chip" title="Size of the network state the model reads every ${st.window_s} s: flow-level + packet-level features.">${st.features}-feature network state</span>` +
            `<span class="chip" title="Each forecast simulates this many possible futures; the risk % and its 80% band come from them.">${st.mc_samples} simulated futures</span>`;
        document.querySelectorAll(".k-sec").forEach(e => e.textContent = `${K} s`);
        const integ = st.integrity;
        if (integ) {
            const hashes = Object.entries(integ.files).map(([n, f]) => `${n} ${f.sha256 ? f.sha256.slice(0, 16) + "…" : "missing"}`).join("; ");
            const offline = document.querySelector("#chips .chip");
            offline.title += integ.verified ? ` Model weights verified against models/SHA256SUMS (${hashes}).`
                                            : ` Model weights do not match models/SHA256SUMS (${hashes}).`;
            if (!integ.verified) $("chips").insertAdjacentHTML("beforeend",
                `<span class="chip warn" title="The model files differ from models/SHA256SUMS: ${hashes}">Weights unverified</span>`);
        }
        $("sample-list").innerHTML = st.samples.map(s => {
            const [title, ...rest] = s.title.split(" - ");
            const detail = rest.join(" - ");
            return `<button class="sample" data-id="${s.id}"><div class="s-title">${title}</div>` +
                `<div class="s-sub"><span class="tag">flows</span>${s.pcap ? '<span class="tag">packets</span>' : ""}${detail}</div></button>`;
        }).join("") || '<div class="lede">No bundled captures found.</div>';
        document.querySelectorAll(".sample").forEach(b => b.addEventListener("click", () => {
            const fd = new FormData(); fd.append("sample", b.dataset.id); run(fd);
        }));
    } catch (e) { setStatus("Model not loaded: " + e, true); }
    loadBenchmark();
    const auto = new URLSearchParams(location.search).get("sample");  // e.g. /?sample=ctu13_s47_menti
    if (auto) { const fd = new FormData(); fd.append("sample", auto); run(fd); }
}

function setStatus(msg, err = false) { const s = $("status"); s.textContent = msg; s.className = "status" + (err ? " err" : ""); }

// ---------------------------------------------------------------- upload
const drop = $("drop"), input = $("file-input");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => upload(e.dataTransfer.files));
input.addEventListener("change", () => upload(input.files));
function upload(files) {
    if (!files || !files.length) return;
    const fd = new FormData();
    [...files].forEach(f => fd.append("files", f));
    run(fd);
    input.value = "";
}

// ---------------------------------------------------------------- stream
async function run(body) {
    stopReplay();
    streaming = true; received = 0; explainCache.clear(); explainWanted = null; follow = true;
    body.append("speed", String(speed));
    runSpeed = speed;
    document.querySelectorAll(".sample, #replay, #export").forEach(b => b.disabled = true);
    setStatus("Uploading…");
    try {
        const res = await fetch("/api/stream", { method: "POST", body });
        if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || res.statusText); }
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let nl;
            while ((nl = buf.indexOf("\n")) >= 0) {
                const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
                if (line.trim()) handle(JSON.parse(line));
            }
        }
    } catch (e) {
        setStatus("Analysis failed: " + e.message, true);
    } finally {
        streaming = false;
        document.querySelectorAll(".sample").forEach(b => b.disabled = false);
    }
}

function handle(ev) {
    if (ev.type === "status") return setStatus(ev.msg);
    if (ev.type === "error") return setStatus(ev.msg, true);
    if (ev.type === "meta") return start(ev);
    if (ev.type === "window") return addWindow(ev);
    if (ev.type === "done") return finish(ev);
}

function start(m) {
    const n = m.n;
    const empty = () => new Array(n).fill(null);
    A = Object.assign(m, {
        nowcast: empty(), p_any: empty(), p_any_lo: empty(), p_any_hi: empty(), lr_any: empty(), surprise: empty(),
        p_step: empty(), p_lo: empty(), p_hi: empty(), stage_now: empty(), stage_future: empty(), stage_pred: empty(),
        incidents: [],
    });
    const mm = m.meta;
    setStatus(`${m.name} · ${(mm.flows ?? 0).toLocaleString()} flows${mm.flows_source ? " (" + mm.flows_source + ")" : ""} · ` +
        `packets: ${mm.has_packets ? "yes" : "no"} · alert profile: ${mm.profile === "global" ? "global (unknown network)" : mm.profile} · ` +
        `parsed in ${mm.parse_seconds}s · streaming ${n} windows through the world model…` +
        (mm.warning ? "  ⚠ " + mm.warning : ""), !!mm.warning);
    ["hero", "sim", "why", "flagged"].forEach(id => $(id).hidden = false);
    clearExplain();
    $("follow").hidden = true;
    document.querySelectorAll(".truth-only").forEach(e => e.hidden = !m.truth_attack);
    $("f-inc").textContent = "0";
    $("f-truth").textContent = m.truth_attack ? "labels available" : "unlabelled input";
    drawTimeline();
    drawRibbonLegend();
    $("ribbon").innerHTML = "";
    $("hero").scrollIntoView({ behavior: "smooth", block: "start" });
}

function addWindow(w) {
    const i = w.i;
    for (const k of Object.keys(w)) if (k !== "type" && k !== "i" && A[k]) A[k][i] = w[k];
    received = i + 1;
    if (follow) cursor = i;
    appendRibbon(i);
    if (!renderQueued) { renderQueued = true; requestAnimationFrame(render); }
}

function appendRibbon(i) {
    const alert = A.p_any[i] >= A.thresholds.forecast || A.nowcast[i] >= A.thresholds.nowcast;
    const cell = document.createElement("div");
    cell.style.background = alert ? STAGE_COL[A.stage_pred[i]] : "transparent";
    cell.title = `${minutes(i)} min · ${alert ? A.stages[A.stage_pred[i]].name : "no alert"}`;
    $("ribbon").appendChild(cell);
}

function render() {
    renderQueued = false;
    const tl = charts.timeline;
    const upto = (arr) => arr.map((v, j) => j < received ? v : null);
    tl.data.datasets[0].data = A.truth_attack ? upto(A.truth_attack) : [];
    tl.data.datasets[1].data = upto(A.p_any_hi);
    tl.data.datasets[2].data = upto(A.p_any_lo);
    tl.data.datasets[3].data = upto(A.p_any);
    tl.data.datasets[4].data = upto(A.lr_any);
    moveCursor();
    $("progress").textContent = streaming || replayTimer
        ? `${received} / ${A.n} windows · ${minutes(received)} of ${minutes(A.n)} min of traffic` : $("progress").textContent;
    updateFacts();
    drawSimulation();
    if (live()) requestExplain(cursor);
}

function finish(d) {
    A.id = d.id; A.incidents = d.incidents;
    $("f-inc").textContent = d.incidents.length;
    streaming = false;
    render();
    const trafficS = A.n * A.window_s;
    $("progress").textContent = `${minutes(A.n)} min of traffic · ${d.infer_seconds}s of compute`;
    setStatus(`${A.name} · ${minutes(A.n)} min of recorded traffic (${A.n} windows of ${A.window_s} s) analysed with ${d.infer_seconds}s of model compute ` +
        `(${d.ms_per_window} ms per window, ≈${Math.round(trafficS / Math.max(d.infer_seconds, 1e-3))}× faster than real time). ` +
        `Playback was paced at ${d.speed ? d.speed + "× real time" : "maximum speed"}.` +
        (A.meta.warning ? "  ⚠ " + A.meta.warning : ""), !!A.meta.warning);
    $("replay").disabled = false;
    $("export").disabled = false;
    $("follow").hidden = true;
    select(follow ? (d.incidents.length ? d.incidents[0].start : A.n - 1) : cursor);
    follow = true;
}

// ------------------------------------------------------------ timeline
function minutes(i) { return (i * A.window_s / 60).toFixed(1); }
function clock(i) { return new Date((A.t0 + i * A.window_s) * 1000).toISOString().substr(11, 8); }

function drawTimeline() {
    if (charts.timeline) charts.timeline.destroy();
    const labels = Array.from({ length: A.n }, (_, i) => i);
    charts.timeline = new Chart($("timeline"), {
        type: "line",
        data: { labels, datasets: [
            { label: "Labelled malicious", data: [], type: "bar", backgroundColor: COL.truth, barPercentage: 1, categoryPercentage: 1, order: 9 },
            { label: "80% interval (high)", data: [], borderWidth: 0, pointRadius: 0, fill: "+1", backgroundColor: COL.band, order: 5 },
            { label: "80% interval (low)", data: [], borderWidth: 0, pointRadius: 0, fill: false, order: 5 },
            { label: "World model", data: [], borderColor: COL.model, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.25, order: 1 },
            { label: "LR baseline", data: [], borderColor: COL.neutral, borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0, order: 2 },
        ] },
        options: {
            animation: false, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
            scales: {
                y: { min: 0, max: 1, ticks: { callback: v => (v * 100) + "%", stepSize: 0.25 }, grid: { color: COL.grid }, border: { display: false } },
                x: { ticks: { maxTicksLimit: 12, callback: (v) => minutes(v) + "m" }, grid: { display: false }, border: { color: COL.line } },
            },
            plugins: {
                tooltip: {
                    filter: (it) => !it.dataset.label.startsWith("80%") && it.raw != null,
                    callbacks: {
                        title: (it) => `${minutes(it[0].dataIndex)} min · ${clock(it[0].dataIndex)}`,
                        label: (c) => c.dataset.label === "Labelled malicious" ? (c.raw ? " labelled malicious" : " labelled benign")
                            : ` ${c.dataset.label}: ${pct(c.raw, 1)}`,
                    },
                },
                annotation: { annotations: {
                    thr: { type: "line", yMin: A.thresholds.forecast, yMax: A.thresholds.forecast, borderColor: COL.thr, borderDash: [2, 4], borderWidth: 1,
                           label: { display: true, content: "alert threshold", position: "end", backgroundColor: "transparent", color: COL.text3, font: { size: 11 }, yAdjust: -9 } },
                    cur: { type: "line", xMin: cursor, xMax: cursor, borderColor: COL.cur, borderWidth: 1 },
                } },
            },
            onClick: (ev, els, chart) => {
                if (!A) return;
                const x = Math.round(chart.scales.x.getValueForPixel(ev.x));
                if (x < 0 || x >= received) return;
                stopReplay();
                if (streaming) setFollow(false);
                select(x);
            },
        },
    });
}

function moveCursor() {
    const a = charts.timeline.options.plugins.annotation.annotations.cur;
    a.xMin = a.xMax = cursor;
    charts.timeline.update("none");
}

function drawRibbonLegend() {
    $("ribbon-legend").innerHTML = "Predicted stage while alerting:" + A.stages.slice(1).map(s =>
        `<span><i style="background:${STAGE_COL[s.idx]}"></i>${s.name}</span>`).join("");
}

// ------------------------------------------------------------- facts
function status(p) {
    const thr = A.thresholds.forecast;
    return p >= 0.9 ? "critical" : p >= thr ? "serious" : p >= thr * 0.5 ? "warning" : "good";
}

function updateFacts() {
    const i = cursor;
    if (A.p_any[i] == null) return;
    const p = A.p_any[i], lvl = status(p);
    $("risk-value").textContent = pct(p);
    const pill = $("risk-pill");
    pill.className = "pill " + lvl;
    pill.innerHTML = STATUS[lvl].icon + STATUS[lvl].label;
    $("risk-band").textContent = `80% of simulated futures: ${pct(A.p_any_lo[i])} – ${pct(A.p_any_hi[i])}`;
    $("f-now").textContent = pct(A.nowcast[i]);
    const s = A.stages[A.stage_pred[i]];
    const peak = Math.max(...A.stage_future[i].map(r => r[A.stage_pred[i]]));
    const expected = p >= A.thresholds.forecast * 0.5;
    $("f-stage").textContent = expected ? s.name : "None expected";
    $("f-tech").textContent = expected ? `${s.tactic} · ${pct(peak)}` : "";
    $("f-tech").title = s.techniques;
    $("f-time").textContent = `${minutes(i)} min`;
    $("f-traffic").textContent = `${clock(i)} · ${A.flow_count[i].toLocaleString()} flows · ${A.packet_count[i].toLocaleString()} pkts`;
}

// -------------------------------------------------------- simulation
function drawSimulation() {
    const i = cursor, K = A.horizon;
    if (!A.p_step[i]) return;
    $("sim-at").textContent = `· from ${minutes(i)} min`;
    const labels = Array.from({ length: K }, (_, k) => `+${(k + 1) * A.window_s}s`);
    const fan = {
        labels, datasets: [
            { label: "90th percentile", data: A.p_hi[i], borderWidth: 0, pointRadius: 0, fill: "+1", backgroundColor: COL.band },
            { label: "10th percentile", data: A.p_lo[i], borderWidth: 0, pointRadius: 0, fill: false },
            { label: "Mean P(infiltration)", data: A.p_step[i], borderColor: COL.model, borderWidth: 2, pointRadius: 4, pointBackgroundColor: COL.model,
              pointBorderColor: T("--surface-1"), pointBorderWidth: 2 },
        ],
    };
    if (!charts.fan) {
        charts.fan = new Chart($("fan"), { type: "line", data: fan, options: { animation: false, maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: { y: { min: 0, max: 1, ticks: { callback: v => (v * 100) + "%", stepSize: 0.25 }, border: { display: false } },
                      x: { grid: { display: false } } },
            plugins: { tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${pct(c.raw, 1)}` } } } } });
    } else { charts.fan.data = fan; charts.fan.update("none"); }

    const ds = A.stages.slice(1).map(s => ({
        label: s.name, data: A.stage_future[i].map(r => r[s.idx]), backgroundColor: STAGE_COL[s.idx],
        borderColor: T("--surface-1"), borderWidth: { top: 2 }, borderSkipped: false, stack: "s", maxBarThickness: 44,
    })).filter(d => Math.max(...d.data) > 0.02);
    const sdata = { labels, datasets: ds };
    if (!charts.stages) {
        charts.stages = new Chart($("stages"), { type: "bar", data: sdata, options: { animation: false, maintainAspectRatio: false,
            scales: { x: { stacked: true, grid: { display: false } },
                      y: { stacked: true, min: 0, max: 1, ticks: { callback: v => (v * 100) + "%", stepSize: 0.25 }, border: { display: false } } },
            plugins: { legend: { display: true, position: "bottom", labels: { boxWidth: 10, boxHeight: 10, color: COL.text2, padding: 14 } },
                       tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${pct(c.raw, 1)}` } } } } });
    } else { charts.stages.data = sdata; charts.stages.update("none"); }
}

// --------------------------------------------------------- selection
function select(i) {
    if (!A) return;
    cursor = Math.max(0, Math.min(i, received - 1));
    moveCursor();
    updateFacts();
    drawSimulation();
    requestExplain(cursor);
}

// Explanations (Integrated Gradients + host occlusion) take longer than the gap between windows at 60x.
// One request runs at a time; when it returns, the panels show it if it is the window asked for (or, while
// following live, anything newer than what is on screen) and the next request goes straight to the newest
// window. The server only explains windows the stream has processed. At Max speed the whole capture is
// processed in about a second, so the panels are filled once at the end instead.
const live = () => follow && (streaming ? runSpeed > 0 : !!replayTimer && speed > 0);

function requestExplain(i) {
    explainWanted = i;
    if (!A || !A.id) return;
    if (explainCache.has(i)) return renderExplain(explainCache.get(i));
    if (explainBusy) return;            // the request in flight chases explainWanted when it returns
    explainBusy = true;
    const id = A.id;
    fetch(`/api/explain/${id}/${i}`).then(r => r.json()).then(d => {
        explainBusy = false;
        if (!A || A.id !== id) return;  // a new analysis has started
        if (d.error) {
            if (!streaming) setStatus(d.error, true);
        } else {
            explainCache.set(d.index, d);
            if (d.index === explainWanted || (live() && d.index > (lastExplain ? lastExplain.index : -1))) renderExplain(d);
        }
        if (explainWanted !== i) requestExplain(explainWanted);
    }).catch(() => { explainBusy = false; });
}

function clearExplain() {
    lastExplain = null;
    ["why-at", "fl-at"].forEach(id => $(id).textContent = "· waiting for the first window");
    for (const k of ["attr", "timeattr"]) if (charts[k]) { charts[k].destroy(); delete charts[k]; }
    $("hosts").querySelector("tbody").innerHTML = "";
    $("flows").querySelector("tbody").innerHTML = "";
}

function setFollow(on) {
    follow = on;
    $("follow").hidden = on || !streaming;
    if (on) select(received - 1);
}
$("follow").addEventListener("click", () => setFollow(true));

function renderExplain(d) {
    lastExplain = d;
    const at = `· at ${minutes(d.index)} min` + (live() ? " · live" : "");
    $("why-at").textContent = at; $("fl-at").textContent = at;
    const feats = d.features.slice(0, 10);
    const data = {
        labels: feats.map(f => `${f.label}  ·  ${f.level === "packet" ? "PKT" : "FLOW"}`),
        datasets: [{ data: feats.map(f => f.value), backgroundColor: feats.map(f => f.value >= 0 ? COL.up : COL.down),
                     borderRadius: 4, borderSkipped: false, barPercentage: 0.72, categoryPercentage: 0.9 }],
    };
    if (!charts.attr) {
        charts.attr = new Chart($("attr"), { type: "bar", data, options: { indexAxis: "y", animation: false, maintainAspectRatio: false,
            scales: { x: { grid: { color: COL.grid }, border: { display: false }, ticks: { callback: v => (v > 0 ? "+" : "") + (v * 100).toFixed(0) + " pp" } },
                      y: { grid: { display: false }, ticks: { autoSkip: false, color: COL.text2, font: { size: 12.5 } } } },
            plugins: { tooltip: { callbacks: { label: (c) => ` ${c.raw >= 0 ? "towards compromise" : "towards normal"}: ${(c.raw * 100).toFixed(2)} pp` } } } } });
    } else { charts.attr.data = data; charts.attr.update("none"); }

    const n = d.time_attribution.length;
    const tdata = { labels: d.time_attribution.map((_, k) => `${(k - n + 1) * A.window_s}s`),
        datasets: [{ data: d.time_attribution, backgroundColor: d.time_attribution.map(v => v >= 0 ? COL.up : COL.down), borderRadius: 3, barPercentage: 0.7 }] };
    if (!charts.timeattr) {
        charts.timeattr = new Chart($("timeattr"), { type: "bar", data: tdata, options: { animation: false, maintainAspectRatio: false,
            scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } }, y: { grid: { color: COL.grid }, border: { display: false }, ticks: { maxTicksLimit: 3 } } },
            plugins: { tooltip: { callbacks: { label: (c) => ` ${(c.raw * 100).toFixed(2)} pp` } } } } });
    } else { charts.timeattr.data = tdata; charts.timeattr.update("none"); }

    const hosts = d.flagged.hosts;
    const heuristic = hosts.length && hosts[0].method;
    $("hosts-lede").textContent = heuristic
        ? "PCAP-only input: hosts are ranked by how extreme their behaviour is (SMTP, failed connections, scans, ICMP, DNS, volume)."
        : "Each host's traffic is removed in turn and the forecast recomputed; the drop is that host's contribution.";
    const maxc = Math.max(1e-9, ...hosts.map(h => Math.abs(h.contribution)));
    $("hosts").querySelector("tbody").innerHTML = hosts.slice(0, 8).map((h, k) => {
        const val = heuristic ? `${(h.contribution * 100).toFixed(0)}th pct` : `${h.contribution >= 0 ? "+" : ""}${(h.contribution * 100).toFixed(1)} pp`;
        return `<tr class="${k === 0 && h.contribution > 0 ? "top" : ""}"><td class="mono">${h.host}</td><td class="num">${h.flows}</td>` +
            `<td><div class="contrib"><div class="bar ${h.contribution < 0 ? "neg" : ""}" style="width:${Math.max(2, Math.abs(h.contribution) / maxc * 220)}px"></div><small>${val}</small></div></td></tr>`;
    }).join("") || '<tr><td colspan="3" class="lbl">No flows start in this window.</td></tr>';
    $("flows").querySelector("tbody").innerHTML = d.flagged.flows.map(f => {
        const lab = (f.label || "").replace("flow=", "");
        return `<tr><td class="mono">${f.src_ip}:${f.src_port ?? ""}</td><td class="mono">${f.dst_ip}:${f.dst_port ?? ""}</td><td>${f.protocol}</td>` +
            `<td class="num">${(f.packet_count ?? 0).toLocaleString()}</td><td class="num">${(f.byte_count ?? 0).toLocaleString()}</td>` +
            `<td class="lbl ${/botnet/i.test(lab) ? "bot" : ""}">${lab || "–"}</td></tr>`;
    }).join("");
}

// ------------------------------------------------------------ replay
function stopReplay() { if (replayTimer) { clearInterval(replayTimer); replayTimer = null; $("replay").textContent = "Replay"; } }
$("replay").addEventListener("click", () => {
    if (!A || streaming) return;
    if (replayTimer) return stopReplay();
    const full = A.n;
    received = 0; cursor = 0; follow = true;
    $("ribbon").innerHTML = "";
    clearExplain();
    $("replay").textContent = "Stop";
    replayTimer = setInterval(() => {
        if (received >= full) {
            stopReplay(); received = full; render();
            $("progress").textContent = `${A.n} windows`;
            select(A.incidents.length ? A.incidents[0].start : full - 1);
            return;
        }
        appendRibbon(received);
        cursor = received; received += 1;
        render();
    }, speed ? A.window_s * 1000 / speed : 8);
});

// --------------------------------------------------------- benchmark
async function loadBenchmark() {
    const res = await fetch("/api/benchmark");
    const b = await res.json();
    if (!res.ok) { $("bench").innerHTML = `<p class="lede">${b.error}</p>`; return; }
    const K = b.horizon * b.window_s, t = b.test;
    const cols = [["f1", "F1"], ["precision", "Precision"], ["recall", "Recall"], ["fpr", "FPR", true], ["auroc", "AUROC"], ["tpr_at_fpr_05", "TPR @ 5% FPR"]];
    const rows = (task, name) => ["world_model", "baseline_lr"].map((m, j) => {
        const me = t[task][m], other = t[task][j ? "world_model" : "baseline_lr"];
        return `<tr><td>${j ? "" : name}</td><td>${j ? "LR baseline" : "World model"}</td>` + cols.map(([k, , low]) =>
            `<td class="mono ${(low ? other[k] - me[k] : me[k] - other[k]) >= 0.01 ? "best" : ""}">${me[k].toFixed(3)}</td>`).join("") + "</tr>";
    }).join("");
    let onW = 0, onL = 0, onN = 0;
    for (const [ds, r] of Object.entries(b.test_by_dataset || {})) {
        if (ds === "ciciot2023") continue;
        onW += r.lead_time.world_model.warned_before_onset; onL += r.lead_time.baseline_lr.warned_before_onset; onN += r.lead_time.world_model.onsets;
    }
    const DS = { ctu13: "CTU-13", cicids2017: "CIC-IDS2017", cicids2018: "CSE-CIC-IDS2018", unswnb15: "UNSW-NB15",
                 lanl: "LANL cyber1", darpa1999: "DARPA 1999", ciciot2023: "CICIoT2023" };
    const f3 = (v) => v == null || isNaN(v) ? "–" : v.toFixed(3);
    const TIE = 0.01;  // AUROC differences below this are ties (no marker)
    const cell = (r, task) => {
        const w = r[task].world_model.auroc, l = r[task].baseline_lr.auroc;
        return `<td class="mono ${w - l >= TIE ? "best" : ""}">${f3(w)}</td><td class="mono ${l - w >= TIE ? "best" : ""}">${f3(l)}</td>`;
    };
    const perDs = Object.entries(b.test_by_dataset || {}).map(([ds, r]) => ds === "ciciot2023"
        ? `<tr><td>${DS[ds] || ds}</td><td class="mono">${r.windows.toLocaleString()}</td><td colspan="4" class="lbl">n/a: composed benign+attack episodes</td>${cell(r, "detect")}</tr>`
        : `<tr><td>${DS[ds] || ds}</td><td class="mono">${r.windows.toLocaleString()}</td>${cell(r, "forecast")}${cell(r, "forecast_from_benign")}${cell(r, "detect")}</tr>`).join("");
    const perDsTable = perDs ? `<h3 class="sub-h">Per dataset (AUROC, world model vs LR baseline)</h3>
        <table class="table ds"><thead><tr><th>Dataset</th><th>Test windows</th><th>Forecast WM</th><th>LR</th><th>Early warning WM</th><th>LR</th><th>Detection WM</th><th>LR</th></tr></thead>
        <tbody>${perDs}</tbody></table>` : "";
    $("bench").innerHTML = perDsTable + `<h3 class="sub-h">Pooled over all temporal test data</h3><table class="table"><thead><tr><th>Task</th><th>Model</th>${cols.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead><tbody>
        ${rows("forecast", `Forecast · next ${K} s`)}${rows("forecast_from_benign", "Early warning (still benign)")}${rows("detect", "Detection · current window")}</tbody></table>
        <ul class="notes">
          <li>Every test source is unseen in training: other days, captures, time slices or botnet families. ${t.windows.toLocaleString()} pooled windows.</li>
          <li>Attack onsets forecast before they began (per-dataset calibrated thresholds, temporal datasets): world model ${onW}/${onN}, LR ${onL}/${onN}.</li>
          <li>Pooled F1 / precision / recall / FPR use one global threshold chosen on validation; per-dataset thresholds are in <span class="mono">results/RESULTS.md</span>. AUROC and TPR @ 5% FPR are threshold-free. ● marks a lead of at least 0.01.</li>
        </ul>`;
}

init();

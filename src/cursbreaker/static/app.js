"use strict";

const $ = (id) => document.getElementById(id);
let staged = [];      // [{id, name, pages}]
let pollTimer = null;
let activeJobId = null;  // the running job, for the Cancel button

async function api(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

// ---- settings ----------------------------------------------------------- //
const NUMERIC = ["temperature", "thinking_budget", "pdf_dpi", "max_dimension", "word_confidence"];
const TEXT = ["thinking_level", "media_resolution", "tesseract_language"];
const BOOL = ["preprocess", "refine_word_boxes"];

function gatherSettings() {
  const s = {};
  for (const id of TEXT) s[id] = $(id).value;
  for (const id of NUMERIC) {
    const v = parseFloat($(id).value);
    if (!Number.isNaN(v)) s[id] = v;
  }
  for (const id of BOOL) s[id] = $(id).checked;
  // One picker drives both models: detection (two-pass) follows transcription.
  const model = $("model").value;
  if (model) { s.transcription_model = model; s.detection_model = model; }
  const mode = document.querySelector("input[name=mode]:checked");
  if (mode) s.mode = mode.value;
  const ct = document.querySelector("input[name=content_type]:checked");
  if (ct) s.content_type = ct.value;
  return s;
}

function applyKeyStatus(data) {
  const badge = $("key-status");
  const info = $("key-info");
  info.hidden = false;
  if (data.api_key_set) {
    badge.textContent = "Key saved"; badge.className = "badge ok";
    info.className = "key-info";
    const where = data.api_key_source === "env"
      ? `from <span class="mono">GEMINI_API_KEY</span> environment variable`
      : "stored locally on this machine";
    info.innerHTML =
      `<span class="glyph" aria-hidden="true">✓</span><span>Gemini key ${where}: <span class="mono">${escapeHtml(data.api_key_hint || "")}</span></span>`;
  } else {
    badge.textContent = "No API key"; badge.className = "badge warn";
    info.className = "key-info warn";
    info.innerHTML =
      `<span class="glyph" aria-hidden="true">!</span><span>No Gemini key stored. Paste one above or set <span class="mono">GEMINI_API_KEY</span>.</span>`;
  }
}

// Free, proactive check that a stored key still works. ListModels costs no
// generation tokens/quota, so a revoked/expired key is caught here in Settings
// instead of failing mid-transcription. Only "valid"/"invalid" change the
// badge; "unknown" (offline/transient) and "no_key" leave it untouched.
async function verifyKey() {
  let data;
  try { data = await api("GET", "/api/key-status"); }
  catch (e) { return; } // our own server unreachable — leave the local badge
  const badge = $("key-status");
  const info = $("key-info");
  if (data.state === "valid") {
    badge.textContent = "Key valid"; badge.className = "badge ok";
  } else if (data.state === "invalid") {
    badge.textContent = "Key rejected"; badge.className = "badge warn";
    info.className = "key-info warn";
    info.innerHTML =
      `<span class="glyph" aria-hidden="true">!</span><span>${escapeHtml(data.message)} Transcription will fail until you paste a current key.</span>`;
  }
}

async function loadSettings() {
  const s = await api("GET", "/api/settings");
  for (const id of [...TEXT, ...NUMERIC]) if (s[id] !== undefined) $(id).value = s[id];
  for (const id of BOOL) if (s[id] !== undefined) $(id).checked = s[id];
  // Model dropdown (options populated by loadModelCatalog before this runs).
  const sel = $("model");
  if (s.transcription_model) sel.value = s.transcription_model;
  if (!sel.value && sel.options.length) {
    // A saved model that isn't in the curated catalog -> fall back to the first
    // (and persist it so the UI and the backend agree on what will run).
    sel.value = sel.options[0].value;
    saveSettings(gatherSettings());
  }
  updateModelPricingHint();
  const radio = document.querySelector(`input[name=mode][value="${s.mode}"]`);
  if (radio) radio.checked = true;
  const ctRadio = document.querySelector(`input[name=content_type][value="${s.content_type}"]`);
  if (ctRadio) ctRadio.checked = true;
  applyKeyStatus(s);
  verifyKey(); // background-verify the stored key (no-op when no key set)
}

async function loadTesseractStatus() {
  let data;
  try {
    data = await api("GET", "/api/tesseract");
  } catch (e) {
    return;
  }
  const info = $("tesseract-info");
  if (info) {
    if (data.available) {
      // Tesseract ships bundled with the app, so confirming "it's installed" is
      // just noise. Stay silent when it works; only speak up when it's missing
      // (Printed-only and word-box refinement genuinely won't run without it).
      info.hidden = true;
      info.innerHTML = "";
    } else {
      info.hidden = false;
      info.className = "key-info warn";
      let detail;
      if (data.wrapper_present === false) {
        detail = `The <span class="mono">pytesseract</span> Python package is missing. Reinstall CursBreaker (<span class="mono">pip install .</span>) and restart.`;
      } else {
        const hint = data.install_hint ? escapeHtml(data.install_hint) + " " : "";
        // No-admin path: a portable build dropped into the app-managed folder.
        const portable = data.managed_dir
          ? ` No admin rights? Unzip a portable Tesseract into <span class="mono">${escapeHtml(data.managed_dir)}</span> (binary + a <span class="mono">tessdata</span> subfolder).`
          : "";
        detail = `Tesseract OCR engine not detected. ${hint}${portable} Or set <span class="mono">TESSERACT_CMD</span> to the full path of the executable and restart.`;
      }
      info.innerHTML =
        `<span class="glyph" aria-hidden="true">!</span><span>${detail} Printed-only mode and word-box refinement need it; Handwriting mode still works as usual.</span>`;
    }
  }
  // Populate the language datalist with whatever's actually installed.
  const dl = $("tesseract-langs");
  if (dl) {
    dl.innerHTML = "";
    for (const code of data.languages || []) {
      const opt = document.createElement("option");
      opt.value = code;
      dl.appendChild(opt);
    }
  }
}

async function saveSettings(partial) {
  const data = await api("POST", "/api/settings", partial);
  applyKeyStatus(data);
}

// Curated model catalog (with published prices) backing the dropdown + the
// automatic cost estimate. Populated once from the backend.
let modelCatalog = [];
let pricesAsOf = "";

async function loadModelCatalog() {
  try {
    const data = await api("GET", "/api/models");
    modelCatalog = data.models || [];
    pricesAsOf = data.prices_as_of || "";
    const sel = $("model");
    sel.innerHTML = "";
    for (const m of modelCatalog) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      sel.appendChild(opt);
    }
  } catch (e) { /* dropdown stays empty; settings still load */ }
}

function modelInfo(id) {
  return modelCatalog.find((m) => m.id === id) || null;
}

// Show the selected model's published price right under the picker, so the cost
// basis is visible before anyone runs anything -- and clearly an estimate.
function updateModelPricingHint() {
  const hint = $("model-pricing");
  if (!hint) return;
  const m = modelInfo($("model").value);
  if (!m) { hint.textContent = ""; return; }
  let rates = `$${m.input_per_mtok.toFixed(2)}/1M input, $${m.output_per_mtok.toFixed(2)}/1M output`;
  if (m.tier_threshold) {
    rates += ` for prompts &le;${formatTokens(m.tier_threshold)} tokens `
      + `($${m.input_per_mtok_high.toFixed(2)}/$${m.output_per_mtok_high.toFixed(2)} above)`;
  }
  hint.innerHTML =
    `<b>${escapeHtml(m.label)}</b>: ${rates}.`
    + (pricesAsOf ? ` Prices as of ${escapeHtml(pricesAsOf)}, used for the cost estimate (an estimate, not a guarantee).` : "")
    + ` <a href="${PRICING_URL}" target="_blank" rel="noopener noreferrer">Live pricing</a>.`;
}

// ---- upload / staging --------------------------------------------------- //
async function uploadFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const fd = new FormData();
  for (const f of fileList) fd.append("files", f);
  const note = $("action-note");
  note.textContent = "Uploading…";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    staged.push(...data.files);
    renderStaged();
    note.textContent = "";
  } catch (e) {
    note.textContent = "Upload failed: " + e.message;
  }
}

// The action note doubles as a transient status/error line. This is its
// resting state -- a summary of what's staged -- restored whenever a transient
// message (e.g. a prior "no API key" error) should be cleared.
function stagedStatus() {
  return staged.length ? `${staged.length} file(s) ready` : "";
}

function renderStaged() {
  const ul = $("staged");
  ul.innerHTML = "";
  for (const f of staged) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${escapeHtml(f.name)} <span class="pill">${f.pages} page(s)</span></span>`;
    const rm = document.createElement("button");
    rm.className = "rm"; rm.type = "button"; rm.textContent = "×";
    rm.title = "Remove " + f.name;
    rm.setAttribute("aria-label", "Remove " + f.name);  // "×" alone says nothing
    rm.onclick = () => { staged = staged.filter((x) => x.id !== f.id); renderStaged(); };
    li.appendChild(rm);
    ul.appendChild(li);
  }
  $("transcribe").disabled = staged.length === 0;
  $("estimate").disabled = staged.length === 0;
  // Any change to the file set invalidates a prior cost estimate.
  const est = $("estimate-info");
  if (est) { est.hidden = true; est.innerHTML = ""; }
  $("action-note").textContent = stagedStatus();
}

// ---- processing --------------------------------------------------------- //
async function transcribe() {
  if (!staged.length) return;
  // Drop any leftover error (e.g. a prior "no API key") so it can't linger
  // through this run.
  $("action-note").textContent = stagedStatus();
  // Free up screen space the moment work starts so progress + results land
  // above the fold on smaller laptops.
  setSettingsOpen(false);  // collapse Settings; remembers the collapsed state
  $("transcribe").disabled = true;
  $("estimate").disabled = true;
  const est = $("estimate-info");
  if (est) { est.hidden = true; est.innerHTML = ""; }
  $("token-text").textContent = "";
  // Reset the activity log + bar for a fresh run.
  const logReset = $("activity-log"); if (logReset) logReset.replaceChildren();
  const liveReset = $("activity-live"); if (liveReset) liveReset.textContent = "";
  $("progress-bar").style.width = "0%";
  $("progress").setAttribute("aria-valuenow", "0");
  $("results-card").hidden = true;
  $("results").innerHTML = "";
  try {
    const { job_id } = await api("POST", "/api/process", { file_ids: staged.map((f) => f.id) });
    activeJobId = job_id;
    const cancel = $("cancel-job");
    if (cancel) { cancel.hidden = false; cancel.disabled = false; cancel.textContent = "Cancel"; }
    $("progress-card").hidden = false;
    pollJob(job_id);
  } catch (e) {
    $("action-note").textContent = "Error: " + e.message;
    $("transcribe").disabled = false;
  }
}

async function cancelJob() {
  if (!activeJobId) return;
  const btn = $("cancel-job");
  if (btn) { btn.disabled = true; btn.textContent = "Cancelling…"; }
  try {
    await api("POST", `/api/jobs/${activeJobId}/cancel`);
  } catch (e) {
    // Re-enable so the user can retry if the request itself failed.
    if (btn) { btn.disabled = false; btn.textContent = "Cancel"; }
  }
}

// Drive the page-driven bar + the verbose activity log from a job payload.
// Top-level so the Node test harness can exercise it directly with a fake DOM.
function renderProgress(job) {
  // Bar fills by pages completed across the whole job; clamp, and force 100%
  // once finished so it always reads full on completion.
  const total = Number(job.total_units || 0);
  const done = Number(job.done_units || 0);
  let pct = total ? Math.round((done / total) * 100) : 0;
  pct = Math.max(0, Math.min(100, pct));
  if (job.status === "done") pct = 100;
  $("progress-bar").style.width = pct + "%";
  $("progress").setAttribute("aria-valuenow", String(pct));

  // Concise headline above the log. (Cancelled keeps its partial bar fraction —
  // only "done" forces 100%.)
  $("progress-text").textContent =
    job.status === "error" ? "Error: " + job.error
    : job.status === "cancelled" ? `Cancelled — ${(job.results || []).length} file(s) completed`
    : job.status === "done" ? `Done — ${(job.results || []).length} file(s)`
    : total ? `Processing — page ${done}/${total}`
    : "Processing…";

  // Append-only activity log: the server sends the full (capped) list each
  // poll; render idempotently by index, rebuilding only if it was trimmed.
  const logEl = $("activity-log");
  if (logEl) {
    const lines = job.log || [];
    // Decide BEFORE adding lines whether to follow the tail: only if the user is
    // already parked at (or within a few px of) the bottom. If they've scrolled
    // up to read, leave their position alone — they can scroll back down to
    // resume following.
    const stickToBottom =
      logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight <= 4;
    if (logEl.childElementCount > lines.length) logEl.replaceChildren();
    for (let i = logEl.childElementCount; i < lines.length; i++) {
      const li = document.createElement("li");
      li.textContent = lines[i];   // textContent -> no escaping needed
      logEl.appendChild(li);
    }
    // Announce only the newest line to screen-reader users.
    if (lines.length) {
      const live = $("activity-live");
      const latest = lines[lines.length - 1];
      if (live && live.textContent !== latest) live.textContent = latest;
    }
    if (stickToBottom) logEl.scrollTop = logEl.scrollHeight;
  }
}

function pollJob(jobId) {
  clearInterval(pollTimer);
  const tick = async () => {
    let job;
    try { job = await api("GET", `/api/jobs/${jobId}`); }
    catch (e) { return; }
    renderProgress(job);
    // Live token counter: ticks up as each page's call returns.
    const tok = $("token-text");
    if (tok) {
      if (job.tokens && job.tokens.calls) {
        const prefix = job.status === "running" ? "Tokens so far — " : "Tokens — ";
        tok.textContent = prefix + tokenSummary(job.tokens) + costSuffix(job.tokens);
      } else {
        tok.textContent = "";
      }
    }
    if (job.status !== "running") {
      clearInterval(pollTimer);
      activeJobId = null;
      const cancel = $("cancel-job");
      if (cancel) { cancel.hidden = true; cancel.disabled = false; cancel.textContent = "Cancel"; }
      $("transcribe").disabled = false;
      $("estimate").disabled = staged.length === 0;
      // Show whatever finished — completed files remain downloadable on cancel.
      if ((job.status === "done" || job.status === "cancelled") && (job.results || []).length) {
        renderResults(jobId, job);
        if (job.status === "done") {
          // Send keyboard/screen-reader focus to the freshly-rendered results.
          const h = $("results-heading");
          if (h) h.focus();
        }
      }
    }
  };
  tick();
  pollTimer = setInterval(tick, 700);
}

function renderResults(jobId, job) {
  $("results-card").hidden = false;
  $("zip-link").href = `/api/download/${jobId}.zip`;
  // Per-job token total with the transparent dollar disclaimer.
  const totals = $("results-tokens");
  if (totals) {
    if (job.tokens && job.tokens.calls) {
      totals.hidden = false;
      totals.innerHTML = `<b>Total tokens:</b> ${tokenSummary(job.tokens)}.${costDisclaimerHtml(job.tokens)}`;
    } else {
      totals.hidden = true;
      totals.innerHTML = "";
    }
  }
  const root = $("results");
  root.innerHTML = "";
  for (const r of job.results) {
    const div = document.createElement("div");
    div.className = "result";
    const tokPill = (r.tokens && r.tokens.calls)
      ? ` <span class="pill">${formatTokens(r.tokens.total)} tokens${
          r.tokens.cost !== null && r.tokens.cost !== undefined ? ", ~" + formatCost(r.tokens.cost) : ""
        }</span>`
      : "";
    let html = `<h3>${escapeHtml(r.source_name)} <span class="pill">${r.n_pages} page(s), ${r.n_lines} lines</span>${tokPill}</h3>`;
    if (r.error) {
      html += `<p class="err">Error: ${escapeHtml(r.error)}</p>`;
      div.innerHTML = html;
    } else {
      const pdfLink = r.pdf ? `<a class="btn small primary" href="${r.pdf}">Searchable PDF</a>` : "";
      html += `<div class="links">
        ${pdfLink}
        <a class="btn small" href="${r.txt}">Download .txt</a>
        <a class="btn small" href="${r.hocr}">Download .hocr</a></div>`;
      div.innerHTML = html;
      const links = div.querySelector(".links");
      r.images.forEach((im, i) => {
        const label = r.images.length > 1 ? `Preview p${i + 1}` : "Preview boxes";
        const pageSuffix = r.images.length > 1 ? `, page ${i + 1}` : "";
        const btn = document.createElement("button");
        btn.className = "btn small"; btn.type = "button";
        btn.textContent = label;
        btn.setAttribute("aria-label", `Preview detected line boxes for ${r.source_name}${pageSuffix}`);
        btn.onclick = () => openPreview(im.preview, `${r.source_name} — detected lines${pageSuffix}`);
        links.appendChild(btn);
        const a = document.createElement("a");
        a.className = "btn small"; a.href = im.download;
        a.textContent = r.images.length > 1 ? `PNG p${i + 1}` : "Page PNG";
        links.appendChild(a);
      });
    }
    root.appendChild(div);
  }
}

// ---- preview modal (native <dialog>: focus trap + Esc + inert bg) ------- //
let lastFocused = null;
function openPreview(url, title) {
  const dlg = $("modal");
  const img = $("modal-img");
  img.src = url;
  img.alt = title;                 // descriptive alt for this specific preview
  $("modal-title").textContent = title;
  lastFocused = document.activeElement;  // so we can restore focus on close
  if (typeof dlg.showModal === "function" && !dlg.open) dlg.showModal();
  else dlg.setAttribute("open", "");     // fallback for very old browsers
}
function closePreview() {
  const dlg = $("modal");
  if (typeof dlg.close === "function" && dlg.open) dlg.close();
  else dlg.removeAttribute("open");
  $("modal-img").removeAttribute("src");
}

// ---- token / cost estimate --------------------------------------------- //
const PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing";

function formatTokens(n) {
  return Number(n || 0).toLocaleString();
}

// Dollar amounts can be tiny; scale precision so a sub-cent estimate isn't
// rounded away to "$0.00".
function formatCost(usd) {
  if (usd === null || usd === undefined) return "";
  const v = Number(usd);
  if (v === 0) return "$0.00";
  if (v < 0.01) return "$" + v.toFixed(4);
  if (v < 1) return "$" + v.toFixed(3);
  return "$" + v.toFixed(2);
}

function priceBasis(t) {
  return `$${Number(t.price_input_per_mtok || 0).toFixed(2)}/1M input and `
    + `$${Number(t.price_output_per_mtok || 0).toFixed(2)}/1M output`;
}

// "Gemini 3.5 Flash's published price of $… (prices as of …)" — the model and
// date a dollar figure was computed from, for full transparency.
function priceSource(t) {
  const model = t.model_label ? `${escapeHtml(t.model_label)}'s ` : "";
  const asOf = t.prices_as_of ? ` (prices as of ${escapeHtml(t.prices_as_of)})` : "";
  return `${model}published price of ${priceBasis(t)}${asOf}`;
}

// Compact token line (no dollars): "12,345 tokens · 10,000 in / 2,345 out · 8 API calls".
function tokenSummary(t) {
  if (!t) return "";
  const parts = [`${formatTokens(t.total)} tokens`];
  parts.push(
    `${formatTokens(t.input)} in / ${formatTokens(t.output)} out`
    + (t.thinking ? ` (incl. ${formatTokens(t.thinking)} thinking)` : "")
  );
  if (t.calls) parts.push(`${formatTokens(t.calls)} API call${t.calls === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

// " · ~$0.0123 (est.)" when prices are set, else "".
function costSuffix(t) {
  if (!t || t.cost === null || t.cost === undefined) return "";
  return ` · ~${formatCost(t.cost)} (est.)`;
}

// The fuller, transparent dollar disclaimer shown with a final/total figure.
function costDisclaimerHtml(t) {
  if (!t || t.cost === null || t.cost === undefined) {
    return ` <span class="muted">No published price on file for this model, so there's no dollar estimate &mdash; the token counts above are exact.</span>`;
  }
  return ` <span class="muted">&mdash; <b>~${formatCost(t.cost)}</b> is a rough estimate, not a guarantee, `
    + `based on ${priceSource(t)}. Token counts are exact; Google `
    + `bills the actual usage &mdash; verify live rates at `
    + `<a href="${PRICING_URL}" target="_blank" rel="noopener noreferrer">Gemini API pricing</a>.</span>`;
}

async function estimateCost() {
  if (!staged.length) return;
  const box = $("estimate-info");
  if (!box) return;
  box.hidden = false;
  box.className = "key-info estimate-info";
  box.innerHTML = `<span>Estimating&hellip;</span>`;
  $("estimate").disabled = true;
  try {
    const data = await api("POST", "/api/estimate", { file_ids: staged.map((f) => f.id) });
    box.className = "key-info estimate-info";
    box.innerHTML = renderEstimate(data);
  } catch (e) {
    box.className = "key-info estimate-info warn";
    box.innerHTML =
      `<div class="estimate-line"><span class="glyph" aria-hidden="true">!</span><span>Couldn't estimate: ${escapeHtml(e.message)}</span></div>`;
  } finally {
    $("estimate").disabled = staged.length === 0;
  }
}

function renderEstimate(d) {
  if (d.billable === false) {
    return `<div class="estimate-line"><span class="glyph" aria-hidden="true">●</span>`
      + `<span>No Gemini tokens for these ${d.files} file(s): ${escapeHtml(d.reason)} makes no API call, so there's no token cost.</span></div>`;
  }
  const hasCost = d.cost !== null && d.cost !== undefined;
  // Headline figure, large: the estimated cost -- or the token total when a
  // model has no published price, so there's still something prominent up top.
  const headline = hasCost
    ? `<div class="estimate-cost"><span class="estimate-cost-num">~${formatCost(d.cost)}</span>`
      + `<span class="estimate-cost-label">estimated cost &mdash; not a guarantee</span></div>`
    : `<div class="estimate-cost"><span class="estimate-cost-num">${formatTokens(d.total)}</span>`
      + `<span class="estimate-cost-label">tokens &mdash; no published price for this model</span></div>`;
  // Supporting detail as bullets rather than a paragraph.
  const points = [];
  points.push(
    `<b>${d.files}</b> file(s), <b>${formatTokens(d.pages)}</b> page(s)`
    + (d.model_label ? ` with <b>${escapeHtml(d.model_label)}</b>` : "")
  );
  points.push(
    `~<b>${formatTokens(d.input)}</b> input + ~<b>${formatTokens(d.output)}</b> output tokens`
    + ` <span class="muted">(~${formatTokens(d.assumed_output_tokens_per_call)}/call across ${formatTokens(d.calls)} call(s))</span>`
  );
  if (hasCost) {
    points.push(
      `Priced at ${priceBasis(d)}`
      + (d.prices_as_of ? `, prices as of ${escapeHtml(d.prices_as_of)}` : "")
      + ` &mdash; <a href="${PRICING_URL}" target="_blank" rel="noopener noreferrer">check live pricing</a>`
    );
  }
  points.push(
    `<span class="muted">Token counts are exact; the dollar amount is an estimate. Input is measured `
    + `from the first page of each file and output length varies, so treat it as a ballpark &mdash; the `
    + `live counter shows real usage during the run.</span>`
  );
  const lis = points.map((p) => `<li>${p}</li>`).join("");
  return `${headline}<ul class="estimate-points">${lis}</ul>`;
}

// ---- misc --------------------------------------------------------------- //
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function wire() {
  $("save-key").onclick = () => saveSettings({ api_key: $("api_key").value }).then(() => {
    $("api_key").value = "";
    // A freshly-saved key invalidates any prior "no API key" transcription
    // error, so clear that stale message immediately.
    $("action-note").textContent = stagedStatus();
    verifyKey(); // confirm the just-pasted key actually works (free check)
  });
  $("clear-key").onclick = async () => {
    if (!confirm("Clear the stored Gemini key from this machine?\n(If GEMINI_API_KEY is set in your environment, that will still be used.)")) return;
    await api("DELETE", "/api/settings/api_key");
    $("api_key").value = "";
    await loadSettings();
  };
  for (const id of [...TEXT, ...NUMERIC]) $(id).addEventListener("change", () => saveSettings(gatherSettings()));
  for (const id of BOOL) $(id).addEventListener("change", () => saveSettings(gatherSettings()));
  for (const r of document.querySelectorAll("input[name=mode]")) r.addEventListener("change", () => saveSettings(gatherSettings()));
  for (const r of document.querySelectorAll("input[name=content_type]")) r.addEventListener("change", () => saveSettings(gatherSettings()));
  // Model picker: update the visible price basis, then persist (both models).
  $("model").addEventListener("change", () => { updateModelPricingHint(); saveSettings(gatherSettings()); });

  const dz = $("dropzone");
  $("browse").onclick = () => $("file-input").click();
  $("file-input").addEventListener("change", (e) => uploadFiles(e.target.files));
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    uploadFiles(e.dataTransfer.files);
  });

  $("transcribe").onclick = transcribe;
  $("estimate").onclick = estimateCost;
  $("cancel-job").onclick = cancelJob;

  // Settings disclosure (heading > button) toggle + theme switcher.
  $("settings-toggle").onclick = () =>
    setSettingsOpen($("settings-toggle").getAttribute("aria-expanded") !== "true");
  // Segmented theme control: both options visible; selecting either (click or
  // arrow keys) applies it. Native radios fire 'change' on selection.
  for (const r of document.querySelectorAll('input[name="theme"]'))
    r.addEventListener("change", () => setTheme(r.value));

  const dlg = $("modal");
  $("modal-close").onclick = closePreview;
  // Clicking the backdrop (target is the dialog itself) closes it.
  dlg.addEventListener("click", (e) => { if (e.target === dlg) closePreview(); });
  // Esc (native) and the Close button both fire 'close' -> restore focus.
  dlg.addEventListener("close", () => {
    if (lastFocused && typeof lastFocused.focus === "function") lastFocused.focus();
  });
}

// ---- heartbeat: server shuts itself down when the tab stops pinging ---- //
function heartbeat() {
  fetch("/api/heartbeat", { method: "POST", keepalive: true }).catch(() => {});
}
heartbeat();
setInterval(heartbeat, 5000);
function bye() {
  try { navigator.sendBeacon("/api/heartbeat?bye=true"); } catch (e) {}
}
window.addEventListener("beforeunload", bye);
window.addEventListener("pagehide", (e) => { if (!e.persisted) bye(); });

// ---- theme + settings disclosure state (both persisted) ----------------- //
function currentTheme() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}
function setTheme(theme) {
  const root = document.documentElement;
  if (theme === "light") root.dataset.theme = "light";
  else delete root.dataset.theme;            // dark is the default (no attribute)
  // Reflect the choice in the always-visible segmented control.
  const radio = $(theme === "light" ? "theme-light" : "theme-dark");
  if (radio) radio.checked = true;
  try { localStorage.setItem("cb.theme", theme); } catch (e) {}
}

function setSettingsOpen(open) {
  const btn = $("settings-toggle");
  const body = $("settings-body");
  if (!btn || !body) return;
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  body.hidden = !open;
  try { localStorage.setItem("cb.settings.open", open ? "1" : "0"); } catch (e) {}
}

async function init() {
  wire();
  // Sync the theme button to whatever the pre-paint inline script applied, and
  // restore the saved Settings open/closed choice (default: open).
  setTheme(currentTheme());
  let settingsOpen = "1";
  try { settingsOpen = localStorage.getItem("cb.settings.open") || "1"; } catch (e) {}
  setSettingsOpen(settingsOpen !== "0");
  // Populate the model dropdown before loading settings, so the saved model can
  // be selected and its price shown.
  await loadModelCatalog();
  await loadSettings();
  loadTesseractStatus();
}
init();

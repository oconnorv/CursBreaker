"use strict";

const $ = (id) => document.getElementById(id);
let staged = [];      // [{id, name, pages}]
let pollTimer = null;

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
const NUMERIC = ["temperature", "pdf_dpi", "max_dimension", "word_confidence"];
const TEXT = ["transcription_model", "detection_model", "thinking_level", "media_resolution"];
const BOOL = ["use_mock", "preprocess"];

function gatherSettings() {
  const s = {};
  for (const id of TEXT) s[id] = $(id).value;
  for (const id of NUMERIC) {
    const v = parseFloat($(id).value);
    if (!Number.isNaN(v)) s[id] = v;
  }
  for (const id of BOOL) s[id] = $(id).checked;
  const mode = document.querySelector("input[name=mode]:checked");
  if (mode) s.mode = mode.value;
  return s;
}

function applyKeyStatus(data) {
  const badge = $("key-status");
  const info = $("key-info");
  info.hidden = false;
  if (data.use_mock) {
    badge.textContent = "Demo mode"; badge.className = "badge ok";
    info.className = "key-info";
    info.innerHTML =
      `<span class="glyph">●</span><span>Demo mode is on — no real API call will be made.</span>`;
  } else if (data.api_key_set) {
    badge.textContent = "Key saved"; badge.className = "badge ok";
    info.className = "key-info";
    const where = data.api_key_source === "env"
      ? `from <span class="mono">GEMINI_API_KEY</span> environment variable`
      : "stored locally on this machine";
    info.innerHTML =
      `<span class="glyph">✓</span><span>Gemini key ${where}: <span class="mono">${escapeHtml(data.api_key_hint || "")}</span></span>`;
  } else {
    badge.textContent = "No API key"; badge.className = "badge warn";
    info.className = "key-info warn";
    info.innerHTML =
      `<span class="glyph">!</span><span>No Gemini key stored. Paste one above or set <span class="mono">GEMINI_API_KEY</span>.</span>`;
  }
}

async function loadSettings() {
  const s = await api("GET", "/api/settings");
  for (const id of [...TEXT, ...NUMERIC]) if (s[id] !== undefined) $(id).value = s[id];
  for (const id of BOOL) if (s[id] !== undefined) $(id).checked = s[id];
  const radio = document.querySelector(`input[name=mode][value="${s.mode}"]`);
  if (radio) radio.checked = true;
  applyKeyStatus(s);
}

async function saveSettings(partial) {
  const data = await api("POST", "/api/settings", partial);
  applyKeyStatus(data);
}

async function loadModels() {
  try {
    const data = await api("GET", "/api/models");
    const list = $("model-list");
    list.innerHTML = "";
    for (const m of data.models || []) {
      const opt = document.createElement("option");
      opt.value = m;
      list.appendChild(opt);
    }
  } catch (e) { /* suggestions are optional */ }
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

function renderStaged() {
  const ul = $("staged");
  ul.innerHTML = "";
  for (const f of staged) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${escapeHtml(f.name)} <span class="pill">${f.pages} page(s)</span></span>`;
    const rm = document.createElement("button");
    rm.className = "rm"; rm.textContent = "×"; rm.title = "Remove";
    rm.onclick = () => { staged = staged.filter((x) => x.id !== f.id); renderStaged(); };
    li.appendChild(rm);
    ul.appendChild(li);
  }
  $("transcribe").disabled = staged.length === 0;
  $("action-note").textContent = staged.length ? `${staged.length} file(s) ready` : "";
}

// ---- processing --------------------------------------------------------- //
async function transcribe() {
  if (!staged.length) return;
  $("transcribe").disabled = true;
  $("results-card").hidden = true;
  $("results").innerHTML = "";
  try {
    const { job_id } = await api("POST", "/api/process", { file_ids: staged.map((f) => f.id) });
    $("progress-card").hidden = false;
    pollJob(job_id);
  } catch (e) {
    $("action-note").textContent = "Error: " + e.message;
    $("transcribe").disabled = false;
  }
}

function pollJob(jobId) {
  clearInterval(pollTimer);
  const tick = async () => {
    let job;
    try { job = await api("GET", `/api/jobs/${jobId}`); }
    catch (e) { return; }
    const pct = job.total ? Math.round((job.done / job.total) * 100) : 0;
    $("progress-bar").style.width = pct + "%";
    $("progress-text").textContent =
      job.status === "running"
        ? `Processing ${job.done}/${job.total}${job.current ? " — " + job.current : ""}`
        : job.status === "error" ? "Error: " + job.error
        : `Done — ${job.total} file(s)`;
    if (job.status !== "running") {
      clearInterval(pollTimer);
      $("transcribe").disabled = false;
      if (job.status === "done") renderResults(jobId, job);
    }
  };
  tick();
  pollTimer = setInterval(tick, 700);
}

function renderResults(jobId, job) {
  $("results-card").hidden = false;
  $("zip-link").href = `/api/download/${jobId}.zip`;
  const root = $("results");
  root.innerHTML = "";
  for (const r of job.results) {
    const div = document.createElement("div");
    div.className = "result";
    let html = `<h3>${escapeHtml(r.source_name)} <span class="pill">${r.n_pages} page(s), ${r.n_lines} lines</span></h3>`;
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
        const btn = document.createElement("button");
        btn.className = "btn small";
        btn.textContent = label;
        btn.onclick = () => openPreview(im.preview, `${r.source_name} — detected lines`);
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

// ---- preview modal ------------------------------------------------------ //
function openPreview(url, title) {
  $("modal-img").src = url;
  $("modal-title").textContent = title;
  $("modal").hidden = false;
}
function closePreview() { $("modal").hidden = true; $("modal-img").src = ""; }

// ---- misc --------------------------------------------------------------- //
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function wire() {
  $("save-key").onclick = () => saveSettings({ api_key: $("api_key").value }).then(() => { $("api_key").value = ""; });
  $("clear-key").onclick = async () => {
    if (!confirm("Clear the stored Gemini key from this machine?\n(If GEMINI_API_KEY is set in your environment, that will still be used.)")) return;
    await api("DELETE", "/api/settings/api_key");
    $("api_key").value = "";
    await loadSettings();
  };
  for (const id of [...TEXT, ...NUMERIC]) $(id).addEventListener("change", () => saveSettings(gatherSettings()));
  for (const id of BOOL) $(id).addEventListener("change", () => saveSettings(gatherSettings()));
  for (const r of document.querySelectorAll("input[name=mode]")) r.addEventListener("change", () => saveSettings(gatherSettings()));

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
  $("modal-close").onclick = closePreview;
  $("modal").addEventListener("click", (e) => { if (e.target === $("modal")) closePreview(); });
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

wire();
loadSettings();
loadModels();

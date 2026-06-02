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
const NUMERIC = ["temperature", "thinking_budget", "pdf_dpi", "max_dimension", "word_confidence"];
const TEXT = ["transcription_model", "detection_model", "thinking_level", "media_resolution", "tesseract_language"];
const BOOL = ["use_mock", "preprocess", "refine_word_boxes"];

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
  const ct = document.querySelector("input[name=content_type]:checked");
  if (ct) s.content_type = ct.value;
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
      `<span class="glyph" aria-hidden="true">●</span><span>Demo mode is on — no real API call will be made.</span>`;
  } else if (data.api_key_set) {
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
// badge; "unknown" (offline/transient) and "no_key"/"mock" leave it untouched.
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
  const radio = document.querySelector(`input[name=mode][value="${s.mode}"]`);
  if (radio) radio.checked = true;
  const ctRadio = document.querySelector(`input[name=content_type][value="${s.content_type}"]`);
  if (ctRadio) ctRadio.checked = true;
  applyKeyStatus(s);
  verifyKey(); // background-verify the stored key (no-op when none/mock)
}

async function loadTesseractStatus() {
  let data;
  try {
    data = await api("GET", "/api/tesseract");
  } catch (e) {
    return;
  }
  const info = $("tesseract-info");
  if (!info) return;
  info.hidden = false;
  if (data.available) {
    info.className = "key-info";
    const langs = (data.languages || []).join(", ") || "eng";
    const ver = data.version ? ` v${escapeHtml(data.version)}` : "";
    const SOURCE_NOTE = {
      bundled: " (bundled with CursBreaker)",
      managed: " (portable build in your CursBreaker folder)",
    };
    const src = SOURCE_NOTE[data.source] || "";
    info.innerHTML =
      `<span class="glyph" aria-hidden="true">✓</span><span>Tesseract${ver} installed${src} &mdash; languages: <span class="mono">${escapeHtml(langs)}</span>. Required for Mixed and Printed-only modes.</span>`;
  } else {
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
      `<span class="glyph" aria-hidden="true">!</span><span>${detail} Mixed and Printed-only modes need it; Handwriting mode still works as usual.</span>`;
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
  const sd = $("settings-details");
  if (sd) sd.open = false;
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
    $("progress").setAttribute("aria-valuenow", String(pct));
    $("progress-text").textContent =
      job.status === "running"
        ? `Processing ${job.done}/${job.total}${job.current ? " — " + job.current : ""}`
        : job.status === "error" ? "Error: " + job.error
        : `Done — ${job.total} file(s)`;
    if (job.status !== "running") {
      clearInterval(pollTimer);
      $("transcribe").disabled = false;
      if (job.status === "done") {
        renderResults(jobId, job);
        // Send keyboard/screen-reader focus to the freshly-rendered results.
        const h = $("results-heading");
        if (h) h.focus();
      }
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

// Restore the user's last Settings open/closed choice, and remember changes.
(function restoreSettingsPanel() {
  const sd = $("settings-details");
  if (!sd) return;
  try {
    const saved = localStorage.getItem("cb.settings.open");
    if (saved === "0") sd.open = false;
    else if (saved === "1") sd.open = true;
  } catch (e) {}
  sd.addEventListener("toggle", () => {
    try { localStorage.setItem("cb.settings.open", sd.open ? "1" : "0"); } catch (e) {}
  });
})();

wire();
loadSettings();
loadModels();
loadTesseractStatus();

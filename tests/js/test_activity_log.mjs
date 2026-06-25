// Stubbed-DOM harness for app.js `renderProgress(job)` — the page-driven bar +
// verbose activity log. Run directly: `node tests/js/test_activity_log.mjs`
// (also run under pytest via tests/test_frontend_js.py). Exit 0 = all pass.
"use strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const APP_JS = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../src/cursbreaker/static/app.js"
);

const CLIENT_H = 200;   // visible height of the log box
const LINE_H = 20;      // simulated px per log line

function makeEl() {
  const el = {
    _children: [], _html: "", _attrs: {}, style: {},
    dataset: {}, classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    value: "", checked: false, hidden: false, textContent: "", className: "",
    disabled: false, href: "", onclick: null, options: [],
    scrollTop: 0, clientHeight: CLIENT_H,
    setAttribute(k, v) { this._attrs[k] = String(v); },
    getAttribute(k) { return this._attrs[k]; },
    removeAttribute(k) { delete this._attrs[k]; },
    addEventListener() {}, focus() {},
    appendChild(c) { this._children.push(c); this.options.push(c); return c; },
    replaceChildren() { this._children = []; this.options = []; },
    querySelector() { return makeEl(); }, querySelectorAll() { return []; },
  };
  Object.defineProperty(el, "childElementCount", { get() { return this._children.length; } });
  // scrollHeight grows with content (clamped to the visible height), like a real
  // scroll container, so the "at bottom?" math behaves realistically.
  Object.defineProperty(el, "scrollHeight", {
    get() { return Math.max(this.clientHeight, this._children.length * LINE_H); },
  });
  Object.defineProperty(el, "innerHTML", {
    get() { return this._html; },
    set(v) { this._html = v; if (v === "") { this._children = []; this.options = []; } },
  });
  return el;
}

let elements = {};
const el = (id) => (elements[id] || (elements[id] = makeEl()));
function freshDom() { elements = {}; }
const lines = (n) => Array.from({ length: n }, (_, i) => `line ${i + 1}`);

const document = {
  getElementById: el,
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => makeEl(),
  documentElement: makeEl(),
  activeElement: makeEl(),
  addEventListener() {},
};
const sandbox = {
  document,
  localStorage: { getItem: () => null, setItem() {} },
  fetch: () => Promise.resolve({ ok: true, json: async () => ({}) }),
  matchMedia: () => ({ matches: false }),
  console, navigator: { sendBeacon() {} },
  setInterval: () => 0, clearInterval: () => {}, setTimeout: () => 0,
  confirm: () => true, FormData: class { append() {} }, addEventListener() {},
};
sandbox.globalThis = sandbox;
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(APP_JS, "utf-8"), sandbox, { filename: "app.js" });

let failures = 0;
const check = (name, cond, extra) => {
  if (cond) console.log("PASS", name);
  else { failures++; console.log("FAIL", name, extra !== undefined ? ":: " + extra : ""); }
};
const { renderProgress } = sandbox;
check("renderProgress is exported", typeof renderProgress === "function");

// --- Group A: page-driven bar + idempotent log append -------------------- //
freshDom();
renderProgress({ status: "running", total_units: 2, done_units: 1, log: ["a", "b"] });
check("bar fills by pages (50%)", el("progress-bar").style.width === "50%", el("progress-bar").style.width);
check("aria-valuenow 50", el("progress").getAttribute("aria-valuenow") === "50", el("progress").getAttribute("aria-valuenow"));
check("headline is page-based", el("progress-text").textContent === "Processing — page 1/2", el("progress-text").textContent);
check("log seeded with 2 lines", el("activity-log").childElementCount === 2, el("activity-log").childElementCount);
check("live region announces newest (b)", el("activity-live").textContent === "b", el("activity-live").textContent);

renderProgress({ status: "running", total_units: 2, done_units: 1, log: ["a", "b", "c"] });
check("one new line appended (3 total)", el("activity-log").childElementCount === 3, el("activity-log").childElementCount);
check("live region updates to c", el("activity-live").textContent === "c", el("activity-live").textContent);

renderProgress({ status: "running", total_units: 2, done_units: 1, log: ["a", "b", "c"] });
check("re-render same log is idempotent (still 3)", el("activity-log").childElementCount === 3, el("activity-log").childElementCount);

renderProgress({ status: "running", total_units: 2, done_units: 1, log: ["x"] });
check("trimmed log rebuilds (1 line)", el("activity-log").childElementCount === 1, el("activity-log").childElementCount);
check("rebuilt line content is x", el("activity-log")._children[0].textContent === "x", el("activity-log")._children[0].textContent);

// --- Group A3: capped log keeps flowing past the cap (regression) -------- //
// The server keeps only the last N lines but reports log_total; the browser must
// append against that total, not the (plateaued) stored length — otherwise the
// log freezes once the cap is hit (the file-83-of-160 stall).
freshDom();
renderProgress({ status: "running", total_units: 1, done_units: 0, log: ["l1", "l2", "l3"], log_total: 3 });
check("A3 seeded 3 from capped window", el("activity-log").childElementCount === 3, el("activity-log").childElementCount);
// Cap is 3: next poll drops l1 and adds l4 — length still 3, but total advances.
renderProgress({ status: "running", total_units: 1, done_units: 0, log: ["l2", "l3", "l4"], log_total: 4 });
check("A3 rotated window still appends l4", el("activity-log").childElementCount === 4, el("activity-log").childElementCount);
check("A3 newest announced is l4", el("activity-live").textContent === "l4", el("activity-live").textContent);
// A jump of three at once (the window fully turns over between polls).
renderProgress({ status: "running", total_units: 1, done_units: 0, log: ["l5", "l6", "l7"], log_total: 7 });
check("A3 appends l5..l7 (7 total)", el("activity-log").childElementCount === 7, el("activity-log").childElementCount);
check("A3 last rendered line is l7", el("activity-log")._children[6].textContent === "l7", el("activity-log")._children[6].textContent);

// --- Group A4: polling fell behind by > cap -> show latest, skip the gap - //
freshDom();
renderProgress({ status: "running", total_units: 1, done_units: 0, log: ["a", "b"], log_total: 2 });
renderProgress({ status: "running", total_units: 1, done_units: 0, log: ["y", "z"], log_total: 9 });
check("A4 appends only the latest window (4 shown)", el("activity-log").childElementCount === 4, el("activity-log").childElementCount);
check("A4 newest is z", el("activity-live").textContent === "z", el("activity-live").textContent);

// --- Group B: completion forces 100% + file-count headline -------------- //
freshDom();
renderProgress({ status: "done", total_units: 0, done_units: 0, log: ["done"], results: [{}, {}] });
check("done forces bar to 100%", el("progress-bar").style.width === "100%", el("progress-bar").style.width);
check("done aria-valuenow 100", el("progress").getAttribute("aria-valuenow") === "100");
check("done headline shows file count", el("progress-text").textContent === "Done — 2 file(s)", el("progress-text").textContent);

// --- Group B2: cancelled keeps its partial bar fraction + headline ------ //
freshDom();
renderProgress({ status: "cancelled", total_units: 4, done_units: 2, log: ["Cancelled."], results: [{}] });
check("cancelled keeps partial bar (not forced to 100)", el("progress-bar").style.width === "50%", el("progress-bar").style.width);
check("cancelled headline shows completed count", el("progress-text").textContent === "Cancelled — 1 file(s) completed", el("progress-text").textContent);

// --- Group C: stick to bottom ONLY when already at the bottom ----------- //
freshDom();
const log = el("activity-log");

// C1: first render with enough lines to overflow -> follows to the bottom.
renderProgress({ status: "running", total_units: 1, done_units: 0, log: lines(30) });
check("first render follows to bottom", log.scrollTop === log.scrollHeight, `${log.scrollTop} vs ${log.scrollHeight}`);

// C2: user scrolls up -> a new line must NOT yank them back down.
log.scrollTop = 100;  // scrolled up, away from the bottom
renderProgress({ status: "running", total_units: 1, done_units: 0, log: lines(31) });
check("scrolled-up position is preserved", log.scrollTop === 100, log.scrollTop);
check("new line still appended while scrolled up", log.childElementCount === 31, log.childElementCount);

// C3: user scrolls back to the bottom -> following resumes.
log.scrollTop = log.scrollHeight - log.clientHeight;  // parked at the bottom
renderProgress({ status: "running", total_units: 1, done_units: 0, log: lines(32) });
check("at-bottom follows the new line", log.scrollTop === log.scrollHeight, `${log.scrollTop} vs ${log.scrollHeight}`);

// --- Group D: error headline -------------------------------------------- //
freshDom();
renderProgress({ status: "error", error: "boom", log: ["x"], total_units: 1, done_units: 0 });
check("error headline shows the error", el("progress-text").textContent === "Error: boom", el("progress-text").textContent);

// --- Group E: disk-full pause banner ------------------------------------ //
freshDom();
// Simulate a stale "Resuming…" button to prove the banner re-enables it.
el("resume-job").disabled = true; el("resume-job").textContent = "Resuming…";
renderProgress({ status: "running", paused: true, pause_reason: "No space left on the disk.",
                 total_units: 830, done_units: 558, log: ["Paused — no space"] });
check("E1 pause banner shown", el("pause-banner").hidden === false, el("pause-banner").hidden);
check("E1 pause reason text set", el("pause-reason").textContent === "No space left on the disk.", el("pause-reason").textContent);
check("E1 paused headline", el("progress-text").textContent === "Paused — no disk space · page 558/830", el("progress-text").textContent);
check("E1 normal cancel hidden while paused", el("cancel-job").hidden === true, el("cancel-job").hidden);
check("E1 resume button re-enabled", el("resume-job").disabled === false && el("resume-job").textContent.includes("Resume"), el("resume-job").textContent);

// E2: resumed -> running, not paused -> banner hidden, normal cancel restored.
renderProgress({ status: "running", paused: false, total_units: 830, done_units: 559, log: ["Resuming"] });
check("E2 banner hidden after resume", el("pause-banner").hidden === true, el("pause-banner").hidden);
check("E2 cancel restored after resume", el("cancel-job").hidden === false, el("cancel-job").hidden);

// E3: stopped terminal status -> banner hidden, disk-aware headline counts only saved files.
freshDom();
renderProgress({ status: "stopped", total_units: 830, done_units: 558, log: ["Stopped"],
                 results: [{}, {}, { error: "Not saved — the disk was full." }] });
check("E3 stopped banner hidden", el("pause-banner").hidden === true, el("pause-banner").hidden);
check("E3 stopped headline shows saved count",
  el("progress-text").textContent === "Stopped — out of disk space · 2 file(s) saved", el("progress-text").textContent);

console.log("\n" + (failures === 0 ? "ALL PASS" : failures + " FAILURE(S)"));
process.exit(failures === 0 ? 0 : 1);

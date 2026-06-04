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

// --- Group B: completion forces 100% + file-count headline -------------- //
freshDom();
renderProgress({ status: "done", total_units: 0, done_units: 0, log: ["done"], results: [{}, {}] });
check("done forces bar to 100%", el("progress-bar").style.width === "100%", el("progress-bar").style.width);
check("done aria-valuenow 100", el("progress").getAttribute("aria-valuenow") === "100");
check("done headline shows file count", el("progress-text").textContent === "Done — 2 file(s)", el("progress-text").textContent);

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

console.log("\n" + (failures === 0 ? "ALL PASS" : failures + " FAILURE(S)"));
process.exit(failures === 0 ? 0 : 1);

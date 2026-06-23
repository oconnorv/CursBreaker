// Stubbed-DOM harness for app.js upload helpers: batch planning, byte
// formatting, and the live upload status line. Run directly:
// `node tests/js/test_upload.mjs` (also run under pytest via
// tests/test_frontend_js.py). Exit 0 = all pass.
"use strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const APP_JS = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../src/cursbreaker/static/app.js"
);

function makeEl() {
  const el = {
    _children: [], _html: "", _attrs: {}, style: {},
    dataset: {}, classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    value: "", checked: false, hidden: false, textContent: "", className: "",
    disabled: false, href: "", onclick: null, options: [],
    scrollTop: 0, clientHeight: 200, scrollHeight: 0,
    setAttribute(k, v) { this._attrs[k] = String(v); },
    getAttribute(k) { return this._attrs[k]; },
    removeAttribute(k) { delete this._attrs[k]; },
    addEventListener() {}, focus() {},
    appendChild(c) { this._children.push(c); this.options.push(c); return c; },
    replaceChildren() { this._children = []; this.options = []; },
    querySelector() { return makeEl(); }, querySelectorAll() { return []; },
  };
  Object.defineProperty(el, "childElementCount", { get() { return this._children.length; } });
  Object.defineProperty(el, "innerHTML", {
    get() { return this._html; },
    set(v) { this._html = v; if (v === "") { this._children = []; this.options = []; } },
  });
  return el;
}

let elements = {};
const el = (id) => (elements[id] || (elements[id] = makeEl()));

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
  confirm: () => true, FormData: class { append() {} },
  XMLHttpRequest: class { open() {} send() {} },
  addEventListener() {},
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

const { planUploadBatches, formatBytes, uploadStatusText } = sandbox;
check("planUploadBatches is exported", typeof planUploadBatches === "function");
check("formatBytes is exported", typeof formatBytes === "function");
check("uploadStatusText is exported", typeof uploadStatusText === "function");

const MB = 1024 * 1024;
const file = (size) => ({ size });

// --- Group A: byte formatting -------------------------------------------- //
check("formatBytes bytes", formatBytes(812) === "812 B", formatBytes(812));
check("formatBytes MB one-decimal", formatBytes(3.4 * MB) === "3.4 MB", formatBytes(3.4 * MB));
check("formatBytes GB rounds >=10", formatBytes(10 * 1024 * MB) === "10 GB", formatBytes(10 * 1024 * MB));
check("formatBytes handles junk", formatBytes(undefined) === "0 B", formatBytes(undefined));

// --- Group B: batch planning -------------------------------------------- //
// Count cap: 45 files at 20/batch -> 20 + 20 + 5.
let b = planUploadBatches(Array.from({ length: 45 }, () => file(1)), 20, 256 * MB);
check("count cap splits into 3 batches", b.length === 3, b.length);
check("count cap batch sizes", b.map((x) => x.length).join(",") === "20,20,5", b.map((x) => x.length).join(","));

// Byte cap: 5 files of 100 MB, cap 256 MB -> [100,100] , [100,100], [100].
b = planUploadBatches([file(100 * MB), file(100 * MB), file(100 * MB), file(100 * MB), file(100 * MB)], 20, 256 * MB);
check("byte cap splits by size", b.map((x) => x.length).join(",") === "2,2,1", b.map((x) => x.length).join(","));

// A single oversized file still gets shipped (its own batch), never dropped.
b = planUploadBatches([file(900 * MB), file(10 * MB)], 20, 256 * MB);
check("oversized file gets its own batch", b.length === 2 && b[0].length === 1, JSON.stringify(b.map((x) => x.length)));
check("every file accounted for", b.flat().length === 2, b.flat().length);

check("empty input -> no batches", planUploadBatches([], 20, 256 * MB).length === 0);

// --- Group C: status line ------------------------------------------------ //
let s = uploadStatusText({ sentBytes: MB, totalBytes: 4 * MB, filesDone: 0, filesTotal: 10, scanning: false });
check("uploading shows percent", s.includes("Uploading… 25%"), s);
check("uploading shows byte progress", s.includes("(1.0 MB of 4.0 MB)"), s);
check("uploading shows file counts", s.includes("0/10 file(s) ready"), s);

s = uploadStatusText({ sentBytes: 4 * MB, totalBytes: 4 * MB, filesDone: 4, filesTotal: 10, scanning: true });
check("scanning phase is labelled", s.startsWith("Scanning pages…"), s);
check("scanning still reports files", s.includes("4/10 file(s) ready"), s);

s = uploadStatusText({ sentBytes: 0, totalBytes: 0, filesDone: 0, filesTotal: 0, scanning: false });
check("zero totals don't divide-by-zero", s === "Uploading… 0%", s);

console.log("\n" + (failures === 0 ? "ALL PASS" : failures + " FAILURE(S)"));
process.exit(failures === 0 ? 0 : 1);

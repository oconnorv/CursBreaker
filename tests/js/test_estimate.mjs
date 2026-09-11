// Harness for app.js renderEstimate: the pre-flight estimate must call BOTH the
// tokens and the cost estimates (output tokens are assumed), never "exact".
// Run: node tests/js/test_estimate.mjs
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
    style: {}, dataset: {}, options: [], _children: [],
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    value: "", checked: false, hidden: false, textContent: "", className: "",
    setAttribute() {}, getAttribute() {}, removeAttribute() {}, addEventListener() {},
    appendChild(c) { this._children.push(c); return c; }, replaceChildren() { this._children = []; },
    querySelector() { return makeEl(); }, querySelectorAll() { return []; }, focus() {},
  };
  Object.defineProperty(el, "innerHTML", { get() { return ""; }, set() {} });
  Object.defineProperty(el, "childElementCount", { get() { return this._children.length; } });
  return el;
}
const cache = {};
const document = {
  getElementById: (id) => (cache[id] || (cache[id] = makeEl())),
  querySelector: () => null, querySelectorAll: () => [], createElement: () => makeEl(),
  documentElement: makeEl(), activeElement: makeEl(), addEventListener() {},
};
const sandbox = {
  document, localStorage: { getItem: () => null, setItem() {} },
  fetch: () => Promise.resolve({ ok: true, json: async () => ({}) }),
  matchMedia: () => ({ matches: false }), console, navigator: { sendBeacon() {} },
  setInterval: () => 0, clearInterval: () => {}, setTimeout: () => 0,
  confirm: () => true, FormData: class { append() {} }, addEventListener() {},
};
sandbox.globalThis = sandbox; sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(APP_JS, "utf-8"), sandbox, { filename: "app.js" });

let failures = 0;
const check = (name, cond, extra) => {
  if (cond) console.log("PASS", name);
  else { failures++; console.log("FAIL", name, extra !== undefined ? ":: " + extra : ""); }
};
const { renderEstimate } = sandbox;
check("renderEstimate is exported", typeof renderEstimate === "function");

const d = {
  billable: true, files: 2, pages: 48, calls: 96, input: 115776,
  output_low: 144000, output_high: 432000, total_low: 259776, total_high: 547776,
  per_page_low: 3000, per_page_high: 9000, cost_low: 1.96, cost_high: 5.42,
  model: "gemini-3.1-pro-preview", model_label: "Gemini 3.1 Pro (preview)",
  price_input_per_mtok: 2.0, price_output_per_mtok: 12.0, prices_as_of: "2026-06-03",
};
const h = renderEstimate(d);

check("does NOT claim tokens are exact", !/are exact/i.test(h), h);
check("frames both ends as estimates", /both ends are estimates/i.test(h), h);
check("explains density drives the range", /pages with less writing cost less/i.test(h), h);
check("headline labels it an estimated range", /estimated range/i.test(h), h);
check("shows the cost range (dollars)", h.includes("$1.96") && h.includes("$5.42"), h);
check("shows the per-page output range", h.includes("3,000") && h.includes("9,000"), h);
check("links to live pricing", h.includes("ai.google.dev/gemini-api/docs/pricing"), h);

// A provider that can't price page images before a run (OpenAI) yields an
// output-only figure. Shown as a plain range it would understate the bill,
// because the image side is usually the larger half -- so it must read as a
// floor and say what's missing.
const partial = renderEstimate({
  billable: true, files: 1, pages: 2, calls: 2, input: 0, input_measured: false,
  provider_label: "OpenAI", output_low: 1000, output_high: 2000,
  total_low: 1000, total_high: 2000, per_page_low: 500, per_page_high: 1000,
  cost_low: 0.05, cost_high: 0.1, model_label: "GPT-6 Astra",
  price_input_per_mtok: 10, price_output_per_mtok: 50,
});
check("output-only estimate reads as a floor, not a total",
  /at least/i.test(partial) && !/estimated range/i.test(partial), partial);
check("output-only estimate says the images cost more",
  /page images/i.test(partial), partial);
check("output-only estimate names what can't be counted",
  /can't count image tokens/i.test(partial), partial);

// The normal path still reads as a range, not a floor.
const full = renderEstimate({
  billable: true, files: 1, pages: 2, calls: 2, input: 5000, input_measured: true,
  output_low: 1000, output_high: 2000, total_low: 6000, total_high: 7000,
  per_page_low: 500, per_page_high: 1000, cost_low: 0.05, cost_high: 0.1,
  model_label: "Gemini 3.1 Pro", price_input_per_mtok: 2, price_output_per_mtok: 12,
});
check("fully-measured estimate is not labelled a floor", !/at least/i.test(full), full);

const nb = renderEstimate({ billable: false, files: 3, reason: "Printed-only mode" });
check("not-billable explains no token cost", /no tokens/i.test(nb) && /Printed-only/.test(nb), nb);
check("not-billable makes no 'exact' claim", !/are exact/i.test(nb), nb);

console.log("\n" + (failures === 0 ? "ALL PASS" : failures + " FAILURE(S)"));
process.exit(failures === 0 ? 0 : 1);

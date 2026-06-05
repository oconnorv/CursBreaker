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
  billable: true, files: 2, pages: 5, calls: 10, input: 12000, output: 20000, total: 32000,
  assumed_output_tokens_per_page: 4000, cost: 0.158, model: "gemini-3.1-pro-preview",
  model_label: "Gemini 3.1 Pro (preview)", price_input_per_mtok: 2.0,
  price_output_per_mtok: 12.0, prices_as_of: "2026-06-03",
};
const h = renderEstimate(d);

check("does NOT claim tokens are exact", !/are exact/i.test(h), h);
check("says both tokens and cost are estimates", /both the token counts and the cost are estimates/i.test(h), h);
check("output is framed as assumed", /output length is assumed/i.test(h), h);
check("headline labels it an estimate", /estimated cost/i.test(h), h);
check("shows the per-page output assumption", h.includes("4,000 output tokens/page"), h);
check("shows the headline cost (cents)", h.includes("15.8¢"), h);
check("links to live pricing", h.includes("ai.google.dev/gemini-api/docs/pricing"), h);

const nb = renderEstimate({ billable: false, files: 3, reason: "Printed-only mode" });
check("not-billable explains no token cost", /no Gemini tokens/i.test(nb) && /Printed-only/.test(nb), nb);
check("not-billable makes no 'exact' claim", !/are exact/i.test(nb), nb);

console.log("\n" + (failures === 0 ? "ALL PASS" : failures + " FAILURE(S)"));
process.exit(failures === 0 ? 0 : 1);

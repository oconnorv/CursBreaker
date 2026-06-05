// Harness for app.js cost formatting: under $1 -> cents to the tenth (e.g.
// "12.3¢"); $1+ -> dollars to the cent ("$1.23"). Run: node tests/js/test_cost_format.mjs
"use strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const APP_JS = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../src/cursbreaker/static/app.js"
);

// Minimal browser stubs so app.js loads (it runs init() at the bottom).
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
const eq = (name, got, want) => {
  if (got === want) console.log("PASS", name);
  else { failures++; console.log("FAIL", name, `:: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
};
const { formatCost } = sandbox;
eq("formatCost is exported", typeof formatCost, "function");

// Under $1 -> cents to the tenth.
eq("$0.123 -> 12.3¢", formatCost(0.123), "12.3¢");
eq("$0.072 -> 7.2¢", formatCost(0.072), "7.2¢");
eq("$0.158 -> 15.8¢", formatCost(0.158), "15.8¢");
eq("sub-cent 0.00325 -> 0.3¢", formatCost(0.00325), "0.3¢");
eq("just under a dollar 0.999 -> 99.9¢", formatCost(0.999), "99.9¢");
eq("zero -> 0.0¢", formatCost(0), "0.0¢");

// $1 and up -> dollars to the cent.
eq("exactly $1 -> $1.00", formatCost(1), "$1.00");
eq("$1.23 -> $1.23", formatCost(1.23), "$1.23");
eq("$12.345 -> $12.35", formatCost(12.345), "$12.35");

// Null/undefined -> empty (unchanged).
eq("null -> ''", formatCost(null), "");
eq("undefined -> ''", formatCost(undefined), "");

console.log("\n" + (failures === 0 ? "ALL PASS" : failures + " FAILURE(S)"));
process.exit(failures === 0 ? 0 : 1);

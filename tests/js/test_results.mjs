// Harness for app.js renderResults: there must be NO standalone page-PNG
// download link (the re-rendered PNG is lower quality than the user's upload),
// while the Preview button and the .pdf/.txt/.hocr links remain.
// Run: node tests/js/test_results.mjs
"use strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const APP_JS = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../src/cursbreaker/static/app.js"
);

const created = [];
function makeEl(tag) {
  const el = {
    tagName: tag || "", _children: [], _html: "", style: {}, dataset: {}, options: [],
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    value: "", checked: false, hidden: false, textContent: "", className: "", href: "",
    onclick: null, setAttribute() {}, getAttribute() {}, removeAttribute() {}, addEventListener() {},
    appendChild(c) { this._children.push(c); return c; },
    replaceChildren() { this._children = []; }, focus() {},
    querySelector() { return makeEl(); }, querySelectorAll() { return []; },
  };
  Object.defineProperty(el, "innerHTML", { get() { return this._html; }, set(v) { this._html = v; } });
  Object.defineProperty(el, "childElementCount", { get() { return this._children.length; } });
  return el;
}
const cache = {};
const document = {
  getElementById: (id) => (cache[id] || (cache[id] = makeEl(id))),
  querySelector: () => null, querySelectorAll: () => [],
  createElement: (tag) => { const e = makeEl(tag); created.push(e); return e; },
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

const job = {
  tokens: { calls: 0 },
  results: [{
    source_name: "scan.tif", n_pages: 2, n_lines: 18, error: null,
    pdf: "/pdf", txt: "/txt", hocr: "/hocr", tokens: { total: 100, cost: null, calls: 1 },
    images: [
      { name: "scan_page_0001.png", download: "/dl/p1.png", preview: "/preview/p1" },
      { name: "scan_page_0002.png", download: "/dl/p2.png", preview: "/preview/p2" },
    ],
  }],
};
created.length = 0;
sandbox.renderResults("job123", job);

const anchors = created.filter((e) => e.tagName === "a");
const buttons = created.filter((e) => e.tagName === "button");
const resultDiv = created.find((e) => e.tagName === "div" && /class="links"/.test(e._html));

check("a Preview button is created per page", buttons.filter((b) => /Preview/.test(b.textContent)).length === 2,
  buttons.map((b) => b.textContent).join("|"));
check("NO page-PNG download link is created", anchors.filter((a) => /Page PNG|^PNG p/.test(a.textContent)).length === 0,
  anchors.map((a) => a.textContent).join("|"));
check("no element mentions 'Page PNG'", !created.some((e) => /Page PNG|PNG p\d/.test(e.textContent || "")),
  created.map((e) => e.textContent).filter(Boolean).join("|"));
check("the .txt/.hocr/.pdf links remain", !!resultDiv
  && /Download \.txt/.test(resultDiv._html) && /Download \.hocr/.test(resultDiv._html)
  && /Searchable PDF/.test(resultDiv._html), resultDiv && resultDiv._html);
check("results markup has no PNG download", !!resultDiv && !/Page PNG|PNG p\d/.test(resultDiv._html));

// --- output picker: only produced types stay downloadable ---------------- //
created.length = 0;
sandbox.renderResults("job456", {
  tokens: { calls: 0 },
  results: [{
    source_name: "only.tif", n_pages: 2, n_lines: 18, error: null,
    pdf: null, txt: null, alto: null, hocr: "/hocr",  // only hOCR produced
    tokens: { total: 0, cost: null, calls: 0 }, images: [],
  }],
});
check("produced type stays enabled (hOCR)", document.getElementById("dl-hocr").disabled === false);
check("absent type disabled (PDF)", document.getElementById("dl-pdf").disabled === true);
check("absent type disabled (ALTO)", document.getElementById("dl-alto").disabled === true);
check("absent type disabled (Text)", document.getElementById("dl-txt").disabled === true);
check("absent type unchecked (PDF)", document.getElementById("dl-pdf").checked === false);
// Result markup links only the produced format — no broken href="null".
const onlyDiv = created.find((e) => e.tagName === "div" && /class="links"/.test(e._html));
check("hOCR-only result links just hOCR", !!onlyDiv && /Download \.hocr/.test(onlyDiv._html)
  && !/Download \.txt/.test(onlyDiv._html) && !/ALTO/.test(onlyDiv._html) && !/Searchable PDF/.test(onlyDiv._html),
  onlyDiv && onlyDiv._html);
check("no broken null hrefs anywhere", !created.some((e) => /href="null"/.test(e._html || "")));

// --- selectedOutputs reads the pre-batch pickers ------------------------- //
document.getElementById("out-hocr").checked = true;
document.getElementById("out-pdf").checked = true;
check("selectedOutputs returns only ticked formats",
  JSON.stringify(sandbox.selectedOutputs()) === JSON.stringify(["hocr", "pdf"]),
  JSON.stringify(sandbox.selectedOutputs()));

console.log("\n" + (failures === 0 ? "ALL PASS" : failures + " FAILURE(S)"));
process.exit(failures === 0 ? 0 : 1);

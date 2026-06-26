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
    disabled: false, parentElement: { hidden: false },
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

// --- unified download: picker only when >1 format produced --------------- //
created.length = 0;
sandbox.renderResults("job456", {
  tokens: { calls: 0 },
  results: [{
    source_name: "only.tif", n_pages: 2, n_lines: 18, error: null,
    pdf: null, txt: null, alto: null, hocr: "/hocr",  // only hOCR produced
    tokens: { total: 0, cost: null, calls: 0 }, images: [],
  }],
});
// One format -> no redundant picker: the type fieldset is hidden, just the button.
check("single-format: type picker hidden", document.getElementById("dl-types-fieldset").hidden === true);
check("single-format: produced type checked", document.getElementById("dl-hocr").checked === true);
check("single-format: produced label shown", document.getElementById("dl-hocr").parentElement.hidden === false);
check("single-format: absent type unchecked", document.getElementById("dl-pdf").checked === false);
check("single-format: absent label hidden", document.getElementById("dl-pdf").parentElement.hidden === true);
// Per-file links only for the produced format — no broken href="null".
const onlyDiv = created.find((e) => e.tagName === "div" && /class="links"/.test(e._html));
check("hOCR-only result links just hOCR", !!onlyDiv && /Download \.hocr/.test(onlyDiv._html)
  && !/Download \.txt/.test(onlyDiv._html) && !/ALTO/.test(onlyDiv._html) && !/Searchable PDF/.test(onlyDiv._html),
  onlyDiv && onlyDiv._html);
check("no broken null hrefs anywhere", !created.some((e) => /href="null"/.test(e._html || "")));

// Two formats -> the picker appears, each produced format checked, the rest hidden.
sandbox.renderResults("job789", {
  tokens: { calls: 0 },
  results: [{
    source_name: "two.tif", n_pages: 1, n_lines: 4, error: null,
    pdf: "/pdf", txt: null, alto: null, hocr: "/hocr",  // hOCR + PDF produced
    tokens: { total: 0, cost: null, calls: 0 }, images: [],
  }],
});
check("multi-format: type picker shown", document.getElementById("dl-types-fieldset").hidden === false);
check("multi-format: produced types checked",
  document.getElementById("dl-hocr").checked === true && document.getElementById("dl-pdf").checked === true);
check("multi-format: unproduced type hidden", document.getElementById("dl-alto").parentElement.hidden === true);

// --- download: a full disk surfaces a message, not a silent failure ------- //
// Reuses the job789 render above (its dl-selected handler is wired). The
// pre-flight probe responds like a 507 so the click should show the note.
const _origFetch = sandbox.fetch;
sandbox.fetch = () => Promise.resolve({
  ok: false, status: 507,
  json: async () => ({ detail: "Not enough free disk space to build this download." }),
});
await document.getElementById("dl-selected").onclick();
sandbox.fetch = _origFetch;
check("disk-full shows a download note",
  /disk space/i.test(document.getElementById("download-note").textContent),
  document.getElementById("download-note").textContent);
check("download note is visible", document.getElementById("download-note").hidden === false);

// --- selectedOutputs reads the pre-batch pickers ------------------------- //
document.getElementById("out-hocr").checked = true;
document.getElementById("out-pdf").checked = true;
check("selectedOutputs returns only ticked formats",
  JSON.stringify(sandbox.selectedOutputs()) === JSON.stringify(["hocr", "pdf"]),
  JSON.stringify(sandbox.selectedOutputs()));

console.log("\n" + (failures === 0 ? "ALL PASS" : failures + " FAILURE(S)"));
process.exit(failures === 0 ? 0 : 1);

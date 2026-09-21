import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const outputDirectory = process.argv[2];
if (!outputDirectory) throw new Error("usage: node review_workbench_runtime.mjs <output-dir>");

const html = await readFile(join(outputDirectory, "review-workbench.v1.html"), "utf8");
assert.match(html, /data-extension-slot/);
assert.match(html, /components\/audit_workbench\/audit_workbench\.js/);
assert.doesNotMatch(html, /https?:\/\//);

const documentModel = JSON.parse(await readFile(join(outputDirectory, "review-workbench.v1.json"), "utf8"));
const extension = documentModel.extensions?.audit_workbench;
assert.equal(extension?.slot, "panels");
assert.equal(extension?.mode, "diagnostic_fixture");
assert.equal(extension?.native, false);
assert.equal(extension?.evidence_status, "diagnostic_only");

const {
  AuditWorkbenchController,
  createFixtureFacade,
  mediaSnapshot,
} = await import(pathToFileURL(join(
  outputDirectory,
  "components/audit_workbench/audit_workbench.js",
)).href);

class Element {
  constructor(ownerDocument, tagName) {
    this.ownerDocument = ownerDocument;
    this.tagName = String(tagName).toUpperCase();
    this.children = [];
    this.listeners = new Map();
    this.dataset = {};
    this.attributes = {};
    this.className = "";
    this.textContent = "";
    this.value = "";
    this.open = false;
    this.isContentEditable = false;
  }

  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, listener) {
    this.listeners.set(type, [...(this.listeners.get(type) || []), listener]);
  }
  removeEventListener(type, listener) {
    this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== listener));
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  remove() {}
  emit(type, event = {}) {
    for (const listener of this.listeners.get(type) || []) listener({ target: this, ...event });
  }
}

class Document {
  constructor() { this.listeners = new Map(); }
  createElement(tagName) { return new Element(this, tagName); }
  addEventListener(type, listener) {
    this.listeners.set(type, [...(this.listeners.get(type) || []), listener]);
  }
  removeEventListener(type, listener) {
    this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== listener));
  }
}

const model = extension.model;
const facade = createFixtureFacade(model);
const documentRef = new Document();
const root = new Element(documentRef, "main");
const controller = new AuditWorkbenchController(model, root, { facade });

await controller.next();
let snapshot = controller.snapshot();
assert.equal(snapshot.selected.episode_id, "fixture-normal-control");
assert.equal(mediaSnapshot(snapshot.selected, 2.5).pts_s, 0.113);
assert.equal(controller.panels.model.context.episode_id, "fixture-normal-control");
assert.equal(controller.editor.model.context.episode_id, "fixture-normal-control");

controller.panels.dispatch({ type: "seek", time_s: 2.5, source: "scrubber" });
await controller.quickAnnotate("normal", "canonical launch control");
const saved = await controller.saveLatestAnnotation();
assert.equal(saved.status, "saved");
const finding = await controller.persistFinding();
assert.equal(finding.status, "saved");

await controller.next();
snapshot = controller.snapshot();
assert.equal(snapshot.selected.episode_id, "fixture-missing-media");
assert.equal(mediaSnapshot(snapshot.selected, 1).status, "unavailable");
assert.equal(mediaSnapshot(snapshot.selected, 1).reason, "recording_not_present");
assert.equal(controller.panels.model.context.episode_id, "fixture-missing-media");
assert.equal(controller.editor.model.context.episode_id, "fixture-missing-media");

controller.unmount();
console.log("review_workbench_runtime: ok");

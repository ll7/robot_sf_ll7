import assert from "node:assert/strict";

import {
  ReviewEditorController,
  overlayCommands,
} from "../../robot_sf/render/web_assets/components/review_editor/review_editor.js";

class Element {
  constructor(documentRef, tagName) {
    this.ownerDocument = documentRef;
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.listeners = new Map();
    this.textContent = "";
    this.className = "";
    this.dataset = {};
    this.isContentEditable = false;
  }

  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) || []), listener]); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== listener)); }
  emit(type, event = {}) { for (const listener of this.listeners.get(type) || []) listener({ target: this, ...event }); }
}

class Document {
  constructor() { this.listeners = new Map(); }
  createElement(tagName) { return new Element(this, tagName); }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) || []), listener]); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== listener)); }
}

const documentRef = new Document();
const root = new Element(documentRef, "main");
const model = {
  schema_version: "review-editor.v1",
  context: { episode_id: "ep", execution_id: "run", cursor: { time_s: 2 }, selection_revision: 4 },
  time: { terminal_s: 2 },
  source_identity: { units: "m" },
  annotations: [],
  storyboard: { intervals: [{ interval_id: "a", start_s: 1, end_s: 2, caption: "old" }], order: ["a"], captions: { a: "old" } },
};
const controller = new ReviewEditorController(model, root);
controller.dispatch({ type: "one-click", label: "bug" });
assert.equal(controller.snapshot().annotations.length, 1);
assert.equal(controller.snapshot().annotations[0].suspected_cause, undefined);
controller.dispatch({ type: "select-interval", interval_id: "a" });
controller.dispatch({ type: "quick-note", classification: "unclear", observed_behavior: "interval note" });
assert.deepEqual(controller.snapshot().annotations.at(-1).interval, { start_s: 1, end_s: 2 });
assert.throws(
  () => controller.dispatch({ type: "add-interval", interval_id: "too-long", start_s: 1, end_s: 3 }),
  /invalid storyboard interval/,
);
controller.dispatch({ type: "undo" });
assert.equal(controller.snapshot().annotations.length, 1);
controller.dispatch({ type: "redo" });
assert.equal(controller.snapshot().annotations.length, 2);
const beforeTyping = controller.snapshot().annotations.length;
const textarea = new Element(documentRef, "textarea");
for (const listener of documentRef.listeners.get("keydown") || []) listener({ key: "b", target: textarea, preventDefault() {} });
assert.equal(controller.snapshot().annotations.length, beforeTyping);
const commands = overlayCommands(model, [{ reference_id: "a", coordinate_frame: "image", point: [10, 20] }, { reference_id: "b", coordinate_frame: "image", point: [20, 20] }], { distances: true });
assert.equal(commands.some((item) => item.kind === "distance"), false);
controller.unmount();
console.log("review_editor_runtime: ok");

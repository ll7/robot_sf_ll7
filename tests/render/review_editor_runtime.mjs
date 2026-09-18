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
  source_identity: { units: "m", sources: { panel: { artifact_id: "panel", sha256: "a".repeat(64), source_commit: "run-1" } } },
  annotations: [],
  storyboard: {
    schema_version: "review-storyboard-edit.v1",
    source_identity: "source-token",
    intervals: [{ interval_id: "a", start_s: 1, end_s: 2, caption: "old" }],
    order: ["a"], captions: { a: "old" },
  },
};
const controller = new ReviewEditorController(model, root);
controller.dispatch({ type: "one-click", label: "bug" });
assert.equal(controller.snapshot().annotations.length, 1);
assert.equal(controller.snapshot().annotations[0].suspected_cause, undefined);
assert.equal(controller.snapshot().annotations[0].source_identity, "a".repeat(64));
controller.dispatch({ type: "select-time", time_s: 1.5 });
controller.dispatch({ type: "quick-note", classification: "unclear" });
assert.equal(controller.snapshot().annotations.at(-1).metadata.selection_revision, controller.snapshot().selection_revision);
controller.dispatch({ type: "select-interval", interval_id: "a" });
controller.dispatch({ type: "quick-note", classification: "unclear", observed_behavior: "interval note" });
assert.deepEqual(controller.snapshot().annotations.at(-1).interval, { start_s: 1, end_s: 2 });
assert.throws(
  () => controller.dispatch({ type: "add-interval", interval_id: "too-long", start_s: 1, end_s: 3 }),
  /invalid storyboard interval/,
);
controller.dispatch({ type: "undo" });
assert.equal(controller.snapshot().annotations.length, 2);
controller.dispatch({ type: "redo" });
assert.equal(controller.snapshot().annotations.length, 3);
const beforeTyping = controller.snapshot().annotations.length;
const textarea = new Element(documentRef, "textarea");
for (const listener of documentRef.listeners.get("keydown") || []) listener({ key: "b", target: textarea, preventDefault() {} });
assert.equal(controller.snapshot().annotations.length, beforeTyping);
let preventedSave = false;
for (const listener of documentRef.listeners.get("keydown") || []) listener({ key: "s", ctrlKey: true, target: root, preventDefault() { preventedSave = true; } });
assert.equal(controller.snapshot().annotations.length, beforeTyping);
assert.equal(preventedSave, true);
let release;
let committed = false;
const pending = new Promise((resolve) => { release = resolve; });
const savedRow = controller.snapshot().annotations.at(-1);
const savePromise = controller.save(savedRow, {
  expected_revision: 0,
  save: async (_record, transaction) => {
    await pending;
    transaction.before_commit();
    committed = true;
    return { revision: 1 };
  },
});
controller.dispatch({ type: "select-time", time_s: 1.75 });
release();
await assert.rejects(savePromise, /stale selection revision/);
assert.equal(committed, false);
assert.equal(controller.snapshot().autosave.state, "error");
const sourceController = new ReviewEditorController(model);
let releaseSource;
let sourceCommitted = false;
const pendingSource = new Promise((resolve) => { releaseSource = resolve; });
const sourceSave = sourceController.save(sourceController.snapshot().annotations[0] || {
  provenance_status: "verified",
  source_identity: "a".repeat(64),
  source_revision: "run-1",
}, {
  expected_revision: 0,
  save: async (_record, transaction) => {
    await pendingSource;
    transaction.before_commit();
    sourceCommitted = true;
    return { revision: 1 };
  },
});
sourceController.model.source_identity.sources.panel.sha256 = "b".repeat(64);
releaseSource();
await assert.rejects(sourceSave, /stale source identity/);
assert.equal(sourceCommitted, false);
const commands = overlayCommands(model, [{ reference_id: "a", coordinate_frame: "image", point: [10, 20] }, { reference_id: "b", coordinate_frame: "image", point: [20, 20] }], { distances: true });
assert.equal(commands.some((item) => item.kind === "distance"), false);
const reloadRecord = (overrides = {}) => ({
  revision: 4,
  record: {
    action_id: controller.storyboardRecordId,
    target_id: "source-token",
    details: {
      schema_version: "review-storyboard-edit.v1",
      record_id: controller.storyboardRecordId,
      source_identity: "source-token",
      source_revision: controller.snapshot().storyboard.source_revision,
      storyboard: {
        schema_version: "review-storyboard-edit.v1",
        source_identity: "source-token",
        source_revision: controller.snapshot().storyboard.source_revision,
        intervals: [], order: [], captions: {},
      },
      ...overrides,
    },
  },
});
await assert.rejects(
  controller.reload({ record_id: controller.storyboardRecordId, load: async () => ({ revision: 4, record: { action_id: controller.storyboardRecordId, details: {} } }) }),
  /schema_version/,
);
await assert.rejects(
  controller.reload({ record_id: controller.storyboardRecordId, load: async () => reloadRecord({ source_identity: "other" }) }),
  /source identity/,
);
await assert.rejects(
  controller.reload({
    record_id: controller.storyboardRecordId,
    load: async () => reloadRecord({
      storyboard: {
        schema_version: "review-storyboard-edit.v1",
        source_identity: "source-token",
        source_revision: controller.snapshot().storyboard.source_revision,
        intervals: [{ interval_id: "outside", start_s: 0, end_s: 3 }], order: ["outside"], captions: { outside: "" },
      },
    }),
  }),
  /invalid storyboard interval/,
);
let releaseStoryboard;
let storyboardCommitted = false;
const pendingStoryboard = new Promise((resolve) => { releaseStoryboard = resolve; });
const storyboardSave = controller.saveStoryboard({
  expected_revision: 0,
  save: async (_record, transaction) => {
    await pendingStoryboard;
    transaction.before_commit();
    storyboardCommitted = true;
    return { revision: 1 };
  },
});
controller.dispatch({ type: "select-time", time_s: 1.5 });
releaseStoryboard();
await assert.rejects(storyboardSave, /stale selection revision/);
assert.equal(storyboardCommitted, false);
await controller.reload({
  record_id: controller.storyboardRecordId,
  load: async (recordId) => ({
    revision: 4,
    record: {
      action_id: recordId,
      target_id: "source-token",
      details: {
        schema_version: "review-storyboard-edit.v1",
        record_id: recordId,
        source_identity: "source-token",
        source_revision: controller.snapshot().storyboard.source_revision,
        storyboard: {
          schema_version: "review-storyboard-edit.v1",
          source_identity: "source-token",
          source_revision: controller.snapshot().storyboard.source_revision,
          intervals: [], order: [], captions: {},
        },
      },
    },
  }),
});
assert.equal(controller.snapshot().autosave.saved_revision, 4);
controller.unmount();
console.log("review_editor_runtime: ok");

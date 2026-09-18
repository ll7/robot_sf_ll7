import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  ReviewEditorController,
  overlayCommands,
  sourceRevision,
  storyboardRecordId,
} from "../../robot_sf/render/web_assets/components/review_editor/review_editor.js";

if (process.argv[2]) {
  const bundle = JSON.parse(readFileSync(process.argv[2], "utf8"));
  const pythonModel = bundle.model || bundle;
  const pythonController = new ReviewEditorController(pythonModel);
  const prepared = [];
  await pythonController.saveStoryboard({
    transaction: {
      atomic: true,
      prepare: async (record, token) => {
        prepared.push({ record, token });
        return { record, token };
      },
      commit: async () => ({ revision: 1 }),
    },
  });
  assert.equal(prepared.length, 1);
  assert.equal(prepared[0].token.expected_revision, 0);
  assert.equal(prepared[0].token.record_id, pythonModel.storyboard_record_id);
  assert.equal(prepared[0].record.details.source_identity, pythonModel.storyboard.source_identity);
  assert.equal(prepared[0].record.details.source_revision, pythonModel.storyboard.source_revision);
  const legacyModel = JSON.parse(JSON.stringify(pythonModel));
  delete legacyModel.storyboard_record_id;
  assert.equal(storyboardRecordId(legacyModel), pythonModel.storyboard_record_id);
  if (bundle.stored) {
    await pythonController.reload({
      record_id: pythonModel.storyboard_record_id,
      load: async () => bundle.stored,
    });
    assert.equal(pythonController.snapshot().autosave.saved_revision, bundle.stored.revision);
  }
  const foreignRecordId = "storyboard-foreign";
  await assert.rejects(
    pythonController.saveStoryboard({
      record_id: foreignRecordId,
      transaction: {
        atomic: true,
        prepare: async () => ({ proposal: "foreign" }),
        commit: async () => ({ revision: 1 }),
      },
    }),
    /canonical storyboard record_id/,
  );
  let foreignLoadCalled = false;
  await assert.rejects(
    pythonController.reload({
      record_id: foreignRecordId,
      load: async () => {
        foreignLoadCalled = true;
        return bundle.stored;
      },
    }),
    /canonical storyboard record_id/,
  );
  assert.equal(foreignLoadCalled, false);
  console.log("review_editor_python_model_runtime: ok");
  process.exit(0);
}

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
  source_identity: { units: "m", sources: { panel: {
    artifact_id: "panel", sha256: "a".repeat(64), source_commit: "run-1",
    schema: "panel.v1", config_identity: "cfg",
  } } },
  annotations: [],
  storyboard: {
    schema_version: "review-storyboard-edit.v1",
    source_identity: "source-token",
    intervals: [{ interval_id: "a", start_s: 1, end_s: 2, caption: "old" }],
    order: ["a"], captions: { a: "old" },
  },
};
const canonicalSourceRevision = sourceRevision(model);
model.storyboard.source_revision = canonicalSourceRevision;
assert.equal(
  storyboardRecordId(model),
  "storyboard-29a47556ff0f216b8a5199defaf57c039ac6a0221a5f72d34774b265ab842aee",
);
assert.equal(sourceRevision(model), canonicalSourceRevision);
const controller = new ReviewEditorController(model, root);
const foreignRecordId = "storyboard-foreign";
assert.equal(storyboardRecordId({ ...model, storyboard_record_id: foreignRecordId }), controller.storyboardRecordId);
assert.throws(
  () => new ReviewEditorController({ ...model, storyboard_record_id: foreignRecordId }),
  /canonical storyboard record_id/,
);
await assert.rejects(
  controller.saveStoryboard({
    record_id: foreignRecordId,
    transaction: {
      atomic: true,
      prepare: async () => ({ proposal: "foreign" }),
      commit: async () => ({ revision: 1 }),
    },
  }),
  /canonical storyboard record_id/,
);
let foreignLoadCalled = false;
await assert.rejects(
  controller.reload({
    record_id: foreignRecordId,
    load: async () => {
      foreignLoadCalled = true;
      return {};
    },
  }),
  /canonical storyboard record_id/,
);
assert.equal(foreignLoadCalled, false);
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
  transaction: {
    atomic: true,
    prepare: async (_record) => {
      await pending;
      return { proposal: "annotation" };
    },
    commit: async () => {
      committed = true;
      return { revision: 1 };
    },
  },
});
controller.dispatch({ type: "select-time", time_s: 1.75 });
release();
await assert.rejects(savePromise, /stale selection revision/);
assert.equal(committed, false);
assert.equal(controller.snapshot().autosave.state, "error");
await assert.rejects(
  controller.save(savedRow, {
    expected_revision: 0,
    save: async () => {
      committed = true;
      return { revision: 1 };
    },
  }),
  /atomic transaction/,
);
assert.equal(committed, false);
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
  transaction: {
    atomic: true,
    prepare: async () => {
      await pendingSource;
      return { proposal: "source" };
    },
    commit: async () => {
      sourceCommitted = true;
      return { revision: 1 };
    },
  },
});
sourceController.model.source_identity.sources.panel.sha256 = "b".repeat(64);
releaseSource();
await assert.rejects(sourceSave, /stale source identity/);
assert.equal(sourceCommitted, false);
const revisionController = new ReviewEditorController(model);
const revisionTokens = [];
const revisionTransaction = {
  atomic: true,
  prepare: async (_record, token) => {
    revisionTokens.push({ record_id: token.record_id, expected_revision: token.expected_revision });
    return { token };
  },
  commit: async (_proposal, token) => ({ revision: token.expected_revision + 1 }),
};
revisionController.dispatch({ type: "one-click", label: "normal" });
const annotationA = revisionController.snapshot().annotations.at(-1);
await revisionController.save(annotationA, { transaction: revisionTransaction });
revisionController.dispatch({ type: "one-click", label: "bug" });
const annotationB = revisionController.snapshot().annotations.at(-1);
await revisionController.save(annotationB, { transaction: revisionTransaction });
await revisionController.saveStoryboard({ transaction: revisionTransaction });
assert.deepEqual(revisionTokens, [
  { record_id: annotationA.annotation_id, expected_revision: 0 },
  { record_id: annotationB.annotation_id, expected_revision: 0 },
  { record_id: revisionController.storyboardRecordId, expected_revision: 0 },
]);
let releaseCommit;
let commitEntered;
const pendingCommit = new Promise((resolve) => { releaseCommit = resolve; });
const commitGate = new Promise((resolve) => { commitEntered = resolve; });
const atomicController = new ReviewEditorController(model);
atomicController.dispatch({ type: "one-click", label: "normal" });
const atomicRecord = atomicController.snapshot().annotations.at(-1);
let atomicCommitted = false;
const atomicSave = atomicController.save(atomicRecord, {
  expected_revision: 0,
  transaction: {
    atomic: true,
    prepare: async (record) => ({ record }),
    commit: async (proposal, token) => {
      commitEntered();
      await pendingCommit;
      if (token.selection_revision !== atomicController.snapshot().selection_revision) {
        throw new Error("stale selection revision");
      }
      atomicCommitted = true;
      return { revision: 1, proposal };
    },
  },
});
await commitGate;
atomicController.dispatch({ type: "select-time", time_s: 1.25 });
releaseCommit();
await assert.rejects(atomicSave, /stale selection revision/);
assert.equal(atomicCommitted, false);
const foreignRecord = {
  ...atomicRecord,
  source_identity: "foreign-source",
  metadata: { ...atomicRecord.metadata, selection_revision: atomicController.state.selectionRevision },
};
await assert.rejects(
  atomicController.save(foreignRecord, {
    expected_revision: 0,
    transaction: { atomic: true, prepare: async (record) => ({ record }), commit: async () => ({ revision: 1 }) },
  }),
  /source identity/,
);
const commands = overlayCommands(model, [{ reference_id: "a", coordinate_frame: "image", point: [10, 20] }, { reference_id: "b", coordinate_frame: "image", point: [20, 20] }], { distances: true });
assert.equal(commands.some((item) => item.kind === "distance"), false);
const reloadRecord = (overrides = {}) => ({
  revision: 4,
  record: {
    record_id: controller.storyboardRecordId,
    action_id: controller.storyboardRecordId,
    record_type: "storyboard_edit",
    action_type: "storyboard_edit",
    target_id: "source-token",
    source_identity: model.source_identity,
    source_revision: controller.snapshot().storyboard.source_revision,
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
  /record\/action type/,
);
await assert.rejects(
  controller.reload({ record_id: controller.storyboardRecordId, load: async () => reloadRecord({ source_identity: "other" }) }),
  /source identity/,
);
await assert.rejects(
  controller.reload({ record_id: controller.storyboardRecordId, load: async () => ({ ...reloadRecord(), deleted: true }) }),
  /tombstone/,
);
await assert.rejects(
  controller.reload({ record_id: controller.storyboardRecordId, load: async () => ({ ...reloadRecord(), revision: -1 }) }),
  /revision/,
);
await assert.rejects(
  controller.reload({
    record_id: controller.storyboardRecordId,
    load: async () => {
      const value = reloadRecord();
      delete value.record.source_revision;
      return value;
    },
  }),
  /top-level source identity and revision/,
);
await assert.rejects(
  controller.reload({
    record_id: controller.storyboardRecordId,
    load: async () => {
      const value = reloadRecord();
      delete value.record.details.storyboard.order;
      return value;
    },
  }),
  /storyboard intervals, order, and captions are required/,
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
  transaction: {
    atomic: true,
    prepare: async () => {
      await pendingStoryboard;
      return { proposal: "storyboard" };
    },
    commit: async () => {
      storyboardCommitted = true;
      return { revision: 1 };
    },
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
      record_id: recordId,
      action_id: recordId,
      record_type: "storyboard_edit",
      action_type: "storyboard_edit",
      target_id: "source-token",
      source_identity: model.source_identity,
      source_revision: controller.snapshot().storyboard.source_revision,
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
const canonicalStoryboard = {
  schema_version: "audit-record.v1",
  record_type: "action_record",
  record_id: controller.storyboardRecordId,
  action_id: controller.storyboardRecordId,
  action_type: "storyboard_edit",
  actor_kind: "human",
  actor_id: "",
  target_id: "source-token",
  status: "committed",
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
  },
};
await controller.reload({
  record_id: controller.storyboardRecordId,
  load: async () => ({ revision: 5, record: canonicalStoryboard }),
});
assert.equal(controller.snapshot().autosave.saved_revision, 5);
assert.equal(controller.snapshot().record_revisions[controller.storyboardRecordId], 5);
controller.unmount();
console.log("review_editor_runtime: ok");

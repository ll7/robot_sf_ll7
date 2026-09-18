import {
  DIAGNOSTICS_MODEL_SCHEMA_VERSION,
  ReviewDiagnosticsController,
  evidenceAt,
} from "../../robot_sf/render/web_assets/components/review_diagnostics/review_diagnostics.js";

class FakeElement {
  constructor(tagName, ownerDocument) {
    this.tagName = tagName;
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.listeners = {};
    this.textContent = "";
    this.className = "";
  }

  get firstChild() {
    return this.children[0] || null;
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) this.children.splice(index, 1);
    return child;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }
}

const fakeDocument = {
  createElement(tagName) {
    return new FakeElement(tagName, fakeDocument);
  },
};

const model = {
  schema_version: DIAGNOSTICS_MODEL_SCHEMA_VERSION,
  status: "complete",
  context: { context_revision: 3 },
  toggles: { planner: { visible: true } },
  panel_status: { planner: "available" },
  panels: { planner: { status: "available", selected: "<not html>" } },
  evidence_references: [{ kind: "planner", context_revision: 3 }],
};

const controller = new ReviewDiagnosticsController(model);
if (!controller.toggle("planner", 3) || controller.toggles.planner !== false) {
  throw new Error("toggle failed");
}
if (controller.toggle("planner", 2)) throw new Error("stale toggle accepted");
if (evidenceAt(model, "planner").length !== 1) throw new Error("evidence lookup failed");
if (controller.snapshot().context_revision !== 3) throw new Error("revision mismatch");
const revisionController = new ReviewDiagnosticsController({
  ...model,
  context: { context_revision: 3, cursor: { context_revision: 3 } },
  evidence_references: [
    { kind: "planner", source_artifact_id: "trace", json_pointer: "/steps/0", context_revision: 3 },
    { kind: "controls", source_artifact_id: "trace", json_pointer: "/steps/0/controls", context_revision: 3 },
  ],
});
if (!revisionController.setContextRevision(4)) throw new Error("context revision update failed");
const revised = revisionController.snapshot();
if (revised.context_revision !== 4) throw new Error("context revision was not applied");
if (revisionController.model.context.cursor.context_revision !== 4) {
  throw new Error("cursor revision was not applied");
}
if (revised.evidence_references.length !== 2 || revised.evidence_references.some((reference) => (
  reference.context_revision !== 4
  || reference.selection_revision !== 4
  || reference.evidence_status !== "stale_context"
  || reference.missing_reason !== "context_revision_changed"
  || !reference.reference_id.endsWith(":rev-4")
))) {
  throw new Error("evidence references were not invalidated and rebuilt");
}

const root = new FakeElement("main", fakeDocument);
const rendered = new ReviewDiagnosticsController(model, root);
const pre = root.children.at(-1).children[0].children.at(-1);
if (!pre || !pre.textContent.includes("<not html>")) {
  throw new Error("recorded text was not rendered as text content");
}
if (Object.prototype.hasOwnProperty.call(root, "innerHTML")) {
  throw new Error("renderer used innerHTML");
}
if (!rendered.toggle("planner", 3) || root.children.at(-1).children.some((child) => child.dataset.panel === "planner")) {
  throw new Error("DOM toggle failed");
}

console.log("review_diagnostics_runtime: ok");

/* Offline SREV-18 recorded diagnostics component.
 *
 * The Python component owns source admission and evidence identity.  This
 * module only owns transient panel visibility and DOM rendering.  It never
 * evaluates source strings as HTML, imports a network module, or changes a
 * source artifact.
 */

export const DIAGNOSTICS_MODEL_SCHEMA_VERSION = "review-diagnostics.v1";
export const PANEL_NAMES = ["planner", "controls", "pedestrians", "failure_diagnosis"];

function finiteNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
function clone(value) {
  if (value === undefined) return undefined;
  return JSON.parse(JSON.stringify(value));
}

function modelContextRevision(model) {
  const context = model?.context;
  const value = context?.context_revision ?? model?.selection_revision ?? 0;
  return Number.isInteger(value) && value >= 0 ? value : 0;
}

export function evidenceAt(model, panelName) {
  const references = Array.isArray(model?.evidence_references)
    ? model.evidence_references
    : [];
  return references.filter((reference) => reference?.kind === panelName);
}

export class ReviewDiagnosticsController {
  constructor(model, root = null) {
    this.model = clone(model || {});
    this.root = root;
    this.revision = modelContextRevision(this.model);
    this.toggles = {};
    PANEL_NAMES.forEach((name) => {
      const configured = this.model?.toggles?.[name];
      this.toggles[name] = configured?.visible !== false;
    });
    this.render();
  }

  toggle(panelName, expectedRevision = this.revision) {
    if (!PANEL_NAMES.includes(panelName)) return false;
    if (expectedRevision !== this.revision) return false;
    this.toggles[panelName] = !this.toggles[panelName];
    this.render();
    return true;
  }

  setContextRevision(nextRevision) {
    if (!Number.isInteger(nextRevision) || nextRevision < this.revision) return false;
    this.revision = nextRevision;
    if (this.model.context) this.model.context.context_revision = nextRevision;
    this.model.selection_revision = nextRevision;
    this.render();
    return true;
  }

  snapshot() {
    return {
      schema_version: DIAGNOSTICS_MODEL_SCHEMA_VERSION,
      context_revision: this.revision,
      toggles: { ...this.toggles },
      panel_status: clone(this.model.panel_status || {}),
      evidence_references: clone(this.model.evidence_references || []),
    };
  }

  render() {
    if (!this.root || !this.root.ownerDocument) return;
    const documentRef = this.root.ownerDocument;
    while (this.root.firstChild) this.root.removeChild(this.root.firstChild);
    const status = documentRef.createElement("p");
    status.className = "review-diagnostics-status";
    status.textContent = `status: ${String(this.model.status || "unavailable")} · selection revision: ${this.revision}`;
    this.root.appendChild(status);
    const toolbar = documentRef.createElement("div");
    toolbar.className = "review-diagnostics-toolbar";
    PANEL_NAMES.forEach((name) => {
      const button = documentRef.createElement("button");
      button.type = "button";
      button.dataset.panel = name;
      button.setAttribute("aria-pressed", String(this.toggles[name]));
      button.textContent = `${this.toggles[name] ? "Hide" : "Show"} ${name}`;
      button.addEventListener("click", () => this.toggle(name));
      toolbar.appendChild(button);
    });
    this.root.appendChild(toolbar);
    const grid = documentRef.createElement("div");
    grid.className = "review-diagnostics-grid";
    PANEL_NAMES.forEach((name) => {
      if (!this.toggles[name]) return;
      const panel = this.model?.panels?.[name] || { status: "unavailable", reason: "panel_not_recorded" };
      const section = documentRef.createElement("section");
      section.className = "review-diagnostics-panel";
      section.dataset.panel = name;
      const heading = documentRef.createElement("h2");
      heading.textContent = name;
      section.appendChild(heading);
      const detail = documentRef.createElement("p");
      detail.textContent = `status: ${String(panel.status || "unavailable")}${panel.reason ? ` · ${String(panel.reason)}` : ""}`;
      section.appendChild(detail);
      const pre = documentRef.createElement("pre");
      pre.textContent = JSON.stringify(panel, null, 2);
      section.appendChild(pre);
      grid.appendChild(section);
    });
    this.root.appendChild(grid);
  }
}

export function mountDiagnostics(container, model) {
  return new ReviewDiagnosticsController(model, container);
}

export function renderDiagnostics(container, model) {
  return mountDiagnostics(container, model);
}

// Keep the browser entry self-starting for the generated standalone HTML, but
// make the DOM lookup inert in Node and in callers that mount explicitly.
if (typeof document !== "undefined") {
  const root = document.getElementById("review-diagnostics-root");
  const data = document.getElementById("review-diagnostics-data");
  if (root && data) {
    try {
      mountDiagnostics(root, JSON.parse(data.textContent || "{}"));
    } catch (error) {
      root.textContent = `diagnostics unavailable: ${String(error.message || error)}`;
    }
  }
}

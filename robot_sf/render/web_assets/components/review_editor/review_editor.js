/* Offline SREV-17 editor.
 *
 * The controller owns transient browser state only.  It never uses
 * browser key/value storage, appends to an audit file, imports a network asset, or guesses
 * a world coordinate from screen pixels.  Durable records are sent to the
 * injected BA-03/BA-05-compatible save callback with an operation ID and
 * expected revision.
 */

export const EDITOR_MODEL_SCHEMA_VERSION = "review-editor.v1";
export const ANNOTATION_SPEEDS = ["one_click", "quick", "full"];
export const TRIAGE_CLASSIFICATIONS = Object.freeze({
  normal: "normal",
  suspicious: "interesting_valid",
  bug: "planner_defect",
  unsure: "unclear",
});
export const OVERLAY_KINDS = ["numbered", "highlights", "arrows", "rings", "distances"];

const CLASSIFICATIONS = [
  "normal",
  "interesting_valid",
  "planner_defect",
  "benchmark_defect",
  "scenario_defect",
  "instrumentation_defect",
  "unclear",
];

function finite(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function intervalValue(value, fallback = null) {
  if (!value || typeof value !== "object") return fallback;
  const start = finite(value.start_s ?? value.start, fallback);
  const end = finite(value.end_s ?? value.end ?? start, start);
  if (start === null || end === null || end < start) return fallback;
  return { start_s: start, end_s: end };
}

function validSourceInterval(model, start, end) {
  if (start === null || end === null || end < start) return false;
  const duration = finite(model?.time?.terminal_s);
  return duration === null || (start >= 0 && end <= duration);
}

function context(model) {
  return model?.context && typeof model.context === "object" ? model.context : {};
}

function currentTime(model) {
  const cursor = context(model).cursor;
  return finite(cursor?.time_s ?? model?.time?.cursor?.time_s ?? model?.time?.cursor, 0);
}

function sourceIdentity(model) {
  return model?.source_identity && typeof model.source_identity === "object"
    ? model.source_identity
    : {};
}

function selectedInterval(model, intervalId, storyboard = null) {
  if (!intervalId) return null;
  const source = storyboard || model?.storyboard;
  const intervals = Array.isArray(source?.intervals) ? source.intervals : [];
  return intervals.find((item) => String(item?.interval_id || item?.id || "") === String(intervalId)) || null;
}

function newId(prefix) {
  // Browser fixture tests do not require cryptographic identity.  Keep the
  // fallback deterministic enough for old runtimes without crypto.randomUUID.
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function isTextEntry(target) {
  const element = target && target.tagName ? String(target.tagName).toLowerCase() : "";
  return element === "input" || element === "textarea" || target?.isContentEditable === true;
}

function clone(value) {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

function annotation(model, mode, classification, extra = {}) {
  const selectedInterval = extra.interval || {
    start_s: finite(extra.time_s, currentTime(model)),
    end_s: finite(extra.time_s, currentTime(model)),
  };
  const normalized = intervalValue(selectedInterval, { start_s: currentTime(model), end_s: currentTime(model) });
  const row = {
    record_type: "annotation",
    annotation_id: extra.annotation_id || newId("annotation"),
    episode_id: String(context(model).episode_id || "episode"),
    classification,
    mode,
    author_kind: extra.author_kind || "human",
    author_id: extra.author_id || "",
    interval: normalized,
    tags: Array.isArray(extra.tags) ? [...extra.tags] : [],
    observed_behavior: extra.observed_behavior || "",
    suspected_cause: extra.hypothesis || extra.suspected_cause || "",
    confidence: extra.confidence ?? null,
    references: Array.isArray(extra.references) ? clone(extra.references) : [],
    evidence: Array.isArray(extra.evidence) ? clone(extra.evidence) : [],
    review_scope: extra.full_episode ? "full_episode" : "interval",
    source_identity: extra.source_identity || sourceIdentity(model).sha256 || "",
    source_revision: extra.source_revision || context(model).source_revision || 0,
    metadata: {
      ...(extra.metadata || {}),
      actors: Array.isArray(extra.actors) ? [...extra.actors] : [],
      notes: extra.notes || "",
      execution_id: String(context(model).execution_id || ""),
      selection_revision: Number(context(model).selection_revision || context(model).context_revision || 0),
    },
  };
  // One-click and quick records intentionally contain no causal requirement;
  // an empty hypothesis/confidence remains a valid quick note.
  if (mode !== "full") {
    delete row.suspected_cause;
    delete row.confidence;
    delete row.evidence;
    delete row.references;
  }
  return row;
}

function sourceUnits(model, references) {
  const first = references.find((item) => item?.coordinate_frame === "world");
  if (first?.source?.units) return first.source.units;
  const identity = sourceIdentity(model);
  if (typeof identity.units === "string" && identity.units) return identity.units;
  return "";
}

export function overlayCommands(model, references, overlays = {}) {
  const state = Object.fromEntries(
    OVERLAY_KINDS.map((kind) => [kind, overlays[kind] ?? (kind === "numbered" || kind === "highlights")]),
  );
  const rows = Array.isArray(references) ? references : [];
  const commands = [];
  rows.forEach((reference, index) => {
    if (!reference || !Array.isArray(reference.point)) return;
    const sourcePoint = reference.coordinate_frame === "image" && Array.isArray(reference.source_point)
      ? [...reference.source_point]
      : null;
    if (state.numbered) {
      commands.push({
        kind: "numbered",
        number: index + 1,
        reference_id: reference.reference_id,
        coordinate_frame: reference.coordinate_frame,
        point: [...reference.point],
        source_point: sourcePoint,
        timestamp_s: reference.timestamp_s ?? null,
      });
    }
    if (state.highlights && reference.actor_id) {
      commands.push({ kind: "highlight", reference_id: reference.reference_id, actor_id: reference.actor_id, source_point: sourcePoint });
    }
    if (state.arrows && reference.actor_id) {
      commands.push({ kind: "arrow", reference_id: reference.reference_id, actor_id: reference.actor_id, point: [...reference.point], source_point: sourcePoint, coordinate_frame: reference.coordinate_frame });
    }
    if (state.rings) {
      commands.push({ kind: "ring", reference_id: reference.reference_id, point: [...reference.point], source_point: sourcePoint, coordinate_frame: reference.coordinate_frame });
    }
  });
  const world = rows.filter((item) => item?.coordinate_frame === "world" && Array.isArray(item.point));
  const units = sourceUnits(model, world);
  // A distance is a measurement in the recorded coordinate frame.  No image
  // or canvas distance is ever emitted.
  if (state.distances && world.length >= 2 && units) {
    const [first, second] = world;
    const dx = Number(first.point[0]) - Number(second.point[0]);
    const dy = Number(first.point[1]) - Number(second.point[1]);
    if (Number.isFinite(dx) && Number.isFinite(dy)) {
      commands.push({ kind: "distance", from: first.reference_id, to: second.reference_id, distance: Math.hypot(dx, dy), units });
    }
  }
  return commands;
}

function createButton(documentRef, text, action, type = "button") {
  const button = documentRef.createElement("button");
  button.type = type;
  button.textContent = text;
  button.addEventListener("click", (event) => {
    event.preventDefault?.();
    action(event);
  });
  return button;
}

export class ReviewEditorController {
  constructor(model, root = null, options = {}) {
    this.model = clone(model || {});
    this.root = null;
    this.options = options || {};
    this.state = {
      mode: "quick",
      selectedTime: currentTime(this.model),
      selectedInterval: context(this.model).interval_id || null,
      selectionRevision: Number(context(this.model).selection_revision || context(this.model).context_revision || 0),
      annotations: Array.isArray(this.model.annotations) ? clone(this.model.annotations) : [],
      storyboard: clone(this.model.storyboard || { intervals: [], order: [], captions: {} }),
      overlayState: {
        ...(this.model.overlay_state || {}),
        numbered: this.model.overlay_state?.numbered ?? true,
        highlights: this.model.overlay_state?.highlights ?? true,
        arrows: this.model.overlay_state?.arrows ?? false,
        rings: this.model.overlay_state?.rings ?? false,
        distances: this.model.overlay_state?.distances ?? false,
      },
      undo: [],
      redo: [],
      autosave: { state: "saved", operation_id: "", expected_revision: null, saved_revision: null, selection_revision: 0, error: "", conflict: null },
      typing: false,
    };
    this.state.autosave.selection_revision = this.state.selectionRevision;
    this._keydown = (event) => this._onKeydown(event);
    if (root) this.mount(root);
  }

  mount(root) {
    this.unmount();
    this.root = root;
    const documentRef = root?.ownerDocument || (typeof document !== "undefined" ? document : null);
    if (!documentRef || !root) return this;
    this._document = documentRef;
    this._render();
    documentRef.addEventListener("keydown", this._keydown);
    return this;
  }

  unmount() {
    if (this._document) this._document.removeEventListener("keydown", this._keydown);
    if (this.root?.replaceChildren) this.root.replaceChildren();
    this.root = null;
    this._document = null;
  }

  _snapshotForUndo() {
    return {
      mode: this.state.mode,
      selectedTime: this.state.selectedTime,
      selectedInterval: this.state.selectedInterval,
      selectionRevision: this.state.selectionRevision,
      annotations: clone(this.state.annotations),
      storyboard: clone(this.state.storyboard),
      overlayState: clone(this.state.overlayState),
    };
  }

  _restore(snapshot) {
    this.state.mode = snapshot.mode;
    this.state.selectedTime = snapshot.selectedTime;
    this.state.selectedInterval = snapshot.selectedInterval;
    this.state.selectionRevision = snapshot.selectionRevision;
    this.state.annotations = clone(snapshot.annotations);
    this.state.storyboard = clone(snapshot.storyboard);
    this.state.overlayState = clone(snapshot.overlayState);
  }

  _mutate(mutator) {
    this.state.undo.push(this._snapshotForUndo());
    this.state.redo = [];
    mutator();
    this._render();
  }

  dispatch(action = {}) {
    const type = String(action.type || "");
    if (type === "set-mode") {
      if (ANNOTATION_SPEEDS.includes(action.mode)) this.state.mode = action.mode;
    } else if (type === "select-time") {
      const time = finite(action.time_s);
      if (time !== null) {
        this.state.selectedTime = time;
        this.state.selectionRevision += 1;
      }
    } else if (type === "select-interval") {
      this.state.selectedInterval = action.interval_id || null;
      this.state.selectionRevision += 1;
    } else if (type === "toggle-overlay") {
      if (OVERLAY_KINDS.includes(action.kind)) this.state.overlayState[action.kind] = !this.state.overlayState[action.kind];
    } else if (type === "one-click") {
      const classification = TRIAGE_CLASSIFICATIONS[action.label] || TRIAGE_CLASSIFICATIONS[String(action.classification || "unsure").toLowerCase()] || action.classification;
      if (!CLASSIFICATIONS.includes(classification)) return this.snapshot();
      this._mutate(() => this.state.annotations.push(annotation(this.model, "one_click", classification, {
        ...action,
        interval: action.interval || selectedInterval(this.model, this.state.selectedInterval, this.state.storyboard),
        time_s: this.state.selectedTime,
      })));
    } else if (type === "quick-note") {
      if (!CLASSIFICATIONS.includes(action.classification)) return this.snapshot();
      this._mutate(() => this.state.annotations.push(annotation(this.model, "quick", action.classification, {
        ...action,
        interval: action.interval || selectedInterval(this.model, this.state.selectedInterval, this.state.storyboard),
        time_s: this.state.selectedTime,
      })));
    } else if (type === "structured-note") {
      if (!CLASSIFICATIONS.includes(action.classification)) return this.snapshot();
      this._mutate(() => this.state.annotations.push(annotation(this.model, "full", action.classification, {
        ...action,
        interval: action.interval || selectedInterval(this.model, this.state.selectedInterval, this.state.storyboard),
        time_s: this.state.selectedTime,
      })));
    } else if (type === "set-caption") {
      const known = (this.state.storyboard.intervals || []).some((item) => item.interval_id === action.interval_id);
      if (!known) throw new Error("unknown storyboard interval");
      this._mutate(() => {
        this.state.storyboard.captions = { ...(this.state.storyboard.captions || {}), [action.interval_id]: String(action.caption || "") };
        this.state.storyboard.intervals = (this.state.storyboard.intervals || []).map((item) => item.interval_id === action.interval_id ? { ...item, caption: String(action.caption || "") } : item);
      });
    } else if (type === "add-interval") {
      const start = finite(action.start_s);
      const end = finite(action.end_s, start);
      const intervalId = String(action.interval_id || newId("interval"));
      if (!validSourceInterval(this.model, start, end)) throw new Error("invalid storyboard interval");
      const known = (this.state.storyboard.intervals || []).map((item) => item.interval_id);
      if (known.includes(intervalId)) throw new Error("duplicate storyboard interval");
      this._mutate(() => {
        this.state.storyboard.intervals = [
          ...(this.state.storyboard.intervals || []),
          { interval_id: intervalId, start_s: start, end_s: end, caption: String(action.caption || "") },
        ];
        this.state.storyboard.order = [...(this.state.storyboard.order || []), intervalId];
        this.state.storyboard.captions = { ...(this.state.storyboard.captions || {}), [intervalId]: String(action.caption || "") };
      });
    } else if (type === "update-interval") {
      const start = finite(action.start_s);
      const end = finite(action.end_s);
      if (!validSourceInterval(this.model, start, end)) throw new Error("invalid storyboard interval");
      if (!(this.state.storyboard.intervals || []).some((item) => item.interval_id === action.interval_id)) throw new Error("unknown storyboard interval");
      this._mutate(() => {
        this.state.storyboard.intervals = (this.state.storyboard.intervals || []).map((item) => item.interval_id === action.interval_id ? { ...item, start_s: start, end_s: end } : item);
      });
    } else if (type === "remove-interval") {
      this._mutate(() => {
        this.state.storyboard.intervals = (this.state.storyboard.intervals || []).filter((item) => item.interval_id !== action.interval_id);
        this.state.storyboard.order = (this.state.storyboard.order || []).filter((item) => item !== action.interval_id);
        const captions = { ...(this.state.storyboard.captions || {}) };
        delete captions[action.interval_id];
        this.state.storyboard.captions = captions;
      });
    } else if (type === "reorder") {
      const order = Array.isArray(action.order) ? action.order.map(String) : [];
      const known = (this.state.storyboard.intervals || []).map((item) => item.interval_id);
      if (order.length !== known.length || new Set(order).size !== known.length || order.some((id) => !known.includes(id))) throw new Error("invalid storyboard order");
      this._mutate(() => { this.state.storyboard.order = order; });
    } else if (type === "undo") {
      if (this.state.undo.length) {
        this.state.redo.push(this._snapshotForUndo());
        this._restore(this.state.undo.pop());
        this._render();
      }
    } else if (type === "redo") {
      if (this.state.redo.length) {
        this.state.undo.push(this._snapshotForUndo());
        this._restore(this.state.redo.pop());
        this._render();
      }
    } else if (type === "save") {
      return this.save(action.record || this.state.annotations.at(-1), action);
    }
    this._render();
    return this.snapshot();
  }

  snapshot() {
    return {
      schema_version: EDITOR_MODEL_SCHEMA_VERSION,
      mode: this.state.mode,
      selected_time_s: this.state.selectedTime,
      selected_interval: this.state.selectedInterval,
      selection_revision: this.state.selectionRevision,
      annotations: clone(this.state.annotations),
      storyboard: clone(this.state.storyboard),
      overlay_state: clone(this.state.overlayState),
      autosave: clone(this.state.autosave),
      source_identity: clone(sourceIdentity(this.model)),
      no_local_storage: true,
    };
  }

  async save(record, options = {}) {
    if (!record) throw new Error("record is required for save");
    const save = options.save || this.options.save;
    if (typeof save !== "function") throw new Error("a BA-03/BA-05 save callback is required");
    const operationId = options.operation_id || newId("operation");
    const expectedRevision = options.expected_revision ?? this.state.autosave.saved_revision ?? 0;
    const recordSelectionRevision = Number(record?.metadata?.selection_revision);
    const expectedSelectionRevision = options.expected_selection_revision ?? (
      Number.isInteger(recordSelectionRevision) ? recordSelectionRevision : this.state.selectionRevision
    );
    if (expectedSelectionRevision !== this.state.selectionRevision) {
      throw new Error(`stale selection revision: expected ${expectedSelectionRevision}, current ${this.state.selectionRevision}`);
    }
    this.state.autosave = {
      state: "pending",
      operation_id: operationId,
      expected_revision: expectedRevision,
      saved_revision: null,
      selection_revision: this.state.selectionRevision,
      error: "",
      conflict: null,
    };
    this._render();
    try {
      const receipt = await save(clone(record), {
        operation_id: operationId,
        expected_revision: expectedRevision,
        selection_revision: this.state.selectionRevision,
        actor_kind: options.actor_kind || this.options.actor_kind || "human",
        actor_id: options.actor_id || this.options.actor_id || "",
      });
      this.state.autosave = {
        state: "saved",
        operation_id: operationId,
        expected_revision: expectedRevision,
        saved_revision: receipt?.revision ?? receipt?.record_revision ?? null,
        selection_revision: this.state.selectionRevision,
        error: "",
        conflict: null,
      };
      this._render();
      return receipt;
    } catch (error) {
      this.state.autosave = {
        state: "error",
        operation_id: operationId,
        expected_revision: expectedRevision,
        saved_revision: null,
        selection_revision: this.state.selectionRevision,
        error: String(error?.message || error),
        conflict: clone(error?.conflict || error?.details?.conflict || null),
      };
      this._render();
      throw error;
    }
  }

  _onKeydown(event) {
    if (isTextEntry(event.target)) return;
    const key = String(event.key || "").toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === "z") {
      event.preventDefault?.();
      this.dispatch({ type: event.shiftKey ? "redo" : "undo" });
    } else if ((event.ctrlKey || event.metaKey) && key === "y") {
      event.preventDefault?.();
      this.dispatch({ type: "redo" });
    } else if (key === "n") {
      event.preventDefault?.();
      this.dispatch({ type: "one-click", label: "normal" });
    } else if (key === "s") {
      event.preventDefault?.();
      this.dispatch({ type: "one-click", label: "suspicious" });
    } else if (key === "b") {
      event.preventDefault?.();
      this.dispatch({ type: "one-click", label: "bug" });
    } else if (key === "u") {
      event.preventDefault?.();
      this.dispatch({ type: "one-click", label: "unsure" });
    }
  }

  _render() {
    if (!this.root || !this._document) return;
    const documentRef = this._document;
    this.root.replaceChildren?.();
    const wrapper = documentRef.createElement("section");
    wrapper.className = "review-editor-controls";
    const modeLabel = documentRef.createElement("strong");
    modeLabel.textContent = `mode: ${this.state.mode}`;
    wrapper.appendChild(modeLabel);
    const modes = documentRef.createElement("div");
    for (const mode of ANNOTATION_SPEEDS) modes.appendChild(createButton(documentRef, mode, () => this.dispatch({ type: "set-mode", mode })));
    wrapper.appendChild(modes);
    const triage = documentRef.createElement("div");
    for (const label of Object.keys(TRIAGE_CLASSIFICATIONS)) triage.appendChild(createButton(documentRef, label, () => this.dispatch({ type: "one-click", label })));
    wrapper.appendChild(triage);
    const overlays = documentRef.createElement("div");
    for (const kind of OVERLAY_KINDS) overlays.appendChild(createButton(documentRef, `${kind}: ${this.state.overlayState[kind] ? "on" : "off"}`, () => this.dispatch({ type: "toggle-overlay", kind })));
    wrapper.appendChild(overlays);
    const history = documentRef.createElement("div");
    history.appendChild(createButton(documentRef, "Undo", () => this.dispatch({ type: "undo" })));
    history.appendChild(createButton(documentRef, "Redo", () => this.dispatch({ type: "redo" })));
    wrapper.appendChild(history);
    const status = documentRef.createElement("p");
    status.className = "autosave-status";
    status.textContent = `autosave: ${this.state.autosave.state}` + (this.state.autosave.error ? ` (${this.state.autosave.error})` : "");
    wrapper.appendChild(status);
    this.root.appendChild(wrapper);
  }
}

export function mountReviewEditor(model, root, options = {}) {
  return new ReviewEditorController(model, root, options);
}

export default ReviewEditorController;

/* Offline SREV-17 editor.
 *
 * The controller owns transient browser state only.  It never uses
 * browser key/value storage, appends to an audit file, imports a network asset, or guesses
 * a world coordinate from screen pixels.  Durable records are sent through
 * an injected atomic transaction with an operation ID, expected revision,
 * and immutable source/selection context. The transaction must expose
 * `atomic: true`, a non-durable `prepare` method, and a durable
 * compare-and-swap `commit` method. Direct save callbacks are rejected
 * because a controller-side guard cannot make an arbitrary delayed callback
 * atomic.
 */

export const EDITOR_MODEL_SCHEMA_VERSION = "review-editor.v1";
export const STORYBOARD_SCHEMA_VERSION = "review-storyboard-edit.v1";
export const AUDIT_RECORD_SCHEMA_VERSION = "audit-record.v1";
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

function stableStringify(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
}

function sameJson(left, right) {
  return stableStringify(left) === stableStringify(right);
}

function sha256Hex(value) {
  const bytes = new TextEncoder().encode(value);
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const bitLength = bytes.length * 8;
  const highLength = Math.floor(bitLength / 0x100000000);
  const lowLength = bitLength >>> 0;
  const lengthOffset = padded.length - 8;
  padded[lengthOffset] = (highLength >>> 24) & 0xff;
  padded[lengthOffset + 1] = (highLength >>> 16) & 0xff;
  padded[lengthOffset + 2] = (highLength >>> 8) & 0xff;
  padded[lengthOffset + 3] = highLength & 0xff;
  padded[lengthOffset + 4] = (lowLength >>> 24) & 0xff;
  padded[lengthOffset + 5] = (lowLength >>> 16) & 0xff;
  padded[lengthOffset + 6] = (lowLength >>> 8) & 0xff;
  padded[lengthOffset + 7] = lowLength & 0xff;
  const constants = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
    0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
    0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
    0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  const state = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ];
  const rotateRight = (word, bits) => (word >>> bits) | (word << (32 - bits));
  for (let offset = 0; offset < padded.length; offset += 64) {
    const words = new Uint32Array(64);
    for (let index = 0; index < 16; index += 1) {
      const position = offset + index * 4;
      words[index] = ((padded[position] << 24) | (padded[position + 1] << 16)
        | (padded[position + 2] << 8) | padded[position + 3]) >>> 0;
    }
    for (let index = 16; index < 64; index += 1) {
      const first = words[index - 15];
      const second = words[index - 2];
      const smallSigma0 = rotateRight(first, 7) ^ rotateRight(first, 18) ^ (first >>> 3);
      const smallSigma1 = rotateRight(second, 17) ^ rotateRight(second, 19) ^ (second >>> 10);
      words[index] = (words[index - 16] + smallSigma0 + words[index - 7] + smallSigma1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = state;
    for (let index = 0; index < 64; index += 1) {
      const bigSigma1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choose = (e & f) ^ (~e & g);
      const temporary1 = (h + bigSigma1 + choose + constants[index] + words[index]) >>> 0;
      const bigSigma0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temporary2 = (bigSigma0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temporary1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temporary1 + temporary2) >>> 0;
    }
    state[0] = (state[0] + a) >>> 0;
    state[1] = (state[1] + b) >>> 0;
    state[2] = (state[2] + c) >>> 0;
    state[3] = (state[3] + d) >>> 0;
    state[4] = (state[4] + e) >>> 0;
    state[5] = (state[5] + f) >>> 0;
    state[6] = (state[6] + g) >>> 0;
    state[7] = (state[7] + h) >>> 0;
  }
  return state.map((word) => word.toString(16).padStart(8, "0")).join("");
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
  const duration = finite(model?.time?.terminal_s ?? model?.time?.end_s);
  const origin = finite(model?.time?.origin_s ?? model?.time?.start_s, 0);
  return start >= origin && (duration === null || end <= duration);
}

function normalizeStoryboard(value, model) {
  if (value === null || value === undefined) {
    return {
      schema_version: STORYBOARD_SCHEMA_VERSION,
      source_identity: storyboardSourceIdentity(model),
      source_revision: sourceRevision(model),
      intervals: [],
      order: [],
      captions: {},
    };
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("storyboard must be an object");
  }
  if (value.schema_version !== STORYBOARD_SCHEMA_VERSION) {
    throw new Error(value.schema_version === undefined
      ? "storyboard schema_version is required"
      : `unsupported storyboard schema_version: ${value.schema_version}`);
  }
  if (!Array.isArray(value.intervals) || !Array.isArray(value.order) || !value.captions || typeof value.captions !== "object" || Array.isArray(value.captions)) {
    throw new Error("storyboard intervals, order, and captions are required");
  }
  const intervals = value.intervals.map((item) => {
    if (!item || typeof item !== "object") throw new Error("storyboard intervals must contain objects");
    const intervalId = String(item.interval_id || item.id || "");
    const start = finite(item.start_s);
    const end = finite(item.end_s, start);
    if (!intervalId || !validSourceInterval(model, start, end)) throw new Error("invalid storyboard interval");
    return {
      ...clone(item),
      interval_id: intervalId,
      start_s: start,
      end_s: end,
      caption: String(item.caption || value.captions[intervalId] || ""),
    };
  });
  const ids = intervals.map((item) => item.interval_id);
  const order = value.order.map(String);
  if (new Set(ids).size !== ids.length || order.length !== ids.length || new Set(order).size !== ids.length || order.some((id) => !ids.includes(id))) {
    throw new Error("storyboard order must contain every interval exactly once");
  }
  return {
    schema_version: STORYBOARD_SCHEMA_VERSION,
    source_identity: value.source_identity === undefined ? storyboardSourceIdentity(model) : clone(value.source_identity),
    source_revision: value.source_revision || sourceRevision(model),
    intervals,
    order,
    captions: Object.fromEntries(ids.map((id) => [id, String(value.captions[id] ?? (intervals.find((item) => item.interval_id === id)?.caption || ""))])),
  };
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

function sourceRevision(model) {
  const identity = sourceIdentity(model);
  const sources = identity.sources && typeof identity.sources === "object" ? identity.sources : {};
  const revisions = Object.fromEntries(Object.entries(sources).sort(([left], [right]) => (
    left < right ? -1 : (left > right ? 1 : 0)
  )).map(([key, item]) => [
    key,
    item && typeof item === "object"
      ? {
        sha256: item.sha256 || item.declared_sha256 || "",
        source_commit: item.source_commit || "",
        schema: item.schema || "",
        config_identity: item.config_identity || "",
      }
      : { sha256: "", source_commit: "", schema: "", config_identity: "" },
  ]));
  return stableStringify({ revisions, context: context(model).source_revision || "" });
}

function storyboardSourceIdentity(model) {
  const declared = model?.storyboard?.source_identity;
  if (typeof declared === "string" && declared) return declared;
  if (declared && typeof declared === "object") return clone(declared);
  return clone(sourceIdentity(model));
}

function storyboardRecordId(model) {
  const identity = storyboardSourceIdentity(model);
  const identityToken = typeof identity === "string"
    ? identity
    : sha256Hex(stableStringify(identity));
  return `storyboard-${sha256Hex(identityToken)}`;
}

function assertCanonicalStoryboardRecordId(recordId, canonicalRecordId, label) {
  if (String(recordId) !== canonicalRecordId) {
    throw new Error(`${label} must match canonical storyboard record_id`);
  }
}

function sourceEntry(model) {
  const identity = sourceIdentity(model);
  const sources = identity.sources && typeof identity.sources === "object" ? identity.sources : {};
  const first = Object.values(sources).find((item) => item && typeof item === "object");
  return first || null;
}

function sourceBinding(model, extra = {}) {
  const entry = extra.source_ref && typeof extra.source_ref === "object"
    ? extra.source_ref
    : sourceEntry(model);
  const identity = sourceIdentity(model);
  const artifactId = entry?.artifact_id || "";
  const status = model?.source_identity_status?.[artifactId]?.status || (entry?.sha256 ? "verified" : "unavailable");
  return {
    source_ref: entry ? clone(entry) : undefined,
    source_identity: extra.source_identity || entry?.sha256 || entry?.artifact_id || identity.sha256 || "",
    source_revision: extra.source_revision || entry?.source_commit || context(model).source_revision || 0,
    provenance_status: extra.provenance_status || status,
  };
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

function frozenClone(value) {
  const copy = clone(value);
  const freeze = (item) => {
    if (!item || typeof item !== "object" || Object.isFrozen(item)) return item;
    Object.values(item).forEach(freeze);
    return Object.freeze(item);
  };
  return freeze(copy);
}

function annotation(model, mode, classification, extra = {}) {
  const selectedInterval = extra.interval || {
    start_s: finite(extra.time_s, currentTime(model)),
    end_s: finite(extra.time_s, currentTime(model)),
  };
  const normalized = intervalValue(selectedInterval, { start_s: currentTime(model), end_s: currentTime(model) });
  if (!normalized || !validSourceInterval(model, normalized.start_s, normalized.end_s)) {
    throw new Error("annotation interval is outside the recorded source time range");
  }
  const binding = sourceBinding(model, extra);
  const selectionRevision = Number(extra.selection_revision ?? context(model).selection_revision ?? context(model).context_revision ?? 0);
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
    ...binding,
    metadata: {
      ...(extra.metadata || {}),
      actors: Array.isArray(extra.actors) ? [...extra.actors] : [],
      notes: extra.notes || "",
      execution_id: String(context(model).execution_id || ""),
      selection_revision: selectionRevision,
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
  const displayPoint = (reference) => {
    const transform = reference?.calibration;
    const sourcePoint = reference?.source_point;
    if (reference?.coordinate_frame !== "image" || !Array.isArray(sourcePoint) || !transform) return null;
    const sourceWidth = finite(transform.source_width);
    const sourceHeight = finite(transform.source_height);
    const displayWidth = finite(transform.display_width);
    const displayHeight = finite(transform.display_height);
    const cropX = finite(transform.crop_x, 0);
    const cropY = finite(transform.crop_y, 0);
    const cropWidth = finite(transform.crop_width, sourceWidth);
    const cropHeight = finite(transform.crop_height, sourceHeight);
    if (![sourceWidth, sourceHeight, displayWidth, displayHeight, cropWidth, cropHeight].every((value) => value !== null && value > 0)) return null;
    return [
      (Number(sourcePoint[0]) - cropX) * displayWidth / cropWidth,
      (Number(sourcePoint[1]) - cropY) * displayHeight / cropHeight,
    ];
  };
  rows.forEach((reference, index) => {
    if (!reference || !Array.isArray(reference.point)) return;
    const sourcePoint = reference.coordinate_frame === "image" && Array.isArray(reference.source_point)
      ? [...reference.source_point]
      : null;
    const transformedPoint = displayPoint(reference);
    if (state.numbered) {
      commands.push({
        kind: "numbered",
        number: index + 1,
        reference_id: reference.reference_id,
        coordinate_frame: reference.coordinate_frame,
        point: [...reference.point],
        source_point: sourcePoint,
        display_point: transformedPoint,
        timestamp_s: reference.timestamp_s ?? null,
      });
    }
    if (state.highlights && reference.actor_id) {
      commands.push({ kind: "highlight", reference_id: reference.reference_id, actor_id: reference.actor_id, source_point: sourcePoint, display_point: transformedPoint });
    }
    if (state.arrows && reference.actor_id) {
      commands.push({ kind: "arrow", reference_id: reference.reference_id, actor_id: reference.actor_id, point: [...reference.point], source_point: sourcePoint, display_point: transformedPoint, coordinate_frame: reference.coordinate_frame });
    }
    if (state.rings) {
      commands.push({ kind: "ring", reference_id: reference.reference_id, point: [...reference.point], source_point: sourcePoint, display_point: transformedPoint, coordinate_frame: reference.coordinate_frame });
    }
  });
  const world = rows.filter((item) => item?.coordinate_frame === "world" && Array.isArray(item.point));
  const units = sourceUnits(model, world);
  const worldUnits = world.map((item) => item?.source?.units || sourceIdentity(model).units || "");
  const sourceKeys = world.map((item) => `${item?.source?.artifact_id || ""}:${item?.source?.sha256 || ""}`);
  // A distance is a measurement in the recorded coordinate frame.  No image
  // or canvas distance is ever emitted.
  if (state.distances && world.length >= 2 && units && worldUnits.every((value) => value === units) && sourceKeys.every((value) => value === sourceKeys[0])) {
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
      storyboard: normalizeStoryboard(this.model.storyboard, this.model),
      recordRevisions: {},
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
    this.storyboardRecordId = storyboardRecordId(this.model);
    for (const [label, declaredId] of [
      ["option storyboard_record_id", this.options.storyboard_record_id],
      ["model storyboard_record_id", this.model.storyboard_record_id],
      ["storyboard record_id", this.model.storyboard?.record_id],
    ]) {
      if (declaredId !== undefined && declaredId !== null) {
        assertCanonicalStoryboardRecordId(declaredId, this.storyboardRecordId, label);
      }
    }
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
        selection_revision: this.state.selectionRevision,
        interval: action.interval || selectedInterval(this.model, this.state.selectedInterval, this.state.storyboard),
        time_s: this.state.selectedTime,
      })));
    } else if (type === "quick-note") {
      if (!CLASSIFICATIONS.includes(action.classification)) return this.snapshot();
      this._mutate(() => this.state.annotations.push(annotation(this.model, "quick", action.classification, {
        ...action,
        selection_revision: this.state.selectionRevision,
        interval: action.interval || selectedInterval(this.model, this.state.selectedInterval, this.state.storyboard),
        time_s: this.state.selectedTime,
      })));
    } else if (type === "structured-note") {
      if (!CLASSIFICATIONS.includes(action.classification)) return this.snapshot();
      this._mutate(() => this.state.annotations.push(annotation(this.model, "full", action.classification, {
        ...action,
        selection_revision: this.state.selectionRevision,
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
    } else if (type === "save-storyboard") {
      return this.saveStoryboard(action);
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
      record_revisions: clone(this.state.recordRevisions),
      overlay_state: clone(this.state.overlayState),
      autosave: clone(this.state.autosave),
      source_identity: clone(sourceIdentity(this.model)),
      no_local_storage: true,
    };
  }

  _saveContext() {
    return {
      selection_revision: this.state.selectionRevision,
      selected_time_s: this.state.selectedTime,
      selected_interval: this.state.selectedInterval,
      source_identity: clone(sourceIdentity(this.model)),
      source_revision: sourceRevision(this.model),
      context: clone(context(this.model)),
    };
  }

  _recordId(record) {
    return String(
      record?.record_id
        || record?.annotation_id
        || record?.review_id
        || record?.action_id
        || "",
    );
  }

  _expectedRevision(record, explicitRevision) {
    if (explicitRevision !== undefined && explicitRevision !== null) return explicitRevision;
    const recordId = this._recordId(record);
    const knownRevision = this.state.recordRevisions[recordId];
    if (Number.isInteger(knownRevision)) return knownRevision;
    const recordRevision = Number.isInteger(record?.revision)
      ? record.revision
      : (Number.isInteger(record?.record_revision) ? record.record_revision : null);
    return recordRevision ?? 0;
  }

  _rememberRevision(recordId, revision) {
    if (recordId && Number.isInteger(revision) && revision >= 0) {
      this.state.recordRevisions[recordId] = revision;
    }
  }

  _assertSaveContext(captured) {
    if (this.state.selectionRevision !== captured.selection_revision) {
      throw new Error(`stale selection revision: expected ${captured.selection_revision}, current ${this.state.selectionRevision}`);
    }
    if (this.state.selectedTime !== captured.selected_time_s || this.state.selectedInterval !== captured.selected_interval) {
      throw new Error("stale selection context");
    }
    if (!sameJson(sourceIdentity(this.model), captured.source_identity) || sourceRevision(this.model) !== captured.source_revision) {
      throw new Error("stale source identity or revision");
    }
    if (!sameJson(context(this.model), captured.context)) {
      throw new Error("stale review context");
    }
  }

  _recordContext(record, captured) {
    if (record?.provenance_status && record.provenance_status !== "verified") {
      throw new Error("cannot durably save an unavailable or stale annotation");
    }
    if (Object.prototype.hasOwnProperty.call(record || {}, "source_identity")) {
      const value = record.source_identity;
      const sourceEntries = captured.source_identity?.sources
        && typeof captured.source_identity.sources === "object"
        ? Object.values(captured.source_identity.sources)
        : [];
      const declaredIdentities = sourceEntries.flatMap((item) => (
        item && typeof item === "object"
          ? [item.sha256, item.declared_sha256].filter(Boolean).map(String)
          : []
      ));
      const matches = typeof value === "string"
        ? value.length > 0 && declaredIdentities.includes(value)
        : value && typeof value === "object" && sameJson(value, captured.source_identity);
      if (!matches) throw new Error("record source identity does not match current source");
    }
    if (record?.provenance_status === "verified"
      && (!Object.prototype.hasOwnProperty.call(record, "source_identity")
        || !Object.prototype.hasOwnProperty.call(record, "source_revision"))) {
      throw new Error("verified record source identity and revision are required");
    }
    if (Object.prototype.hasOwnProperty.call(record || {}, "source_revision")) {
      const sourceEntries = captured.source_identity?.sources && typeof captured.source_identity.sources === "object"
        ? Object.values(captured.source_identity.sources)
        : [];
      const declaredRevisions = sourceEntries.flatMap((item) => item && typeof item === "object" && item.source_commit ? [String(item.source_commit)] : []);
      const revision = record.source_revision;
      if (typeof revision !== "string" || !revision
        || (revision !== captured.source_revision && !declaredRevisions.includes(revision))) {
        throw new Error("record source revision does not match current source");
      }
    }
  }

  _transaction(options = {}, storyboard = false) {
    const transaction = options.transaction
      || (storyboard ? options.storyboard_transaction : options.annotation_transaction)
      || this.options.transaction
      || (storyboard ? this.options.storyboardTransaction : this.options.annotationTransaction);
    if (
      !transaction
      || typeof transaction !== "object"
      || transaction.atomic !== true
      || typeof transaction.prepare !== "function"
      || typeof transaction.commit !== "function"
    ) {
      throw new Error(
        "an atomic transaction with prepare/commit compare-and-swap primitives is required",
      );
    }
    return transaction;
  }

  async save(record, options = {}) {
    if (!record) throw new Error("record is required for save");
    const transaction = this._transaction(options);
    const operationId = options.operation_id || newId("operation");
    const expectedRevision = this._expectedRevision(record, options.expected_revision);
    if (!Number.isInteger(expectedRevision) || expectedRevision < 0) {
      throw new Error("expected_revision is required for every durable save");
    }
    const recordSelectionRevision = Number(record?.metadata?.selection_revision);
    const expectedSelectionRevision = options.expected_selection_revision ?? (
      Number.isInteger(recordSelectionRevision) ? recordSelectionRevision : this.state.selectionRevision
    );
    const saveContext = this._saveContext();
    if (expectedSelectionRevision !== saveContext.selection_revision) {
      throw new Error(`stale selection revision: expected ${expectedSelectionRevision}, current ${saveContext.selection_revision}`);
    }
    this._recordContext(record, saveContext);
    const token = frozenClone({
      operation_id: operationId,
      expected_revision: expectedRevision,
      selection_revision: saveContext.selection_revision,
      source_identity: saveContext.source_identity,
      source_revision: saveContext.source_revision,
      context: saveContext.context,
      record_id: this._recordId(record),
      actor_kind: options.actor_kind || this.options.actor_kind || "human",
      actor_id: options.actor_id || this.options.actor_id || "",
    });
    this.state.autosave = {
      state: "pending",
      operation_id: operationId,
      expected_revision: expectedRevision,
      saved_revision: null,
      selection_revision: saveContext.selection_revision,
      error: "",
      conflict: null,
    };
    this._render();
    try {
      // prepare is explicitly non-durable.  Revalidation immediately before
      // commit closes the delayed-prepare race; the atomic adapter owns the
      // final CAS while the durable write is in flight.
      const proposal = await transaction.prepare(clone(record), token);
      if (proposal === undefined) {
        throw new Error("atomic transaction prepare must return an uncommitted proposal");
      }
      this._assertSaveContext(saveContext);
      const receipt = await transaction.commit(proposal, token);
      const savedRevision = receipt?.revision ?? receipt?.record_revision ?? null;
      this._rememberRevision(token.record_id, savedRevision);
      this.state.autosave = {
        state: "saved",
        operation_id: operationId,
        expected_revision: expectedRevision,
        saved_revision: savedRevision,
        selection_revision: saveContext.selection_revision,
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
        selection_revision: saveContext.selection_revision,
        error: String(error?.message || error),
        conflict: clone(error?.conflict || error?.details?.conflict || null),
      };
      this._render();
      throw error;
    }
  }

  async saveStoryboard(options = {}) {
    const operationId = options.operation_id || newId("operation");
    const recordId = String(options.record_id ?? this.storyboardRecordId);
    assertCanonicalStoryboardRecordId(recordId, this.storyboardRecordId, "storyboard record_id");
    const transaction = this._transaction(options, true);
    const expectedRevision = this._expectedRevision({ record_id: recordId }, options.expected_revision);
    if (!Number.isInteger(expectedRevision) || expectedRevision < 0) {
      throw new Error("expected_revision is required for every durable save");
    }
    const saveContext = this._saveContext();
    const expectedSelectionRevision = options.expected_selection_revision ?? saveContext.selection_revision;
    if (expectedSelectionRevision !== saveContext.selection_revision) {
      throw new Error(`stale selection revision: expected ${expectedSelectionRevision}, current ${saveContext.selection_revision}`);
    }
    const storyboard = normalizeStoryboard(this.state.storyboard, this.model);
    if (!sameJson(storyboard.source_identity, storyboardSourceIdentity(this.model)) || storyboard.source_revision !== sourceRevision(this.model)) {
      throw new Error("storyboard source identity or revision is stale");
    }
    const record = {
      record_id: recordId,
      action_id: recordId,
      record_type: "storyboard_edit",
      action_type: "storyboard_edit",
      target_id: storyboardSourceIdentity(this.model),
      source_identity: clone(saveContext.source_identity),
      source_revision: saveContext.source_revision,
      details: {
        schema_version: STORYBOARD_SCHEMA_VERSION,
        record_id: recordId,
        source_identity: clone(storyboardSourceIdentity(this.model)),
        source_revision: saveContext.source_revision,
        storyboard,
      },
    };
    const token = frozenClone({
      operation_id: operationId,
      expected_revision: expectedRevision,
      selection_revision: saveContext.selection_revision,
      source_identity: saveContext.source_identity,
      source_revision: saveContext.source_revision,
      context: saveContext.context,
      record_id: recordId,
      actor_kind: options.actor_kind || this.options.actor_kind || "human",
      actor_id: options.actor_id || this.options.actor_id || "",
    });
    this.state.autosave = {
      state: "pending", operation_id: operationId, expected_revision: expectedRevision,
      saved_revision: null, selection_revision: saveContext.selection_revision, error: "", conflict: null,
    };
    this._render();
    try {
      const proposal = await transaction.prepare(clone(record), token);
      if (proposal === undefined) {
        throw new Error("atomic transaction prepare must return an uncommitted proposal");
      }
      this._assertSaveContext(saveContext);
      const receipt = await transaction.commit(proposal, token);
      const savedRevision = receipt?.revision ?? receipt?.record_revision ?? null;
      this._rememberRevision(recordId, savedRevision);
      this.state.autosave = {
        state: "saved", operation_id: operationId, expected_revision: expectedRevision,
        saved_revision: savedRevision,
        selection_revision: saveContext.selection_revision, error: "", conflict: null,
      };
      this._render();
      return receipt;
    } catch (error) {
      this.state.autosave = {
        state: "error", operation_id: operationId, expected_revision: expectedRevision,
        saved_revision: null, selection_revision: saveContext.selection_revision,
        error: String(error?.message || error), conflict: clone(error?.conflict || error?.details?.conflict || null),
      };
      this._render();
      throw error;
    }
  }

  async reload(options = {}) {
    const load = options.load || this.options.load;
    if (typeof load !== "function") throw new Error("a BA-03/BA-05 load callback is required");
    const requestedRecordId = options.record_id ?? options.storyboard_id ?? this.storyboardRecordId;
    const expectedRecordId = String(requestedRecordId);
    assertCanonicalStoryboardRecordId(expectedRecordId, this.storyboardRecordId, "storyboard record_id");
    const reloadContext = this._saveContext();
    const loaded = await load(expectedRecordId);
    this._assertSaveContext(reloadContext);
    if (loaded?.deleted === true || loaded?.tombstone === true) {
      throw new Error("loaded storyboard tombstone is not reloadable");
    }
    const record = loaded?.record && typeof loaded.record === "object" ? loaded.record : loaded;
    if (!record || typeof record !== "object" || record.deleted === true || record.tombstone === true) {
      throw new Error("loaded storyboard tombstone or record is invalid");
    }
    const canonicalAction = record.record_type === "action_record";
    const browserStoryboard = record.record_type === "storyboard_edit";
    if ((!canonicalAction && !browserStoryboard) || record.action_type !== "storyboard_edit") {
      throw new Error("loaded storyboard record/action type is invalid");
    }
    if (loaded?.record_type !== undefined && loaded.record_type !== record.record_type) {
      throw new Error("loaded storyboard record type is invalid");
    }
    if (canonicalAction
      && (record.schema_version !== AUDIT_RECORD_SCHEMA_VERSION || record.status !== "committed")) {
      throw new Error("loaded storyboard canonical action record is invalid");
    }
    if (record.record_id !== expectedRecordId || record.action_id !== expectedRecordId) {
      throw new Error("loaded storyboard record identity does not match requested record");
    }
    const revision = loaded?.revision ?? loaded?.record_revision ?? record?.revision;
    if (!Number.isInteger(revision) || revision < 0) throw new Error("loaded storyboard record revision is required");
    const details = record?.details || loaded?.details;
    if (!details || typeof details !== "object" || details.schema_version !== STORYBOARD_SCHEMA_VERSION) {
      throw new Error("loaded storyboard schema_version is invalid");
    }
    const expectedSourceIdentity = storyboardSourceIdentity(this.model);
    const loadedSourceIdentity = details.source_identity;
    if (!sameJson(loadedSourceIdentity, expectedSourceIdentity)) throw new Error("loaded storyboard source identity is stale");
    const expectedSourceRevision = sourceRevision(this.model);
    if (details.source_revision !== expectedSourceRevision) throw new Error("loaded storyboard source revision is stale");
    if (details.record_id !== expectedRecordId) throw new Error("loaded storyboard record_id is invalid");
    if (!sameJson(record?.target_id, expectedSourceIdentity)) throw new Error("loaded storyboard target identity is stale");
    if (canonicalAction) {
      if (Object.prototype.hasOwnProperty.call(record, "source_identity")
        && !sameJson(record.source_identity, expectedSourceIdentity)) {
        throw new Error("loaded storyboard source identity is stale");
      }
      if (Object.prototype.hasOwnProperty.call(record, "source_revision")
        && record.source_revision !== expectedSourceRevision) {
        throw new Error("loaded storyboard source revision is stale");
      }
    } else {
      if (!Object.prototype.hasOwnProperty.call(record, "source_identity")
        || !Object.prototype.hasOwnProperty.call(record, "source_revision")) {
        throw new Error("loaded storyboard top-level source identity and revision are required");
      }
      if (!sameJson(record.source_identity, sourceIdentity(this.model))) throw new Error("loaded storyboard source identity is stale");
      if (record.source_revision !== expectedSourceRevision) throw new Error("loaded storyboard source revision is stale");
    }
    const payload = details.storyboard;
    if (!payload || typeof payload !== "object" || !Object.prototype.hasOwnProperty.call(payload, "schema_version") || !Object.prototype.hasOwnProperty.call(payload, "source_identity") || !Object.prototype.hasOwnProperty.call(payload, "source_revision")) {
      throw new Error("loaded storyboard schema, source identity, and source revision are required");
    }
    if (!sameJson(payload.source_identity, expectedSourceIdentity) || payload.source_revision !== expectedSourceRevision) {
      throw new Error("loaded storyboard source identity or revision is stale");
    }
    const normalized = normalizeStoryboard(payload, this.model);
    if (!sameJson(normalized.source_identity, expectedSourceIdentity)) throw new Error("loaded storyboard source identity is stale");
    if (normalized.source_revision !== expectedSourceRevision) throw new Error("loaded storyboard source revision is stale");
    this.state.storyboard = normalized;
    this._rememberRevision(expectedRecordId, revision);
    this.state.autosave = {
      state: "saved",
      operation_id: "",
      expected_revision: revision,
      saved_revision: revision,
      selection_revision: reloadContext.selection_revision,
      error: "",
      conflict: null,
    };
    this._render();
    return this.snapshot();
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
    } else if ((event.ctrlKey || event.metaKey) && key === "s") {
      event.preventDefault?.();
      void this.dispatch({
        type: "save",
        record: this.state.annotations.at(-1),
      }).catch(() => {});
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
    const heading = documentRef.createElement("h2");
    heading.textContent = "Offline annotation editor";
    wrapper.appendChild(heading);
    const modeLabel = documentRef.createElement("strong");
    modeLabel.textContent = `mode: ${this.state.mode}`;
    wrapper.appendChild(modeLabel);
    const modes = documentRef.createElement("div");
    for (const mode of ANNOTATION_SPEEDS) {
      modes.appendChild(createButton(documentRef, mode, () => this.dispatch({ type: "set-mode", mode })));
    }
    wrapper.appendChild(modes);
    const triage = documentRef.createElement("div");
    for (const label of Object.keys(TRIAGE_CLASSIFICATIONS)) {
      triage.appendChild(createButton(documentRef, `triage: ${label}`, () => this.dispatch({ type: "one-click", label })));
    }
    wrapper.appendChild(triage);
    const noteActions = documentRef.createElement("div");
    noteActions.appendChild(createButton(documentRef, "Quick note: unclear", () => this.dispatch({ type: "quick-note", classification: "unclear" })));
    noteActions.appendChild(createButton(documentRef, "Full note: unclear", () => this.dispatch({ type: "structured-note", classification: "unclear" })));
    const latest = () => this.state.annotations.at(-1);
    noteActions.appendChild(createButton(documentRef, "Save annotation", () => {
      void this.dispatch({
        type: "save",
        record: latest(),
      }).catch(() => {});
    }));
    wrapper.appendChild(noteActions);
    const overlays = documentRef.createElement("div");
    for (const kind of OVERLAY_KINDS) {
      overlays.appendChild(createButton(documentRef, `${kind}: ${this.state.overlayState[kind] ? "on" : "off"}`, () => this.dispatch({ type: "toggle-overlay", kind })));
    }
    wrapper.appendChild(overlays);
    const history = documentRef.createElement("div");
    history.appendChild(createButton(documentRef, "Undo", () => this.dispatch({ type: "undo" })));
    history.appendChild(createButton(documentRef, "Redo", () => this.dispatch({ type: "redo" })));
    wrapper.appendChild(history);
    const storyboardHeading = documentRef.createElement("h3");
    storyboardHeading.textContent = "Storyboard edits";
    wrapper.appendChild(storyboardHeading);
    const storyboard = documentRef.createElement("div");
    const intervals = Array.isArray(this.state.storyboard?.intervals) ? this.state.storyboard.intervals : [];
    const order = Array.isArray(this.state.storyboard?.order) ? this.state.storyboard.order : intervals.map((item) => item.interval_id);
    for (const intervalId of order) {
      const interval = intervals.find((item) => item.interval_id === intervalId);
      if (!interval) continue;
      const row = documentRef.createElement("div");
      row.className = "storyboard-row";
      const label = documentRef.createElement("span");
      label.textContent = `${interval.interval_id}: ${interval.start_s}–${interval.end_s}`;
      row.appendChild(label);
      const caption = documentRef.createElement("input");
      caption.type = "text";
      caption.value = String(interval.caption || this.state.storyboard.captions?.[intervalId] || "");
      caption.addEventListener("change", () => this.dispatch({ type: "set-caption", interval_id: intervalId, caption: caption.value }));
      row.appendChild(caption);
      storyboard.appendChild(row);
    }
    storyboard.appendChild(createButton(documentRef, "Save storyboard", () => {
      void this.dispatch({ type: "save-storyboard" }).catch(() => {});
    }));
    wrapper.appendChild(storyboard);
    const sourceStatus = documentRef.createElement("p");
    const sources = sourceIdentity(this.model).sources || {};
    sourceStatus.textContent = `sources: ${Object.keys(sources).length}; time: ${this.state.selectedTime}`;
    wrapper.appendChild(sourceStatus);
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

export { sourceRevision, storyboardRecordId };

const dataElement = typeof document === "undefined" ? null : document.getElementById("review-editor-data");
const rootElement = typeof document === "undefined" ? null : document.getElementById("review-editor-root");
if (dataElement && rootElement) {
  try {
    const model = JSON.parse(dataElement.textContent || "{}");
    mountReviewEditor(model, rootElement);
  } catch (error) {
    rootElement.textContent = `Unable to load review editor model: ${error}`;
  }
}

export default ReviewEditorController;

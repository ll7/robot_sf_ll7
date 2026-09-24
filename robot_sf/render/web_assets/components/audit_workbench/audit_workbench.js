/* Offline BA-06 workbench shell.
 *
 * The shell owns transient queue/pane state only.  Queue selection and durable
 * writes are always sent to an injected facade.  The fixture facade below is
 * intentionally disposable browser-test state; it is not BA-05, SREV-17, or a
 * native/live execution path.  An optional native diagnostic action is exposed
 * only through a server-held capability and sends a closed, CAS-bound request;
 * it never accepts source paths, runner identities, or credentials. Annotation
 * construction, CAS guards and autosave rendering remain owned by
 * review_editor.js.
 */

import { mountReviewEditor } from "../review_editor/review_editor.js";
import { mountReviewPanels, nearestSample } from "../review_panels/review_panels.js";

export const AUDIT_WORKBENCH_MODEL_SCHEMA_VERSION = "audit-workbench.v1";
export const AUDIT_WORKBENCH_SERVICE_ID = "ba06-fixture-facade";

const CODEX_DEFAULT_TOKEN_BUDGET = 2048;
const CODEX_DEFAULT_COMPUTE_BUDGET = 1;
const CODEX_MAX_PROMPT_LENGTH = 8192;
const CODEX_MAX_OPERATION_ID_LENGTH = 128;
const CODEX_MAX_REASON_LENGTH = 512;
const CODEX_MAX_ACTIVITY = 24;
const CODEX_MAX_ACTIVITY_MESSAGE = 512;
const CODEX_MAX_EVIDENCE_IDS = 24;
const CODEX_OPAQUE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const NATIVE_DIAGNOSTIC_DEFAULT_EPSILON_M = 1e-9;
const NATIVE_DIAGNOSTIC_DEFAULT_DEADLINE_S = 30;
const NATIVE_DIAGNOSTIC_MAX_DEADLINE_S = 60;
const NATIVE_DIAGNOSTIC_STATUSES = new Set([
  "running", "complete", "failed", "unavailable", "cancelled", "conflict", "denied",
]);
const RELATED_DEFAULT_MODE = "same_scenario_across_planners";
const RELATED_DEFAULT_LIMIT = 20;
const RELATED_MAX_LIMIT = 100;
const RELATED_MAX_ID_LENGTH = 256;
const RELATED_MODES = new Set([
  "same_scenario_across_planners", "same_scenario",
  "same_planner_across_seeds", "same_planner", "symptom",
  "anomaly_signature", "anomaly", "geometry", "outcome",
  "metric_behaviour", "metric", "existing_finding", "finding",
]);
const RELATED_COMPATIBILITY = new Set(["compatible", "unknown", "incompatible"]);
const RELATED_CLAIM_BOUNDARY = "retrieval_only_not_benchmark_evidence";
const RELATED_MEMBERSHIP_BOUNDARY = "candidates_are_unconfirmed";

function codexArgumentSource(value, options = {}) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  return { ...options, prompt: value };
}

function codexStartArguments(value, options = {}) {
  const source = codexArgumentSource(value, options);
  return {
    prompt: String(source.prompt ?? "").slice(0, CODEX_MAX_PROMPT_LENGTH),
    operation_id: String(source.operation_id || operationId("codex")).slice(0, CODEX_MAX_OPERATION_ID_LENGTH),
    token_budget: source.token_budget ?? CODEX_DEFAULT_TOKEN_BUDGET,
    compute_budget: source.compute_budget ?? CODEX_DEFAULT_COMPUTE_BUDGET,
  };
}

function codexReadArguments(value = {}) {
  const source = typeof value === "string" ? { operation_id: value } : value || {};
  return source.operation_id === undefined
    ? {}
    : { operation_id: String(source.operation_id).slice(0, CODEX_MAX_OPERATION_ID_LENGTH) };
}

function codexReconnectArguments(value = {}, options = {}) {
  const source = { ...options, ...(value || {}) };
  const sessionId = String(source.codex_session_id || "").slice(0, CODEX_MAX_OPERATION_ID_LENGTH);
  const operation = String(source.operation_id || operationId("codex-reconnect"))
    .slice(0, CODEX_MAX_OPERATION_ID_LENGTH);
  if (!CODEX_OPAQUE_ID.test(sessionId)) throw new Error("codex_session_id must be opaque");
  if (!CODEX_OPAQUE_ID.test(operation)) throw new Error("operation_id must be opaque");
  const revision = (name) => {
    const current = source[name];
    if (!Number.isInteger(current) || current < 0) {
      throw new Error(`${name} must be a non-negative integer`);
    }
    return current;
  };
  return {
    codex_session_id: sessionId,
    operation_id: operation,
    expected_selection_revision: revision("expected_selection_revision"),
    expected_context_revision: revision("expected_context_revision"),
  };
}

function codexCancelArguments(value, options = {}) {
  const source = codexArgumentSource(value, options);
  return {
    reason: String(source.reason || "cancelled from audit workbench").slice(0, CODEX_MAX_REASON_LENGTH),
    operation_id: String(source.operation_id || "").slice(0, CODEX_MAX_OPERATION_ID_LENGTH),
  };
}

function relatedCasesArguments(value = {}, options = {}) {
  const source = typeof value === "string" ? { mode: value } : { ...options, ...(value || {}) };
  const mode = source.mode ?? RELATED_DEFAULT_MODE;
  if (typeof mode !== "string" || !RELATED_MODES.has(mode)) {
    throw new Error("unsupported related-case mode");
  }
  const limit = source.limit ?? RELATED_DEFAULT_LIMIT;
  if (!Number.isInteger(limit) || limit < 0 || limit > RELATED_MAX_LIMIT) {
    throw new Error("related-case limit is outside the permitted bound");
  }
  const result = { mode, limit };
  if (source.episode_id !== undefined) {
    if (typeof source.episode_id !== "string") throw new Error("episode_id must be text");
    const episodeId = source.episode_id.slice(0, RELATED_MAX_ID_LENGTH);
    if (!episodeId || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(episodeId)) {
      throw new Error("episode_id must be an opaque identifier");
    }
    result.episode_id = episodeId;
  }
  if (source.expected_context_revision !== undefined) {
    const revision = source.expected_context_revision;
    if (!Number.isInteger(revision) || revision < 0) {
      throw new Error("expected_context_revision must be a non-negative integer");
    }
    result.expected_context_revision = revision;
  }
  if (source.operation_id !== undefined) {
    if (typeof source.operation_id !== "string") throw new Error("operation_id must be text");
    const operation = source.operation_id.slice(0, CODEX_MAX_OPERATION_ID_LENGTH);
    if (!CODEX_OPAQUE_ID.test(operation)) throw new Error("operation_id must be opaque");
    result.operation_id = operation;
  }
  return result;
}

function materializationArguments(value = {}, options = {}) {
  const source = { ...options, ...(value || {}) };
  const operation = String(source.operation_id || operationId("materialize"))
    .slice(0, CODEX_MAX_OPERATION_ID_LENGTH);
  if (!CODEX_OPAQUE_ID.test(operation)) throw new Error("operation_id must be opaque");
  const revision = (name) => {
    const value = source[name];
    if (!Number.isInteger(value) || value < 0) {
      throw new Error(`${name} must be a non-negative integer`);
    }
    return value;
  };
  return {
    operation_id: operation,
    expected_selection_revision: revision("expected_selection_revision"),
    expected_context_revision: revision("expected_context_revision"),
  };
}

function nativeDiagnosticSource(value = {}, options = {}) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ...options };
  }
  return { ...options, ...value };
}

/** Build the exact browser-to-server native diagnostic request. */
export function nativeDiagnosticArguments(value = {}, options = {}) {
  const source = nativeDiagnosticSource(value, options);
  const operation = source.operation_id === undefined
    ? operationId("native-diagnostic") : source.operation_id;
  if (typeof operation !== "string" || !CODEX_OPAQUE_ID.test(operation)) {
    throw new Error("native diagnostic operation_id must be opaque");
  }
  const revision = (field) => {
    const value = source[field];
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error(`native diagnostic ${field} must be a non-negative integer`);
    }
    return value;
  };
  const intervention = source.intervention_id;
  if (typeof intervention !== "string" || !CODEX_OPAQUE_ID.test(intervention)) {
    throw new Error("native diagnostic intervention_id must be opaque");
  }
  const goal = source.robot_goal;
  if (!Array.isArray(goal) || goal.length !== 2
    || goal.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new Error("native diagnostic robot_goal must contain two finite numbers");
  }
  const epsilon = source.activation_epsilon_m === undefined
    ? NATIVE_DIAGNOSTIC_DEFAULT_EPSILON_M : source.activation_epsilon_m;
  if (typeof epsilon !== "number" || !Number.isFinite(epsilon) || epsilon < 0 || epsilon > 1) {
    throw new Error("native diagnostic activation_epsilon_m must be within 0..1");
  }
  const deadline = source.deadline_s === undefined
    ? NATIVE_DIAGNOSTIC_DEFAULT_DEADLINE_S : source.deadline_s;
  if (typeof deadline !== "number" || !Number.isFinite(deadline)
    || deadline <= 0 || deadline > NATIVE_DIAGNOSTIC_MAX_DEADLINE_S) {
    throw new Error("native diagnostic deadline_s is outside the permitted bound");
  }
  return {
    operation_id: operation,
    expected_selection_revision: revision("expected_selection_revision"),
    expected_context_revision: revision("expected_context_revision"),
    intervention_id: intervention,
    robot_goal: [goal[0], goal[1]],
    activation_epsilon_m: epsilon,
    deadline_s: deadline,
  };
}

function nativeDiagnosticInputIdentity(value = {}) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  return JSON.stringify({
    intervention_id: source.intervention_id ?? null,
    robot_goal: Array.isArray(source.robot_goal) ? [source.robot_goal[0], source.robot_goal[1]] : null,
    activation_epsilon_m: source.activation_epsilon_m ?? null,
    deadline_s: source.deadline_s ?? null,
  });
}

/** Same-origin transport for a server-held BA-05 facade; no token enters JS. */
export function createServiceFacade(endpoint = "/api/audit") {
  if (endpoint !== "/api/audit") throw new Error("audit endpoint must be same-origin");
  const call = async (operation, args = {}) => {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      cache: "no-store",
      body: JSON.stringify({ operation, arguments: args }),
    });
    const result = await response.json();
    if (!response.ok) throw Object.assign(new Error(result.reason || "audit service unavailable"), { status: result.status });
    if (!result || typeof result !== "object") throw new Error("audit service returned no result");
    // Presentation aliases are only for the existing controller. Preserve the
    // authoritative BA-05 status and receipt in the returned envelope.
    const presented = { ...result, service_status: result.status };
    if (result.presentation_status === "saved" && result.status === "committed") presented.status = "saved";
    if (result.presentation_status === "selected" && result.status === "complete") presented.status = "selected";
    // A queue packet remains authoritative even when its episode/recording is
    // unavailable.  Keep the BA-05 status in ``service_status`` while giving
    // the controller a selectable packet to render with an unavailable
    // inspection/editor state.
    if (result.status === "complete" && result.packet
      && ["selected", "unavailable"].includes(result.presentation_status)) {
      presented.status = "selected";
    }
    if (result.presentation_status === "empty" && result.status === "complete") presented.status = "empty";
    if (operation === "snapshot") {
      presented.selection_revision = result.selection_revision ?? 0;
      if (Array.isArray(result.annotations)) presented.annotations = result.annotations;
      if (Array.isArray(result.findings)) presented.findings = result.findings;
      const artifactBinding = artifactStatusBindingFromPacket(
        result.packet,
        result.context?.value && typeof result.context.value === "object"
          ? result.context.value : result.context,
      );
      if (result.artifact_status_result !== undefined) {
        presented.artifact_status_result = artifactStatusResultProjection(
          result.artifact_status_result,
          artifactBinding,
        );
      }
      if (result.artifact_status !== undefined) {
        presented.artifact_status = normalizeArtifactStatus(
          result.artifact_status,
          artifactBinding.episode_id,
          artifactBinding,
          true,
        );
      }
    }
    return presented;
  };
  const saveAnnotation = (annotation, options = {}) => call("save_annotation", { annotation, ...options });
  const persistFinding = (annotation, options = {}) => call("persist_finding", { annotation, ...options });
  const syncFinding = (options = {}) => call("sync_finding", { ...options });
  const recordHumanReview = (outcome, options = {}) => call("record_human_review", { outcome, ...options });
  const relatedCases = (value = {}, options = {}) => call(
    "related_cases", relatedCasesArguments(value, options),
  );
  const runNativeDiagnostic = (value = {}, options = {}) => call(
    "run_native_diagnostic", nativeDiagnosticArguments(value, options),
  );
  const materializeSelected = (value = {}, options = {}) => call(
    "materialize_selected", materializationArguments(value, options),
  );
  const codexStart = (value, options = {}) => call("codex_start", codexStartArguments(value, options));
  const codexRead = (value = {}) => call("codex_read", codexReadArguments(value));
  const codexReconnect = (value = {}, options = {}) => call(
    "codex_reconnect", codexReconnectArguments(value, options),
  );
  const codexCancel = (value, options = {}) => call("codex_cancel", codexCancelArguments(value, options));
  return {
    next: (options = {}) => call("next", options),
    saveAnnotation, save_annotation: saveAnnotation,
    persistFinding, persist_finding: persistFinding,
    syncFinding, sync_finding: syncFinding,
    recordHumanReview, record_human_review: recordHumanReview,
    relatedCases, related_cases: relatedCases,
    materializeSelected, materialize_selected: materializeSelected,
    runNativeDiagnostic, run_native_diagnostic: runNativeDiagnostic,
    snapshot: () => call("snapshot"),
    codexStart, codex_start: codexStart,
    codexRead, codex_read: codexRead,
    codexReconnect, codex_reconnect: codexReconnect,
    codexCancel, codex_cancel: codexCancel,
  };
}
export const PANE_NAMES = Object.freeze(["queue", "finding", "related", "coverage", "agent"]);

function clone(value) {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

function finite(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function finitePoint(value) {
  if (Array.isArray(value) && value.length === 2
    && value.every((item) => typeof item === "number" && Number.isFinite(item))) {
    return [value[0], value[1]];
  }
  if (value && typeof value === "object" && !Array.isArray(value)
    && typeof value.x === "number" && Number.isFinite(value.x)
    && typeof value.y === "number" && Number.isFinite(value.y)) {
    return [value.x, value.y];
  }
  return null;
}

function nativeDiagnosticGoal(packet) {
  const models = [
    packet,
    packet?.editor_model,
    packet?.editor_model?.panel_model,
  ];
  for (const model of models) {
    const geometry = model?.goal_geometry;
    const candidates = [
      geometry?.point?.value,
      geometry?.goal_point?.value,
      geometry?.point,
      geometry?.goal_point,
      model?.goal?.point?.value,
      model?.goal?.point,
      model?.goal,
    ];
    for (const candidate of candidates) {
      const point = finitePoint(candidate);
      if (point) return point;
    }
  }
  return null;
}

function declaredPresentationTime(value) {
  if (!value || typeof value !== "object") return null;
  for (const key of ["pts_s", "media_t_s", "media_time_s", "presentation_t_s", "presentation_time_s"]) {
    const raw = value[key];
    if (typeof raw !== "number" && typeof raw !== "string") continue;
    if (typeof raw === "string" && raw.trim() === "") continue;
    const number = Number(raw);
    if (Number.isFinite(number)) return number;
  }
  return null;
}

function isTextEntry(target) {
  const tagName = String(target?.tagName || "").toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select"
    || target?.isContentEditable === true;
}

function operationId(prefix = "audit") {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function serviceError(result, fallback = "audit service operation failed") {
  if (result instanceof Error) return result;
  const message = result?.reason || result?.error || fallback;
  const error = new Error(String(message));
  error.status = result?.status || "error";
  error.code = result?.code || result?.reason_code || error.status;
  error.conflict = clone(result?.conflict || null);
  error.details = clone(result?.details || result?.conflict || null);
  return error;
}

function boundedCodexText(value, maximum, fallback = "") {
  if (typeof value !== "string") return fallback;
  return value.replaceAll("\u0000", "").slice(0, maximum).replace(
    /(?:[A-Za-z]:[\\/]|\/(?:private|tmp|home|workspace|var|etc|proc|dev)[^\s<>"']*)/gi,
    "<path redacted>",
  );
}

function codexOpaqueId(value) {
  const candidate = boundedCodexText(value, CODEX_MAX_OPERATION_ID_LENGTH);
  return CODEX_OPAQUE_ID.test(candidate) ? candidate : null;
}

function codexSafeReference(value, maximum = CODEX_MAX_OPERATION_ID_LENGTH) {
  if (typeof value !== "string" && typeof value !== "number") return null;
  const candidate = boundedCodexText(String(value), maximum);
  if (!candidate || candidate.includes("/") || candidate.includes("\\")) return null;
  return candidate;
}

function codexCandidates(result) {
  const values = [result];
  for (const key of ["current", "snapshot", "session", "receipt", "result"]) {
    if (result?.[key] && typeof result[key] === "object") values.push(result[key]);
  }
  return values;
}

function firstCodexValue(candidates, keys) {
  for (const candidate of candidates) {
    for (const key of keys) {
      if (candidate?.[key] !== undefined && candidate?.[key] !== null) return candidate[key];
    }
  }
  return null;
}

function normalizeCodexContext(candidates) {
  const raw = firstCodexValue(candidates, ["context", "current_context"]);
  const context = raw && typeof raw === "object" ? raw : {};
  const value = (keys) => firstCodexValue([context, ...candidates], keys);
  return {
    context_revision: codexSafeReference(value(["context_revision"]), 128),
    selection_revision: codexSafeReference(value(["selection_revision"]), 128),
    episode_id: codexSafeReference(value(["episode_id"]), 128),
    scenario_id: codexSafeReference(value(["scenario_id"]), 128),
    execution_id: codexSafeReference(value(["execution_id"]), 128),
  };
}

function normalizeCodexSource(candidates) {
  const raw = firstCodexValue(candidates, ["source", "current_source"]);
  const source = raw && typeof raw === "object" ? raw : {};
  const value = (keys) => firstCodexValue([source, ...candidates], keys);
  return {
    source_revision: codexSafeReference(value(["source_revision"]), 128),
    source_digest: codexSafeReference(value(["source_digest", "digest"]), 128),
  };
}

function normalizeCodexUsage(candidates) {
  const raw = firstCodexValue(candidates, ["usage"]);
  const usage = raw && typeof raw === "object" ? raw : {};
  const normalized = {};
  for (const key of [
    "input_tokens", "output_tokens", "total_tokens", "reserved_compute", "measured_compute",
    "token_budget", "compute_budget",
  ]) {
    const value = usage[key] ?? firstCodexValue(candidates, [key]);
    const number = Number(value);
    if (value !== null && value !== undefined && Number.isFinite(number)) normalized[key] = number;
  }
  return normalized;
}

function normalizeCodexEvidence(candidates) {
  const raw = firstCodexValue(candidates, ["evidence_ids", "evidence_references"]);
  if (!Array.isArray(raw)) return [];
  return raw.slice(0, CODEX_MAX_EVIDENCE_IDS).map((value) => {
    const id = value && typeof value === "object"
      ? (value.evidence_id ?? value.id ?? value.reference)
      : value;
    return codexSafeReference(id, 256);
  }).filter(Boolean);
}

function normalizeCodexActivity(candidates) {
  const raw = firstCodexValue(candidates, ["activity", "events"]);
  if (!Array.isArray(raw)) return [];
  return raw.slice(-CODEX_MAX_ACTIVITY).map((event) => {
    const value = event && typeof event === "object" ? event : { message: event };
    return {
      message: boundedCodexText(value.message ?? value.text, CODEX_MAX_ACTIVITY_MESSAGE, "activity update"),
      evidence_ids: Array.isArray(value.evidence_ids)
        ? value.evidence_ids.slice(0, CODEX_MAX_EVIDENCE_IDS)
          .map((item) => codexSafeReference(item, 256)).filter(Boolean)
        : [],
      operation_id: codexOpaqueId(value.operation_id),
      timestamp: codexSafeReference(value.timestamp, 128),
    };
  });
}

function normalizeCodexResult(result, fallback = "Codex activity unavailable") {
  const source = result && typeof result === "object" ? result : {};
  const candidates = codexCandidates(source);
  const status = boundedCodexText(firstCodexValue(candidates, ["status"]), 64, "unavailable");
  const activityScope = firstCodexValue(candidates, ["activity_scope"]);
  const normalized = {
    status: status || "unavailable",
    reason: boundedCodexText(firstCodexValue(candidates, ["reason", "message"]), 512, fallback),
    operation_id: codexOpaqueId(firstCodexValue(candidates, ["operation_id"])),
    codex_session_id: codexOpaqueId(firstCodexValue(candidates, ["codex_session_id", "session_id"])),
    context: normalizeCodexContext(candidates),
    source: normalizeCodexSource(candidates),
    route_id: codexSafeReference(firstCodexValue(candidates, ["route_id"]), 128),
    evidence_ids: normalizeCodexEvidence(candidates),
    usage: normalizeCodexUsage(candidates),
    activity: normalizeCodexActivity(candidates),
    activity_scope: ["process", "durable_lifecycle"].includes(activityScope)
      ? activityScope
      : "unavailable",
  };
  return normalized;
}

function relatedText(value, maximum = 256) {
  if (typeof value !== "string") return null;
  const bounded = value.replaceAll("\u0000", "").slice(0, maximum);
  if (/%[0-9a-f]{2}/i.test(bounded)) return "<encoded redacted>";
  return bounded.replace(
    /(?:[A-Za-z]:[\\/]|\\\\|\/(?:private|tmp|home|workspace|var|etc|proc|dev)[^\s<>"']*)[^\s<>"']*/gi,
    "<path redacted>",
  );
}

function relatedFeatureProjection(value, depth = 0, state = { nodes: 0 }) {
  if (depth > 4 || state.nodes >= 128) return null;
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string") return relatedText(value);
  state.nodes += 1;
  if (Array.isArray(value)) {
    return value.slice(0, 32).map((item) => relatedFeatureProjection(item, depth + 1, state));
  }
  if (typeof value === "object") {
    const projected = {};
    for (const [key, item] of Object.entries(value).slice(0, 32)) {
      const normalized = key.toLowerCase().replaceAll("-", "_");
      if (normalized.includes("path") || [
        "candidate_members", "confirmed_members", "negative_controls", "finding_id",
        "membership", "provenance", "token", "secret", "credential",
      ].includes(normalized)) continue;
      const child = relatedFeatureProjection(item, depth + 1, state);
      if (child !== null || item === null) projected[key.slice(0, 128)] = child;
    }
    return projected;
  }
  return null;
}

function relatedCandidatesValue(result) {
  if (Array.isArray(result?.value)) return result.value;
  if (result?.value && typeof result.value === "object") {
    for (const key of ["candidates", "related_cases", "results", "items"]) {
      if (Array.isArray(result.value[key])) return result.value[key];
    }
  }
  return null;
}

function normalizeRelatedCasesResult(result, expectedEpisodeId = null) {
  const source = result && typeof result === "object" ? result : {};
  const status = typeof source.status === "string" ? source.status : "unavailable";
  const normalized = {
    status,
    service_status: source.service_status ?? status,
    reason: relatedText(source.reason, 512) || "",
    query_id: null,
    mode: RELATED_DEFAULT_MODE,
    candidates: [],
    candidate_count: 0,
    membership_boundary: RELATED_MEMBERSHIP_BOUNDARY,
    claim_boundary: RELATED_CLAIM_BOUNDARY,
  };
  if (!["complete", "committed", "ok"].includes(status)) return normalized;
  const rawCandidates = relatedCandidatesValue(source);
  if (!rawCandidates) {
    return {
      ...normalized,
      status: "unavailable",
      reason: "related-case service returned no bounded candidate list",
    };
  }
  for (const raw of rawCandidates.slice(0, RELATED_MAX_LIMIT)) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
      return { ...normalized, status: "unavailable", reason: "related-case candidate is malformed" };
    }
    const queryId = codexSafeReference(raw.query_id, RELATED_MAX_ID_LENGTH);
    const candidateId = codexSafeReference(raw.candidate_id, RELATED_MAX_ID_LENGTH);
    const mode = typeof raw.mode === "string" && RELATED_MODES.has(raw.mode) ? raw.mode : null;
    const score = Number(raw.score);
    const compatibility = RELATED_COMPATIBILITY.has(raw.compatibility) ? raw.compatibility : null;
    if (!queryId || !candidateId || !mode || !compatibility || !Number.isFinite(score)
      || score < 0 || score > 1) {
      return { ...normalized, status: "unavailable", reason: "related-case candidate is unsafe" };
    }
    if (expectedEpisodeId !== null && String(queryId) !== String(expectedEpisodeId)) {
      return {
        ...normalized,
        status: "conflict",
        reason: "related-case result belongs to a foreign selected episode",
      };
    }
    if (normalized.query_id === null) {
      normalized.query_id = queryId;
      normalized.mode = mode;
    } else if (normalized.query_id !== queryId) {
      return { ...normalized, status: "conflict", reason: "related-case result has mixed query episodes" };
    }
    const listField = (key) => Array.isArray(raw[key])
      ? raw[key].slice(0, 16).map((item) => relatedText(item)).filter((item) => item !== null)
      : [];
    normalized.candidates.push({
      query_id: queryId,
      candidate_id: candidateId,
      mode,
      score,
      compatibility,
      reasons: listField("reasons"),
      features: relatedFeatureProjection(raw.features || {}) || {},
      missingness: listField("missingness"),
      warnings: listField("warnings"),
      membership_status: "unconfirmed_candidate",
    });
  }
  normalized.candidate_count = normalized.candidates.length;
  return normalized;
}

function requireStatus(result, allowed, fallback) {
  if (!result || !allowed.includes(result.status)) throw serviceError(result, fallback);
  return result;
}

function normalizeSelectionResult(result) {
  if (!result || result.status !== "complete") return result;
  if (result.packet) {
    return {
      ...result,
      service_status: result.service_status ?? result.status,
      status: "selected",
    };
  }
  if (result.presentation_status === "empty") {
    return {
      ...result,
      service_status: result.service_status ?? result.status,
      status: "empty",
    };
  }
  return result;
}

const ARTIFACT_STATUS_SCHEMA_VERSION = "audit-artifact-status.v1";
const ARTIFACT_STATUS_VALUES = new Set([
  "available", "unavailable", "not_configured", "no_selection",
]);
const ARTIFACT_CLASSIFICATIONS = new Set([
  "historical_original", "derived_render", "unavailable",
]);
const ARTIFACT_FIDELITIES = new Set([
  "verified", "diverged", "unverifiable", "unavailable",
]);
const ARTIFACT_RESULT_STATUSES = new Set([
  "complete", "partial", "committed", "ok", "selected", "conflict", "unavailable",
  "denied", "failed", "cancelled",
]);
const ARTIFACT_SENSITIVE_TEXT = /(?:token|secret|credential|bearer|password|private)/i;

function artifactReference(value, maximum = 256) {
  if (typeof value === "number") return Number.isInteger(value) && value >= 0 ? value : null;
  if (typeof value !== "string") return null;
  const candidate = value.replaceAll("\u0000", "").trim();
  return candidate && candidate.length <= maximum
    && !ARTIFACT_SENSITIVE_TEXT.test(candidate)
    && !candidate.includes("%")
    && !candidate.includes("/") && !candidate.includes("\\") ? candidate : null;
}

function artifactReason(value, fallback) {
  if (typeof value !== "string") return fallback;
  const candidate = value.replaceAll("\u0000", "").trim();
  return candidate && candidate.length <= 256
    && !ARTIFACT_SENSITIVE_TEXT.test(candidate)
    && !candidate.includes("%")
    && !candidate.includes("/") && !candidate.includes("\\") ? candidate : fallback;
}

function artifactReasonIsUnsafe(value) {
  if (value === undefined || value === null || value === "") return false;
  return artifactReason(value, "artifact status is unavailable") !== String(value).trim();
}

function artifactSourceIdentityToken(value) {
  if (typeof value === "string") return value;
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const sources = value.sources;
  if (sources && typeof sources === "object" && !Array.isArray(sources)) {
    for (const item of Object.values(sources)) {
      if (!item || typeof item !== "object" || Array.isArray(item)) continue;
      for (const field of ["sha256", "artifact_id"]) {
        if (item[field]) return String(item[field]);
      }
    }
  }
  for (const field of ["sha256", "artifact_id"]) {
    if (value[field]) return String(value[field]);
  }
  return null;
}

function artifactBindingField(field, values, fallbackValues = []) {
  const present = [...values, ...fallbackValues]
    .filter((value) => value !== null && value !== undefined && value !== "");
  if (!present.length) return { value: null, conflict: false };
  const normalized = present.map((value) => field === "source_digest"
    ? artifactSourceIdentityToken(value) : value);
  const safeValues = normalized.map((value) => artifactReference(value));
  const first = safeValues[0];
  return {
    value: first,
    conflict: safeValues.some((value) => value === null)
      || safeValues.some((value) => value !== first),
  };
}

function artifactStatusBindingFromPacket(packet, context = {}) {
  const selectedPacket = packet && typeof packet === "object" ? packet : {};
  const editor = selectedPacket.editor_model && typeof selectedPacket.editor_model === "object"
    ? selectedPacket.editor_model : {};
  const packetContext = editor.context && typeof editor.context === "object" ? editor.context : {};
  const storyboard = editor.storyboard && typeof editor.storyboard === "object" ? editor.storyboard : {};
  const panel = editor.panel_model && typeof editor.panel_model === "object" ? editor.panel_model : {};
  const panelContext = panel.context && typeof panel.context === "object" ? panel.context : {};
  const selectedContext = context && typeof context === "object" ? context : {};
  const contextRevisionValues = [
    selectedPacket.context_revision,
    selectedContext.context_revision,
    packetContext.context_revision,
    panelContext.context_revision,
    selectedPacket.selection_revision,
    selectedContext.selection_revision,
    packetContext.selection_revision,
    panelContext.selection_revision,
  ];
  const fields = {
    episode_id: artifactBindingField("episode_id", [
      selectedPacket.episode_id,
      selectedContext.episode_id,
      packetContext.episode_id,
      panelContext.episode_id,
      editor.episode_id,
      panel.episode_id,
    ]),
    context_revision: artifactBindingField("context_revision", contextRevisionValues),
    source_revision: artifactBindingField("source_revision", [
      selectedPacket.source_revision,
      selectedContext.source_revision,
      packetContext.source_revision,
      storyboard.source_revision,
      panel.source_revision,
      panelContext.source_revision,
    ]),
    source_digest: artifactBindingField("source_digest", [
      selectedPacket.source_digest,
      selectedPacket.source_identity,
      selectedContext.source_digest,
      selectedContext.source_identity,
      packetContext.source_digest,
      packetContext.source_identity,
      editor.source_digest,
      editor.source_identity,
      storyboard.source_digest,
      storyboard.source_identity,
      panel.source_digest,
      panel.source_identity,
      panelContext.source_digest,
      panelContext.source_identity,
    ]),
  };
  return {
    episode_id: fields.episode_id.value,
    context_revision: fields.context_revision.value,
    source_revision: fields.source_revision.value,
    source_digest: fields.source_digest.value,
    binding_conflict: Object.values(fields).some((field) => field.conflict),
  };
}

function artifactStatusBinding(expectedEpisodeId, expectedBinding = {}) {
  const binding = expectedBinding && typeof expectedBinding === "object"
    ? expectedBinding : {};
  return {
    episode_id: binding.episode_id ?? expectedEpisodeId,
    context_revision: binding.context_revision ?? null,
    source_revision: binding.source_revision ?? null,
    source_digest: binding.source_digest ?? binding.source_identity ?? null,
    binding_conflict: binding.binding_conflict === true,
  };
}

function artifactStatusBindingComplete(binding) {
  return binding?.binding_conflict !== true && ["episode_id", "context_revision", "source_revision", "source_digest"]
    .every((field) => artifactReference(binding[field]) !== null);
}

function artifactStatusMapping(value) {
  if (value && typeof value === "object" && !value.materialization
    && value.value && typeof value.value === "object") return value.value;
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function artifactStatusBindingMismatch(value, binding) {
  if (binding?.binding_conflict === true) return true;
  const source = artifactStatusMapping(value);
  for (const [field, expected] of Object.entries(binding)) {
    if (field === "binding_conflict") continue;
    const safeExpected = artifactReference(expected);
    const actual = source[field];
    if (expected !== null && expected !== undefined && expected !== "" && safeExpected === null) return true;
    if (actual !== null && actual !== undefined && artifactReference(actual) === null) return true;
    if (safeExpected !== null && (actual === null || actual === undefined
      || artifactReference(actual) !== safeExpected)) return true;
  }
  return false;
}

function artifactStatusContextMismatch(actual, binding) {
  if (!actual || typeof actual !== "object") return true;
  const sourceAliases = [actual.source_identity, actual.source_digest]
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map(artifactSourceIdentityToken);
  const safeSourceAliases = sourceAliases.map((value) => artifactReference(value));
  if (safeSourceAliases.some((value) => value === null)
    || safeSourceAliases.some((value) => value !== safeSourceAliases[0])) return true;
  const source = {
    episode_id: actual.episode_id,
    context_revision: actual.context_revision,
    source_revision: actual.source_revision,
    source_digest: sourceAliases[0] ?? null,
  };
  return artifactStatusBindingMismatch(source, binding);
}

/** Normalize service-owned artifact/native capability without adding actions. */
export function normalizeArtifactStatus(
  value, expectedEpisodeId = null, expectedBinding = {}, requireCompleteBinding = false,
) {
  const binding = artifactStatusBinding(expectedEpisodeId, expectedBinding);
  const envelope = value && typeof value === "object" && !value.materialization
    && value.value && typeof value.value === "object" ? value : null;
  const source = artifactStatusMapping(value);
  const bindingMismatch = (requireCompleteBinding && !artifactStatusBindingComplete(binding))
    || artifactStatusBindingMismatch(source, binding)
    || (envelope && envelope.context !== undefined
      && artifactStatusContextMismatch(envelope.context, binding));
  const candidate = bindingMismatch ? {} : source;
  const episodeId = artifactReference(binding.episode_id) ?? artifactReference(candidate.episode_id);
  const fallbackStatus = episodeId === null ? "no_selection" : "unavailable";
  const fallbackReason = episodeId === null
    ? "no selected episode" : "selected artifact status is unavailable";
  const normalizeCapability = (raw, native = false) => {
    const candidate = raw && typeof raw === "object" ? raw : {};
    const reasonIsSafe = candidate.reason === undefined || (
      typeof candidate.reason === "string"
      && candidate.reason.trim().length > 0
      && candidate.reason.length <= 256
      && !ARTIFACT_SENSITIVE_TEXT.test(candidate.reason)
      && !candidate.reason.includes("%")
      && !candidate.reason.includes("/") && !candidate.reason.includes("\\")
    );
    const resultStatus = candidate.result_status === undefined
      ? null : artifactReference(candidate.result_status, 64);
    const resultStatusIsSafe = candidate.result_status === undefined
      || (resultStatus !== null && ARTIFACT_RESULT_STATUSES.has(resultStatus));
    const classificationIsSafe = candidate.classification === undefined
      || (typeof candidate.classification === "string"
        && ARTIFACT_CLASSIFICATIONS.has(candidate.classification));
    const fidelityIsSafe = candidate.fidelity === undefined
      || (typeof candidate.fidelity === "string" && ARTIFACT_FIDELITIES.has(candidate.fidelity));
    const unsafeValue = bindingMismatch || !reasonIsSafe
      || !resultStatusIsSafe
      || !ARTIFACT_STATUS_VALUES.has(candidate.status)
      || !classificationIsSafe || !fidelityIsSafe;
    const status = unsafeValue ? fallbackStatus : candidate.status;
    const projected = {
      status,
      reason: unsafeValue ? fallbackReason : artifactReason(candidate.reason, fallbackReason),
      diagnostic_only: true,
    };
    if (native) {
      projected.evidence_boundary = "diagnostic_only";
      projected.scientific_claim_allowed = false;
    } else {
      projected.classification = !unsafeValue && ARTIFACT_CLASSIFICATIONS.has(candidate.classification)
        ? candidate.classification : null;
      projected.fidelity = !unsafeValue && ARTIFACT_FIDELITIES.has(candidate.fidelity)
        ? candidate.fidelity : null;
      if (!unsafeValue && typeof candidate.simulation_executed === "boolean") {
        projected.simulation_executed = candidate.simulation_executed;
      }
    }
    if (!unsafeValue && resultStatus !== null) projected.result_status = resultStatus;
    return projected;
  };
  return {
    schema_version: ARTIFACT_STATUS_SCHEMA_VERSION,
    episode_id: episodeId,
    context_revision: artifactReference(binding.context_revision) ?? artifactReference(candidate.context_revision),
    source_revision: artifactReference(binding.source_revision) ?? artifactReference(candidate.source_revision),
    source_digest: artifactReference(binding.source_digest) ?? artifactReference(candidate.source_digest),
    materialization: normalizeCapability(candidate.materialization),
    native_diagnostic: normalizeCapability(candidate.native_diagnostic, true),
  };
}

function artifactStatusResultProjection(result, expectedBinding = {}) {
  const status = ["complete", "committed", "ok", "selected", "conflict", "unavailable", "denied", "failed", "cancelled"]
    .includes(result?.status) ? result.status : "unavailable";
  const reason = artifactReason(result?.reason, "selected artifact status is unavailable");
  const succeeded = ["complete", "committed", "ok", "selected"].includes(status);
  const contextPresent = result?.context && typeof result.context === "object";
  const unsafeReason = artifactReasonIsUnsafe(result?.reason);
  const contextForComparison = contextPresent ? {
    ...result.context,
    ...(result.context.source_identity === undefined && result.context.source_digest === undefined
      ? { source_digest: expectedBinding.source_digest } : {}),
  } : null;
  const candidateValue = succeeded && contextForComparison && !unsafeReason ? {
    ...result,
    context: contextForComparison,
  } : null;
  const binding = artifactStatusBinding(expectedBinding.episode_id, expectedBinding);
  const bindingMismatch = succeeded && (
    !contextPresent
    || !artifactStatusBindingComplete(binding)
    || artifactStatusBindingMismatch(candidateValue, binding)
    || artifactStatusContextMismatch(contextForComparison, binding)
    || unsafeReason
  );
  const projectedStatus = bindingMismatch ? "conflict" : status;
  const projectedReason = bindingMismatch
    ? (unsafeReason ? "selected artifact status is unavailable" : "selected artifact status is stale")
    : reason;
  return {
    schema_version: ARTIFACT_STATUS_SCHEMA_VERSION,
    status: projectedStatus,
    reason: projectedReason,
    value: normalizeArtifactStatus(
      bindingMismatch ? null : candidateValue,
      expectedBinding.episode_id,
      expectedBinding,
      true,
    ),
    context: {
      ...((result?.context && typeof result.context === "object") ? {
        context_revision: artifactReference(result.context.context_revision),
        episode_id: artifactReference(result.context.episode_id),
        source_revision: artifactReference(result.context.source_revision),
      } : {}),
    },
  };
}

function materializationResultProjection(result, expectedBinding = {}, currentStatus = {}) {
  const status = [
    "complete", "partial", "committed", "ok", "selected", "conflict",
    "unavailable", "denied", "failed", "cancelled",
  ].includes(result?.status) ? result.status : "failed";
  const reason = artifactReason(result?.reason, "materialization result is unavailable");
  const classification = typeof result?.classification === "string"
    && ARTIFACT_CLASSIFICATIONS.has(result.classification) ? result.classification : null;
  const fidelity = typeof result?.fidelity === "string"
    && ARTIFACT_FIDELITIES.has(result.fidelity) ? result.fidelity : null;
  const successful = ["complete", "partial", "committed", "ok", "selected"].includes(status)
    && classification !== null && fidelity !== null;
  const materialization = {
    status: successful ? "available" : "unavailable",
    reason: successful ? reason : (reason || "materialization result is unavailable"),
    diagnostic_only: true,
    classification: successful ? classification : null,
    fidelity: successful ? fidelity : null,
    ...(status !== "failed" ? { result_status: status } : {}),
  };
  return normalizeArtifactStatus({
    ...expectedBinding,
    materialization,
    native_diagnostic: currentStatus?.native_diagnostic,
  }, expectedBinding?.episode_id, expectedBinding, true);
}

function artifactStatusSummary(status) {
  const normalized = normalizeArtifactStatus(status, status?.episode_id);
  const materialization = normalized.materialization;
  const native = normalized.native_diagnostic;
  const materializationDetail = materialization.classification
    ? ` (${materialization.classification})` : "";
  return `Materialization: ${materialization.status}${materializationDetail} — ${materialization.reason}; `
    + `Native diagnostic: ${native.status} — ${native.reason}; Diagnostic only.`;
}

/** Project the closed native action result without exposing runner details. */
export function normalizeNativeDiagnosticResult(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const rawStatus = source.status ?? "unavailable";
  const status = NATIVE_DIAGNOSTIC_STATUSES.has(rawStatus) ? rawStatus : "failed";
  const reason = artifactReason(source.reason, status === "running"
    ? "native diagnostic is running"
    : "native diagnostic result is unavailable");
  const projected = {
    status,
    reason,
    diagnostic_only: true,
    scientific_claim_allowed: false,
    control_fidelity: null,
    activation: null,
  };
  if (status !== "complete") return projected;
  if (source.diagnostic_only !== true || source.scientific_claim_allowed !== false
    || source.control_fidelity !== "verified" || source.activation !== "verified") {
    projected.status = "failed";
    projected.reason = "native diagnostic result did not preserve diagnostic-only fidelity";
    return projected;
  }
  projected.control_fidelity = "verified";
  projected.activation = "verified";
  projected.reason = artifactReason(source.reason, "native diagnostic completed");
  return projected;
}

function hasAdmittedEditorModel(packet) {
  const model = packet?.editor_model;
  const sources = model?.source_identity?.sources;
  const bounds = model?.time;
  const scene = model?.panel_model?.streams?.scene;
  return model?.schema_version === "review-editor.v1"
    && model?.panel_model?.schema_version === "review-panels.v1"
    && sources && typeof sources === "object" && Object.keys(sources).length > 0
    && Number.isFinite(bounds?.origin_s) && Number.isFinite(bounds?.terminal_s)
    && bounds.origin_s <= bounds.terminal_s
    && ["available", "partial"].includes(scene?.status)
    && Array.isArray(scene.samples)
    && scene.samples.some((sample) => Number.isFinite(sample?.time_s) && sample?.missing !== true);
}

function packetEditorModel(packet) {
  if (!hasAdmittedEditorModel(packet)) {
    throw new Error("selected packet has no admitted editor model");
  }
  const model = clone(packet.editor_model);
  model.schema_version = model.schema_version || "review-editor.v1";
  model.context = { ...(model.context || {}) };
  model.context.episode_id = String(packet?.episode_id || model.context.episode_id || "episode");
  model.context.execution_id = String(packet?.execution_id || model.context.execution_id || "");
  const queueSelectionRevision = Number(packet?.selection_revision || 0);
  const declaredQueueRevision = model.context.queue_selection_revision;
  const contextBelongsToQueue = declaredQueueRevision !== undefined
    && Number(declaredQueueRevision) === queueSelectionRevision;
  const declaredEditorRevision = contextBelongsToQueue
    ? (model.context.selection_revision ?? model.context.context_revision)
    : queueSelectionRevision;
  const editorSelectionRevision = declaredEditorRevision === null || declaredEditorRevision === undefined
    ? queueSelectionRevision
    : Number(declaredEditorRevision);
  model.context.selection_revision = Number.isFinite(editorSelectionRevision)
    ? editorSelectionRevision : queueSelectionRevision;
  model.context.context_revision = model.context.selection_revision;
  model.context.queue_selection_revision = queueSelectionRevision;
  model.context.cursor = { time_s: finite(packet?.cursor?.time_s, 0) };
  model.annotations = Array.isArray(model.annotations) ? model.annotations : [];
  return model;
}

/**
 * Pick a media sample under the SREV-16 nearest-sample contract.
 *
 * The returned ``pts_s`` is copied from the declared sample value.  No fps,
 * duration normalization, or index arithmetic is performed here.
 */
export function mediaSnapshot(packet, cursorTimeS) {
  const stream = packet?.media || packet?.editor_model?.streams?.video || packet?.streams?.video;
  if (!stream || stream.status === "unavailable") {
    return {
      status: "unavailable",
      target_time_s: cursorTimeS,
      sample_time_s: null,
      pts_s: null,
      reason: stream?.reason || "video_stream_missing",
      alignment: "declared_pts_to_simulation",
    };
  }
  const selected = nearestSample(stream, cursorTimeS);
  const pts = declaredPresentationTime(selected?.value);
  if (selected.status !== "available" || pts === null) {
    return {
      ...selected,
      status: "unavailable",
      pts_s: null,
      reason: selected.reason || "presentation_timestamp_missing",
      alignment: "declared_pts_to_simulation",
    };
  }
  return {
    ...selected,
    pts_s: pts,
    alignment: "declared_pts_to_simulation",
  };
}

/**
 * Browser-only fixture facade used by the generated offline HTML and runtime tests.
 * It mirrors the narrow service shape without importing BA-03 storage or BA-05 logic.
 */
export function createFixtureFacade(document = {}) {
  const queue = Array.isArray(document.queue) ? clone(document.queue) : [];
  let index = Number.isInteger(document.queue_index) ? document.queue_index : -1;
  let selectionRevision = Number(document.selection_revision || 0);
  let coverage = clone(document.coverage || {
    schema_version: "audit-coverage.v1", reviewed: 0, total: queue.length,
    remaining: queue.length, reviewed_episode_ids: [], status: "under_review",
  });
  let packet = clone(document.packet || null);
  const annotations = new Map((document.annotations || []).map((item) => [
    String(item.annotation_id || item.record_id), clone(item),
  ]));
  const revisions = new Map((document.annotations || []).map((item) => [
    String(item.annotation_id || item.record_id),
    Number(item.revision ?? item.record_revision ?? 0),
  ]));
  const findings = new Map((document.findings || []).map((item) => [
    String(item.finding_id || item.record_id), clone(item),
  ]));
  const operationIds = new Map();

  function checkSelection(expected) {
    if (Number(expected) !== selectionRevision) {
      return {
        status: "conflict",
        code: "stale_selection_revision",
        reason: `stale selection revision: expected ${expected}, current ${selectionRevision}`,
        conflict: { expected_revision: Number(expected), actual_revision: selectionRevision },
      };
    }
    return null;
  }

  function updatePacket(caseValue) {
    packet = clone(caseValue);
    packet.selection_revision = selectionRevision;
    packet.cursor = { ...(packet.cursor || {}), selection_revision: selectionRevision };
    packet.editor_model = packetEditorModel(packet);
    return packet;
  }

  function syncPacketFromAnnotation(annotation) {
    if (!packet) return;
    const rawTime = annotation?.interval?.start_s;
    const cursorTime = typeof rawTime === "number" && Number.isFinite(rawTime) ? rawTime : null;
    if (cursorTime === null) return;
    packet.cursor = { ...(packet.cursor || {}), time_s: cursorTime, selection_revision: selectionRevision };
    const context = packet.editor_model?.context;
    if (!context) return;
    context.cursor = { time_s: cursorTime };
    const editorRevision = annotation?.metadata?.selection_revision;
    if (typeof editorRevision === "number" && Number.isInteger(editorRevision)) {
      context.selection_revision = editorRevision;
      context.context_revision = editorRevision;
    }
    context.queue_selection_revision = selectionRevision;
  }

  function next({ expected_selection_revision: expected } = {}) {
    const conflict = checkSelection(expected);
    if (conflict) return conflict;
    const nextIndex = index + 1;
    if (nextIndex >= queue.length) {
      return {
        status: "empty", reason: "queue_exhausted", selection_revision: selectionRevision,
        coverage: clone(coverage), next: null,
      };
    }
    index = nextIndex;
    selectionRevision += 1;
    updatePacket(queue[index]);
    const nextCase = queue[index + 1];
    return {
      status: "selected", packet: clone(packet), selection_revision: selectionRevision,
      coverage: clone(coverage),
      next: nextCase ? {
        episode_id: nextCase.episode_id, scenario_id: nextCase.scenario_id,
        selection_reasons: clone(nextCase.selection_reasons || []),
      } : null,
    };
  }

  function saveAnnotation(annotation, options = {}) {
    const conflict = checkSelection(options.expected_selection_revision);
    if (conflict) return conflict;
    const operation = String(options.operation_id || "");
    if (operation && operationIds.has(operation)) return clone(operationIds.get(operation));
    if (!packet) return { status: "unavailable", reason: "no_audit_packet_selected" };
    const annotationId = String(annotation?.annotation_id || annotation?.record_id || "");
    if (!annotationId) return { status: "error", reason: "annotation_id_required" };
    const actual = revisions.get(annotationId) || 0;
    if (Number(options.expected_revision) !== actual) {
      return {
        status: "conflict",
        code: "record_revision_conflict",
        reason: `revision conflict for ${annotationId}: expected ${options.expected_revision}, current ${actual}`,
        conflict: { record_id: annotationId, expected_revision: Number(options.expected_revision), actual_revision: actual },
      };
    }
    const stored = { ...clone(annotation), episode_id: String(packet.episode_id), selection_revision: selectionRevision };
    annotations.set(annotationId, stored);
    const revision = actual + 1;
    revisions.set(annotationId, revision);
    stored.revision = revision;
    syncPacketFromAnnotation(stored);
    const receipt = {
      status: "saved", record_id: annotationId, revision, operation_id: options.operation_id,
      selection_revision: selectionRevision, record: clone(stored),
    };
    if (operation) operationIds.set(operation, clone(receipt));
    return receipt;
  }

  function persistFinding(annotation, options = {}) {
    const conflict = checkSelection(options.expected_selection_revision);
    if (conflict) return conflict;
    const operation = String(options.operation_id || "");
    if (operation && operationIds.has(operation)) return clone(operationIds.get(operation));
    if (!packet) return { status: "unavailable", reason: "no_audit_packet_selected" };
    const annotationId = String(annotation?.annotation_id || annotation?.record_id || "");
    const storedAnnotation = annotations.get(annotationId);
    if (!storedAnnotation) return { status: "unavailable", reason: "finding_requires_saved_annotation" };
    const episodeId = String(packet.episode_id);
    if (String(storedAnnotation.episode_id) !== episodeId
      || Number(storedAnnotation.selection_revision) !== selectionRevision) {
      return {
        status: "conflict",
        code: "annotation_context_conflict",
        reason: "saved annotation does not belong to the selected episode/context",
        conflict: {
          annotation_episode_id: storedAnnotation.episode_id,
          annotation_selection_revision: Number(storedAnnotation.selection_revision),
          selected_episode_id: episodeId,
          selected_selection_revision: selectionRevision,
        },
      };
    }
    const findingId = `finding-${episodeId}`;
    if (!findings.has(findingId)) {
      findings.set(findingId, {
        schema_version: "audit-record.v1", record_type: "finding", record_id: findingId,
        finding_id: findingId, status: "proposed", title: String(storedAnnotation.classification || "review finding"),
        candidate_members: [episodeId], confirmed_members: [], negative_controls: [], annotation_ids: [annotationId],
        service_boundary: "injected_fixture_facade",
      });
      const reviewed = [...new Set([...(coverage.reviewed_episode_ids || []), episodeId])];
      coverage = {
        ...coverage, reviewed: reviewed.length,
        remaining: Math.max(0, Number(coverage.total || queue.length) - reviewed.length),
        reviewed_episode_ids: reviewed,
        status: reviewed.length >= Number(coverage.total || queue.length) ? "complete" : "under_review",
      };
    }
    const nextCase = queue[index + 1];
    const receipt = {
      status: "saved", finding: clone(findings.get(findingId)), coverage: clone(coverage),
      selection_revision: selectionRevision, operation_id: options.operation_id,
      next: nextCase ? { episode_id: nextCase.episode_id, scenario_id: nextCase.scenario_id, selection_reasons: clone(nextCase.selection_reasons || []) } : null,
    };
    if (operation) operationIds.set(operation, clone(receipt));
    return receipt;
  }

  function snapshot() {
    const artifactBinding = artifactStatusBindingFromPacket(packet);
    return {
      schema_version: AUDIT_WORKBENCH_MODEL_SCHEMA_VERSION,
      service: clone(document.service || { id: AUDIT_WORKBENCH_SERVICE_ID, native: false, evidence_status: "diagnostic_only" }),
      queue: clone(queue), queue_index: index, selection_revision: selectionRevision,
      packet: clone(packet), coverage: clone(coverage), annotations: [...annotations.values()].map(clone),
      findings: [...findings.values()].map(clone), next: queue[index + 1] ? { episode_id: queue[index + 1].episode_id } : null,
      artifact_status: normalizeArtifactStatus(
        document.artifact_status || null,
        artifactBinding.episode_id,
        artifactBinding,
        true,
      ),
    };
  }

  return { next, saveAnnotation, save_annotation: saveAnnotation, persistFinding, persist_finding: persistFinding, snapshot };
}

function button(documentRef, label, action, attributes = {}) {
  const element = documentRef.createElement("button");
  element.type = "button";
  element.textContent = label;
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute?.(key, String(value)));
  element.addEventListener("click", (event) => {
    event.preventDefault?.();
    void action(event);
  });
  return element;
}

function text(documentRef, tag, value, className = "") {
  const element = documentRef.createElement(tag);
  element.textContent = value;
  if (className) element.className = className;
  return element;
}

function coverageSummary(coverage) {
  // BA-05 keeps its capability envelope; only the BA-04 value inside a
  // completed, named capability may be read as an authoritative report.
  const report = coverage?.capability === "ba-04.coverage"
    ? (coverage.status === "complete" ? coverage.value : null) : coverage;
  const inventory = report?.counts?.coverage;
  const reviews = report?.counts?.reviews;
  if (inventory && reviews && report?.schema_version === "audit-coverage.v1") {
    if (!["incomplete", "complete_with_declared_exceptions", "complete_under_protocol"].includes(report.status)) {
      return "Coverage report unavailable";
    }
    const readable = inventory.readable;
    const fullHuman = reviews.full_episode_human;
    if (!Number.isInteger(readable) || readable < 0
      || !Number.isInteger(fullHuman) || fullHuman < 0 || fullHuman > readable
      || !Array.isArray(report.deficits)) return "Coverage report unavailable";
    const unmet = report.deficits.filter((item) => item?.status !== "waived").length;
    return `${fullHuman}/${readable} readable episodes have full human review · ${unmet} unmet protocol requirements · ${report.status}`;
  }
  if (!coverage?.capability && Number.isInteger(coverage?.reviewed) && Number.isInteger(coverage?.total)
    && Number.isInteger(coverage?.remaining)) {
    return `Fixture coverage: ${coverage.reviewed}/${coverage.total} episodes reviewed · ${coverage.remaining} remaining`;
  }
  return "Coverage unavailable; no source-bound BA-04 report is available.";
}

function detailsPane(documentRef, name, title, open) {
  const details = documentRef.createElement("details");
  details.dataset.pane = name;
  details.open = open;
  const summary = documentRef.createElement("summary");
  summary.textContent = title;
  details.appendChild(summary);
  return { details, body: documentRef.createElement("div") };
}

function appendCodexList(documentRef, body, title, entries, emptyText) {
  body.appendChild(text(documentRef, "h3", title));
  const list = documentRef.createElement("ul");
  const available = entries.filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (!available.length) {
    body.appendChild(text(documentRef, "p", emptyText));
    return;
  }
  for (const [label, value] of available) {
    list.appendChild(text(documentRef, "li", `${label}: ${String(value)}`));
  }
  body.appendChild(list);
}

function appendCodexDisplay(documentRef, body, codex) {
  appendCodexList(documentRef, body, "Current context", [
    ["status", codex.status],
    ["context revision", codex.context.context_revision],
    ["selection revision", codex.context.selection_revision],
    ["episode", codex.context.episode_id],
    ["scenario", codex.context.scenario_id],
    ["execution", codex.context.execution_id],
  ], "Current context is unavailable.");
  appendCodexList(documentRef, body, "Current source", [
    ["Codex session", codex.codex_session_id],
    ["source revision", codex.source.source_revision],
    ["source digest", codex.source.source_digest],
    ["route", codex.route_id],
    ["evidence", codex.evidence_ids.length ? codex.evidence_ids.join(", ") : null],
  ], "Current source and route references are unavailable.");
  appendCodexList(
    documentRef,
    body,
    "Usage",
    Object.entries(codex.usage),
    "Usage is unavailable.",
  );
  body.appendChild(text(
    documentRef,
    "p",
    codex.activity_scope === "durable_lifecycle"
      ? "Durable operation summaries only; provider conversation events and transcripts are unavailable."
      : codex.activity_scope === "process"
        ? "Activity is process-scoped; no durable transcript is exposed."
        : "No durable transcript is exposed; bounded activity may be unavailable.",
    "audit-boundary",
  ));
  const activity = documentRef.createElement("ul");
  for (const event of codex.activity) {
    const evidence = event.evidence_ids.length ? ` [${event.evidence_ids.join(", ")}]` : "";
    const metadata = [event.timestamp, event.operation_id].filter(Boolean).join(" · ");
    activity.appendChild(text(
      documentRef,
      "li",
      `${event.message}${metadata ? ` (${metadata})` : ""}${evidence}`,
    ));
  }
  if (codex.activity.length) body.appendChild(activity);
  else body.appendChild(text(documentRef, "p", "No bounded activity is available."));
}

export class AuditWorkbenchController {
  constructor(model = {}, root = null, options = {}) {
    this.model = clone(model || {});
    this.facade = options.facade;
    if (!this.facade || typeof this.facade.next !== "function") {
      throw new Error("an injected audit facade with next() is required");
    }
    this.root = null;
    this._document = null;
    this._refs = {};
    this.editor = null;
    this.panels = null;
    this._syncingCursor = false;
    this._selectionEpoch = 0;
    this._serviceCas = {};
    this._nativeDiagnosticRequestEpoch = 0;
    this._codexRequestEpoch = 0;
    this._codexReconnectOperationId = null;
    this._codexReconnectSessionId = null;
    this._recordProjection = clone(
      this.model.record_projection || this.model.service_snapshot?.record_projection || null,
    );
    this._recordProjectionScope = this._recordProjection?.scope_id || null;
    const modelHasRecordArrays = Array.isArray(this.model.annotations)
      && Array.isArray(this.model.findings);
    this.state = {
      selected: clone(this.model.packet || null),
      selectionRevision: Number(this.model.selection_revision || this.model.packet?.selection_revision || 0),
      coverage: clone(this.model.coverage || {}),
      finding: (this.model.findings || this.model.service_snapshot?.findings || []).at(-1) || null,
      githubSync: clone(this.model.github_sync || this.model.githubSync || null),
      autosave: { state: "saved", error: "", revision: null },
      serviceStatus: "ready",
      serviceAuthorityStatus: this.model.service_status || this.model.status || "ready",
      presentationStatus: this.model.presentation_status || null,
      inspectionStatus: this.model.inspection_status || null,
      inspectionReason: this.model.inspection_reason || "",
      serviceError: "",
      nativeDiagnostic: normalizeNativeDiagnosticResult(
        this.model.native_diagnostic_result || this.model.nativeDiagnosticResult || null,
      ),
      nativeDiagnosticInFlight: false,
      nativeDiagnosticInput: {
        intervention_id: "native-goal-change",
        robot_goal: nativeDiagnosticGoal(this.model.packet),
        activation_epsilon_m: NATIVE_DIAGNOSTIC_DEFAULT_EPSILON_M,
        deadline_s: NATIVE_DIAGNOSTIC_DEFAULT_DEADLINE_S,
      },
      codex: normalizeCodexResult(this.model.codex || this.model.codex_activity || null),
      codexPrompt: "",
      codexStartInFlight: false,
      codexReconnectInFlight: false,
      materializationInFlight: false,
      next: clone(this.model.next || null),
      artifactStatus: normalizeArtifactStatus(
        this.model.artifact_status || this.model.service_snapshot?.artifact_status,
        this.model.packet?.episode_id,
        this._artifactStatusBinding(this.model.packet?.episode_id, this.model),
        true,
      ),
      relatedCases: normalizeRelatedCasesResult(
        this.model.related_cases || this.model.relatedCases || null,
        this.model.packet?.episode_id || null,
      ),
      recordsStatus: this._recordProjection
        ? "same_facade_only" : (modelHasRecordArrays ? "complete" : "unknown"),
      recordsReason: this._recordProjection
        ? "projection is complete only for this facade instance"
        : (modelHasRecordArrays ? "" : "durable record reads are unavailable"),
      panes: Object.fromEntries(PANE_NAMES.map((name) => [name, name !== "agent"])),
    };
    this._rememberServiceState(this.model);
    this._keydown = (event) => this.handleKey(event);
    if (root) this.mount(root);
  }

  mount(root) {
    this.unmount();
    this.root = root;
    this._document = root?.ownerDocument || (typeof document !== "undefined" ? document : null);
    if (this.root && this._document) {
      this.render();
      this._document.addEventListener("keydown", this._keydown);
    }
    return this;
  }

  unmount() {
    this._nativeDiagnosticRequestEpoch += 1;
    this.state.nativeDiagnosticInFlight = false;
    this._codexRequestEpoch += 1;
    this.state.codexStartInFlight = false;
    this.state.codexReconnectInFlight = false;
    this._codexReconnectOperationId = null;
    this._codexReconnectSessionId = null;
    this.state.materializationInFlight = false;
    this.panels?.unmount?.();
    this.panels = null;
    this.editor?.unmount?.();
    this.editor = null;
    if (this._document) this._document.removeEventListener("keydown", this._keydown);
    this.root = null;
    this._document = null;
    this._refs = {};
  }

  snapshot() {
    return {
      schema_version: AUDIT_WORKBENCH_MODEL_SCHEMA_VERSION,
      selected: clone(this.state.selected), selection_revision: this.state.selectionRevision,
      coverage: clone(this.state.coverage), finding: clone(this.state.finding),
      github_sync: clone(this.state.githubSync),
      next: clone(this.state.next),
      artifact_status: clone(this.state.artifactStatus),
      autosave: clone(this.state.autosave), service_status: this.state.serviceStatus,
      service_authority_status: this.state.serviceAuthorityStatus,
      service_revisions: {
        context_revision: this._serviceCas.contextRevision,
        source_revision: this._serviceCas.sourceRevision,
        queue_state_revision: this._serviceCas.queueStateRevision,
        queue_input_revision: this._serviceCas.queueInputRevision,
        queue_input_identity: this._serviceCas.queueInputIdentity,
      },
      presentation_status: this.state.presentationStatus,
      inspection_status: this.state.inspectionStatus,
      inspection_reason: this.state.inspectionReason,
      records_status: this.state.recordsStatus,
      records_reason: this.state.recordsReason,
      record_projection: clone(this._recordProjection),
      service_error: this.state.serviceError,
      native_diagnostic: clone(this.state.nativeDiagnostic),
      native_diagnostic_in_flight: this.state.nativeDiagnosticInFlight,
      native_diagnostic_input: clone(this.state.nativeDiagnosticInput),
      codex: clone(this.state.codex),
      codex_start_in_flight: this.state.codexStartInFlight,
      codex_reconnect_in_flight: this.state.codexReconnectInFlight,
      materialization_in_flight: this.state.materializationInFlight,
      related_cases: clone(this.state.relatedCases),
      panes: clone(this.state.panes),
      editor: this.editor?.snapshot?.() || null,
      panels: this.panels?.snapshot?.() || null,
    };
  }

  _artifactStatusBinding(episodeId = null, source = {}) {
    const context = source?.context
      ?? source?.context_result?.value
      ?? this._serviceCas.context
      ?? this.model.context
      ?? {};
    const packet = source?.packet || this.state?.selected || this.model.packet || {};
    const bindingContext = context && typeof context === "object" ? { ...context } : {};
    for (const field of ["episode_id", "context_revision", "source_revision", "source_digest"]) {
      if (source?.[field] !== undefined && source?.[field] !== null) {
        bindingContext[field] = source[field];
      }
    }
    if (episodeId !== null && episodeId !== undefined
      && (bindingContext.episode_id === undefined
        || bindingContext.episode_id === null || bindingContext.episode_id === "")) {
      bindingContext.episode_id = episodeId;
    }
    if (bindingContext.source_digest === undefined
      && this._serviceCas.context?.source_identity !== undefined) {
      bindingContext.source_identity = this._serviceCas.context.source_identity;
    }
    if (bindingContext.source_revision === undefined
      && this._serviceCas.sourceRevision !== undefined) {
      bindingContext.source_revision = this._serviceCas.sourceRevision;
    }
    if (bindingContext.context_revision === undefined
      && this._serviceCas.contextRevision !== undefined) {
      bindingContext.context_revision = this._serviceCas.contextRevision;
    }
    return artifactStatusBindingFromPacket(packet, bindingContext);
  }

  _rememberServiceState(result = {}, { recordProjection = true } = {}) {
    if (!result || typeof result !== "object") return;
    const snapshot = result.service_snapshot && typeof result.service_snapshot === "object"
      ? result.service_snapshot : result;
    const context = snapshot.context
      ?? snapshot.context_result?.value
      ?? result.context
      ?? result.context_result?.value;
    const contextRevision = snapshot.context_revision
      ?? context?.context_revision
      ?? result.context_revision;
    const sourceRevision = snapshot.source_revision
      ?? context?.source_revision
      ?? result.source_revision;
    const queueStateRevision = snapshot.queue_state_revision
      ?? snapshot.queue?.state_revision
      ?? snapshot.queue?.queue_state_revision
      ?? snapshot.queue_result?.value?.state_revision
      ?? snapshot.queue_result?.value?.queue_state_revision
      ?? result.queue_state_revision
      ?? result.queue?.state_revision
      ?? result.value?.state_revision;
    const queueInputRevision = snapshot.queue_input_revision
      ?? snapshot.queue?.input_revision
      ?? snapshot.queue?.queue_input_revision
      ?? snapshot.queue_result?.value?.input_revision
      ?? snapshot.queue_result?.value?.queue_input_revision
      ?? result.queue_input_revision
      ?? result.queue?.input_revision
      ?? result.value?.input_revision;
    const queueInputIdentity = snapshot.queue_input_identity
      ?? snapshot.queue?.input_identity
      ?? snapshot.queue_result?.value?.input_identity
      ?? result.queue_input_identity
      ?? result.queue?.input_identity
      ?? result.value?.input_identity;
    const projection = snapshot.record_projection ?? result.record_projection;
    if (recordProjection && projection && typeof projection === "object") {
      const durableProjection = projection.authoritative === true
        && projection.durability === "service_store";
      this._recordProjection = clone(projection);
      this._recordProjectionScope = projection.scope_id || null;
      this.state.recordsStatus = projection.completeness === "complete"
        ? (durableProjection ? "complete" : "same_facade_only") : "unknown";
      this.state.recordsReason = projection.completeness === "complete"
        ? (durableProjection ? "" : "projection is complete only for this facade instance")
        : "service record projection is incomplete";
    }
    if (context !== undefined && context !== null) this._serviceCas.context = clone(context);
    if (contextRevision !== undefined && contextRevision !== null) {
      this._serviceCas.contextRevision = contextRevision;
    }
    if (sourceRevision !== undefined && sourceRevision !== null) {
      this._serviceCas.sourceRevision = sourceRevision;
    }
    if (queueStateRevision !== undefined && queueStateRevision !== null) {
      this._serviceCas.queueStateRevision = queueStateRevision;
    }
    if (queueInputRevision !== undefined && queueInputRevision !== null) {
      this._serviceCas.queueInputRevision = queueInputRevision;
    }
    if (queueInputIdentity !== undefined && queueInputIdentity !== null) {
      this._serviceCas.queueInputIdentity = queueInputIdentity;
    }
    const artifactStatus = snapshot.artifact_status ?? result.artifact_status;
    if (artifactStatus !== undefined) {
      const episodeId = snapshot.packet?.episode_id || this.state.selected?.episode_id;
      this.state.artifactStatus = normalizeArtifactStatus(
        artifactStatus,
        episodeId,
        this._artifactStatusBinding(episodeId, snapshot),
        true,
      );
    }
  }

  _nextOptions() {
    const options = { expected_selection_revision: this.state.selectionRevision };
    if (this._serviceCas.contextRevision !== undefined) {
      options.expected_context_revision = this._serviceCas.contextRevision;
    }
    if (this._serviceCas.queueStateRevision !== undefined) {
      options.expected_queue_state_revision = this._serviceCas.queueStateRevision;
    }
    if (this._serviceCas.queueInputRevision !== undefined) {
      options.expected_queue_input_revision = this._serviceCas.queueInputRevision;
    }
    return options;
  }

  _nativeDiagnosticMethod() {
    const method = this.facade?.run_native_diagnostic || this.facade?.runNativeDiagnostic;
    return typeof method === "function" ? method.bind(this.facade) : null;
  }

  _nativeDiagnosticCapability() {
    return this.state.artifactStatus?.native_diagnostic || {
      status: "unavailable", reason: "server native diagnostic capability is unavailable",
    };
  }

  _nativeDiagnosticCasReady() {
    return Boolean(this.state.selected?.episode_id)
      && Number.isSafeInteger(this.state.selectionRevision)
      && this.state.selectionRevision >= 0
      && Number.isSafeInteger(this._serviceCas.contextRevision)
      && this._serviceCas.contextRevision >= 0;
  }

  _nativeDiagnosticReady(input = this.state.nativeDiagnosticInput) {
    const capability = this._nativeDiagnosticCapability();
    if (!this._nativeDiagnosticMethod() || capability.status !== "available"
      || capability.diagnostic_only !== true || capability.scientific_claim_allowed !== false
      || !this._nativeDiagnosticCasReady()) return false;
    try {
      nativeDiagnosticArguments({
        ...(input && typeof input === "object" ? input : {}),
        expected_selection_revision: this.state.selectionRevision,
        expected_context_revision: this._serviceCas.contextRevision,
      });
      return true;
    } catch (_error) {
      return false;
    }
  }

  _nativeDiagnosticUnavailable(reason = "server native diagnostic capability is unavailable") {
    this.state.nativeDiagnostic = normalizeNativeDiagnosticResult({ status: "unavailable", reason });
    this.state.nativeDiagnosticInFlight = false;
    this.render({ captureEditor: false });
    return clone(this.state.nativeDiagnostic);
  }

  _nativeDiagnosticReason(input = this.state.nativeDiagnosticInput) {
    if (!this._nativeDiagnosticMethod()) return "server action is unavailable";
    const capability = this._nativeDiagnosticCapability();
    if (capability.status !== "available") {
      return capability.reason || "server capability is not configured";
    }
    if (!this._nativeDiagnosticCasReady()) {
      return "current selection/context revisions are unavailable";
    }
    if (!this._nativeDiagnosticReady(input)) {
      return "enter a bounded intervention and two finite goal coordinates";
    }
    return "ready; result remains diagnostic-only";
  }

  _resetNativeDiagnosticForSelection() {
    this._nativeDiagnosticRequestEpoch += 1;
    this.state.nativeDiagnosticInFlight = false;
    this.state.nativeDiagnostic = normalizeNativeDiagnosticResult({
      status: "unavailable",
      reason: "native diagnostic result is bound to the selected case",
    });
    this.state.nativeDiagnosticInput = {
      intervention_id: "native-goal-change",
      robot_goal: nativeDiagnosticGoal(this.state.selected),
      activation_epsilon_m: NATIVE_DIAGNOSTIC_DEFAULT_EPSILON_M,
      deadline_s: NATIVE_DIAGNOSTIC_DEFAULT_DEADLINE_S,
    };
  }

  _discardStaleNativeDiagnostic(requestEpoch) {
    if (requestEpoch !== this._nativeDiagnosticRequestEpoch) return;
    this.state.nativeDiagnosticInFlight = false;
    this.state.nativeDiagnostic = normalizeNativeDiagnosticResult({
      status: "conflict",
      reason: "native diagnostic result was discarded because current context or intervention changed",
    });
    this.render({ captureEditor: false });
  }

  _recordProjectionContextMatches(packet, projection) {
    const identity = projection?.context_identity;
    if (!identity || typeof identity !== "object") return false;
    for (const field of ["episode_id", "execution_id", "campaign_id", "reference_id"]) {
      if (identity[field] === undefined || identity[field] === null || identity[field] === "") continue;
      if (packet?.[field] !== undefined && packet?.[field] !== null
        && String(identity[field]) !== String(packet[field])) return false;
    }
    if (identity.context_revision !== undefined
      && this._serviceCas.contextRevision !== undefined
      && Number(identity.context_revision) !== Number(this._serviceCas.contextRevision)) return false;
    if (identity.source_revision !== undefined && this._serviceCas.sourceRevision !== undefined
      && String(identity.source_revision) !== String(this._serviceCas.sourceRevision)) return false;
    if (identity.source_identity !== undefined && identity.source_identity !== null
      && identity.source_identity !== "") {
      const actualSourceIdentity = this._serviceCas.context?.source_identity
        ?? packet?.source_identity;
      if (actualSourceIdentity === undefined || actualSourceIdentity === null
        || String(identity.source_identity) !== String(actualSourceIdentity)) return false;
    }
    return true;
  }

  _durableRecordRowsMatch(saved, projection, annotations, findings) {
    const identity = projection?.context_identity;
    if (!identity || typeof identity !== "object") return false;
    const expectedEpisode = identity.episode_id
      ?? saved?.packet?.episode_id ?? this.state.selected?.episode_id;
    const expectedSourceIdentity = identity.source_identity;
    const expectedSourceRevision = identity.source_revision;
    if (expectedEpisode === undefined || expectedEpisode === null || expectedEpisode === ""
      || expectedSourceIdentity === undefined || expectedSourceIdentity === null
      || expectedSourceIdentity === "" || expectedSourceRevision === undefined
      || expectedSourceRevision === null) return false;
    const commonMatches = (row) => {
      if (!row || typeof row !== "object") return false;
      if (row.source_revision === undefined || row.source_revision === null
        || String(row.source_revision) !== String(expectedSourceRevision)) return false;
      return true;
    };
    for (const row of annotations) {
      if (!commonMatches(row) || String(row.episode_id || "") !== String(expectedEpisode)
        || String(row.source_identity || "") !== String(expectedSourceIdentity)) return false;
    }
    for (const row of findings) {
      if (!commonMatches(row)) return false;
      if (row.source_identity !== undefined && row.source_identity !== null
        && String(row.source_identity) !== String(expectedSourceIdentity)) return false;
      const members = [
        ...(Array.isArray(row.candidate_members) ? row.candidate_members : []),
        ...(Array.isArray(row.confirmed_members) ? row.confirmed_members : []),
        ...(Array.isArray(row.negative_controls) ? row.negative_controls : []),
      ];
      if (!members.some((member) => String(member) === String(expectedEpisode))) return false;
    }
    return true;
  }

  _applySnapshotRecords(saved, previousScope) {
    const annotations = Array.isArray(saved?.annotations) ? saved.annotations : null;
    const findings = Array.isArray(saved?.findings) ? saved.findings : null;
    const projection = saved?.record_projection;
    const hasArrays = annotations !== null && findings !== null;
    if (projection && typeof projection === "object") {
      const durableProjection = projection.authoritative === true
        && projection.durability === "service_store";
      if (!durableProjection) {
        const scopeMatches = Boolean(previousScope && projection.scope_id
          && previousScope === projection.scope_id);
        if (!scopeMatches) {
          this.state.recordsStatus = "unknown";
          this.state.recordsReason = "snapshot belongs to another facade instance";
          return;
        }
      }
      if (!this._recordProjectionContextMatches(saved.packet || this.state.selected, projection)) {
        this.state.recordsStatus = "unavailable";
        this.state.recordsReason = "snapshot record context does not match the selected packet";
        return;
      }
      if (projection.completeness !== "complete" || !hasArrays) {
        this.state.recordsStatus = "unknown";
        this.state.recordsReason = "service record projection is incomplete";
        return;
      }
      if (durableProjection && !this._durableRecordRowsMatch(
        saved, projection, annotations, findings,
      )) {
        this.state.recordsStatus = "unavailable";
        this.state.recordsReason = "snapshot durable record provenance does not match context";
        return;
      }
      this._recordProjection = clone(projection);
      this._recordProjectionScope = projection.scope_id;
      this.state.recordsStatus = durableProjection ? "complete" : "same_facade_only";
      this.state.recordsReason = durableProjection
        ? "" : "projection is complete only for this facade instance";
    } else if (!hasArrays) {
      this.state.recordsStatus = "unknown";
      this.state.recordsReason = "snapshot omitted authoritative record state";
      return;
    } else {
      // The disposable fixture has no projection scope and its arrays are
      // authoritative for that fixture facade.
      this.state.recordsStatus = "complete";
      this.state.recordsReason = "";
    }
    this.state.finding = clone(findings.at(-1) || null);
    if (this.state.selected) {
      this.state.selected.editor_model = this.state.selected.editor_model || {};
      this.state.selected.editor_model.annotations = clone(annotations);
    }
  }

  _captureEditor() {
    if (!this.editor || !this.state.selected) return;
    const snapshot = this.editor.snapshot();
    const selected = this.state.selected;
    const cursorTime = finite(snapshot.selected_time_s, finite(selected.cursor?.time_s, 0));
    const editorRevision = Number.isFinite(Number(snapshot.selection_revision))
      ? Number(snapshot.selection_revision) : 0;
    const priorAnnotations = Array.isArray(selected.editor_model?.annotations)
      ? selected.editor_model.annotations : [];
    const priorById = new Map(priorAnnotations.map((item) => [
      String(item.annotation_id || item.record_id || ""), item,
    ]));
    selected.cursor = {
      ...(selected.cursor || {}),
      time_s: cursorTime,
      selection_revision: this.state.selectionRevision,
    };
    selected.selection_revision = this.state.selectionRevision;
    selected.editor_model = selected.editor_model || {};
    selected.editor_model.annotations = (snapshot.annotations || []).map((item) => {
      const prior = priorById.get(String(item.annotation_id || item.record_id || ""));
      if (item.revision === undefined && prior?.revision !== undefined) {
        return { ...clone(item), revision: prior.revision };
      }
      return clone(item);
    });
    // Keep the editor's transient full-note draft and reference controls in
    // the selected packet.  render() unmounts/remounts the editor; losing
    // these fields there would silently discard text or pending references.
    selected.editor_model.full_draft = clone(snapshot.full_draft || {});
    selected.editor_model.reference_target = snapshot.reference_target || "actor";
    selected.editor_model.reference_target_id = snapshot.reference_target_id || "";
    selected.editor_model.reference_error = snapshot.reference_error || "";
    selected.editor_model.autosave = clone(snapshot.autosave || this.state.autosave);
    selected.editor_model.context = {
      ...(selected.editor_model.context || {}),
      selection_revision: editorRevision,
      context_revision: editorRevision,
      queue_selection_revision: this.state.selectionRevision,
      cursor: { time_s: cursorTime },
    };
    if (this.editor.model && typeof this.editor.model === "object") {
      this.editor.model.context = {
        ...(this.editor.model.context || {}),
        selection_revision: editorRevision,
        context_revision: editorRevision,
        queue_selection_revision: this.state.selectionRevision,
        cursor: { time_s: cursorTime },
      };
    }
    this.state.autosave = clone(snapshot.autosave || this.state.autosave);
  }

  _setStatus(status, error = "") {
    this.state.serviceStatus = status;
    this.state.serviceError = error;
    if (this._refs.serviceStatus) {
      this._refs.serviceStatus.textContent = error ? `${status}: ${error}` : status;
      this._refs.serviceStatus.className = error ? "audit-status audit-error" : "audit-status";
    }
  }

  _setAutosave(value) {
    this.state.autosave = { ...this.state.autosave, ...value };
    if (this._refs.autosave) {
      const detail = this.state.autosave.error ? ` (${this.state.autosave.error})` : "";
      this._refs.autosave.textContent = `Autosave: ${this.state.autosave.state}${detail}`;
      this._refs.autosave.className = `autosave-status autosave-${this.state.autosave.state}`;
    }
  }

  _renderQueueSummary() {
    if (this._refs.artifactStatus) {
      this._refs.artifactStatus.textContent = artifactStatusSummary(this.state.artifactStatus);
    }
    if (!this.state.selected) return;
    const packet = this.state.selected;
    const cursorTime = packet.cursor?.time_s ?? "unavailable";
    if (this._refs.queueCursor) {
      this._refs.queueCursor.textContent = `cursor ${cursorTime} s · selection revision ${this.state.selectionRevision}`;
    }
    const media = mediaSnapshot(packet, finite(packet.cursor?.time_s, 0));
    if (this._refs.queueMedia) {
      this._refs.queueMedia.textContent = media.status === "available"
        ? `media PTS ${media.pts_s} s (declared mapping)`
        : `media unavailable: ${media.reason}`;
      this._refs.queueMedia.className = media.status === "available" ? "" : "audit-warning";
    }
    if (this._refs.queueNext) {
      this._refs.queueNext.textContent = this.state.next
        ? `Next: ${this.state.next.episode_id}` : "Next: queue exhausted";
    }
  }

  _bridgeEditor() {
    if (!this.editor || this.editor.__auditWorkbenchDispatchBridged) return;
    const dispatch = this.editor.dispatch.bind(this.editor);
    this.editor.dispatch = (action = {}) => {
      const result = dispatch(action);
      const sync = () => {
        this._captureEditor();
        if (action.type === "select-time" && this.panels && !this._syncingCursor) {
          this._syncingCursor = true;
          try {
            this.panels.dispatch({ type: "seek", time_s: this.editor.snapshot().selected_time_s,
              source: "annotation-editor" });
          } finally {
            this._syncingCursor = false;
          }
        }
        this._renderQueueSummary();
      };
      if (result && typeof result.then === "function") return result.then((value) => {
        sync();
        return value;
      });
      sync();
      return result;
    };
    this.editor.__auditWorkbenchDispatchBridged = true;
  }

  async next() {
    this._captureEditor();
    const expectedSelectionRevision = this.state.selectionRevision;
    const expectedSelectionEpoch = this._selectionEpoch;
    this._setStatus("loading");
    try {
      const raw = await this.facade.next({
        ...this._nextOptions(),
        expected_selection_revision: expectedSelectionRevision,
      });
      const result = requireStatus(normalizeSelectionResult(raw), ["selected", "empty"], "queue next failed");
      if (this._selectionEpoch !== expectedSelectionEpoch) return this.snapshot();
      this._rememberServiceState(raw);
      // Keep the outgoing editor bound to its own packet through the async boundary.
      this._captureEditor();
      this.state.selectionRevision = Number(result.selection_revision ?? this.state.selectionRevision);
      this._selectionEpoch += 1;
      this._resetCodexForSelection();
      this._resetRelatedCasesForSelection();
      this.state.coverage = clone(result.coverage || this.state.coverage);
      this.state.next = clone(result.next || null);
      this.state.serviceAuthorityStatus = raw?.service_status ?? raw?.status ?? this.state.serviceAuthorityStatus;
      this.state.presentationStatus = result.presentation_status || result.status;
      this.state.inspectionStatus = result.inspection_status || null;
      this.state.inspectionReason = result.inspection_reason || result.reason || "";
      this.state.materializationInFlight = false;
      if (result.status === "empty") {
        this.state.selected = null;
        this.state.artifactStatus = normalizeArtifactStatus(null);
        this.state.finding = null;
        this.state.inspectionStatus = null;
        this.state.inspectionReason = "";
        this._setStatus("empty", result.reason || "queue exhausted");
      } else {
        this.state.selected = clone(result.packet);
        this.state.artifactStatus = normalizeArtifactStatus(
          result.artifact_status || raw?.artifact_status,
          result.packet?.episode_id,
          this._artifactStatusBinding(result.packet?.episode_id, raw),
          true,
        );
        this.state.finding = null;
        this._setStatus("selected");
      }
      this._resetNativeDiagnosticForSelection();
      this._setAutosave({ state: "saved", error: "", revision: null });
      this.render({ captureEditor: false });
      return this.snapshot();
    } catch (error) {
      const normalized = serviceError(error, "queue next failed");
      if (this._selectionEpoch === expectedSelectionEpoch) {
        this._setStatus(normalized.status || "error", normalized.message);
      }
      throw normalized;
    }
  }

  _annotationTransaction() {
    const selectedEpisodeId = this.state.selected?.episode_id;
    const selectedRevision = this.state.selectionRevision;
    const stillSelected = () => this.state.selectionRevision === selectedRevision
      && this.state.selected?.episode_id === selectedEpisodeId;
    return {
      atomic: true,
      prepare: async (record, token) => {
        if (stillSelected()) this._setAutosave({ state: "pending", error: "", revision: null });
        return {
          record: clone(record), token: clone(token),
          queue_selection_revision: selectedRevision,
        };
      },
      commit: async (proposal, token) => {
        const save = this.facade.saveAnnotation || this.facade.save_annotation;
        if (typeof save !== "function") throw new Error("injected audit facade requires saveAnnotation()");
        const result = await save(proposal.record, {
          expected_selection_revision: proposal.queue_selection_revision,
          expected_revision: token.expected_revision,
          operation_id: token.operation_id,
          ...(this._serviceCas.contextRevision !== undefined
            ? { expected_context_revision: this._serviceCas.contextRevision } : {}),
          ...(this._serviceCas.sourceRevision !== undefined
            ? { expected_source_revision: this._serviceCas.sourceRevision } : {}),
        });
        requireStatus(result, ["saved"], "annotation save failed");
        this._rememberServiceState(result);
        if (stillSelected()) {
          this._setAutosave({ state: "saved", error: "", revision: result.revision ?? null });
        }
        return result;
      },
    };
  }

  async quickAnnotate(classification = "unclear", observedBehavior = "") {
    if (!this.editor) throw new Error("select an audit packet before annotating");
    return this.editor.dispatch({ type: "quick-note", classification, observed_behavior: observedBehavior });
  }

  async fullAnnotate(fields = {}) {
    if (!this.editor) throw new Error("select an audit packet before annotating");
    const classification = fields.classification || this.editor.snapshot().full_draft?.classification || "unclear";
    return this.editor.dispatch({ type: "structured-note", ...fields, classification });
  }

  async saveLatestAnnotation() {
    if (!this.editor) throw new Error("select an audit packet before saving");
    const selectedEpisodeId = this.state.selected?.episode_id;
    const selectedRevision = this.state.selectionRevision;
    const editorSnapshot = this.editor.snapshot();
    this._captureEditor();
    this._renderQueueSummary();
    const latest = editorSnapshot.annotations.at(-1);
    if (!latest) throw new Error("create an annotation before saving");
    try {
      return await this.editor.save(latest, {
        transaction: this._annotationTransaction(),
        expected_selection_revision: editorSnapshot.selection_revision,
      });
    } catch (error) {
      const normalized = serviceError(error, "annotation save failed");
      if (
        this.state.selectionRevision === selectedRevision
        && this.state.selected?.episode_id === selectedEpisodeId
      ) {
        this._setAutosave({ state: "error", error: normalized.message, revision: null });
        this._setStatus(normalized.status || "error", normalized.message);
      }
      throw normalized;
    }
  }

  async persistFinding(annotation = null) {
    this._captureEditor();
    const selectedEpisodeId = this.state.selected?.episode_id;
    const selectedRevision = this.state.selectionRevision;
    const selectedAnnotation = annotation || this.state.selected?.editor_model?.annotations?.at(-1);
    if (!selectedAnnotation) throw new Error("create and save an annotation before persisting a finding");
    const saveFinding = this.facade.persistFinding || this.facade.persist_finding;
    if (typeof saveFinding !== "function") throw new Error("injected audit facade requires persistFinding()");
    try {
      const findingOptions = {
        expected_selection_revision: selectedRevision,
        operation_id: operationId("finding"),
        ...(this._serviceCas.context && typeof this._serviceCas.context === "object"
          ? { expected_context: clone(this._serviceCas.context) } : {}),
        ...(this._serviceCas.contextRevision !== undefined
          ? { expected_context_revision: this._serviceCas.contextRevision } : {}),
        ...(this._serviceCas.sourceRevision !== undefined
          ? { expected_source_revision: this._serviceCas.sourceRevision } : {}),
        ...(this._serviceCas.queueStateRevision !== undefined
          ? { expected_queue_state_revision: this._serviceCas.queueStateRevision } : {}),
        ...(this._serviceCas.queueInputRevision !== undefined
          ? { expected_queue_input_revision: this._serviceCas.queueInputRevision } : {}),
        ...(this._serviceCas.queueInputIdentity !== undefined
          ? { expected_queue_input_identity: this._serviceCas.queueInputIdentity } : {}),
      };
      const result = requireStatus(await saveFinding(selectedAnnotation, findingOptions), ["saved"], "finding save failed");
      this._rememberServiceState(result);
      if (
        this.state.selectionRevision !== selectedRevision
        || this.state.selected?.episode_id !== selectedEpisodeId
      ) return result;
      this.state.finding = clone(result.finding);
      if (!this._recordProjection) {
        this.state.recordsStatus = "current_write";
        this.state.recordsReason = "current finding receipt is available; record-list reopen is unavailable";
      }
      this.state.coverage = clone(result.coverage || this.state.coverage);
      this.state.next = clone(result.next || null);
      this._setStatus("finding_saved");
      this.render();
      return result;
    } catch (error) {
      const normalized = serviceError(error, "finding save failed");
      if (
        this.state.selectionRevision === selectedRevision
        && this.state.selected?.episode_id === selectedEpisodeId
      ) this._setStatus(normalized.status || "error", normalized.message);
      throw normalized;
    }
  }

  async syncFinding(repository) {
    const selectedEpisodeId = this.state.selected?.episode_id;
    const selectedRevision = this.state.selectionRevision;
    const finding = this.state.finding;
    const sync = this.facade.syncFinding || this.facade.sync_finding;
    if (typeof sync !== "function") {
      return { status: "unavailable", reason: "GitHub sync is unavailable" };
    }
    if (!selectedEpisodeId || !finding?.finding_id) {
      throw new Error("select and persist a finding before publishing it");
    }
    const findingRevision = finding.revision ?? finding.record_revision;
    if (!Number.isSafeInteger(findingRevision) || findingRevision < 0) {
      throw new Error("canonical finding revision is unavailable");
    }
    if (typeof repository !== "string" || !repository.trim()) {
      throw new Error("an allowlisted GitHub repository is required");
    }
    const cas = this._serviceCas;
    if (!Number.isSafeInteger(cas.contextRevision) || cas.sourceRevision === undefined) {
      throw new Error("current service and source revisions are unavailable");
    }
    try {
      const result = await sync.call(this.facade, {
        finding_id: String(finding.finding_id),
        repository: repository.trim(),
        expected_finding_revision: findingRevision,
        expected_selection_revision: selectedRevision,
        expected_context_revision: cas.contextRevision,
        expected_source_revision: cas.sourceRevision,
        retry_ambiguous: false,
        operation_id: operationId("github-sync"),
      });
      this._rememberServiceState(result);
      if (this.state.selectionRevision !== selectedRevision
        || this.state.selected?.episode_id !== selectedEpisodeId) return result;
      this.state.githubSync = clone(result);
      const syncedFinding = result?.value?.finding;
      if (syncedFinding && typeof syncedFinding === "object") this.state.finding = clone(syncedFinding);
      this._setStatus(result?.status || "unavailable", result?.reason || "");
      this.render({ captureEditor: false });
      return result;
    } catch (error) {
      const normalized = serviceError(error, "GitHub sync failed");
      if (this.state.selectionRevision === selectedRevision
        && this.state.selected?.episode_id === selectedEpisodeId) {
        this.state.githubSync = { status: normalized.status || "failed", reason: normalized.message };
        this._setStatus(normalized.status || "error", normalized.message);
      }
      throw normalized;
    }
  }

  async saveAndPersistFinding() {
    await this.saveLatestAnnotation();
    return this.persistFinding();
  }

  async recordHumanReview(outcome) {
    const selectedEpisodeId = this.state.selected?.episode_id;
    const selectedRevision = this.state.selectionRevision;
    const saveReview = this.facade.recordHumanReview || this.facade.record_human_review;
    if (!selectedEpisodeId || typeof saveReview !== "function") {
      throw new Error("full-human review requires a live selected audit case");
    }
    const cas = this._serviceCas;
    if (![cas.contextRevision, cas.queueStateRevision, cas.queueInputRevision].every(
      (value) => Number.isSafeInteger(value) && value >= 0,
    )) {
      throw new Error("full-human review requires current service and queue revisions");
    }
    try {
      const result = requireStatus(await saveReview(outcome, {
        expected_selection_revision: selectedRevision,
        expected_context_revision: cas.contextRevision,
        expected_queue_state_revision: cas.queueStateRevision,
        expected_queue_input_revision: cas.queueInputRevision,
        operation_id: operationId("full-human-review"),
      }), ["committed"], "full-human review failed");
      this._rememberServiceState(result);
      if (this.state.selectionRevision === selectedRevision
        && this.state.selected?.episode_id === selectedEpisodeId) {
        await this.reopen();
        this._setStatus("review_saved", "explicit full-episode review recorded");
      }
      return result;
    } catch (error) {
      const normalized = serviceError(error, "full-human review failed");
      if (this.state.selectionRevision === selectedRevision
        && this.state.selected?.episode_id === selectedEpisodeId) {
        this._setStatus(normalized.status || "error", normalized.message);
      }
      throw normalized;
    }
  }

  async runNativeDiagnostic(options = {}) {
    const method = this._nativeDiagnosticMethod();
    if (!method) return this._nativeDiagnosticUnavailable();
    if (!this.state.selected) {
      return this._nativeDiagnosticUnavailable("select an audit case before running a native diagnostic");
    }
    if (this._nativeDiagnosticCapability().status !== "available") {
      return this._nativeDiagnosticUnavailable(this._nativeDiagnosticReason(options));
    }
    if (!this._nativeDiagnosticCasReady()) {
      return this._nativeDiagnosticUnavailable("current selection/context revisions are unavailable");
    }
    if (this.state.nativeDiagnosticInFlight) {
      return clone(this.state.nativeDiagnostic);
    }
    const source = options && typeof options === "object" && !Array.isArray(options)
      ? options : {};
    const input = {
      intervention_id: source.intervention_id ?? this.state.nativeDiagnosticInput.intervention_id,
      robot_goal: source.robot_goal ?? this.state.nativeDiagnosticInput.robot_goal,
      activation_epsilon_m: source.activation_epsilon_m
        ?? this.state.nativeDiagnosticInput.activation_epsilon_m,
      deadline_s: source.deadline_s ?? this.state.nativeDiagnosticInput.deadline_s,
      ...(source.operation_id !== undefined ? { operation_id: source.operation_id } : {}),
    };
    const request = nativeDiagnosticArguments({
      ...input,
      expected_selection_revision: this.state.selectionRevision,
      expected_context_revision: this._serviceCas.contextRevision,
    });
    const selectedEpoch = this._selectionEpoch;
    const selectedRevision = this.state.selectionRevision;
    const selectedEpisodeId = this.state.selected?.episode_id;
    const requestedContextRevision = request.expected_context_revision;
    const requestedInputIdentity = nativeDiagnosticInputIdentity(request);
    const requestEpoch = ++this._nativeDiagnosticRequestEpoch;
    this.state.nativeDiagnosticInput = {
      intervention_id: request.intervention_id,
      robot_goal: clone(request.robot_goal),
      activation_epsilon_m: request.activation_epsilon_m,
      deadline_s: request.deadline_s,
    };
    this.state.nativeDiagnosticInFlight = true;
    this.state.nativeDiagnostic = normalizeNativeDiagnosticResult({
      status: "running",
      reason: "native diagnostic is running; no benchmark claim is established",
    });
    this.render({ captureEditor: false });
    const requestIsCurrent = () => requestEpoch === this._nativeDiagnosticRequestEpoch
      && selectedEpoch === this._selectionEpoch
      && selectedRevision === this.state.selectionRevision
      && selectedEpisodeId === this.state.selected?.episode_id
      && requestedContextRevision === this._serviceCas.contextRevision
      && requestedInputIdentity === nativeDiagnosticInputIdentity(this.state.nativeDiagnosticInput);
    try {
      const result = await method(request);
      if (!requestIsCurrent()) {
        this._discardStaleNativeDiagnostic(requestEpoch);
        return this.snapshot();
      }
      this.state.nativeDiagnostic = normalizeNativeDiagnosticResult(result);
      this.render({ captureEditor: false });
      return clone(this.state.nativeDiagnostic);
    } catch (error) {
      if (!requestIsCurrent()) {
        this._discardStaleNativeDiagnostic(requestEpoch);
        throw error;
      }
      const status = error?.status === 409 || error?.status === "conflict"
        ? "conflict"
        : (error?.status === "unavailable" || error?.status === 501
          ? "unavailable" : (error?.status === "denied" ? "denied" : "failed"));
      this.state.nativeDiagnostic = normalizeNativeDiagnosticResult({
        status,
        reason: error?.message || "native diagnostic failed",
      });
      this.render({ captureEditor: false });
      throw serviceError(error, "native diagnostic failed");
    } finally {
      if (requestIsCurrent()) {
        this.state.nativeDiagnosticInFlight = false;
        this.render({ captureEditor: false });
      }
    }
  }

  run_native_diagnostic(options = {}) {
    return this.runNativeDiagnostic(options);
  }

  async reopen() {
    if (typeof this.facade.snapshot !== "function") throw new Error("injected audit facade requires snapshot()");
    const selectedEpoch = this._selectionEpoch;
    let saved;
    try {
      saved = await this.facade.snapshot();
      if (saved?.status && !["complete", "ok"].includes(saved.status)) {
        throw serviceError(saved, "audit snapshot unavailable");
      }
    } catch (error) {
      const normalized = serviceError(error, "audit snapshot unavailable");
      if (this._selectionEpoch === selectedEpoch) this._setStatus(normalized.status || "error", normalized.message);
      throw normalized;
    }
    if (this._selectionEpoch !== selectedEpoch) return this.snapshot();
    const previousProjectionScope = this._recordProjectionScope;
    this._rememberServiceState(saved, { recordProjection: false });
    // Reopen is a durable reload: do not let the transient editor snapshot
    // overwrite revisions carried by the facade's saved annotation records.
    this.editor?.unmount?.();
    this.editor = null;
    this.state.selectionRevision = Number(saved.selection_revision ?? this.state.selectionRevision);
    this._selectionEpoch += 1;
    this._resetCodexForSelection();
    this._resetRelatedCasesForSelection();
    this.state.selected = clone(saved.packet || this.state.selected);
    this._resetNativeDiagnosticForSelection();
    this.state.artifactStatus = normalizeArtifactStatus(
      saved.artifact_status,
      this.state.selected?.episode_id,
      this._artifactStatusBinding(this.state.selected?.episode_id, saved),
      true,
    );
    this.state.coverage = clone(saved.coverage || this.state.coverage);
    this.state.next = clone(saved.next || null);
    this._applySnapshotRecords(saved, previousProjectionScope);
    this.state.serviceAuthorityStatus = saved.service_status ?? saved.status ?? this.state.serviceAuthorityStatus;
    this.state.presentationStatus = saved.presentation_status || this.state.presentationStatus;
    this.state.inspectionStatus = saved.inspection_status || this.state.inspectionStatus;
    this.state.inspectionReason = saved.inspection_reason || this.state.inspectionReason;
    this._setStatus("reopened");
    this.render();
    return this.snapshot();
  }

  _materializationMethod() {
    const method = this.facade?.materialize_selected || this.facade?.materializeSelected;
    return typeof method === "function" ? method.bind(this.facade) : null;
  }

  _materializationUnavailable(reason = "Materialization capability is unavailable") {
    const binding = this._artifactStatusBinding(this.state.selected?.episode_id, {
      context: this._serviceCas.context,
      packet: this.state.selected,
    });
    this.state.artifactStatus = normalizeArtifactStatus({
      ...binding,
      materialization: { status: "unavailable", reason },
      native_diagnostic: this.state.artifactStatus.native_diagnostic,
    }, this.state.selected?.episode_id, binding, true);
    this.render({ captureEditor: false });
    return clone(this.state.artifactStatus);
  }

  async materializeSelected(options = {}) {
    const method = this._materializationMethod();
    if (!method) return this._materializationUnavailable();
    const selectedEpisodeId = this.state.selected?.episode_id;
    if (!selectedEpisodeId) return this._materializationUnavailable("select an audit case before materializing");
    if (this.state.materializationInFlight) {
      return clone(this.state.artifactStatus);
    }
    const selectionEpoch = this._selectionEpoch;
    const selectionRevision = this.state.selectionRevision;
    const contextRevision = this._serviceCas.contextRevision
      ?? this.state.selected?.context_revision
      ?? this.state.selected?.editor_model?.context?.context_revision;
    if (!Number.isInteger(contextRevision) || contextRevision < 0) {
      return this._materializationUnavailable("selected artifact context revision is unavailable");
    }
    const source = options && typeof options === "object" ? options : {};
    const request = materializationArguments({
      ...source,
      operation_id: source.operation_id || operationId("materialize"),
      expected_selection_revision: selectionRevision,
      expected_context_revision: contextRevision,
    });
    this.state.materializationInFlight = true;
    this.render({ captureEditor: false });
    try {
      const result = await method(request);
      if (selectionEpoch !== this._selectionEpoch
        || selectionRevision !== this.state.selectionRevision
        || selectedEpisodeId !== this.state.selected?.episode_id) {
        return this.snapshot();
      }
      const binding = this._artifactStatusBinding(selectedEpisodeId, {
        context: this._serviceCas.context,
        packet: this.state.selected,
      });
      this.state.artifactStatus = materializationResultProjection(
        result, binding, this.state.artifactStatus,
      );
      this.render({ captureEditor: false });
      return clone(this.state.artifactStatus);
    } catch (error) {
      if (selectionEpoch !== this._selectionEpoch
        || selectionRevision !== this.state.selectionRevision
        || selectedEpisodeId !== this.state.selected?.episode_id) throw error;
      const status = error?.status === 409 || error?.status === "conflict"
        ? "conflict" : (error?.status === 501 || error?.status === "unavailable"
          ? "unavailable" : "failed");
      const binding = this._artifactStatusBinding(selectedEpisodeId, {
        context: this._serviceCas.context,
        packet: this.state.selected,
      });
      this.state.artifactStatus = materializationResultProjection({
        status, reason: error?.message || "materialization failed",
      }, binding, this.state.artifactStatus);
      this.render({ captureEditor: false });
      throw serviceError(error, "materialization failed");
    } finally {
      if (selectionEpoch === this._selectionEpoch
        && selectionRevision === this.state.selectionRevision
        && selectedEpisodeId === this.state.selected?.episode_id) {
        this.state.materializationInFlight = false;
        this.render({ captureEditor: false });
      }
    }
  }

  materialize_selected(options = {}) {
    return this.materializeSelected(options);
  }

  _codexMethod(snakeName, camelName) {
    const method = this.facade?.[snakeName] || this.facade?.[camelName];
    return typeof method === "function" ? method.bind(this.facade) : null;
  }

  _codexUnavailable(reason = "Codex capability is unavailable") {
    this.state.codex = normalizeCodexResult({ status: "unavailable", reason }, reason);
    this.render({ captureEditor: false });
    return clone(this.state.codex);
  }

  _relatedMethod() {
    const method = this.facade?.related_cases || this.facade?.relatedCases;
    return typeof method === "function" ? method.bind(this.facade) : null;
  }

  _relatedUnavailable(reason = "Related-case capability is unavailable", status = "unavailable") {
    this.state.relatedCases = normalizeRelatedCasesResult({ status, reason });
    this.render({ captureEditor: false });
    return clone(this.state.relatedCases);
  }

  async relatedCases(options = {}) {
    const method = this._relatedMethod();
    if (!method) return this._relatedUnavailable();
    const selectedEpisodeId = this.state.selected?.episode_id;
    if (!selectedEpisodeId) return this._relatedUnavailable("select an audit case before loading related cases");
    const selectedEpoch = this._selectionEpoch;
    const selectedRevision = this.state.selectionRevision;
    const source = typeof options === "string" ? { mode: options } : (options || {});
    const request = relatedCasesArguments({
      ...source,
      episode_id: selectedEpisodeId,
      ...(this._serviceCas.contextRevision !== undefined
        ? { expected_context_revision: this._serviceCas.contextRevision } : {}),
      operation_id: source.operation_id || operationId("related"),
    });
    this.state.relatedCases = normalizeRelatedCasesResult({
      status: "loading", reason: "Loading related candidates...",
    }, selectedEpisodeId);
    this.render({ captureEditor: false });
    try {
      const raw = await method(request);
      if (selectedEpoch !== this._selectionEpoch
        || selectedRevision !== this.state.selectionRevision
        || selectedEpisodeId !== this.state.selected?.episode_id) {
        return this.snapshot();
      }
      this.state.relatedCases = normalizeRelatedCasesResult(raw, selectedEpisodeId);
      this.render({ captureEditor: false });
      return clone(this.state.relatedCases);
    } catch (error) {
      if (selectedEpoch !== this._selectionEpoch
        || selectedRevision !== this.state.selectionRevision
        || selectedEpisodeId !== this.state.selected?.episode_id) throw error;
      const status = error?.status === 409 ? "conflict" : (error?.status || "failed");
      this.state.relatedCases = normalizeRelatedCasesResult({
        status, reason: error?.message || "related-case lookup failed",
      }, selectedEpisodeId);
      this.render({ captureEditor: false });
      throw serviceError(error, "related-case lookup failed");
    }
  }

  loadRelatedCases(options = {}) {
    return this.relatedCases(options);
  }

  _codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision) {
    return requestEpoch === this._codexRequestEpoch
      && selectionEpoch === this._selectionEpoch
      && selectionRevision === this.state.selectionRevision;
  }

  _resetCodexForSelection(reason = "Codex activity is bound to the selected case") {
    this._codexRequestEpoch += 1;
    this.state.codexStartInFlight = false;
    this.state.codexReconnectInFlight = false;
    this._codexReconnectOperationId = null;
    this._codexReconnectSessionId = null;
    this.state.codexPrompt = "";
    this.state.codex = normalizeCodexResult({ status: "unavailable", reason }, reason);
  }

  _resetRelatedCasesForSelection(reason = "Related cases are bound to the selected case") {
    this.state.relatedCases = normalizeRelatedCasesResult({ status: "unavailable", reason });
  }

  async codexStart(prompt = "", options = {}) {
    const method = this._codexMethod("codex_start", "codexStart");
    if (!method) return this._codexUnavailable();
    if (this.state.codexReconnectInFlight) {
      return this._codexUnavailable("A Codex reconnect is already in flight");
    }
    if (this.state.codexStartInFlight) {
      return this._codexUnavailable("A Codex turn is already in flight");
    }
    if (!this.state.selected) return this._codexUnavailable("select an audit case before starting Codex");
    const request = codexStartArguments(prompt, options);
    if (!request.prompt.trim()) return this._codexUnavailable("prompt is required");
    const selectionEpoch = this._selectionEpoch;
    const selectionRevision = this.state.selectionRevision;
    const requestEpoch = ++this._codexRequestEpoch;
    this.state.codexPrompt = request.prompt;
    this.state.codexStartInFlight = true;
    this.state.codex = normalizeCodexResult({
      ...this.state.codex,
      status: "running",
      operation_id: request.operation_id,
      reason: "Codex turn in progress; cancellation is unavailable until it settles",
    });
    this.render({ captureEditor: false });
    try {
      const result = await method(request);
      if (!this._codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision)) {
        return this.snapshot();
      }
      this.state.codex = normalizeCodexResult(result);
      return clone(this.state.codex);
    } catch (error) {
      if (!this._codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision)) {
        throw error;
      }
      this.state.codex = normalizeCodexResult({
        status: error?.status || "failed",
        reason: error?.message || "Codex start failed",
        operation_id: request.operation_id,
      });
      throw serviceError(error, "Codex start failed");
    } finally {
      if (this._codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision)) {
        this.state.codexStartInFlight = false;
        this.state.codexPrompt = "";
        this.render({ captureEditor: false });
      }
    }
  }

  startCodex(prompt = "", options = {}) {
    return this.codexStart(prompt, options);
  }

  async codexRead(options = {}) {
    const method = this._codexMethod("codex_read", "codexRead");
    if (!method) return this._codexUnavailable();
    if (this.state.codexStartInFlight || this.state.codexReconnectInFlight) {
      return clone(this.state.codex);
    }
    const selectionEpoch = this._selectionEpoch;
    const selectionRevision = this.state.selectionRevision;
    const requestEpoch = ++this._codexRequestEpoch;
    const operationIdValue = options && typeof options === "object"
      ? (options.operation_id || this.state.codex.operation_id)
      : this.state.codex.operation_id;
    try {
      const result = await method(codexReadArguments({ operation_id: operationIdValue }));
      if (!this._codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision)) {
        return this.snapshot();
      }
      this.state.codex = normalizeCodexResult(result);
      this.render({ captureEditor: false });
      return clone(this.state.codex);
    } catch (error) {
      if (!this._codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision)) throw error;
      this.state.codex = normalizeCodexResult({
        status: error?.status || "unavailable",
        reason: error?.message || "Codex activity is unavailable",
        operation_id: operationIdValue,
      });
      this.render({ captureEditor: false });
      throw serviceError(error, "Codex activity is unavailable");
    }
  }

  readCodex(options = {}) {
    return this.codexRead(options);
  }

  async codexReconnect(options = {}) {
    const method = this._codexMethod("codex_reconnect", "codexReconnect");
    if (!method) return this._codexUnavailable("Codex reconnect capability is unavailable");
    if (this.state.codexReconnectInFlight) return clone(this.state.codex);
    if (!this.state.selected) return this._codexUnavailable("select an audit case before reconnecting Codex");
    const source = options && typeof options === "object" ? options : {};
    const sessionId = source.codex_session_id
      || this.state.codex.codex_session_id
      || this._codexReconnectSessionId;
    if (!codexOpaqueId(sessionId)) {
      return this._codexUnavailable("no durable Codex session is available to reconnect");
    }
    const contextRevision = this._serviceCas.contextRevision
      ?? this.state.selected?.context_revision
      ?? this.state.selected?.editor_model?.context?.context_revision;
    if (!Number.isInteger(contextRevision) || contextRevision < 0) {
      return this._codexUnavailable("selected Codex context revision is unavailable");
    }
    const selectionEpoch = this._selectionEpoch;
    const selectionRevision = this.state.selectionRevision;
    const requestEpoch = ++this._codexRequestEpoch;
    const pendingOperationId = this._codexReconnectSessionId === sessionId
      ? this._codexReconnectOperationId : null;
    let request;
    try {
      request = codexReconnectArguments({
        ...source,
        codex_session_id: sessionId,
        operation_id: source.operation_id || pendingOperationId || operationId("codex-reconnect"),
        expected_selection_revision: selectionRevision,
        expected_context_revision: contextRevision,
      });
    } catch (error) {
      return this._codexUnavailable(error?.message || "Codex reconnect request is invalid");
    }
    this._codexReconnectOperationId = request.operation_id;
    this._codexReconnectSessionId = sessionId;
    this.state.codexReconnectInFlight = true;
    this.state.codex = normalizeCodexResult({
      ...this.state.codex,
      status: "running",
      operation_id: request.operation_id,
      codex_session_id: sessionId,
      reason: "Codex reconnect is in flight; retrying uses the same operation",
    });
    this.render({ captureEditor: false });
    const selectionIsCurrent = () => selectionEpoch === this._selectionEpoch
      && selectionRevision === this.state.selectionRevision;
    try {
      const result = await method(request);
      if (!this._codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision)) {
        return this.snapshot();
      }
      const normalized = normalizeCodexResult(result);
      // Keep the opaque request identity visible after a fail-closed provider
      // response so the user can retry the same idempotent reconnect.  The
      // server remains the authority; this is only local retry state.
      normalized.operation_id ||= request.operation_id;
      normalized.codex_session_id ||= sessionId;
      this.state.codex = normalized;
      this.state.codexReconnectInFlight = false;
      this.render({ captureEditor: false });
      return clone(this.state.codex);
    } catch (error) {
      if (!this._codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision)) throw error;
      this.state.codex = normalizeCodexResult({
        status: error?.status || "unavailable",
        reason: error?.message || "Codex reconnect failed",
        operation_id: request.operation_id,
        codex_session_id: sessionId,
      });
      this.state.codexReconnectInFlight = false;
      this.render({ captureEditor: false });
      throw serviceError(error, "Codex reconnect failed");
    } finally {
      if (selectionIsCurrent()) {
        this.state.codexReconnectInFlight = false;
        this.render({ captureEditor: false });
      }
    }
  }

  reconnectCodex(options = {}) {
    return this.codexReconnect(options);
  }

  async codexCancel(reason = "cancelled from audit workbench", options = {}) {
    if (this.state.codexStartInFlight) {
      this.state.codex = normalizeCodexResult({
        ...this.state.codex,
        status: "running",
        reason: "Cancel is unavailable while the synchronous Codex turn is in flight",
      });
      this.render({ captureEditor: false });
      return clone(this.state.codex);
    }
    const method = this._codexMethod("codex_cancel", "codexCancel");
    if (!method) return this._codexUnavailable();
    const operationIdValue = options && typeof options === "object"
      ? (options.operation_id || this.state.codex.operation_id)
      : this.state.codex.operation_id;
    if (!codexOpaqueId(operationIdValue)) {
      return this._codexUnavailable("No post-turn Codex operation is available to cancel");
    }
    const selectionEpoch = this._selectionEpoch;
    const selectionRevision = this.state.selectionRevision;
    const requestEpoch = ++this._codexRequestEpoch;
    const request = codexCancelArguments({ reason, operation_id: operationIdValue });
    try {
      const result = await method(request);
      if (!this._codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision)) {
        return this.snapshot();
      }
      this.state.codex = normalizeCodexResult(result);
      this.render({ captureEditor: false });
      return clone(this.state.codex);
    } catch (error) {
      if (!this._codexRequestCurrent(requestEpoch, selectionEpoch, selectionRevision)) throw error;
      this.state.codex = normalizeCodexResult({
        status: error?.status || "failed",
        reason: error?.message || "Codex cancel failed",
        operation_id: operationIdValue,
      });
      this.render({ captureEditor: false });
      throw serviceError(error, "Codex cancel failed");
    }
  }

  cancelCodex(reason = "cancelled from audit workbench", options = {}) {
    return this.codexCancel(reason, options);
  }

  togglePane(name) {
    if (!PANE_NAMES.includes(name)) return this.snapshot();
    this.state.panes[name] = !this.state.panes[name];
    const pane = this._refs.panes?.[name];
    if (pane) pane.open = this.state.panes[name];
    return this.snapshot();
  }

  handleKey(event = {}) {
    if (isTextEntry(event.target)) return false;
    const key = String(event.key || "").toLowerCase();
    if (key === "j" || event.code === "ArrowDown") {
      event.preventDefault?.();
      void this.next().catch(() => {});
      return true;
    }
    return false;
  }

  render({ captureEditor = true } = {}) {
    if (!this.root || !this._document) return;
    if (captureEditor) this._captureEditor();
    this.panels?.unmount?.();
    this.panels = null;
    this.editor?.unmount?.();
    this.editor = null;
    const documentRef = this._document;
    this.root.replaceChildren?.();
    this._refs = { panes: {} };
    const shell = documentRef.createElement("div");
    shell.className = "audit-workbench-shell";
    const heading = text(documentRef, "h1", "Benchmark audit workbench");
    heading.id = "audit-workbench-title";
    shell.appendChild(heading);
    shell.appendChild(text(
      documentRef,
      "p",
      this._codexMethod("codex_start", "codexStart")
        ? "Server-held Codex activity is diagnostic-only; benchmark claims are not established."
        : "Fixture-backed offline UI slice; server-held Codex capability is unavailable.",
      "audit-boundary",
    ));
    this._refs.serviceStatus = text(documentRef, "p", this.state.serviceError || this.state.serviceStatus, "audit-status");
    this._refs.serviceStatus.setAttribute?.("aria-live", "polite");
    shell.appendChild(this._refs.serviceStatus);
    this._refs.autosave = text(documentRef, "p", `Autosave: ${this.state.autosave.state}`, "autosave-status");
    this._refs.autosave.setAttribute?.("aria-live", "polite");
    shell.appendChild(this._refs.autosave);

    const queuePane = detailsPane(documentRef, "queue", "Queue and selected case", this.state.panes.queue);
    this._refs.panes.queue = queuePane.details;
    const nextButton = button(documentRef, "Queue Next", () => this.next(), { "aria-label": "Select next audit case" });
    queuePane.body.appendChild(nextButton);
    this._refs.artifactStatus = text(
      documentRef,
      "p",
      artifactStatusSummary(this.state.artifactStatus),
      "audit-status",
    );
    queuePane.body.appendChild(this._refs.artifactStatus);
    if (this.state.selected) {
      const packet = this.state.selected;
      queuePane.body.appendChild(text(documentRef, "h2", `${packet.episode_id} · ${packet.scenario_id || "case"}`));
      this._refs.queueCursor = text(documentRef, "p", "");
      queuePane.body.appendChild(this._refs.queueCursor);
      const reasons = documentRef.createElement("ul");
      for (const reason of packet.selection_reasons || []) reasons.appendChild(text(documentRef, "li", `${reason.code || "reason"}: ${reason.label || reason.reason || ""}`));
      queuePane.body.appendChild(reasons);
      const metrics = documentRef.createElement("ul");
      for (const metric of packet.metrics || []) metrics.appendChild(text(documentRef, "li", `${metric.label || metric.id}: ${metric.value} ${metric.unit || ""}`.trim()));
      queuePane.body.appendChild(metrics);
      this._refs.queueMedia = text(documentRef, "p", "");
      queuePane.body.appendChild(this._refs.queueMedia);
      if (this.state.presentationStatus === "unavailable" && this.state.inspectionReason) {
        queuePane.body.appendChild(text(
          documentRef,
          "p",
          `Inspection unavailable: ${this.state.inspectionReason}`,
          "audit-warning",
        ));
      }
      this._refs.queueNext = text(documentRef, "p", "");
      queuePane.body.appendChild(this._refs.queueNext);
      this._renderQueueSummary();

      const materialize = this._materializationMethod();
      if (materialize) {
        const materializationButton = button(
          documentRef,
          this.state.materializationInFlight
            ? "Materialization in progress" : "Materialize selected artifact",
          () => this.materializeSelected().catch(() => {}),
          { "aria-label": "Materialize selected artifact" },
        );
        materializationButton.disabled = this.state.materializationInFlight;
        if (materializationButton.disabled) materializationButton.setAttribute?.("disabled", "");
        else materializationButton.removeAttribute?.("disabled");
        queuePane.body.appendChild(materializationButton);
        const materializationStatus = this.state.materializationInFlight
          ? "Materialization is running; the selected case remains bound to its current context."
          : `Materialization action: ${this.state.artifactStatus.materialization.status} — ${this.state.artifactStatus.materialization.reason}`;
        queuePane.body.appendChild(text(
          documentRef,
          "p",
          materializationStatus,
          this.state.materializationInFlight ? "audit-status" : "audit-warning",
        ));
      }

      const nativeSection = documentRef.createElement("section");
      nativeSection.className = "audit-native-diagnostic";
      nativeSection.appendChild(text(documentRef, "h3", "Native diagnostic"));
      const nativeCapability = this._nativeDiagnosticCapability();
      const nativeActionAvailable = Boolean(
        this._nativeDiagnosticMethod() && nativeCapability.status === "available",
      );
      const nativeResult = this.state.nativeDiagnostic;
      const nativeStatus = text(
        documentRef,
        "p",
        nativeResult.status === "complete"
          ? "Native diagnostic complete · control fidelity verified · activation verified."
          : `Native diagnostic: ${nativeResult.status} — ${nativeResult.reason}`,
        nativeResult.status === "complete" ? "audit-status" : "audit-warning",
      );
      nativeStatus.setAttribute?.("aria-live", "polite");
      nativeStatus.setAttribute?.("role", "status");
      nativeSection.appendChild(nativeStatus);
      if (nativeActionAvailable) {
        const inputGroup = documentRef.createElement("div");
        inputGroup.className = "audit-native-diagnostic-inputs";
        const input = (labelText, id, type, value, attributes = {}) => {
          const label = documentRef.createElement("label");
          label.setAttribute?.("for", id);
          label.appendChild(text(documentRef, "span", labelText));
          const field = documentRef.createElement("input");
          field.type = type;
          field.id = id;
          field.value = value === null || value === undefined ? "" : String(value);
          field.setAttribute?.("aria-label", labelText);
          Object.entries(attributes).forEach(([key, attributeValue]) => {
            field.setAttribute?.(key, String(attributeValue));
          });
          label.appendChild(field);
          inputGroup.appendChild(label);
          return field;
        };
        const nativeInput = this.state.nativeDiagnosticInput || {};
        const goal = Array.isArray(nativeInput.robot_goal) ? nativeInput.robot_goal : [];
        const intervention = input(
          "Intervention ID",
          "audit-native-diagnostic-intervention",
          "text",
          nativeInput.intervention_id,
          { maxlength: 128 },
        );
        const goalX = input(
          "Goal X (m)",
          "audit-native-diagnostic-goal-x",
          "number",
          goal[0],
          { step: "any" },
        );
        const goalY = input(
          "Goal Y (m)",
          "audit-native-diagnostic-goal-y",
          "number",
          goal[1],
          { step: "any" },
        );
        const epsilon = input(
          "Activation epsilon (m)",
          "audit-native-diagnostic-epsilon",
          "number",
          nativeInput.activation_epsilon_m,
          { min: 0, max: 1, step: "any" },
        );
        const deadline = input(
          "Deadline (s)",
          "audit-native-diagnostic-deadline",
          "number",
          nativeInput.deadline_s,
          { min: 0, max: NATIVE_DIAGNOSTIC_MAX_DEADLINE_S, step: "any" },
        );
        const runButton = button(
          documentRef,
          this.state.nativeDiagnosticInFlight
            ? "Native diagnostic in progress" : "Run native diagnostic",
          () => this.runNativeDiagnostic({
            intervention_id: intervention.value,
            robot_goal: [Number(goalX.value), Number(goalY.value)],
            activation_epsilon_m: Number(epsilon.value),
            deadline_s: Number(deadline.value),
          }).catch(() => {}),
          { "aria-label": "Run native diagnostic" },
        );
        const readNumber = (field) => {
          const raw = String(field.value ?? "").trim();
          if (!raw) return null;
          const number = Number(raw);
          return Number.isFinite(number) ? number : null;
        };
        const guidance = text(documentRef, "p", "", "audit-status");
        guidance.setAttribute?.("aria-live", "polite");
        guidance.setAttribute?.("role", "status");
        const syncInput = () => {
          const nextInput = {
            intervention_id: String(intervention.value || ""),
            robot_goal: [readNumber(goalX), readNumber(goalY)].every(
              (value) => value !== null,
            ) ? [readNumber(goalX), readNumber(goalY)] : null,
            activation_epsilon_m: readNumber(epsilon),
            deadline_s: readNumber(deadline),
          };
          const inputChanged = nativeDiagnosticInputIdentity(nextInput)
            !== nativeDiagnosticInputIdentity(this.state.nativeDiagnosticInput);
          this.state.nativeDiagnosticInput = nextInput;
          if (inputChanged && !this.state.nativeDiagnosticInFlight) {
            this.state.nativeDiagnostic = normalizeNativeDiagnosticResult({
              status: "unavailable",
              reason: "native diagnostic result is bound to the current intervention",
            });
          }
          const ready = this._nativeDiagnosticReady(nextInput);
          runButton.disabled = this.state.nativeDiagnosticInFlight || !ready;
          if (runButton.disabled) runButton.setAttribute?.("disabled", "");
          else runButton.removeAttribute?.("disabled");
          nativeStatus.textContent = this.state.nativeDiagnostic.status === "complete"
            ? "Native diagnostic complete · control fidelity verified · activation verified."
            : `Native diagnostic: ${this.state.nativeDiagnostic.status} — ${this.state.nativeDiagnostic.reason}`;
          nativeStatus.className = this.state.nativeDiagnostic.status === "complete"
            ? "audit-status" : "audit-warning";
          guidance.textContent = this.state.nativeDiagnosticInFlight
            ? "Native diagnostic is running; editing inputs will discard its result."
            : this._nativeDiagnosticReason(nextInput);
          guidance.className = ready ? "audit-status" : "audit-warning";
        };
        [intervention, goalX, goalY, epsilon, deadline].forEach((field) => {
          field.addEventListener?.("input", syncInput);
        });
        inputGroup.appendChild(runButton);
        inputGroup.appendChild(guidance);
        syncInput();
        nativeSection.appendChild(inputGroup);
      } else {
        const unavailableButton = button(
          documentRef,
          "Native diagnostic unavailable",
          () => this.runNativeDiagnostic().catch(() => {}),
          { "aria-label": "Run native diagnostic" },
        );
        unavailableButton.disabled = true;
        unavailableButton.setAttribute?.("disabled", "");
        nativeSection.appendChild(unavailableButton);
        nativeSection.appendChild(text(
          documentRef,
          "p",
          `Native diagnostic unavailable: ${this._nativeDiagnosticReason()}`,
          "audit-warning",
        ));
      }
      nativeSection.appendChild(text(
        documentRef,
        "p",
        "Diagnostic-only action; no benchmark claim, source path, runner identity, "
          + "or credential is sent by the browser.",
        "audit-boundary",
      ));
      queuePane.body.appendChild(nativeSection);
    } else {
      queuePane.body.appendChild(text(documentRef, "p", "No case selected. Press Queue Next to begin."));
    }
    queuePane.details.appendChild(queuePane.body);
    shell.appendChild(queuePane.details);

    const main = documentRef.createElement("section");
    main.className = "audit-main-content";
    main.setAttribute?.("aria-label", "Audit annotation");
    const panelsRoot = documentRef.createElement("div");
    panelsRoot.className = "audit-review-panels-root";
    main.appendChild(panelsRoot);
    const editorRoot = documentRef.createElement("div");
    editorRoot.className = "audit-editor-root";
    this._refs.editorRoot = editorRoot;
    main.appendChild(editorRoot);
    if (this.state.selected) {
      const panelModel = clone(this.state.selected.editor_model?.panel_model || null);
      if (panelModel?.schema_version === "review-panels.v1") {
        panelModel.context = { ...(panelModel.context || {}),
          cursor: { time_s: finite(this.state.selected.cursor?.time_s, 0) } };
        this.panels = mountReviewPanels(panelModel, panelsRoot, {
          onCursorChange: (cursor) => {
            if (this._syncingCursor || !this.editor) return;
            this._syncingCursor = true;
            try {
              this.editor.dispatch({ type: "select-time", time_s: cursor.time_s });
            } finally {
              this._syncingCursor = false;
            }
            this._captureEditor();
            this._renderQueueSummary();
          },
        });
      } else {
        panelsRoot.appendChild(text(documentRef, "p", "Scene, video and metric panels unavailable for this case.", "audit-warning"));
      }
      if (hasAdmittedEditorModel(this.state.selected)) {
        const findingActions = documentRef.createElement("div");
        findingActions.className = "audit-finding-actions";
        findingActions.appendChild(button(documentRef, "Save latest annotation", () => this.saveLatestAnnotation()));
        findingActions.appendChild(button(documentRef, "Persist finding", () => this.persistFinding()));
        findingActions.appendChild(button(documentRef, "Save annotation + finding", () => this.saveAndPersistFinding()));
        if (typeof (this.facade.recordHumanReview || this.facade.record_human_review) === "function") {
          findingActions.appendChild(button(documentRef, "Record full review: pass", () => this.recordHumanReview("pass")));
          findingActions.appendChild(button(documentRef, "Record full review: fail", () => this.recordHumanReview("fail")));
        }
        main.appendChild(findingActions);
        this.editor = mountReviewEditor(packetEditorModel(this.state.selected), editorRoot, {
          annotationTransaction: this._annotationTransaction(),
        });
        this._bridgeEditor();
      } else {
        editorRoot.appendChild(text(documentRef, "p", "Annotation editor unavailable: this case has no admitted time or scene source.", "audit-warning"));
      }
    }
    shell.appendChild(main);

    const findingPane = detailsPane(documentRef, "finding", "Finding", this.state.panes.finding);
    this._refs.panes.finding = findingPane.details;
    const findingText = this.state.recordsStatus === "unknown" || this.state.recordsStatus === "unavailable"
      ? `Finding state ${this.state.recordsStatus}: ${this.state.recordsReason}`
      : (this.state.finding ? `${this.state.finding.finding_id}: ${this.state.finding.status}` : "No finding persisted yet.");
    findingPane.body.appendChild(text(documentRef, "p", findingText));
    const syncFinding = this.facade.syncFinding || this.facade.sync_finding;
    if (typeof syncFinding === "function") {
      const publish = documentRef.createElement("div");
      publish.className = "audit-github-publication";
      const repository = documentRef.createElement("input");
      repository.type = "text";
      repository.placeholder = "owner/repository (allowlisted by the service)";
      repository.value = this.state.githubSync?.repository || this.model.github_repository || "";
      repository.setAttribute?.("aria-label", "GitHub repository");
      const publishButton = button(
        documentRef,
        "Publish finding to GitHub",
        () => this.syncFinding(repository.value).catch(() => {}),
        { "aria-label": "Publish finding to GitHub" },
      );
      const syncStatus = this.state.githubSync;
      publishButton.disabled = !this.state.finding || !repository.value.trim();
      if (publishButton.disabled) publishButton.setAttribute?.("disabled", "");
      repository.addEventListener?.("input", () => {
        publishButton.disabled = !this.state.finding || !repository.value.trim();
        if (publishButton.disabled) publishButton.setAttribute?.("disabled", "");
        else publishButton.removeAttribute?.("disabled");
      });
      publish.appendChild(repository);
      publish.appendChild(publishButton);
      publish.appendChild(text(
        documentRef,
        "p",
        syncStatus
          ? `GitHub sync: ${syncStatus.status} — ${syncStatus.reason || ""}`
          : "Server-held, append-only publication; provider capability is checked at request time.",
        syncStatus?.status === "committed" ? "audit-status" : "audit-warning",
      ));
      findingPane.body.appendChild(publish);
    }
    findingPane.details.appendChild(findingPane.body);
    shell.appendChild(findingPane.details);
    const relatedPane = detailsPane(documentRef, "related", "Related cases", this.state.panes.related);
    this._refs.panes.related = relatedPane.details;
    const related = this.state.relatedCases;
    const relatedButton = button(
      documentRef,
      related.status === "loading" ? "Loading related cases" : "Find related cases",
      () => this.relatedCases().catch(() => {}),
      { "aria-label": "Find related audit cases" },
    );
    relatedButton.disabled = related.status === "loading" || !this.state.selected;
    if (relatedButton.disabled) relatedButton.setAttribute?.("disabled", "");
    relatedPane.body.appendChild(relatedButton);
    relatedPane.body.appendChild(text(
      documentRef,
      "p",
      related.reason || `Related-case status: ${related.status}`,
      related.status === "complete" ? "audit-status" : "audit-warning",
    ));
    if (related.candidates.length) {
      relatedPane.body.appendChild(text(
        documentRef,
        "p",
        `${related.candidate_count} retrieval candidate(s); none is confirmed membership.`,
      ));
      const relatedList = documentRef.createElement("ul");
      for (const candidate of related.candidates) {
        const reasons = candidate.reasons.length ? ` · ${candidate.reasons.join("; ")}` : "";
        relatedList.appendChild(text(
          documentRef,
          "li",
          `${candidate.candidate_id} · ${candidate.compatibility} · priority ${candidate.score}${reasons}`,
        ));
      }
      relatedPane.body.appendChild(relatedList);
    }
    relatedPane.body.appendChild(text(
      documentRef,
      "p",
      "Similarity is a retrieval aid only; it is not benchmark evidence or finding confirmation.",
      "audit-boundary",
    ));
    relatedPane.details.appendChild(relatedPane.body);
    shell.appendChild(relatedPane.details);
    const coveragePane = detailsPane(documentRef, "coverage", "Coverage", this.state.panes.coverage);
    this._refs.panes.coverage = coveragePane.details;
    coveragePane.body.appendChild(text(documentRef, "p", coverageSummary(this.state.coverage)));
    coveragePane.details.appendChild(coveragePane.body);
    shell.appendChild(coveragePane.details);
    const agentPane = detailsPane(documentRef, "agent", "Agent activity", this.state.panes.agent);
    this._refs.panes.agent = agentPane.details;
    const codexStart = this._codexMethod("codex_start", "codexStart");
    const codexRead = this._codexMethod("codex_read", "codexRead");
    const codexReconnect = this._codexMethod("codex_reconnect", "codexReconnect");
    const codexCancel = this._codexMethod("codex_cancel", "codexCancel");
    if (!codexStart || !codexRead || !codexCancel) {
      agentPane.body.appendChild(text(documentRef, "p", "Server-held Codex capability is unavailable."));
    } else {
      const codex = this.state.codex;
      const prompt = documentRef.createElement("textarea");
      prompt.setAttribute?.("aria-label", "Codex prompt");
      prompt.setAttribute?.("maxlength", String(CODEX_MAX_PROMPT_LENGTH));
      prompt.placeholder = "Ask for a bounded diagnostic review of the selected case";
      prompt.value = this.state.codexPrompt;
      prompt.addEventListener?.("input", (event) => {
        this.state.codexPrompt = String(event.target?.value || "").slice(0, CODEX_MAX_PROMPT_LENGTH);
      });
      agentPane.body.appendChild(prompt);
      const startButton = button(
        documentRef,
        this.state.codexStartInFlight ? "Codex turn in progress" : "Start Codex turn",
        () => this.codexStart(prompt.value).catch(() => {}),
        { "aria-label": "Start Codex diagnostic turn" },
      );
      startButton.disabled = this.state.codexStartInFlight
        || this.state.codexReconnectInFlight || !this.state.selected;
      if (startButton.disabled) startButton.setAttribute?.("disabled", "");
      agentPane.body.appendChild(startButton);
      const readButton = button(
        documentRef,
        "Refresh activity",
        () => this.codexRead().catch(() => {}),
        { "aria-label": "Refresh Codex activity" },
      );
      readButton.disabled = this.state.codexStartInFlight || this.state.codexReconnectInFlight;
      if (readButton.disabled) readButton.setAttribute?.("disabled", "");
      agentPane.body.appendChild(readButton);
      if (codexReconnect && codex.codex_session_id) {
        const reconnectButton = button(
          documentRef,
          this.state.codexReconnectInFlight
            ? "Reconnecting Codex session" : "Reconnect Codex session",
          () => this.codexReconnect().catch(() => {}),
          { "aria-label": "Reconnect Codex session" },
        );
        reconnectButton.disabled = this.state.codexStartInFlight
          || this.state.codexReconnectInFlight || !this.state.selected;
        if (reconnectButton.disabled) reconnectButton.setAttribute?.("disabled", "");
        agentPane.body.appendChild(reconnectButton);
      }
      const cancelButton = button(
        documentRef,
        "Cancel settled turn",
        () => this.codexCancel().catch(() => {}),
        { "aria-label": "Cancel settled Codex turn" },
      );
      cancelButton.disabled = this.state.codexStartInFlight
        || this.state.codexReconnectInFlight || !codex.operation_id;
      if (cancelButton.disabled) cancelButton.setAttribute?.("disabled", "");
      agentPane.body.appendChild(cancelButton);
      if (this.state.codexStartInFlight) {
        agentPane.body.appendChild(text(
          documentRef,
          "p",
          "Cancel is unavailable while the synchronous Codex turn is in flight.",
          "audit-warning",
        ));
      }
      this._refs.agentStatus = text(
        documentRef,
        "p",
        `Codex status: ${codex.status}${codex.reason ? ` · ${codex.reason}` : ""}`,
        "audit-status",
      );
      this._refs.agentStatus.setAttribute?.("aria-live", "polite");
      agentPane.body.appendChild(this._refs.agentStatus);
      appendCodexDisplay(documentRef, agentPane.body, codex);
    }
    agentPane.details.appendChild(agentPane.body);
    shell.appendChild(agentPane.details);
    this.root.appendChild(shell);
  }
}

export function mountAuditWorkbench(model, root, options = {}) {
  return new AuditWorkbenchController(model, root, options);
}

const dataElement = typeof document === "undefined" ? null : document.getElementById("audit-workbench-data");
const rootElement = typeof document === "undefined" ? null : document.getElementById("audit-workbench-root");
if (dataElement && rootElement) {
  try {
    const model = JSON.parse(dataElement.textContent || "{}");
    mountAuditWorkbench(model, rootElement, { facade: createFixtureFacade(model) });
  } catch (error) {
    rootElement.textContent = `Unable to load audit workbench model: ${error}`;
  }
}

export default AuditWorkbenchController;

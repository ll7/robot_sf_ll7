import assert from "node:assert/strict";

import {
  AuditWorkbenchController,
  createServiceFacade,
  createFixtureFacade,
  mediaSnapshot,
  normalizeArtifactStatus,
  nativeDiagnosticArguments,
  normalizeNativeDiagnosticResult,
} from "../../robot_sf/render/web_assets/components/audit_workbench/audit_workbench.js";

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
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) || []), listener]); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== listener)); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  remove() {}
  emit(type, event = {}) { for (const listener of this.listeners.get(type) || []) listener({ target: this, ...event }); }
}

class Document {
  constructor() { this.listeners = new Map(); }
  createElement(tagName) { return new Element(this, tagName); }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) || []), listener]); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== listener)); }
}

function sourceIdentity(id) {
  return { sources: { scene: {
    artifact_id: `${id}-scene`, uri: `${id}/scene.json`, format: "threejs-viewer.v1",
    schema: "threejs-viewer.v1", sha256: "a".repeat(64), source_commit: "b".repeat(40),
    config_identity: "fixture", units: "m", coordinate_frame: "map",
  } } };
}

function editorModel(id, missingMedia) {
  const identity = sourceIdentity(id);
  const video = missingMedia
    ? { status: "unavailable", reason: "recording_not_present", resolution_s: 1.5, samples: [] }
    : { status: "available", resolution_s: 1.5, samples: [
      { time_s: 0, value: { pts_s: 0 } },
      { time_s: 1, value: { pts_s: 0.041 } },
      { time_s: 2.5, value: { pts_s: 0.113 } },
    ] };
  return {
    schema_version: "review-editor.v1",
    context: { episode_id: id, execution_id: `${id}-run`, selection_revision: 0, cursor: { time_s: 0 } },
    source_identity: identity,
    source_identity_status: { [`${id}-scene`]: { status: "verified", sha256: "a".repeat(64) } },
    time: { origin_s: 0, terminal_s: 3, cursor: { time_s: 0 } },
    streams: { video },
    panel_model: {
      schema_version: "review-panels.v1",
      context: { episode_id: id, execution_id: `${id}-run`, actor_id: "robot" },
      time: { origin_s: 0, terminal_s: 3, cursor: { time_s: 0 } },
      streams: { scene: { status: "available", resolution_s: 1.5, samples: [
        { time_s: 0, value: { robot: { actor_id: "robot", position: [0, 0] }, pedestrians: [{ actor_id: "ped-1", position: [1, 1] }] } },
        { time_s: 1, value: { robot: { actor_id: "robot", position: [1, 0] }, pedestrians: [{ actor_id: "ped-1", position: [1, 1] }] } },
        { time_s: 2.5, value: { robot: { actor_id: "robot", position: [2, 0] }, pedestrians: [{ actor_id: "ped-1", position: [1, 1] }] } },
        { time_s: 3, value: { robot: { actor_id: "robot", position: [3, 0] }, pedestrians: [{ actor_id: "ped-1", position: [1, 1] }] } },
      ] }, video },
      metrics: {}, events: [], intervals: [],
    },
    metrics: {}, events: [], annotations: [],
    scene_surface: { objects: [{ id: "wall-1", point: [2, 2] }] },
    goal_geometry: { point: [3, 2] },
    storyboard: { schema_version: "review-storyboard-edit.v1", source_identity: identity,
      source_revision: "fixture-revision-1", intervals: [], order: [], captions: {} },
  };
}

function caseValue(id, missingMedia) {
  return {
    episode_id: id, execution_id: `${id}-run`, scenario_id: "fixture-scenario",
    selection_reasons: [{ code: "coverage_deficit", label: "control is under-reviewed" }],
    cursor: { time_s: 1, authority: "simulation_time" },
    metrics: [{ id: "clearance", label: "Clearance", value: 0.7, unit: "m" }],
    metric_units: { time: "s", distance: "m", clearance: "m" },
    media: missingMedia
      ? { status: "unavailable", reason: "recording_not_present", resolution_s: 1.5, samples: [] }
      : { status: "available", resolution_s: 1.5, samples: editorModel(id, false).streams.video.samples },
    editor_model: editorModel(id, missingMedia),
  };
}

const document = {
  schema_version: "audit-workbench.v1",
  service: { id: "ba06-fixture-facade", native: false, evidence_status: "diagnostic_only" },
  queue: [caseValue("fixture-normal", false), caseValue("fixture-missing", true)],
  queue_index: -1, selection_revision: 0,
  coverage: { schema_version: "audit-coverage.v1", reviewed: 0, total: 2, remaining: 2, reviewed_episode_ids: [], status: "under_review" },
};
const facade = createFixtureFacade(document);
const documentRef = new Document();
const root = new Element(documentRef, "main");
const controller = new AuditWorkbenchController(document, root, { facade });

const missingEditorDocument = new Document();
const missingEditorController = new AuditWorkbenchController({}, new Element(missingEditorDocument, "main"), {
  facade: {
    next: async () => ({ status: "selected", selection_revision: 1, packet: { episode_id: "retained-no-time", selection_revision: 1 } }),
    snapshot: async () => ({ status: "complete" }),
  },
});
await missingEditorController.next();
assert.equal(missingEditorController.snapshot().selected.episode_id, "retained-no-time");
assert.equal(missingEditorController.snapshot().editor, null);
assert.equal(missingEditorController.snapshot().service_status, "selected");
await assert.rejects(() => missingEditorController.quickAnnotate(), /select an audit packet before annotating/);

const markerOnlyDocument = new Document();
const markerOnlyController = new AuditWorkbenchController({}, new Element(markerOnlyDocument, "main"), {
  facade: {
    next: async () => ({ status: "selected", selection_revision: 1, packet: {
      episode_id: "marker-only", selection_revision: 1,
      editor_model: { schema_version: "review-editor.v1" },
    } }),
    snapshot: async () => ({ status: "complete" }),
  },
});
await markerOnlyController.next();
assert.equal(markerOnlyController.snapshot().editor, null);

const unavailablePacketController = new AuditWorkbenchController({}, new Element(new Document(), "main"), {
  facade: {
    next: async () => ({
      status: "complete", service_status: "complete", selection_status: "complete",
      presentation_status: "unavailable", inspection_status: "unavailable",
      inspection_reason: "recording_not_present", selection_revision: 1,
      packet: { episode_id: "retained-without-media", selection_revision: 1 },
    }),
    snapshot: async () => ({ status: "complete" }),
  },
});
await unavailablePacketController.next();
assert.equal(unavailablePacketController.snapshot().selected.episode_id, "retained-without-media");
assert.equal(unavailablePacketController.snapshot().service_status, "selected");
assert.equal(unavailablePacketController.snapshot().service_authority_status, "complete");
assert.equal(unavailablePacketController.snapshot().presentation_status, "unavailable");
assert.equal(unavailablePacketController.snapshot().inspection_status, "unavailable");
assert.equal(unavailablePacketController.snapshot().inspection_reason, "recording_not_present");
assert.equal(unavailablePacketController.snapshot().editor, null);

const partialScene = caseValue("native-partial", true);
partialScene.editor_model.panel_model.streams.scene.status = "partial";
partialScene.editor_model.panel_model.streams.scene.samples[0].missing = false;
const partialController = new AuditWorkbenchController({}, new Element(new Document(), "main"), {
  facade: {
    next: async () => ({ status: "selected", selection_revision: 1, packet: partialScene }),
    snapshot: async () => ({ status: "complete" }),
  },
});
await partialController.next();
assert.notEqual(partialController.snapshot().editor, null);
const missingPosition = caseValue("native-no-position", true);
missingPosition.editor_model.panel_model.streams.scene.status = "partial";
missingPosition.editor_model.panel_model.streams.scene.samples.forEach((sample) => { sample.missing = true; });
const missingPositionController = new AuditWorkbenchController({}, new Element(new Document(), "main"), {
  facade: {
    next: async () => ({ status: "selected", selection_revision: 1, packet: missingPosition }),
    snapshot: async () => ({ status: "complete" }),
  },
});
await missingPositionController.next();
assert.equal(missingPositionController.snapshot().editor, null);

const priorFetch = globalThis.fetch;
globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "complete", selection_revision: 1,
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "selected-episode",
      source_revision: "source-r4", source_identity: "digest-r4",
    },
    artifact_status_result: {
      status: "complete", reason: "safe",
      operation: { operation_id: "must-not-cross" },
      receipt: { source_path: "/private/receipt" },
      context: {
        context_revision: 4, episode_id: "selected-episode", source_revision: "source-r4",
      },
      value: {
        episode_id: "foreign-episode", context_revision: 99,
        materialization: { status: "available", reason: "configured" },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const serviceTransportSnapshot = await createServiceFacade().snapshot();
assert.equal(Object.prototype.hasOwnProperty.call(serviceTransportSnapshot, "annotations"), false);
assert.equal(Object.prototype.hasOwnProperty.call(serviceTransportSnapshot, "findings"), false);
assert.equal(Object.prototype.hasOwnProperty.call(serviceTransportSnapshot.artifact_status_result, "operation"), false);
assert.equal(Object.prototype.hasOwnProperty.call(serviceTransportSnapshot.artifact_status_result, "receipt"), false);
assert.equal(serviceTransportSnapshot.artifact_status_result.value.episode_id, "selected-episode");
assert.equal(serviceTransportSnapshot.artifact_status_result.value.materialization.status, "unavailable");
assert.equal(serviceTransportSnapshot.artifact_status_result.value.materialization.classification, null);
assert.doesNotMatch(JSON.stringify(serviceTransportSnapshot.artifact_status_result), /\/private/);

const serviceStatusToken = "service-secret-token";
globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "complete", reason: "safe",
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "selected-episode",
      source_revision: "source-r4", source_identity: "digest-r4",
    },
    artifact_status_result: {
      status: "complete", reason: serviceStatusToken,
      context: {
        context_revision: 4, episode_id: "selected-episode", source_revision: "source-r4",
      },
      value: {
        episode_id: "selected-episode", context_revision: 4,
        source_revision: "source-r4", source_digest: "digest-r4",
        materialization: {
          status: "available", reason: serviceStatusToken,
          result_status: serviceStatusToken,
          classification: "historical_original", fidelity: "verified",
        },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const serviceStatusTokenSnapshot = await createServiceFacade().snapshot();
assert.equal(serviceStatusTokenSnapshot.artifact_status_result.status, "conflict");
assert.equal(serviceStatusTokenSnapshot.artifact_status_result.value.materialization.status, "unavailable");
assert.equal(serviceStatusTokenSnapshot.artifact_status_result.value.materialization.classification, null);
assert.doesNotMatch(JSON.stringify(serviceStatusTokenSnapshot), new RegExp(serviceStatusToken));

globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "complete", reason: "safe",
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "selected-episode",
      source_revision: "source-r4", source_identity: "digest-r4",
    },
    artifact_status_result: {
      status: "complete", reason: "safe",
      context: {
        context_revision: 4, episode_id: "selected-episode", source_revision: "source-r4",
      },
      value: {
        episode_id: "selected-episode", context_revision: 4,
        source_revision: "source-r4", source_digest: "digest-r4",
        materialization: {
          status: "available", reason: "configured",
          classification: "historical_original", fidelity: "verified",
        },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const matchingServiceTransportSnapshot = await createServiceFacade().snapshot();
assert.equal(matchingServiceTransportSnapshot.artifact_status_result.value.materialization.status, "available");
assert.equal(matchingServiceTransportSnapshot.artifact_status_result.value.materialization.classification, "historical_original");

globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "complete", reason: "safe",
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "selected-episode",
      source_revision: "source-r4", source_identity: "digest-r4",
    },
    artifact_status_result: {
      status: "complete", reason: "safe",
      value: {
        episode_id: "selected-episode", context_revision: 4,
        source_revision: "source-r4", source_digest: "digest-r4",
        materialization: { status: "available", reason: "configured" },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const missingContextTransportSnapshot = await createServiceFacade().snapshot();
assert.equal(missingContextTransportSnapshot.artifact_status_result.status, "conflict");
assert.equal(missingContextTransportSnapshot.artifact_status_result.value.materialization.status, "unavailable");

globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "conflict", reason: "status is stale",
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "selected-episode",
      source_revision: "source-r4", source_identity: "digest-r4",
    },
    artifact_status_result: {
      status: "conflict", reason: "status is stale",
      value: {
        episode_id: "selected-episode", context_revision: 4,
        source_revision: "source-r4", source_digest: "digest-r4",
        materialization: { status: "available", reason: "configured" },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const failedServiceTransportSnapshot = await createServiceFacade().snapshot();
assert.equal(failedServiceTransportSnapshot.artifact_status_result.value.materialization.status, "unavailable");
assert.equal(failedServiceTransportSnapshot.artifact_status_result.value.native_diagnostic.status, "unavailable");

globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "complete", reason: "/private/status-reason",
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "selected-episode",
      source_revision: "source-r4", source_identity: "digest-r4",
    },
    artifact_status_result: {
      status: "complete", reason: "/private/status-reason",
      value: {
        episode_id: "selected-episode", context_revision: 4,
        source_revision: "source-r4", source_digest: "digest-r4",
        materialization: {
          status: "available", reason: "configured",
          classification: "historical_original", fidelity: "verified",
        },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const unsafeReasonTransportSnapshot = await createServiceFacade().snapshot();
assert.equal(unsafeReasonTransportSnapshot.artifact_status_result.value.materialization.status, "unavailable");
assert.equal(unsafeReasonTransportSnapshot.artifact_status_result.value.materialization.classification, null);
assert.doesNotMatch(JSON.stringify(unsafeReasonTransportSnapshot.artifact_status_result), /\/private/);

globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "saved", reason: "generic write result",
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "selected-episode",
      source_revision: "source-r4", source_identity: "digest-r4",
    },
    artifact_status_result: {
      status: "saved", reason: "generic write result",
      context: {
        context_revision: 4, episode_id: "selected-episode", source_revision: "source-r4",
      },
      value: {
        episode_id: "selected-episode", context_revision: 4,
        source_revision: "source-r4", source_digest: "digest-r4",
        materialization: {
          status: "available", reason: "configured",
          classification: "historical_original", fidelity: "verified",
        },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const savedStatusTransportSnapshot = await createServiceFacade().snapshot();
assert.equal(savedStatusTransportSnapshot.artifact_status_result.status, "unavailable");
assert.equal(savedStatusTransportSnapshot.artifact_status_result.value.materialization.status, "unavailable");
assert.equal(savedStatusTransportSnapshot.artifact_status_result.value.materialization.classification, null);

globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "complete", reason: "file:%2Fprivate%2Fstatus",
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "selected-episode",
      source_revision: "source-r4", source_identity: "digest-r4",
    },
    artifact_status_result: {
      status: "complete", reason: "file:%2Fprivate%2Fstatus",
      context: {
        context_revision: 4, episode_id: "selected-episode", source_revision: "source-r4",
      },
      value: {
        episode_id: "selected-episode", context_revision: 4,
        source_revision: "source-r4", source_digest: "digest-r4",
        materialization: {
          status: "available", reason: "configured",
          classification: "historical_original", fidelity: "verified",
        },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const encodedReasonTransportSnapshot = await createServiceFacade().snapshot();
assert.equal(encodedReasonTransportSnapshot.artifact_status_result.value.materialization.status, "unavailable");
assert.equal(encodedReasonTransportSnapshot.artifact_status_result.value.materialization.classification, null);
assert.doesNotMatch(JSON.stringify(encodedReasonTransportSnapshot.artifact_status_result), /%2F/i);

const encodedNestedReason = normalizeArtifactStatus({
  episode_id: "selected-episode", context_revision: 4,
  source_revision: "source-r4", source_digest: "digest-r4",
  materialization: {
    status: "available", reason: "file:%2Fprivate%2Fstatus",
    classification: "historical_original", fidelity: "verified",
  },
  native_diagnostic: { status: "available", reason: "configured" },
}, "selected-episode", {
  context_revision: 4, source_revision: "source-r4", source_digest: "digest-r4",
}, true);
assert.equal(encodedNestedReason.materialization.status, "unavailable");
assert.equal(encodedNestedReason.materialization.classification, null);
assert.equal(encodedNestedReason.materialization.fidelity, null);
assert.doesNotMatch(JSON.stringify(encodedNestedReason), /%2F/i);

globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "complete", reason: "safe",
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "foreign-episode",
      source_revision: "foreign-source", source_identity: "foreign-digest",
    },
    artifact_status_result: {
      status: "complete", reason: "safe",
      context: {
        context_revision: 4, episode_id: "foreign-episode", source_revision: "foreign-source",
      },
      value: {
        episode_id: "selected-episode", context_revision: 4,
        source_revision: "source-r4", source_digest: "digest-r4",
        materialization: { status: "available", reason: "configured" },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const contradictoryBindingTransportSnapshot = await createServiceFacade().snapshot();
assert.equal(contradictoryBindingTransportSnapshot.artifact_status_result.status, "conflict");
assert.equal(contradictoryBindingTransportSnapshot.artifact_status_result.value.materialization.status, "unavailable");
assert.equal(contradictoryBindingTransportSnapshot.artifact_status_result.value.native_diagnostic.status, "unavailable");

globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({
    status: "complete", reason: "safe",
    packet: {
      episode_id: "selected-episode", selection_revision: 4,
      source_revision: "source-r4", source_digest: "digest-r4",
    },
    context: {
      context_revision: 4, episode_id: "selected-episode",
      source_revision: "source-r4", source_identity: "digest-r4",
    },
    artifact_status_result: {
      status: "complete", reason: "safe",
      context: {
        context_revision: 4, episode_id: "selected-episode",
        source_revision: "source-r4", source_identity: "digest-r4",
        source_digest: "foreign-digest",
      },
      value: {
        episode_id: "selected-episode", context_revision: 4,
        source_revision: "source-r4", source_digest: "digest-r4",
        materialization: { status: "available", reason: "configured" },
        native_diagnostic: { status: "available", reason: "configured" },
      },
    },
  }),
});
const conflictingSourceAliasSnapshot = await createServiceFacade().snapshot();
assert.equal(conflictingSourceAliasSnapshot.artifact_status_result.status, "conflict");
assert.equal(conflictingSourceAliasSnapshot.artifact_status_result.value.materialization.status, "unavailable");
assert.equal(conflictingSourceAliasSnapshot.artifact_status_result.value.materialization.classification, null);

const relatedTransportCalls = [];
globalThis.fetch = async (_endpoint, init) => {
  relatedTransportCalls.push(JSON.parse(init.body));
  return {
    ok: true,
    json: async () => ({
      status: "complete",
      value: [{
        query_id: "fixture-normal", candidate_id: "fixture-peer", mode: "same_scenario",
        score: 0.75, compatibility: "compatible", reasons: ["same scenario"],
        features: { planner_id: "peer-planner", confirmed_members: ["must-not-cross"] },
      }],
    }),
  };
};
const relatedTransport = await createServiceFacade().relatedCases({
  episode_id: "fixture-normal", expected_context_revision: 4, operation_id: "related-http",
  mode: "same_scenario", limit: 4, token: "must-not-cross-browser-boundary",
});
assert.equal(relatedTransport.status, "complete");
assert.deepEqual(relatedTransportCalls, [{
  operation: "related_cases",
  arguments: {
    episode_id: "fixture-normal", expected_context_revision: 4, operation_id: "related-http",
    mode: "same_scenario", limit: 4,
  },
}]);
globalThis.fetch = priorFetch;

const unavailableController = new AuditWorkbenchController({}, null, { facade: {
  next: async () => ({ status: "unavailable" }),
  snapshot: async () => ({ status: "unavailable", reason: "service snapshot unavailable" }),
} });
await assert.rejects(unavailableController.reopen(), /service snapshot unavailable/);
assert.equal(unavailableController.snapshot().service_status, "unavailable");

function deferred() {
  let resolve;
  const promise = new Promise((ready) => { resolve = ready; });
  return { promise, resolve };
}

await controller.next();
assert.equal(controller.snapshot().selected.episode_id, "fixture-normal");
assert.equal(controller.snapshot().next.episode_id, "fixture-missing");
assert.equal(mediaSnapshot(controller.snapshot().selected, 2.5).pts_s, 0.113);
assert.equal(mediaSnapshot(caseValue("missing", true), 1).status, "unavailable");
for (const key of ["pts_s", "media_t_s", "media_time_s", "presentation_t_s"]) {
  assert.equal(mediaSnapshot({ media: {
    status: "available", resolution_s: 1, samples: [{ time_s: 10, value: { [key]: 0.37 } }],
  } }, 10).pts_s, 0.37);
}
assert.equal(mediaSnapshot({ media: {
  status: "available", resolution_s: 1, samples: [{ time_s: 10, value: { pts_s: null } }],
} }, 10).reason, "presentation_timestamp_missing");
assert.equal(mediaSnapshot({ media: {
  status: "available", resolution_s: 1, samples: [{ time_s: 10, value: {} }],
} }, 10).status, "unavailable");
for (const malformed of [true, false, [], {}]) {
  assert.equal(mediaSnapshot({ media: {
    status: "available", resolution_s: 1, samples: [{ time_s: 10, value: { pts_s: malformed } }],
  } }, 10).status, "unavailable");
}

controller.editor.dispatch({ type: "select-time", time_s: 2.5 });
assert.equal(controller.snapshot().editor.selected_time_s, 2.5);
assert.equal(controller.snapshot().selected.cursor.time_s, 2.5);
assert.equal(controller.snapshot().selection_revision, 1);
assert.equal(controller.snapshot().editor.selection_revision, 2);
assert.equal(controller.snapshot().selected.editor_model.context.selection_revision, 2);
assert.equal(controller.snapshot().selected.editor_model.context.queue_selection_revision, 1);
assert.equal(controller.snapshot().panels.cursor.time_s, 2.5);
controller.panels.dispatch({ type: "step", delta: 1 });
assert.equal(controller.snapshot().panels.cursor.time_s, 3);
assert.equal(controller.snapshot().editor.selected_time_s, 3);
assert.equal(controller.snapshot().selected.cursor.time_s, 3);
assert.equal(controller.snapshot().selection_revision, 1);
controller.panels.dispatch({ type: "seek", time_s: 2.5, source: "scrubber" });
assert.equal(controller.snapshot().editor.selected_time_s, 2.5);
controller.panels.dispatch({ type: "toggle-play" });
controller.panels.tick(controller.panels._playbackLastMs + 500);
assert.equal(controller.snapshot().panels.cursor.time_s, 3);
assert.equal(controller.snapshot().editor.selected_time_s, 3);
assert.equal(controller.snapshot().selection_revision, 1);
controller.panels.dispatch({ type: "seek", time_s: 2.5, source: "scrubber" });
await controller.quickAnnotate("normal", "control remains nominal");
const firstSave = await controller.saveLatestAnnotation();
assert.equal(firstSave.record.interval.start_s, 2.5);
const finding = await controller.persistFinding();
assert.equal(finding.finding.candidate_members[0], "fixture-normal");
assert.equal(controller.snapshot().coverage.reviewed, 1);
assert.equal(controller.snapshot().autosave.state, "saved");

const reopened = await controller.reopen();
assert.equal(reopened.selected.editor_model.annotations.length, 1);
assert.equal(reopened.finding.finding_id, "finding-fixture-normal");
const reopenedSave = await controller.saveLatestAnnotation();
assert.equal(reopenedSave.revision, 2);

controller.editor.dispatch({ type: "set-full-field", field: "observed_behavior", value: "draft survives remount" });
controller.editor.dispatch({ type: "set-reference-target", target: "metric" });
controller.editor.dispatch({ type: "set-reference-target-id", target_id: "clearance" });
controller.render();
assert.equal(controller.editor.snapshot().full_draft.observed_behavior, "draft survives remount");
assert.equal(controller.editor.snapshot().reference_target, "metric");
assert.equal(controller.editor.snapshot().reference_target_id, "clearance");
controller.render({ captureEditor: false });
assert.equal(controller.editor.snapshot().full_draft.observed_behavior, "draft survives remount");
assert.equal(controller.editor.snapshot().reference_target, "metric");
assert.equal(controller.editor.snapshot().reference_target_id, "clearance");

controller.editor.dispatch({ type: "snap-reference", target: "actor", target_id: "robot" });
controller.editor.dispatch({ type: "snap-reference", target: "goal", target_id: "goal" });
controller.editor.dispatch({ type: "snap-reference", target: "map", target_id: "wall-1" });
await controller.fullAnnotate({
  classification: "planner_defect",
  observed_behavior: "robot pauses before the crossing",
  hypothesis: "clearance guard engaged",
  confidence: "0.8",
  evidence_text: "clearance remains below the declared threshold",
  evidence_metric_id: "clearance",
  evidence_value: "0.7",
  evidence_units: "m",
});
const fullEditorRecord = controller.snapshot().editor.annotations.at(-1);
assert.equal(fullEditorRecord.mode, "full");
assert.equal(fullEditorRecord.references.length, 3);
assert.equal(fullEditorRecord.references[0].actor_id, "robot");
assert.equal(fullEditorRecord.references[1].goal_id, "goal");
assert.equal(fullEditorRecord.references[2].object_id, "wall-1");
assert.equal(fullEditorRecord.evidence[0].metric_id, "clearance");
const fullSave = await controller.saveLatestAnnotation();
assert.equal(fullSave.record.mode, "full");
const reopenedFull = await controller.reopen();
const durableFull = reopenedFull.editor.annotations.at(-1);
assert.equal(durableFull.observed_behavior, "robot pauses before the crossing");
assert.equal(durableFull.references.length, 3);
assert.equal(durableFull.references[2].coordinate_frame, "world");

const stale = facade.next({ expected_selection_revision: 0 });
assert.equal(stale.status, "conflict");
let typingPrevented = false;
const typingHandled = controller.handleKey({
  key: "j", target: { tagName: "TEXTAREA" }, preventDefault() { typingPrevented = true; },
});
assert.equal(typingHandled, false);
assert.equal(typingPrevented, false);
assert.equal(controller.state.selectionRevision, 1);

const crossFacade = createFixtureFacade(document);
const crossFirst = crossFacade.next({ expected_selection_revision: 0 });
assert.equal(crossFirst.packet.episode_id, "fixture-normal");
const crossSaved = crossFacade.saveAnnotation({
  annotation_id: "cross-case", classification: "normal",
}, {
  expected_selection_revision: 1, expected_revision: 0, operation_id: "cross-save",
});
assert.equal(crossSaved.status, "saved");
crossFacade.next({ expected_selection_revision: 1 });
const crossFinding = crossFacade.persistFinding(crossSaved.record, {
  expected_selection_revision: 2, operation_id: "cross-finding",
});
assert.equal(crossFinding.status, "conflict");

await controller.next();
const secondSelection = controller.snapshot();
assert.equal(secondSelection.selected.episode_id, "fixture-missing");
assert.equal(secondSelection.selected.cursor.time_s, 1);
assert.equal(secondSelection.selected.editor_model.annotations.length, 0);
assert.equal(secondSelection.selected.editor_model.context.episode_id, "fixture-missing");
assert.equal(secondSelection.editor.annotations.length, 0);
assert.equal(secondSelection.editor.selected_time_s, 1);
assert.equal(controller.editor.model.context.episode_id, "fixture-missing");
assert.equal(secondSelection.panels.cursor.time_s, 1);
assert.equal(controller.panels.model.context.episode_id, "fixture-missing");
assert.equal(mediaSnapshot(secondSelection.selected, 1).reason, "recording_not_present");
assert.equal(controller.snapshot().panes.queue, true);
controller.togglePane("agent");
assert.equal(controller.snapshot().panes.agent, true);
await controller.next();
const exhausted = controller.snapshot();
assert.equal(exhausted.selected, null);
assert.equal(exhausted.editor, null);
assert.equal(exhausted.panels, null);
assert.equal(exhausted.selection_revision, 2);
assert.equal(exhausted.service_status, "empty");
controller.unmount();

// A completed save from an old case may be durable, but must not repaint the
// new case's autosave revision or service status after Queue Next.
const delayedSaveFacade = createFixtureFacade(document);
const originalSave = delayedSaveFacade.saveAnnotation.bind(delayedSaveFacade);
const saveGate = deferred();
delayedSaveFacade.saveAnnotation = async (...args) => {
  const result = originalSave(...args);
  await saveGate.promise;
  return result;
};
const delayedSaveController = new AuditWorkbenchController(document, new Element(documentRef, "main"), {
  facade: delayedSaveFacade,
});
await delayedSaveController.next();
await delayedSaveController.quickAnnotate("normal", "old case note");
const oldSave = delayedSaveController.saveLatestAnnotation();
await Promise.resolve();
await Promise.resolve();
await delayedSaveController.next();
assert.equal(delayedSaveController.snapshot().selected.episode_id, "fixture-missing");
saveGate.resolve();
await oldSave;
assert.equal(delayedSaveController.snapshot().autosave.state, "saved");
assert.equal(delayedSaveController.snapshot().autosave.revision, null);
assert.equal(delayedSaveController.snapshot().service_status, "selected");
delayedSaveController.unmount();

const failedSaveFacade = createFixtureFacade(document);
const failedSaveGate = deferred();
failedSaveFacade.saveAnnotation = async () => {
  await failedSaveGate.promise;
  throw new Error("old case save failed");
};
const failedSaveController = new AuditWorkbenchController(document, new Element(documentRef, "main"), {
  facade: failedSaveFacade,
});
await failedSaveController.next();
await failedSaveController.quickAnnotate("normal", "will fail after selection");
const oldFailedSave = failedSaveController.saveLatestAnnotation();
await Promise.resolve();
await failedSaveController.next();
failedSaveGate.resolve();
await assert.rejects(oldFailedSave, /old case save failed/);
assert.equal(failedSaveController.snapshot().selected.episode_id, "fixture-missing");
assert.equal(failedSaveController.snapshot().autosave.state, "saved");
assert.equal(failedSaveController.snapshot().service_status, "selected");
failedSaveController.unmount();

const delayedFindingFacade = createFixtureFacade(document);
const originalFinding = delayedFindingFacade.persistFinding.bind(delayedFindingFacade);
const findingGate = deferred();
delayedFindingFacade.persistFinding = async (...args) => {
  const result = originalFinding(...args);
  await findingGate.promise;
  return result;
};
const delayedFindingController = new AuditWorkbenchController(document, new Element(documentRef, "main"), {
  facade: delayedFindingFacade,
});
await delayedFindingController.next();
await delayedFindingController.quickAnnotate("normal", "finding for old case");
await delayedFindingController.saveLatestAnnotation();
const oldFinding = delayedFindingController.persistFinding();
await Promise.resolve();
await delayedFindingController.next();
findingGate.resolve();
await oldFinding;
assert.equal(delayedFindingController.snapshot().selected.episode_id, "fixture-missing");
assert.equal(delayedFindingController.snapshot().finding, null);
assert.equal(delayedFindingController.snapshot().service_status, "selected");
delayedFindingController.unmount();

const delayedReopenFacade = createFixtureFacade(document);
const originalSnapshot = delayedReopenFacade.snapshot.bind(delayedReopenFacade);
const reopenGate = deferred();
delayedReopenFacade.snapshot = async () => {
  const result = originalSnapshot();
  await reopenGate.promise;
  return result;
};
const delayedReopenController = new AuditWorkbenchController(document, new Element(documentRef, "main"), {
  facade: delayedReopenFacade,
});
await delayedReopenController.next();
const oldReopen = delayedReopenController.reopen();
await delayedReopenController.next();
reopenGate.resolve();
await oldReopen;
assert.equal(delayedReopenController.snapshot().selected.episode_id, "fixture-missing");
assert.equal(delayedReopenController.snapshot().service_status, "selected");
delayedReopenController.unmount();

const oneCaseDocument = { ...document, queue: [caseValue("only", false)] };
const exhaustedFacade = createFixtureFacade(oneCaseDocument);
const exhaustedSnapshot = exhaustedFacade.snapshot.bind(exhaustedFacade);
const exhaustedReopenGate = deferred();
exhaustedFacade.snapshot = async () => {
  const result = exhaustedSnapshot();
  await exhaustedReopenGate.promise;
  return result;
};
const exhaustedController = new AuditWorkbenchController(oneCaseDocument, new Element(documentRef, "main"), {
  facade: exhaustedFacade,
});
await exhaustedController.next();
const reopenBeforeEmpty = exhaustedController.reopen();
await exhaustedController.next();
assert.equal(exhaustedController.snapshot().selected, null);
assert.equal(exhaustedController.snapshot().selection_revision, 1);
exhaustedReopenGate.resolve();
await reopenBeforeEmpty;
assert.equal(exhaustedController.snapshot().selected, null);
assert.equal(exhaustedController.snapshot().service_status, "empty");
exhaustedController.unmount();

const staleEmptyFacade = createFixtureFacade(oneCaseDocument);
const staleEmptyNext = staleEmptyFacade.next.bind(staleEmptyFacade);
const staleEmptyGate = deferred();
let exhaustedNextCalls = 0;
staleEmptyFacade.next = async (...args) => {
  exhaustedNextCalls += 1;
  if (exhaustedNextCalls === 3) {
    await staleEmptyGate.promise;
    throw new Error("late old Next error");
  }
  return staleEmptyNext(...args);
};
const staleEmptyController = new AuditWorkbenchController(oneCaseDocument, new Element(documentRef, "main"), {
  facade: staleEmptyFacade,
});
await staleEmptyController.next();
const completeEmpty = staleEmptyController.next();
const failedAfterEmpty = staleEmptyController.next();
await completeEmpty;
staleEmptyGate.resolve();
await assert.rejects(failedAfterEmpty, /late old Next error/);
assert.equal(staleEmptyController.snapshot().selected, null);
assert.equal(staleEmptyController.snapshot().service_status, "empty");
staleEmptyController.unmount();

// A stale concurrent Next conflict is visible to its caller but cannot
// overwrite the successful selection's status in the mounted UI.
const delayedNextFacade = createFixtureFacade(document);
const originalNext = delayedNextFacade.next.bind(delayedNextFacade);
const nextGate = deferred();
let nextCalls = 0;
delayedNextFacade.next = async (...args) => {
  nextCalls += 1;
  const result = originalNext(...args);
  if (nextCalls === 2) await nextGate.promise;
  return result;
};
const delayedNextController = new AuditWorkbenchController(document, new Element(documentRef, "main"), {
  facade: delayedNextFacade,
});
const firstNext = delayedNextController.next();
const staleNext = delayedNextController.next();
await firstNext;
nextGate.resolve();
await assert.rejects(staleNext, /selection revision|queue next failed|conflict/i);
assert.equal(delayedNextController.snapshot().selected.episode_id, "fixture-normal");
assert.equal(delayedNextController.snapshot().service_status, "selected");
delayedNextController.unmount();

const casDocument = {
  ...document,
  queue: [caseValue("cas-case", false)],
  coverage: { schema_version: "audit-coverage.v1", reviewed: 0, total: 1, remaining: 1,
    reviewed_episode_ids: [], status: "under_review" },
  service_snapshot: {
    context: { context_revision: 7, source_revision: "source-r7", episode_id: "" },
    context_revision: 7, queue_state_revision: 17, queue_input_revision: 23,
    queue_input_identity: "queue-input-23",
    record_projection: {
      scope: "facade_instance", scope_id: "scope-cas", completeness: "complete", authoritative: false,
      context_identity: { context_revision: 7, episode_id: "" },
    },
  },
};
const casFixture = createFixtureFacade(casDocument);
const casCalls = [];
const casFacade = {
  next: async (options = {}) => {
    casCalls.push({ operation: "next", options: { ...options } });
    return {
      ...casFixture.next(options), service_status: "complete", context_revision: 7,
      source_revision: "source-r7", queue_state_revision: 17, queue_input_revision: 23,
      queue_input_identity: "queue-input-23",
    };
  },
  saveAnnotation: async (annotation, options = {}) => {
    casCalls.push({ operation: "save_annotation", options: { ...options } });
    return casFixture.saveAnnotation(annotation, options);
  },
  persistFinding: async (annotation, options = {}) => {
    casCalls.push({ operation: "persist_finding", options: { ...options } });
    return casFixture.persistFinding(annotation, options);
  },
  recordHumanReview: async (outcome, options = {}) => {
    casCalls.push({ operation: "record_human_review", outcome, options: { ...options } });
    return { status: "committed", value: { scope: "full_episode", outcome } };
  },
  snapshot: async () => casFixture.snapshot(),
};
const casController = new AuditWorkbenchController(casDocument, new Element(new Document(), "main"), {
  facade: casFacade,
});
await casController.next();
assert.deepEqual(casCalls[0], {
  operation: "next",
  options: {
    expected_selection_revision: 0,
    expected_context_revision: 7,
    expected_queue_state_revision: 17,
    expected_queue_input_revision: 23,
  },
});
await casController.quickAnnotate("normal", "service CAS");
await casController.saveLatestAnnotation();
const saveCasCall = casCalls.find((call) => call.operation === "save_annotation");
  assert.deepEqual(saveCasCall.options, {
  expected_selection_revision: 1,
  expected_revision: 0,
  expected_context_revision: 7,
  expected_source_revision: "source-r7",
    operation_id: saveCasCall.options.operation_id,
  });
const findingCasResult = await casController.persistFinding();
assert.equal(findingCasResult.status, "saved");
const findingCasCall = casCalls.find((call) => call.operation === "persist_finding");
assert.deepEqual(findingCasCall.options, {
  expected_selection_revision: 1,
  expected_context: { context_revision: 7, source_revision: "source-r7", episode_id: "" },
  expected_context_revision: 7,
  expected_source_revision: "source-r7",
  expected_queue_state_revision: 17,
  expected_queue_input_revision: 23,
  expected_queue_input_identity: "queue-input-23",
  operation_id: findingCasCall.options.operation_id,
});
assert.match(renderedText(casController.root), /Record full review: pass/);
const humanReview = await casController.recordHumanReview("pass");
assert.equal(humanReview.status, "committed");
const reviewCasCall = casCalls.find((call) => call.operation === "record_human_review");
assert.equal(reviewCasCall.outcome, "pass");
assert.deepEqual(reviewCasCall.options, {
  expected_selection_revision: 1,
  expected_context_revision: 7,
  expected_queue_state_revision: 17,
  expected_queue_input_revision: 23,
  operation_id: reviewCasCall.options.operation_id,
});
casFacade.snapshot = async () => ({
  ...casFixture.snapshot(),
  annotations: [], findings: [],
  record_projection: {
    scope: "facade_instance", scope_id: "scope-cas", completeness: "complete", authoritative: false,
    context_identity: { context_revision: 7, episode_id: "" },
  },
});
const sameFacadeReopen = await casController.reopen();
assert.equal(sameFacadeReopen.records_status, "same_facade_only");
assert.equal(sameFacadeReopen.finding, null);
assert.deepEqual(sameFacadeReopen.selected.editor_model.annotations, []);
casController.state.finding = { finding_id: "stale-finding", status: "proposed" };
casFacade.snapshot = async () => ({
  status: "complete", selection_revision: 1, packet: casFixture.snapshot().packet,
  record_projection: {
    scope: "facade_instance", scope_id: "new-process-scope", completeness: "complete", authoritative: false,
    context_identity: { context_revision: 7, episode_id: "" },
  },
});
const unknownReopen = await casController.reopen();
assert.equal(unknownReopen.records_status, "unknown");
assert.match(unknownReopen.records_reason, /another facade instance/);
assert.equal(unknownReopen.finding.finding_id, "stale-finding");
casController.unmount();

// A BA-05-backed record projection is reconnectable: a fresh facade scope may
// apply it without importing the previous facade's private in-memory scope.
const durableDocument = {
  ...casDocument,
  service_snapshot: {
    ...casDocument.service_snapshot,
    context: {
      ...casDocument.service_snapshot.context,
      source_identity: "source-digest",
      source_revision: 0,
    },
    source_revision: 0,
    record_projection: {
      scope: "service_store", scope_id: "ba05-audit-store", completeness: "complete",
      authoritative: true, durability: "service_store", fresh_process: "reconnectable",
      context_identity: {
        context_revision: 7, episode_id: "cas-case", source_identity: "source-digest", source_revision: 0,
      },
    },
  },
};
const durableFacade = {
  next: async () => ({ status: "selected", selection_revision: 1, packet: caseValue("cas-case", true) }),
  snapshot: async () => ({
    status: "complete", selection_revision: 1, packet: caseValue("cas-case", true),
    context: {
      context_revision: 7, episode_id: "cas-case", source_identity: "source-digest", source_revision: 0,
    },
    annotations: [{
      record_type: "annotation", record_id: "durable-annotation", annotation_id: "durable-annotation",
      episode_id: "cas-case", author_kind: "agent", source_identity: "source-digest", source_revision: 0,
      revision: 1,
    }],
    findings: [{
      record_type: "finding", record_id: "durable-finding", finding_id: "durable-finding",
      status: "proposed", candidate_members: ["cas-case"], confirmed_members: [], source_revision: 0,
      revision: 1,
    }],
    record_projection: {
      scope: "service_store", scope_id: "ba05-audit-store", completeness: "complete",
      authoritative: true, durability: "service_store", fresh_process: "reconnectable",
      context_identity: {
        context_revision: 7, episode_id: "cas-case", source_identity: "source-digest", source_revision: 0,
      },
    },
  }),
};
const durableController = new AuditWorkbenchController(durableDocument, new Element(documentRef, "main"), {
  facade: durableFacade,
});
await durableController.next();
const durableReopen = await durableController.reopen();
assert.equal(durableReopen.records_status, "complete");
assert.equal(durableReopen.editor.annotations[0].author_kind, "agent");
assert.equal(durableReopen.finding.status, "proposed");
durableController.unmount();

const durableContextMismatchController = new AuditWorkbenchController(
  durableDocument, new Element(documentRef, "main"), {
    facade: {
      next: durableFacade.next,
      snapshot: async () => {
        const snapshot = await durableFacade.snapshot();
        return {
          ...snapshot,
          record_projection: {
            ...snapshot.record_projection,
            context_identity: {
              context_revision: 7, episode_id: "cas-case", source_identity: "foreign-source", source_revision: 0,
            },
          },
        };
      },
    },
  },
);
await durableContextMismatchController.next();
const durableContextMismatch = await durableContextMismatchController.reopen();
assert.equal(durableContextMismatch.records_status, "unavailable");
assert.match(durableContextMismatch.records_reason, /context/);
durableContextMismatchController.unmount();

const durableForeignRowController = new AuditWorkbenchController(
  durableDocument, new Element(documentRef, "main"), {
    facade: {
      next: durableFacade.next,
      snapshot: async () => {
        const snapshot = await durableFacade.snapshot();
        return {
          ...snapshot,
          annotations: [{
            ...snapshot.annotations[0], episode_id: "foreign-case", source_identity: "foreign-source",
          }],
        };
      },
    },
  },
);
await durableForeignRowController.next();
const durableForeignRow = await durableForeignRowController.reopen();
assert.equal(durableForeignRow.records_status, "unavailable");
assert.match(durableForeignRow.records_reason, /provenance/);
durableForeignRowController.unmount();
function renderedText(node) {
  return `${node.textContent || ""}${(node.children || []).map(renderedText).join("")}`;
}

const artifactStatusDocument = {
  ...document,
  packet: caseValue("artifact-status-case", false),
  artifact_status: {
    schema_version: "audit-artifact-status.v1",
    episode_id: "artifact-status-case",
    context_revision: 0,
    source_revision: "fixture-revision-1",
    source_digest: "a".repeat(64),
    materialization: {
      status: "unavailable", classification: "historical_original", fidelity: "verified",
      reason: "retained trace available", diagnostic_only: true,
    },
    native_diagnostic: {
      status: "not_configured", reason: "native binding is not configured",
      evidence_boundary: "diagnostic_only", scientific_claim_allowed: false,
    },
  },
};
const artifactStatusRoot = new Element(new Document(), "main");
const artifactStatusController = new AuditWorkbenchController(
  artifactStatusDocument,
  artifactStatusRoot,
  { facade: createFixtureFacade(artifactStatusDocument) },
);
const artifactStatusText = renderedText(artifactStatusRoot);
assert.match(artifactStatusText, /Materialization: unavailable \(historical_original\)/);
assert.match(artifactStatusText, /Native diagnostic: not_configured/);
assert.match(artifactStatusText, /Diagnostic only/);
assert.equal(artifactStatusController.snapshot().artifact_status.materialization.fidelity, "verified");
artifactStatusController.unmount();

const staleArtifactStatus = normalizeArtifactStatus({
  episode_id: "artifact-status-case",
  materialization: { status: "available", reason: "/private/source-root" },
  native_diagnostic: { status: "available", reason: "native binding is configured" },
});
assert.equal(staleArtifactStatus.materialization.status, "unavailable");
assert.equal(staleArtifactStatus.materialization.reason, "selected artifact status is unavailable");
assert.equal(staleArtifactStatus.native_diagnostic.scientific_claim_allowed, false);
assert.doesNotMatch(JSON.stringify(staleArtifactStatus), /\/private/);

const foreignArtifactStatus = normalizeArtifactStatus({
  episode_id: "foreign-episode", context_revision: 99,
  source_revision: "foreign-source", source_digest: "foreign-digest",
  materialization: {
    status: "available", reason: "retained trace available",
    classification: "historical_original", fidelity: "verified",
  },
  native_diagnostic: { status: "available", reason: "native binding is configured" },
}, "selected-episode", {
  context_revision: 4, source_revision: "source-r4", source_digest: "digest-r4",
});
assert.equal(foreignArtifactStatus.episode_id, "selected-episode");
assert.equal(foreignArtifactStatus.materialization.status, "unavailable");
assert.equal(foreignArtifactStatus.materialization.classification, null);
assert.equal(foreignArtifactStatus.native_diagnostic.status, "unavailable");

const unsafeArtifactStatus = normalizeArtifactStatus({
  episode_id: "selected-episode", context_revision: 4,
  source_revision: "source-r4", source_digest: "digest-r4",
  materialization: {
    status: "available", reason: "configured", classification: "historical_original",
    fidelity: "verified", result_status: "/private/result",
  },
  native_diagnostic: { status: "available", reason: "configured" },
}, "selected-episode", {
  context_revision: 4, source_revision: "source-r4", source_digest: "digest-r4",
});
assert.equal(unsafeArtifactStatus.materialization.status, "unavailable");
assert.equal(unsafeArtifactStatus.materialization.classification, null);
assert.equal(Object.prototype.hasOwnProperty.call(unsafeArtifactStatus.materialization, "result_status"), false);

const encodedSourceArtifactStatus = normalizeArtifactStatus({
  episode_id: "selected-episode", context_revision: 4,
  source_revision: "source-r4", source_digest: "digest-%2Fprivate",
  materialization: {
    status: "available", reason: "configured",
    classification: "historical_original", fidelity: "verified",
  },
  native_diagnostic: { status: "available", reason: "configured" },
}, "selected-episode", {
  context_revision: 4, source_revision: "source-r4", source_digest: "digest-%2Fprivate",
}, true);
assert.equal(encodedSourceArtifactStatus.materialization.status, "unavailable");
assert.equal(encodedSourceArtifactStatus.materialization.classification, null);
assert.doesNotMatch(JSON.stringify(encodedSourceArtifactStatus), /%2F/i);

for (const [field, malformed] of [["status", []], ["status", {}], ["classification", []], ["fidelity", {}]]) {
  const malformedCapability = {
    status: "available", reason: "configured",
    classification: "historical_original", fidelity: "verified",
  };
  malformedCapability[field] = malformed;
  const malformedStatus = normalizeArtifactStatus({
    episode_id: "selected-episode", context_revision: 4,
    source_revision: "source-r4", source_digest: "digest-r4",
    materialization: malformedCapability,
    native_diagnostic: { status: "available", reason: "configured" },
  }, "selected-episode", {
    context_revision: 4, source_revision: "source-r4", source_digest: "digest-r4",
  }, true);
  assert.equal(malformedStatus.materialization.status, "unavailable");
  assert.equal(malformedStatus.materialization.classification, null);
  assert.equal(malformedStatus.materialization.fidelity, null);
}

const contradictoryControllerModel = {
  packet: {
    episode_id: "selected-episode", selection_revision: 4,
    source_revision: "source-r4", source_digest: "digest-r4",
  },
  context: {
    context_revision: 4, episode_id: "foreign-episode",
    source_revision: "foreign-source", source_identity: "foreign-digest",
  },
  artifact_status: {
    episode_id: "selected-episode", context_revision: 4,
    source_revision: "source-r4", source_digest: "digest-r4",
    materialization: { status: "available", reason: "configured" },
    native_diagnostic: { status: "available", reason: "configured" },
  },
};
const contradictoryController = new AuditWorkbenchController(
  contradictoryControllerModel, new Element(new Document(), "main"), {
    facade: { next: async () => ({ status: "empty" }), snapshot: async () => ({ status: "complete" }) },
  },
);
assert.equal(contradictoryController.snapshot().artifact_status.materialization.status, "unavailable");
assert.equal(contradictoryController.snapshot().artifact_status.native_diagnostic.status, "unavailable");
contradictoryController.unmount();

const foreignFixtureDocument = {
  ...artifactStatusDocument,
  artifact_status: {
    episode_id: "foreign-episode", context_revision: 99,
    materialization: { status: "available", reason: "configured" },
    native_diagnostic: { status: "available", reason: "configured" },
  },
};
const foreignFixtureStatus = createFixtureFacade(foreignFixtureDocument).snapshot().artifact_status;
assert.equal(foreignFixtureStatus.episode_id, "artifact-status-case");
assert.equal(foreignFixtureStatus.materialization.status, "unavailable");
assert.equal(foreignFixtureStatus.native_diagnostic.status, "unavailable");

const unselectedFixtureStatus = createFixtureFacade({
  ...foreignFixtureDocument, packet: null, queue_index: -1, selection_revision: 0,
}).snapshot().artifact_status;
assert.equal(unselectedFixtureStatus.episode_id, null);
assert.equal(unselectedFixtureStatus.materialization.status, "no_selection");
assert.equal(unselectedFixtureStatus.materialization.classification, null);
assert.equal(unselectedFixtureStatus.native_diagnostic.status, "no_selection");

const foreignSourceFixtureDocument = {
  ...artifactStatusDocument,
  artifact_status: {
    episode_id: "artifact-status-case", context_revision: 0,
    source_revision: "foreign-source", source_digest: "foreign-digest",
    materialization: {
      status: "available", reason: "configured",
      classification: "historical_original", fidelity: "verified",
    },
    native_diagnostic: { status: "available", reason: "configured" },
  },
};
const foreignSourceFixtureStatus = createFixtureFacade(foreignSourceFixtureDocument).snapshot().artifact_status;
assert.equal(foreignSourceFixtureStatus.materialization.status, "unavailable");
assert.equal(foreignSourceFixtureStatus.materialization.classification, null);
assert.equal(foreignSourceFixtureStatus.native_diagnostic.status, "unavailable");

const missingSourceFixtureDocument = {
  ...artifactStatusDocument,
  artifact_status: {
    episode_id: "artifact-status-case", context_revision: 0,
    materialization: {
      status: "available", reason: "configured",
      classification: "historical_original", fidelity: "verified",
    },
    native_diagnostic: { status: "available", reason: "configured" },
  },
};
const missingSourceFixtureStatus = createFixtureFacade(missingSourceFixtureDocument).snapshot().artifact_status;
assert.equal(missingSourceFixtureStatus.materialization.status, "unavailable");
assert.equal(missingSourceFixtureStatus.materialization.classification, null);
assert.equal(missingSourceFixtureStatus.native_diagnostic.status, "unavailable");

const matchingFixtureDocument = {
  ...artifactStatusDocument,
  artifact_status: {
    episode_id: "artifact-status-case", context_revision: 0,
    source_revision: "fixture-revision-1", source_digest: "a".repeat(64),
    materialization: {
      status: "available", reason: "configured",
      classification: "historical_original", fidelity: "verified",
    },
    native_diagnostic: { status: "available", reason: "configured" },
  },
};
const matchingFixtureStatus = createFixtureFacade(matchingFixtureDocument).snapshot().artifact_status;
assert.equal(matchingFixtureStatus.materialization.status, "available");
assert.equal(matchingFixtureStatus.materialization.classification, "historical_original");
assert.equal(matchingFixtureStatus.materialization.fidelity, "verified");
for (const [field, value] of [
  ["context_revision", 99],
  ["source_revision", "packet-source-revision"],
  ["source_digest", "packet-source-digest"],
]) {
  const conflictingPacket = { ...matchingFixtureDocument.packet, [field]: value };
  const conflictingStatusDocument = {
    ...matchingFixtureDocument,
    packet: conflictingPacket,
    artifact_status: { ...matchingFixtureDocument.artifact_status, [field]: value },
  };
  const conflictingStatus = createFixtureFacade(conflictingStatusDocument).snapshot().artifact_status;
  assert.equal(conflictingStatus.materialization.status, "unavailable");
  assert.equal(conflictingStatus.materialization.classification, null);
  assert.equal(conflictingStatus.materialization.fidelity, null);
  assert.equal(conflictingStatus.native_diagnostic.status, "unavailable");
}
const foreignReopenController = new AuditWorkbenchController(
  foreignFixtureDocument, new Element(new Document(), "main"),
  { facade: createFixtureFacade(foreignFixtureDocument) },
);
const foreignReopen = await foreignReopenController.reopen();
assert.equal(foreignReopen.artifact_status.episode_id, "artifact-status-case");
assert.equal(foreignReopen.artifact_status.materialization.status, "unavailable");
foreignReopenController.unmount();

const ba04Report = {
  schema_version: "audit-coverage.v1", status: "incomplete",
  counts: { coverage: { readable: 3 }, reviews: { full_episode_human: 1 } },
  deficits: [{ status: "under_review" }, { status: "waived" }],
};
const coverageRoot = new Element(new Document(), "main");
const coverageController = new AuditWorkbenchController(
  { ...document, coverage: { capability: "ba-04.coverage", status: "complete", value: ba04Report } },
  coverageRoot, { facade: createFixtureFacade(document) },
);
assert.match(renderedText(coverageRoot), /1\/3 readable episodes have full human review/);
assert.match(renderedText(coverageRoot), /1 unmet protocol requirements · incomplete/);
coverageController.unmount();
const unavailableCoverageRoot = new Element(new Document(), "main");
const unavailableCoverageController = new AuditWorkbenchController(
  { ...document, coverage: null }, unavailableCoverageRoot, { facade: createFixtureFacade(document) },
);
assert.match(renderedText(unavailableCoverageRoot), /Coverage unavailable/);
assert.doesNotMatch(renderedText(unavailableCoverageRoot), /0\/0 episodes reviewed/);
unavailableCoverageController.unmount();
const deniedCoverageRoot = new Element(new Document(), "main");
const deniedCoverageController = new AuditWorkbenchController(
  { ...document, coverage: { capability: "ba-04.coverage", status: "unavailable", value: ba04Report } },
  deniedCoverageRoot, { facade: createFixtureFacade(document) },
);
assert.match(renderedText(deniedCoverageRoot), /Coverage unavailable/);
assert.doesNotMatch(renderedText(deniedCoverageRoot), /1\/3 readable episodes/);
deniedCoverageController.unmount();

function renderedTags(node) {
  return [node.tagName, ...(node.children || []).flatMap(renderedTags)];
}

function findElements(node, predicate) {
  if (!node) return [];
  const matches = predicate(node) ? [node] : [];
  return matches.concat((node.children || []).flatMap((child) => findElements(child, predicate)));
}

const relatedCalls = [];
const relatedFacade = {
  next: async () => ({
    status: "selected", selection_revision: 1, context_revision: 4,
    packet: caseValue("related-query", true),
  }),
  snapshot: async () => ({ status: "complete" }),
  related_cases: async (args) => {
    relatedCalls.push(args);
    return {
      status: "complete",
      value: [{
        query_id: "related-query", candidate_id: "related-peer", mode: "same_scenario",
        score: 0.8, compatibility: "compatible", reasons: ["same scenario"],
        features: { planner_id: "peer-planner", confirmed_members: ["must-not-cross"] },
      }],
    };
  },
};
const relatedController = new AuditWorkbenchController(
  {}, new Element(new Document(), "main"), { facade: relatedFacade },
);
await relatedController.next();
const relatedResult = await relatedController.relatedCases({ mode: "same_scenario", limit: 3 });
assert.equal(relatedResult.status, "complete");
assert.equal(relatedResult.candidates[0].candidate_id, "related-peer");
assert.equal(relatedResult.candidates[0].membership_status, "unconfirmed_candidate");
assert.equal(Object.prototype.hasOwnProperty.call(relatedResult.candidates[0], "confirmed_members"), false);
assert.equal(relatedCalls[0].episode_id, "related-query");
assert.equal(relatedCalls[0].expected_context_revision, 4);
relatedFacade.related_cases = async () => ({
  status: "complete",
  value: [{
    query_id: "foreign-query", candidate_id: "foreign-peer", mode: "same_scenario",
    score: 0.9, compatibility: "compatible", reasons: ["/private/secret"],
    features: { source_path: "/private/secret", confirmed_members: ["foreign-peer"] },
  }],
});
const foreignRelated = await relatedController.relatedCases();
assert.equal(foreignRelated.status, "conflict");
assert.equal(foreignRelated.candidates.length, 0);
relatedFacade.related_cases = async () => ({
  status: "denied", reason: "stale audit context", value: null,
});
const deniedRelated = await relatedController.relatedCases();
assert.equal(deniedRelated.status, "denied");
assert.match(deniedRelated.reason, /stale/);
relatedController.unmount();

const nativeArguments = nativeDiagnosticArguments({
  operation_id: "native-action-1",
  expected_selection_revision: 3,
  expected_context_revision: 4,
  intervention_id: "goal-change",
  robot_goal: [4, 4],
  activation_epsilon_m: 1e-9,
  deadline_s: 30,
  source_path: "/private/source",
  runner_identity: "must-not-cross-browser-boundary",
});
assert.deepEqual(nativeArguments, {
  operation_id: "native-action-1",
  expected_selection_revision: 3,
  expected_context_revision: 4,
  intervention_id: "goal-change",
  robot_goal: [4, 4],
  activation_epsilon_m: 1e-9,
  deadline_s: 30,
});
assert.throws(
  () => nativeDiagnosticArguments({
    expected_selection_revision: 0, expected_context_revision: 0,
    intervention_id: "goal-change", robot_goal: [true, 4],
  }),
  /two finite numbers/,
);
assert.deepEqual(
  normalizeNativeDiagnosticResult({
    status: "complete", diagnostic_only: true, scientific_claim_allowed: false,
    control_fidelity: "verified", activation: "verified", reason: "/private/result",
  }),
  {
    status: "complete", reason: "native diagnostic completed", diagnostic_only: true,
    scientific_claim_allowed: false, control_fidelity: "verified", activation: "verified",
  },
);
assert.equal(normalizeNativeDiagnosticResult({
  status: "complete", diagnostic_only: true, scientific_claim_allowed: false,
}).status, "failed");

const priorCodexFetch = globalThis.fetch;
const serviceCalls = [];
globalThis.fetch = async (_endpoint, init) => {
  serviceCalls.push(JSON.parse(init.body));
  return { ok: true, json: async () => ({ status: "complete", operation_id: "codex-http" }) };
};
const serviceFacade = createServiceFacade();
await serviceFacade.codex_start({
  prompt: "service prompt",
  operation_id: "codex-http",
  token_budget: 64,
  compute_budget: 1,
  token: "must-not-cross-browser-boundary",
  provider_path: "/private/provider",
});
await serviceFacade.codex_read({ operation_id: "codex-http", session_id: "forged" });
await serviceFacade.codex_reconnect({
  codex_session_id: "codex-session-http",
  operation_id: "codex-reconnect-http",
  expected_selection_revision: 1,
  expected_context_revision: 2,
  token: "must-not-cross-browser-boundary",
});
await serviceFacade.codex_cancel({ reason: "post-turn", operation_id: "codex-http", path: "/private" });
await serviceFacade.run_native_diagnostic({
  ...nativeArguments,
  source_path: "/private/source",
  runner_identity: "must-not-cross-browser-boundary",
});
await serviceFacade.materialize_selected({
  operation_id: "materialize-http",
  expected_selection_revision: 1,
  expected_context_revision: 2,
  output_root: "/private/output",
});
await serviceFacade.recordHumanReview("pass", {
  expected_selection_revision: 1,
  expected_context_revision: 2,
  expected_queue_state_revision: 3,
  expected_queue_input_revision: 4,
  operation_id: "browser-review",
});
assert.deepEqual(serviceCalls, [
  {
    operation: "codex_start",
    arguments: { prompt: "service prompt", operation_id: "codex-http", token_budget: 64, compute_budget: 1 },
  },
  { operation: "codex_read", arguments: { operation_id: "codex-http" } },
  { operation: "codex_reconnect", arguments: {
    codex_session_id: "codex-session-http", operation_id: "codex-reconnect-http",
    expected_selection_revision: 1, expected_context_revision: 2,
  } },
  { operation: "codex_cancel", arguments: { reason: "post-turn", operation_id: "codex-http" } },
  { operation: "run_native_diagnostic", arguments: nativeArguments },
  { operation: "materialize_selected", arguments: {
    operation_id: "materialize-http", expected_selection_revision: 1,
    expected_context_revision: 2,
  } },
  { operation: "record_human_review", arguments: {
    outcome: "pass", expected_selection_revision: 1, expected_context_revision: 2,
    expected_queue_state_revision: 3, expected_queue_input_revision: 4,
    operation_id: "browser-review",
  } },
]);
globalThis.fetch = priorCodexFetch;

const nativePacket = caseValue("native-selected", false);
nativePacket.selection_revision = 1;
nativePacket.context_revision = 1;
nativePacket.source_revision = "fixture-revision-1";
nativePacket.source_digest = "a".repeat(64);
nativePacket.editor_model.context.selection_revision = 1;
nativePacket.editor_model.context.context_revision = 1;
const nativeModel = {
  packet: nativePacket,
  selection_revision: 1,
  context: {
    episode_id: "native-selected", context_revision: 1,
    source_revision: "fixture-revision-1", source_identity: "a".repeat(64),
  },
  artifact_status: {
    episode_id: "native-selected", context_revision: 1,
    source_revision: "fixture-revision-1", source_digest: "a".repeat(64),
    native_diagnostic: { status: "available", reason: "configured" },
  },
};
const nativeCalls = [];
const nativeFacade = {
  next: async () => ({ status: "empty" }),
  snapshot: async () => ({ status: "complete" }),
  run_native_diagnostic: async (args) => {
    nativeCalls.push(args);
    return {
      status: "complete", diagnostic_only: true, scientific_claim_allowed: false,
      control_fidelity: "verified", activation: "verified",
    };
  },
};
const nativeController = new AuditWorkbenchController(
  nativeModel, new Element(new Document(), "main"), { facade: nativeFacade },
);
assert.equal(nativeController.snapshot().artifact_status.native_diagnostic.status, "available");
const nativeResult = await nativeController.runNativeDiagnostic({
  operation_id: "native-controller-1",
  intervention_id: "goal-change",
  robot_goal: [4, 4],
  activation_epsilon_m: 1e-9,
  deadline_s: 30,
  source_path: "/private/source",
});
assert.equal(nativeResult.status, "complete");
assert.deepEqual(nativeCalls, [{
  operation_id: "native-controller-1",
  expected_selection_revision: 1,
  expected_context_revision: 1,
  intervention_id: "goal-change",
  robot_goal: [4, 4],
  activation_epsilon_m: 1e-9,
  deadline_s: 30,
}]);
assert.doesNotMatch(JSON.stringify(nativeCalls), /private|runner|token/i);
assert.match(renderedText(nativeController.root), /Native diagnostic complete/);

const materializationCalls = [];
const materializationFacade = {
  next: async () => ({ status: "empty" }),
  snapshot: async () => ({ status: "complete" }),
  materialize_selected: async (args) => {
    materializationCalls.push(args);
    return {
      status: "complete",
      reason: "retained state rendered",
      classification: "derived_render",
      fidelity: "unverifiable",
    };
  },
};
const materializationController = new AuditWorkbenchController(
  nativeModel, new Element(new Document(), "main"), { facade: materializationFacade },
);
assert.match(renderedText(materializationController.root), /Materialize selected artifact/);
const materializationResult = await materializationController.materializeSelected({
  operation_id: "materialize-controller-1",
});
assert.equal(materializationResult.materialization.status, "available");
assert.equal(materializationResult.materialization.classification, "derived_render");
assert.equal(materializationResult.materialization.fidelity, "unverifiable");
assert.deepEqual(materializationCalls, [{
  operation_id: "materialize-controller-1",
  expected_selection_revision: 1,
  expected_context_revision: 1,
}]);
assert.equal(materializationController.snapshot().materialization_in_flight, false);
materializationController.unmount();

const nativeLiveRegions = findElements(
  nativeController.root,
  (node) => node.attributes["aria-live"] === "polite" && node.attributes.role === "status",
);
assert.ok(nativeLiveRegions.length >= 2);
assert.match(renderedText(nativeController.root), /ready; result remains diagnostic-only/);
const nativeInterventionInput = findElements(
  nativeController.root,
  (node) => node.tagName === "INPUT" && node.attributes["aria-label"] === "Intervention ID",
)[0];
const nativeRunButton = findElements(
  nativeController.root,
  (node) => node.tagName === "BUTTON" && node.attributes["aria-label"] === "Run native diagnostic",
)[0];
nativeInterventionInput.value = "edited intervention";
nativeInterventionInput.emit("input");
assert.equal(nativeController.snapshot().native_diagnostic.status, "unavailable");
assert.equal(nativeRunButton.disabled, true);
assert.match(renderedText(nativeController.root), /enter a bounded intervention/);
nativeInterventionInput.value = "goal-change";
nativeInterventionInput.emit("input");
assert.equal(nativeRunButton.disabled, false);
assert.match(renderedText(nativeController.root), /ready; result remains diagnostic-only/);

const contextGate = deferred();
const contextPendingController = new AuditWorkbenchController(
  nativeModel, new Element(new Document(), "main"), {
    facade: {
      next: async () => ({ status: "empty" }),
      snapshot: async () => ({ status: "complete" }),
      run_native_diagnostic: async () => contextGate.promise,
    },
  },
);
const contextPending = contextPendingController.runNativeDiagnostic({
  operation_id: "native-context-pending",
  intervention_id: "context-change",
  robot_goal: [4, 4],
});
await Promise.resolve();
assert.equal(contextPendingController.snapshot().native_diagnostic.status, "running");
contextPendingController._serviceCas.contextRevision = 2;
contextGate.resolve({
  status: "complete", diagnostic_only: true, scientific_claim_allowed: false,
  control_fidelity: "verified", activation: "verified",
});
const contextPendingResult = await contextPending;
assert.equal(contextPendingResult.native_diagnostic.status, "conflict");
assert.match(contextPendingResult.native_diagnostic.reason, /discarded/);
assert.equal(contextPendingResult.native_diagnostic_in_flight, false);
contextPendingController.unmount();

const inputGate = deferred();
const inputPendingController = new AuditWorkbenchController(
  nativeModel, new Element(new Document(), "main"), {
    facade: {
      next: async () => ({ status: "empty" }),
      snapshot: async () => ({ status: "complete" }),
      run_native_diagnostic: async () => inputGate.promise,
    },
  },
);
const inputPending = inputPendingController.runNativeDiagnostic({
  operation_id: "native-input-pending",
  intervention_id: "input-change",
  robot_goal: [4, 4],
});
await Promise.resolve();
const pendingInterventionInput = findElements(
  inputPendingController.root,
  (node) => node.tagName === "INPUT" && node.attributes["aria-label"] === "Intervention ID",
)[0];
pendingInterventionInput.value = "edited-mid-flight";
pendingInterventionInput.emit("input");
assert.match(renderedText(inputPendingController.root), /editing inputs will discard/);
inputGate.resolve({
  status: "complete", diagnostic_only: true, scientific_claim_allowed: false,
  control_fidelity: "verified", activation: "verified",
});
const inputPendingResult = await inputPending;
assert.equal(inputPendingResult.native_diagnostic.status, "conflict");
assert.match(inputPendingResult.native_diagnostic.reason, /discarded/);
assert.equal(inputPendingResult.native_diagnostic_in_flight, false);
inputPendingController.unmount();
nativeController.unmount();

const fixtureNativeController = new AuditWorkbenchController(
  document, new Element(new Document(), "main"), { facade: createFixtureFacade(document) },
);
await fixtureNativeController.next();
const fixtureNativeResult = await fixtureNativeController.runNativeDiagnostic({
  intervention_id: "should-not-run", robot_goal: [4, 4],
});
assert.equal(fixtureNativeResult.status, "unavailable");
assert.match(renderedText(fixtureNativeController.root), /Native diagnostic unavailable/);
fixtureNativeController.unmount();

const absentCodexController = new AuditWorkbenchController({}, new Element(new Document(), "main"), {
  facade: { next: async () => ({ status: "empty" }), snapshot: async () => ({ status: "complete" }) },
});
assert.match(renderedText(absentCodexController.root), /Codex capability is unavailable/);
absentCodexController.unmount();

const staleCodexDocument = {
  ...document,
  queue: [caseValue("codex-old", false), caseValue("codex-new", false)],
  queue_index: -1,
  selection_revision: 0,
};
const startGate = deferred();
const staleCodexCalls = { starts: [], cancels: 0 };
const staleCodexFacade = {
  next: async ({ expected_selection_revision }) => {
    const next = expected_selection_revision === 0 ? staleCodexDocument.queue[0] : staleCodexDocument.queue[1];
    return {
      status: "selected",
      packet: { ...next },
      selection_revision: expected_selection_revision + 1,
    };
  },
  snapshot: async () => ({ status: "complete" }),
  codex_start: async (args) => {
    staleCodexCalls.starts.push(args);
    return startGate.promise;
  },
  codex_read: async () => ({ status: "unavailable", reason: "no activity" }),
  codex_cancel: async () => { staleCodexCalls.cancels += 1; return { status: "cancelled" }; },
};
const staleCodexController = new AuditWorkbenchController(
  staleCodexDocument,
  new Element(new Document(), "main"),
  { facade: staleCodexFacade },
);
await staleCodexController.next();
const staleStart = staleCodexController.codexStart("inspect old case", {
  token_budget: 64, compute_budget: 1,
});
await Promise.resolve();
assert.equal(staleCodexController.snapshot().codex_start_in_flight, true);
const blockedCancel = await staleCodexController.codexCancel();
assert.equal(blockedCancel.status, "running");
assert.match(blockedCancel.reason, /unavailable/);
assert.equal(staleCodexCalls.cancels, 0);
await staleCodexController.next();
startGate.resolve({
  status: "complete",
  operation_id: staleCodexCalls.starts[0].operation_id,
  context: { episode_id: "codex-old" },
  activity: [{ message: "old result must not repaint new selection" }],
});
await staleStart;
assert.equal(staleCodexController.snapshot().selected.episode_id, "codex-new");
assert.equal(staleCodexController.snapshot().codex.status, "unavailable");
staleCodexController.unmount();

const reconnectGate = deferred();
const reconnectCalls = [];
const reconnectDocument = {
  ...document,
  packet: { ...caseValue("codex-reconnect", false), context_revision: 2 },
  selection_revision: 1,
  context: { context_revision: 2, episode_id: "codex-reconnect" },
  codex: { status: "complete", codex_session_id: "codex-session-runtime" },
};
const reconnectFacade = {
  next: async () => ({ status: "selected" }),
  snapshot: async () => ({ status: "complete" }),
  codex_start: async () => ({ status: "unavailable" }),
  codex_read: async () => ({ status: "unavailable" }),
  codex_cancel: async () => ({ status: "unavailable" }),
  codex_reconnect: async (args) => {
    reconnectCalls.push(args);
    return reconnectGate.promise;
  },
};
const reconnectController = new AuditWorkbenchController(
  reconnectDocument,
  new Element(new Document(), "main"),
  { facade: reconnectFacade },
);
const firstReconnect = reconnectController.codexReconnect();
await Promise.resolve();
assert.equal(reconnectController.snapshot().codex_reconnect_in_flight, true);
const duplicateReconnect = await reconnectController.codexReconnect();
assert.equal(duplicateReconnect.status, "running");
assert.equal(reconnectCalls.length, 1);
const reconnectOperationId = reconnectCalls[0].operation_id;
reconnectGate.resolve({
  status: "complete",
  operation_id: reconnectOperationId,
  codex_session_id: "codex-session-runtime",
  context: { context_revision: 2, episode_id: "codex-reconnect" },
});
await firstReconnect;
assert.equal(reconnectController.snapshot().codex_reconnect_in_flight, false);
await reconnectController.codexReconnect();
assert.equal(reconnectCalls.length, 2);
assert.equal(reconnectCalls[1].operation_id, reconnectOperationId);
reconnectController.unmount();

const retryReconnectCalls = [];
const retryReconnectResults = [
  { status: "unavailable", reason: "provider is temporarily unavailable" },
  {
    status: "complete",
    context: { context_revision: 2, episode_id: "codex-reconnect" },
    source: { source_revision: "source-1", source_digest: "digest-1" },
  },
];
const retryReconnectFacade = {
  next: async () => ({ status: "selected" }),
  snapshot: async () => ({ status: "complete" }),
  codex_start: async () => ({ status: "unavailable" }),
  codex_read: async () => ({ status: "unavailable" }),
  codex_cancel: async () => ({ status: "unavailable" }),
  codex_reconnect: async (args) => {
    retryReconnectCalls.push(args);
    return retryReconnectResults.shift();
  },
};
const retryReconnectController = new AuditWorkbenchController(
  reconnectDocument,
  new Element(new Document(), "main"),
  { facade: retryReconnectFacade },
);
const failedReconnect = await retryReconnectController.codexReconnect();
assert.equal(failedReconnect.status, "unavailable");
assert.equal(failedReconnect.codex_session_id, "codex-session-runtime");
const retriedReconnect = await retryReconnectController.codexReconnect();
assert.equal(retriedReconnect.status, "complete");
assert.equal(retryReconnectCalls.length, 2);
assert.equal(retryReconnectCalls[1].codex_session_id, "codex-session-runtime");
assert.equal(retryReconnectCalls[1].operation_id, retryReconnectCalls[0].operation_id);
retryReconnectController.unmount();

const xssActivity = Array.from({ length: 80 }, (_, index) => ({
  message: `<img src=x onerror=alert(${index})>${"x".repeat(700)}`,
  evidence_ids: ["evidence-safe"],
  provider_path: "/private/provider",
}));
const xssCodexFacade = {
  next: async () => ({ status: "selected", packet: caseValue("codex-safe", false), selection_revision: 1 }),
  snapshot: async () => ({ status: "complete" }),
  codex_start: async (args) => ({
    status: "complete",
    operation_id: args.operation_id,
    context: { context_revision: 7, episode_id: "<script>episode</script>" },
    source: { source_revision: "source-7", source_digest: "digest-7", provider_path: "/private/provider" },
    route_id: "route-safe",
    evidence_ids: ["evidence-safe"],
    usage: { input_tokens: 3, output_tokens: 2, provider_path: "/private/provider" },
    activity_scope: "process",
    activity: xssActivity,
  }),
  codex_read: async () => ({ status: "complete" }),
  codex_cancel: async (args) => ({ status: "cancelled", operation_id: args.operation_id }),
};
const xssRoot = new Element(new Document(), "main");
const xssCodexController = new AuditWorkbenchController({}, xssRoot, { facade: xssCodexFacade });
await xssCodexController.next();
const xssResult = await xssCodexController.codexStart("safe prompt");
assert.equal(xssResult.activity.length, 24);
assert.match(renderedText(xssRoot), /<img src=x onerror=alert/);
assert.equal(renderedTags(xssRoot).includes("IMG"), false);
assert.equal(renderedText(xssRoot).includes("/private/provider"), false);
assert.match(renderedText(xssRoot), /no durable transcript is exposed/i);
assert.match(renderedText(xssRoot), /Activity is process-scoped/i);
xssCodexFacade.codex_read = async () => ({
  status: "complete",
  operation_id: "audit-operation-1",
  activity_scope: "lifecycle",
  activity: [{
    message: "Codex start operation was admitted.",
    operation_id: "audit-operation-1",
    timestamp: "2026-09-24T13:00:00Z",
    source_digest: "private-digest-must-not-render",
  }],
});
const lifecycleRead = await xssCodexController.codexRead();
assert.equal(lifecycleRead.activity_scope, "lifecycle");
assert.match(renderedText(xssRoot), /Durable operation summaries only/i);
assert.match(renderedText(xssRoot), /audit-operation-1/);
assert.match(renderedText(xssRoot), /2026-09-24T13:00:00Z/);
assert.equal(renderedText(xssRoot).includes("private-digest-must-not-render"), false);
const cancelled = await xssCodexController.codexCancel("post-turn review");
assert.equal(cancelled.status, "cancelled");
xssCodexController.unmount();

console.log("audit_workbench_runtime: ok");

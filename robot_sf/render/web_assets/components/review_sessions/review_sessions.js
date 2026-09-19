/* Offline SREV-28 review-session controller.
 *
 * This file is deliberately dependency-free.  It renders a snapshot supplied
 * by the local component and sends only authenticated, loopback-bound control
 * envelopes to an injected adapter.  It never calls fetch, opens a socket,
 * starts a simulator, evaluates a recipe, or stores credentials in browser
 * storage.  Construction and rendering are passive; start/resume/stop are
 * explicit method or button actions.
 */

export const SESSION_VIEW_SCHEMA_VERSION = "review-session.v1";
export const CONTROL_SCHEMA_VERSION = "review-session-control.v1";
export const CONTROL_ACTIONS = Object.freeze(["start", "stop", "resume", "progress", "result"]);

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function loopbackOrigin(value) {
  if (typeof value !== "string" || !value) return false;
  try {
    const parsed = new URL(value);
    return (parsed.protocol === "http:" || parsed.protocol === "https:")
      && ["localhost", "127.0.0.1", "[::1]", "::1"].includes(parsed.hostname)
      && parsed.username === "" && parsed.password === "";
  } catch (_error) {
    return false;
  }
}

export function isLoopbackOrigin(value) {
  if (!loopbackOrigin(value)) return false;
  const parsed = new URL(value);
  return (parsed.pathname === "" || (parsed.pathname === "/" && !value.endsWith("/")))
    && parsed.search === "" && parsed.hash === "";
}

function controlEnvelope(action, origin, sessionToken, sessionId, contextRevision, payload = {}) {
  if (!CONTROL_ACTIONS.includes(action)) throw new Error("unsupported control action");
  if (!isLoopbackOrigin(origin)) throw new Error("control origin must be loopback");
  if (typeof sessionToken !== "string" || !sessionToken || sessionToken.length > 256) throw new Error("session token is required");
  if (typeof sessionId !== "string" || !sessionId) throw new Error("session context is required");
  return {
    schema_version: CONTROL_SCHEMA_VERSION,
    action,
    origin,
    session_token: sessionToken,
    session_id: sessionId,
    context_revision: contextRevision || "",
    payload: asObject(payload),
  };
}

function text(value, fallback = "") {
  return value === undefined || value === null ? fallback : String(value);
}

function renderSnapshot(root, state) {
  if (!root) return;
  const budget = asObject(state?.budget);
  const authorization = asObject(state?.authorization);
  const provenance = asObject(state?.provenance);
  const context = asObject(state?.context);
  const status = text(state?.status, "not_started");
  const reason = text(state?.stop_reason || state?.reason);
  root.replaceChildren();
  const heading = root.ownerDocument.createElement("h1");
  heading.textContent = "Review session";
  root.appendChild(heading);
  const summary = root.ownerDocument.createElement("p");
  summary.dataset.reviewSessionStatus = status;
  summary.textContent = reason ? `${status}: ${reason}` : status;
  root.appendChild(summary);
  const budgetLine = root.ownerDocument.createElement("p");
  budgetLine.textContent = `Budget: ${text(budget.executions_consumed, 0)} consumed, `
    + `${text(budget.remaining_executions, 0)} remaining; `
    + `${text(budget.elapsed_s, 0)}s elapsed / ${text(budget.wall_timeout_s, 0)}s`;
  root.appendChild(budgetLine);
  const source = root.ownerDocument.createElement("p");
  source.textContent = `Source admission: ${text(asObject(state?.source_admission).status, "not_checked")}; `
    + `preservation: ${text(asObject(state?.preservation).status, "required")}`;
  root.appendChild(source);
  const boundary = root.ownerDocument.createElement("p");
  boundary.textContent = `Evidence boundary: ${text(state?.evidence_boundary, "diagnostic_only")}; `
    + `claims allowed: ${text(state?.scientific_claim_allowed, false)}`;
  root.appendChild(boundary);
  const buttons = ["start", "stop", "resume", "previous", "next"];
  for (const action of buttons) {
    const button = root.ownerDocument.createElement("button");
    button.type = "button";
    button.dataset.reviewSessionAction = action;
    button.textContent = action;
    if (authorization.read_only === true && ["start", "stop", "resume"].includes(action)) {
      button.disabled = true;
    }
    root.appendChild(button);
  }
  const candidates = Array.isArray(state?.candidates) ? state.candidates : [];
  const list = root.ownerDocument.createElement("ol");
  for (const candidate of candidates) {
    const item = root.ownerDocument.createElement("li");
    item.textContent = `${text(candidate.intervention_id, "candidate")} (${text(candidate.state, "pending")})`;
    list.appendChild(item);
  }
  root.appendChild(list);
  if (provenance.component_id) root.dataset.componentId = text(provenance.component_id);
  if (context.session_id) root.dataset.sessionId = text(context.session_id);
}

export class ReviewSessionsController {
  constructor({
    root = null,
    document = root?.ownerDocument || globalThis.document,
    view = {},
    scheduler = globalThis,
    origin = globalThis.location?.origin || "",
    sessionToken = "",
    sessionId = text(view?.context?.session_id || view?.session_id),
    contextRevision = text(view?.context?.context_revision),
    controlRequest = null,
    readOnly = true,
  } = {}) {
    this.root = root || document?.getElementById?.("review-session-root") || null;
    this.document = document;
    this.scheduler = scheduler;
    this.origin = origin;
    this.sessionToken = sessionToken;
    this.sessionId = sessionId;
    this.contextRevision = contextRevision;
    this.controlRequest = controlRequest;
    this.readOnly = Boolean(readOnly);
    this.state = { ...asObject(view), authorization: { ...asObject(view.authorization), read_only: this.readOnly } };
    this._listeners = [];
    this._mounted = false;
  }

  mount() {
    if (this._mounted) return this;
    this._mounted = true;
    this.render();
    if (this.root?.addEventListener) {
      const listener = (event) => {
        const action = event.target?.dataset?.reviewSessionAction;
        if (!action) return;
        if (action === "previous") this.previous();
        else if (action === "next") this.next();
        else this.dispatch(action);
      };
      this.root.addEventListener("click", listener);
      this._listeners.push(() => this.root.removeEventListener("click", listener));
    }
    return this;
  }

  destroy() {
    for (const remove of this._listeners.splice(0)) remove();
    this._mounted = false;
    return this;
  }

  render() {
    renderSnapshot(this.root, this.state);
    return this.snapshot();
  }

  snapshot() {
    return JSON.parse(JSON.stringify(this.state));
  }

  setView(view) {
    this.state = { ...asObject(view), authorization: { ...asObject(view?.authorization), read_only: this.readOnly } };
    return this.render();
  }

  _control(action, payload = {}) {
    if (this.readOnly && ["start", "stop", "resume"].includes(action)) {
      return Promise.reject(new Error("read_only_never_executes"));
    }
    if (typeof this.controlRequest !== "function") {
      return Promise.reject(new Error("authenticated local control adapter is required"));
    }
    const envelope = controlEnvelope(
      action,
      this.origin,
      this.sessionToken,
      this.sessionId,
      this.contextRevision,
      payload,
    );
    return Promise.resolve(this.controlRequest(envelope)).then((result) => {
      if (result && typeof result === "object") this.setView(result);
      return result;
    });
  }

  dispatch(action, payload = {}) {
    return this._control(action, payload);
  }

  start() { return this._control("start"); }
  stop() { return this._control("stop"); }
  resume() { return this._control("resume"); }
  refresh() { return this._control("progress"); }

  _navigationIndex() {
    const navigation = asObject(this.state.navigation);
    return Number.isInteger(navigation.index) ? navigation.index : 0;
  }

  previous() { return this._control("result", { index: Math.max(0, this._navigationIndex() - 1) }); }
  next() { return this._control("result", { index: this._navigationIndex() + 1 }); }
}

export function createReviewSessions(options = {}) {
  return new ReviewSessionsController(options).mount();
}

export function bootstrapReviewSessions({ document = globalThis.document, view = null, ...options } = {}) {
  const element = options.root || document?.getElementById?.("review-session-root");
  let model = view;
  const data = document?.getElementById?.("review-session-data");
  if (model === null || model === undefined) {
    try { model = data ? JSON.parse(data.textContent || "{}") : {}; } catch (_error) { model = { status: "failed", reason: "invalid embedded session view" }; }
  }
  return createReviewSessions({ ...options, document, root: element, view: model || {} });
}

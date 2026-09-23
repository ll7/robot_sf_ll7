# BA05 #9488: bounded local Codex App Server adapter

This change adds an experimental, local-only adapter for the installed Codex
App Server, a real MCP stdio transport for the existing audit dispatcher, and
a bounded durable session/operation seam owned by `AuditService.authority`.
It is an integration seam and capability probe; it is not production support
or evidence that the complete BA05 workflow is closed.

## Versioned boundary

The adapter is pinned to the locally verified `codex-cli 0.154.0` App Server
v2 contract. The protocol schema was generated with:

```text
codex app-server generate-json-schema --out <DIR> --experimental
```

The generated v2 schema digest recorded by the adapter is
`7b9e7d385fffef8d428cc5490b56ce9c393bd3ed7bc7ccd730956387e723ec05`.
The adapter performs the `initialize`/`initialized` handshake, discovers the
actual model and provider with `model/list` and `thread/start`, and refuses to
invent a route when the installed capability is absent, mismatched, or
ambiguous. App Server remains an experimental local capability.

The authority journal's optional `state.extensions.codex` namespace stores
versioned, token-free session and operation snapshots. A fresh process must
present the original audit token, revalidate the complete `SourceRef`, policy,
actor, context, evidence, and current source bytes, and rediscover an identical
route before reconnecting or resuming. A terminal operation replay returns its
stored receipt without provider work. An inflight operation is reported as
ambiguous and keeps its reservation; generic provider errors and missing meter
data are not silently retried or released. The reservation is conservative
accounting, not a physical hard token cap: remote provider work can overshoot
before telemetry is available.

## MCP authority boundary

`AuditMCPStdioServer` translates JSON-RPC `initialize`, `ping`, `tools/list`,
and `tools/call` messages into the existing `AuditMCPDispatcher`. The server
owns the session id, token, and loopback origin; tool arguments cannot replace
them. The dispatcher and `AuditService` remain the authority for selection,
source, authorization, budgets, revisions, and writes.

For an App Server child, `AuditMCPBridge` owns a task-private Unix socket. The
Codex MCP child receives only the socket path and session metadata through its
environment; the bearer token is never placed in the command line or route
receipts. Each process receives MCP configuration through per-process `-c`
overrides, so no global Codex configuration is edited.

## Bounded behavior

Both transports enforce newline-delimited frame limits, message/event limits,
startup/request/idle/lifetime deadlines, bounded stderr diagnostics, ordered
event sequence numbers, and deterministic shutdown. Lost, malformed,
oversized, timed-out, or mismatched processes fail closed. `turn/interrupt`,
`thread/resume`, and transport restart are exposed through the existing Codex
provider lifecycle. Provider-created threads are durable (`ephemeral: false`)
because the installed App Server cannot resume an ephemeral rollout; a durable
rollout must have completed a turn before `thread/resume` is available. The
adapter fails closed on a resume error rather than claiming reconnection.
Token usage is accepted only from the App Server's actual
`thread/tokenUsage/updated` event. The installed server does not expose
measured compute, so every completed model turn carries the finite
`compute=1.0` accounting charge `bounded-per-turn-unmeasured-charge` and
`compute_measured=false`. This charge advances the service aggregate budget
and prevents zero-cost live work; it is not a provider-compute measurement.
The provider therefore requires an explicit `reserved_compute >= 1.0`
pre-admission argument before `start` or `resume` can initialize the transport.
The authority caller must obtain that reservation from `AuditService` and
forward it; missing or insufficient admission fails closed without a child
transport call. The durable `AuditCodexClient` forwards that argument only
after its service-owned reservation succeeds. A lost reservation reply is recovered by
authenticated operation identity and settled to zero before provider work.
Ambiguous provider work, missing or over-budget telemetry, and a successful
zero-compute report consume the reserved ceiling without promoting success.
Token settlement uses the App Server's `last` turn meter, not cumulative
thread `total`. The installed App Server exposes no physical per-turn token
cap: this is pre-admission accounting with bounded runtime and post-turn
enforcement, not a strict single-turn spend guarantee.

## Evidence boundary

Focused tests cover direct MCP dispatch, external MCP proxying, session-token
denial, JSONL bounds, route discovery, event ordering, durable resume,
idle-lifetime termination, bridge-client shutdown, reconnect/cancel, pre-
admission denial, and fake subprocess failures. A local credentials smoke may
show that the current machine can complete one read-only turn and resume a
durable thread after explicit pre-admission, but it is not benchmark evidence:
credentials, model availability, network state, and usage can change. The
credentialed smoke's token usage is provider telemetry; its one-unit compute
charge is deliberately conservative and unmeasured, not benchmark evidence.

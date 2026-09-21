# Local Benchmark Auditor launch (integration slice)

The live workbench serves one admitted campaign through the existing local
review-workbench shell. It scans the campaign, creates a digest-bound queue,
opens one server-held audit session, and prints a loopback URL. No audit token
is placed in the URL or browser model.

```bash
uv run python -m robot_sf.render.audit_workbench_launch \
  --campaign /absolute/path/to/campaign.json \
  --source-root /absolute/path/to \
  --store-root /absolute/path/to/private-audit-store
```

The campaign must be a regular file inside `--source-root`. The command serves
only on `127.0.0.1` using an ephemeral port and stops on Ctrl-C. Keep the
store in a private location; it contains durable audit state, including the
queue and findings. The browser has a short-lived, HTTP-only transport cookie,
but the audit-service credential remains on the server.

The ordinary command above leaves embedded Codex disabled. To opt into the
experimental local Codex App Server route, provide two trusted JSON files:

```json
{
  "app_server": {
    "executable": "/absolute/path/to/codex",
    "cwd": "/absolute/path/to",
    "expected_cli_version": "0.154.0",
    "request_timeout_seconds": 45.0,
    "startup_timeout_seconds": 15.0,
    "lifetime_seconds": 600.0
  },
  "route_id": "provider:model"
}
```

```json
{
  "schema_version": "audit-session-policy.v1",
  "allowed_roots": ["/absolute/path/to"],
  "token_budget": 30000,
  "compute_budget": 1.0
}
```

Launch with:

```bash
uv run python -m robot_sf.render.audit_workbench_launch \
  --campaign /absolute/path/to/campaign.json \
  --source-root /absolute/path/to \
  --store-root /absolute/path/to/private-audit-store \
  --codex-app-server /absolute/path/to/codex-app-server.json \
  --session-policy /absolute/path/to/session-policy.json
```

`--codex-route-id <provider:model>` may be used as a trusted route pin when it
is not stored in the App Server file. The CLI rejects missing or malformed
files, unsupported versions, absent installed capability, ambiguous routes,
and policies without positive token and compute budgets before serving a
browser. The launcher creates the private MCP bridge and keeps the service
session authority server-side; browser requests cannot provide a route,
credential, process path, or policy. The App Server adapter reports token
telemetry but does not expose a measured compute field, so the one-unit
compute charge is a conservative accounting guardrail, not a physical
per-turn token ceiling or a claim about provider work.

To opt into one campaign-bound native diagnostic, provide a second trusted
configuration and the same session policy. The native file must contain
exactly one binding with `admission`, `request`, `recipe`, and `campaign_row`
objects; the row binds the campaign URI/digest, native source URI/digest, and
the historical episode/scenario/seed/planner/source-commit/config identity.
The launcher resolves the selected generated BA-01 `EpisodeRef` from that
campaign row and fails closed when the row is stale or ambiguous.
The JSON below is a shape sketch: replace placeholder values and omit the
displayed `...` placeholders in a real file.

```json
{
  "bindings": [
    {
      "admission": {"source_root": "/absolute/path/to/native-bundle", "...": "..."},
      "request": {"request_id": "...", "sources": [{"uri": "...", "sha256": "..."}]},
      "recipe": {"recipe_id": "...", "source_identity": {"...": "..."}},
      "campaign_row": {
        "campaign_uri": "campaign.json",
        "campaign_sha256": "...",
        "native_uri": "original.json",
        "native_sha256": "...",
        "episode_id": "...",
        "scenario_id": "...",
        "seed": 0,
        "planner_id": "...",
        "source_commit": "40-hex-characters",
        "config_identity": "..."
      }
    }
  ],
  "max_timeout_s": 30.0,
  "compute_cost": 2.0
}
```

The native launch requires `--session-policy`; that policy must allow the
native source root and recipe (add the recipe ID to an `allowed_recipes`
array) and have `compute_budget >= compute_cost`.
Malformed, stale, missing, or insufficient authority is rejected before the
HTTP server or browser is exposed:

```bash
uv run python -m robot_sf.render.audit_workbench_launch \
  --campaign /absolute/path/to/campaign.json \
  --source-root /absolute/path/to \
  --store-root /absolute/path/to/private-audit-store \
  --native-diagnostic-config /absolute/path/to/native-diagnostic.json \
  --session-policy /absolute/path/to/session-policy.json
```

The control accepts only the selected packet's bounded goal intervention and
compare-and-swap revisions. Its real-runner pair remains
`diagnostic_only` with `scientific_claim_allowed: false`; this launcher path
does not claim physical process pinning, measured benchmark compute, or
benchmark success.

This is an integration slice, not a completed BA-06 workflow or benchmark
result. A retained native episode trace has been exercised through the local
HTTP route: the editor received sampled scene times, saved a quick annotation,
and proposed a finding without confirming human review coverage. A real
service-backed related-case read also returns an explained peer candidate,
explicitly unconfirmed as finding membership or benchmark evidence. The live
coverage pane reads a source-bound BA-04 health report and shows its remaining
deficits; quick annotations and stale review receipts do not earn full-human
credit. An explicit browser `Record full review` action now writes a typed
full-episode human receipt only after the selected packet and service/queue
revisions pass compare-and-swap guards. A local HTTP test proves that receipt
can raise the matching BA-04 human count by one while the stored queue ID is
left unchanged. The test does not prove that a person actually inspected the
whole episode, and it does not establish protocol completion or a benchmark
result. The visible review buttons appear only with an admitted editor model;
the summary-only fixture has no playable scene and therefore no such button.
Fresh snapshots read the durable queue and coverage again after a review;
reopening the queue reconstructs only review receipts named by its own state.

The admitted browser editor also exposes the SREV-17 full-annotation draft:
observed behaviour, hypothesis, confidence, measured evidence, notes, actors,
and explicit full-episode scope. Its source-time snap controls reuse recorded
actor, goal, waypoint, map-object, event, and metric samples when those
identities and geometry are present, and render references as numbered rows.
World references are emitted only from admitted scene geometry; image
references retain their media source, source time, source point, and declared
crop/resize transform, and are never treated as metre coordinates. These
fields use the existing annotation transaction, so pending/durable/error
autosave status and durable reopen are the same CAS path as quick notes. This
browser proof remains diagnostic UI evidence; it does not establish native
media availability or full BA-06 acceptance.

The native trace had no retained video or synchronized metrics, and the tracked
summary-only fixture cannot supply them; the workbench does not recreate
missing media from summary data. Diagnostics, GitHub sync, embedded Codex
activity, browser-side visual inspection, and complete recovery proof remain
unvalidated for the full epic.

The service snapshot also exposes a read-only selected-artifact status line.
By default, materialization is explicitly `not_configured` because no
launcher-owned output capability is provided. A local launcher may opt in with
`--materialization-output-root <trusted-path>` and
`--materialization-compute-budget <finite-units>`; the exact-origin HTTP route
then accepts only an operation ID and selection/context revisions, passes the
trusted output root privately to the service, and returns diagnostic status,
classification, and fidelity without output paths or raw artifacts. This is a
server-held action seam, not a complete BA-06 workflow.
Native diagnostic execution remains `not_configured` unless a trusted native
binding is installed. Source drift is reported as unavailable after
service-side revalidation, while a
stale selection remains an explicit conflict.
The status is diagnostic-only: it does not accept a browser-supplied
source/output path, expose secrets or receipts, or establish a positive BA-06
capability claim. The optional materialization action result is a separate
bounded projection; the snapshot status line does not yet display its result.
When a trusted native binding is available, the browser may show a bounded
`Run native diagnostic` control. It is disabled unless the selected packet,
server-reported capability, selection revision, and context revision are all
current. The request contains only an opaque operation/intervention ID, the
two-coordinate goal intervention, activation epsilon, deadline, and the two
CAS revisions; source paths, runner identities, planner configuration, and
credentials remain server-owned. The returned control-fidelity/activation
projection is diagnostic-only and never benchmark evidence. A campaign-source
session whose native source binding is absent or mismatched remains unavailable
or conflict; the control does not create a fallback or positive real-runner
proof. Editing an intervention field updates the bounded guidance and clears
the prior result; an edit during an in-flight request discards that response
as a conflict.
Each returned snapshot status is checked against the selected episode, context
revision, and source identity; delayed or foreign responses reopen as unavailable/conflict rather
than changing the remembered selection. Path-like or otherwise unsafe reasons
and references also clear positive capability, classification, and fidelity
claims. The status-specific envelope omits generic operation and receipt data.

## Codex accounting and route gate (BA-05 amendment)

The App Server route is not accepted merely because a local executable is
installed. For hosted or live evidence, configure an exact
`required_route_id`, `required_provider`, and `required_model_id` with
`require_explicit_route: true`; model discovery does not fall back to a
default or premium model when the requested model is absent.

Choose one accounting mode explicitly:

- `offline` refuses provider work and remains provider-free/read-only.
- `local_accounting` reserves finite local token, compute, and issue-write
  budgets, records observed usage and overspend, and treats retries,
  cancellation, reconnect, missing usage, and nested sessions as durable
  ledger outcomes. It is **not** a physical provider cap.
- `strict_provider_ceiling` is admitted only with the exact route above and a
  verified finite `provider_compute_ceiling`. It refuses admission when the
  ceiling is absent or a reservation exceeds it. The current App Server
  contract does not expose a verified token ceiling, so token accounting stays
  local unless a future provider capability proves otherwise.

Receipts expose `accounting_mode`, `provider_ceiling_verified`,
`physical_provider_cap_enforced`, reservation, and `overspent`. A green local
or hosted test therefore proves the selected accounting contract only; it
must not be reported as evidence of an external provider quota or a completed
BA-06 benchmark workflow.

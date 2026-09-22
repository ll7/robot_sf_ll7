# Benchmark Auditor operator guide (bounded V1)

This guide covers the accepted BA-05 service slices and the diagnostic BA-06
route. It is an offline/read-only and injected-provider guide, not a claim
that this repository ships a production browser, live GitHub client, or live
Codex provider setup. The full acceptance boundary is in
[`benchmark_auditor_v1_acceptance.md`](./benchmark_auditor_v1_acceptance.md).

## First screen: disposable offline demo

The smallest tested launch generates a disposable fixture and serves it only
on loopback. It does not contact GitHub, Codex, or an external MCP server:

```bash
fixture_root="$(mktemp -d /dev/shm/benchmark-auditor-fixture.XXXXXX)"
.venv/bin/python -c 'from pathlib import Path; from robot_sf.render.audit_workbench import write_fixture_workbench; write_fixture_workbench(Path("'"$fixture_root"'/fixture"))'
python -m http.server 8765 --bind 127.0.0.1 --directory "$fixture_root/fixture"
```

Open `http://127.0.0.1:8765/audit-workbench.v1.html`. The first screen is
headed **Benchmark audit workbench** and shows `Queue Next`, queue/selected
case state, declared media PTS, coverage, and the diagnostic-only boundary.
The fixture's missing-recording case stays reviewable and reports
`recording_not_present`; it is not promoted to native or benchmark evidence.
The command was checked by fetching the page over loopback and asserting the
title on 2026-09-22.

## One-command local service launch

For a source-bound campaign, use the canonical server-held launcher. This is
the exact command shape exercised by the launch tests; all three paths must be
absolute and the campaign must be inside `--source-root`:

```bash
.venv/bin/python -m robot_sf.render.audit_workbench_launch \
  --campaign /absolute/path/to/campaign.json \
  --source-root /absolute/path/to \
  --store-root /absolute/path/to/private-audit-store \
  --materialization-output-root /absolute/path/to/private-materialization-output \
  --materialization-compute-budget 1.0
```

The process prints an ephemeral `http://127.0.0.1:<port>/` URL. The browser
receives only an HTTP-only transport cookie. After `Queue Next`, a configured
service facade exposes **Materialize selected artifact** in the queue pane;
the request carries the selected episode, selection revision, context
revision, and one operation ID. A stale selection or context is rejected by
the server, and only the safe diagnostic status (never source/output paths)
returns to the browser.

For a real campaign, replace the three absolute roots with the admitted source
and private store/output roots. Do not infer source identity from a filename:
the scanner must admit the campaign and exact source/config/checkpoint first.
If that admission or the retained media is unavailable, the launcher reports
`unavailable`/`blocked_external` and does not synthesize a successful audit.

## Preconditions and evidence modes

Use a clean worktree based on current `origin/main`, a repository-local
`.venv`, and an output directory outside any public GitHub evidence path. The
commands below were exercised with the repository `.venv` on 2026-09-22.
The shared-venv wrapper is preferred on a normal host; the acceptance run used
direct `.venv/bin/python` because the host had less than the wrapper's 1 GiB
temporary-space floor.

The supported accounting modes are:

| Mode | Meaning | Current status |
| --- | --- | --- |
| `offline` | Provider-free, read-only operation; no App Server turn is admitted. | Tested and available. |
| `local_accounting` | Finite local token/compute/issue-write reservations; observed usage and provider overspend are recorded. | Tested; not a physical provider cap. |
| `strict_provider_ceiling` | Requires an exact route/provider/model and a verified finite provider compute ceiling before admission. | Tested fail-closed; no verified live provider cap is available. |

Do not describe `local_accounting` overspend as provider enforcement. Do not
switch to a premium or fallback provider to make strict mode pass.

## Verify the bounded service contracts

The exact focused BA-05 publication/MCP command is:

```bash
.venv/bin/python -m pytest \
  tests/analysis_workbench/test_audit_github_rest.py \
  tests/analysis_workbench/test_audit_github.py \
  tests/analysis_workbench/test_audit_github_service.py -q
```

Receipt on the accepted branch: `126 passed in 49.85s`.

The accounting boundary command is:

```bash
.venv/bin/python -m pytest \
  tests/analysis_workbench/test_audit_codex_app_server.py::test_d2_strict_provider_ceiling_requires_verified_capability \
  tests/analysis_workbench/test_audit_codex_app_server.py::test_d2_offline_mode_refuses_provider_work_before_transport \
  tests/analysis_workbench/test_audit_codex_app_server.py::test_d2_strict_ceiling_rejects_a_reservation_above_verified_cap \
  tests/analysis_workbench/test_audit_codex_app_server.py::test_d2_local_accounting_records_provider_overspend_without_retry -q
```

Receipt: `4 passed in 2.97s`. The strict tests prove refusal without a
verified cap and refusal above a verified cap; the local-accounting test proves
overspend is recorded without an automatic retry.

The integrated store/render/server command used for the BA-06 runtime repair
was:

```bash
.venv/bin/python -m pytest \
  tests/analysis_workbench/test_audit_store.py \
  tests/render/test_audit_workbench_live_coverage.py \
  tests/render/test_audit_workbench_server.py \
  tests/render/test_audit_workbench.py -q
```

Receipt: `181 passed`; the live coverage call was 17.55 seconds after the
repair (64.88 seconds in the baseline profile).

## Publication setup and safe retry

The REST transport is injectable: construct `GitHubRESTProvider` with an
explicit HTTP transport, an allowlisted `owner/name` repository, and the
configured API URL. The transport validates complete bounded pagination and
supports issue search/get/create plus `append_auditor_comment`. The service
owns credentials, session tokens, source context, finding revision, and the
durable outbox; MCP payloads cannot provide them.

The V1 publication sequence is:

1. Search the complete issue collection for the exact finding marker.
2. Create one immutable issue snapshot when no marker exists.
3. Append a comment carrying one exact request marker for each later finding
   revision.
4. Commit the local outbox receipt and finding link only after the remote
   result is observed and the local compare-and-swap succeeds.

After a timeout, reopen the same `AuditStore` and reuse the same outbox
operation identity. Reconcile by a complete exact-marker search before any
retry. An incomplete page, duplicate marker, changed auditor block, unknown
remote outcome, or local link-CAS conflict is `ambiguous`/`conflict` and stops
automatic retry. A read-then-`PATCH` body update is unsupported: the ordinary
GitHub issue API is not a proven CAS. The historical body-CAS path exists only
for explicit compatibility fakes.

No command in this guide performs a live GitHub write. The focused tests use a
fake provider and do not require credentials.

## Session lifecycle, resume, export and restore

Use the existing `AuditService`/`AuditMCPDispatcher` APIs for session start,
`resume`/`reconnect`, and `cancel`/`kill_switch`. Every transition writes a
durable receipt with source/context revisions, route, budget charge and
external identity. A cancelled or stale session cannot silently resume work;
recover it only through the explicit service transition and a fresh context
check.

For a portable handoff, export the canonical JSON/NDJSON records and the
compact receipt/manifest, then restore into a new `AuditStore` and rebuild its
SQLite projection. Do not treat `output/`, raw videos, traces, credentials,
or local URLs as durable public evidence. The repository has no supported
single shell launcher for this full workflow; use the service APIs and the
focused tests above rather than inventing a CLI command.

## Unavailable and blocked cases

Report these states explicitly rather than substituting a degraded success:

- no exact source/config/checkpoint admission or missing media: `unavailable`;
- native planner/source cannot satisfy the admitted adapter: `blocked_external`;
- no verified physical provider ceiling: strict mode refuses; local accounting
  remains diagnostic only;
- incomplete GitHub pagination or timeout without reconciliation:
  `ambiguous`;
- duplicate markers, human edits, or local finding-link races: `conflict`;
- real browser, real-data, live Codex/App Server, and native diagnostic
  receipts not yet exercised: `diagnostic_only` / `implemented_but_unproven`.

These statuses are evidence, not reasons to close BA-06 or epic #9483.

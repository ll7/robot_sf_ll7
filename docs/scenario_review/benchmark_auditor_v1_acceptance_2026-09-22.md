# Benchmark Auditor V1 acceptance report (2026-09-22)

Status: **BA-05 bounded service acceptance is complete; BA-06 and epic #9483
are not complete.** The report is an evidence receipt, not a benchmark result.
Fallback, degraded, unavailable, synthetic, and diagnostic-only paths are not
promoted to scientific or release evidence.

## Post-BA-06 materialization update (2026-09-22)

The bounded BA-06 materialization continuation is now merged on top of the
accepted BA-05 package. This additive update supersedes the historical
`current origin/main` line below for the repository's current state; the
dated slice table and matrix retain their original receipts.

- Current `origin/main`: `f231fc2faa729a5c96f732d4d690365882878feb`.
- PR #9571 exact reviewed head: `1aa3bb47bc60771d0511a36a201e1084c2c9c01e`.
- PR #9571 guarded squash merge: `f231fc2faa729a5c96f732d4d690365882878feb`.
- Final PR metadata digest: `492a32dfe252d660757c700fd9e8e9b7dab95936204ef9fa5ac1fadef0b407f3`.
- Hosted CI run `35688989382` passed; changed-coverage job `106625885543`
  passed; merge-queue gate `35690704033` passed.
- Focused proof: runtime contract passed; render/launch/server tests `174
  passed in 83.92 s`; full docs/link scan passed for `2123` Markdown files.

The slice exposes one revision-bound, server-held materialization action and
keeps the result diagnostic-only. It does not accept live GitHub CAS,
exactly-once delivery, a physical provider ceiling, retained media, native
source diagnostics, live Codex, real-data browser execution, or recovery.
BA-06 issue #9489 and epic #9483 remain open; this report does not declare
either complete.

## Current revision and acceptance boundary

- Repository: `ll7/robot_sf_ll7`.
- Acceptance base before the final runtime slice:
  `9d37fd1f619fcda682e42773923ea8ddf93fb8d1`.
- PR #9569 exact reviewed head:
  `4833a1eb06b60f04aca78003789bdf992361c8be`.
- PR #9569 guarded squash merge:
  `7521db3b5f1b7ab562971ffe1000383416e4eadc`.
- Current fetched `origin/main` at report preparation:
  `7521db3b5f1b7ab562971ffe1000383416e4eadc`.
- Exact-head metadata digest for #9569:
  `fe00b9dad7d6b15329d5720e69c514b69cc19c280da67b8311ec73e3f637e7dd`.
- Guarded merge receipt digest:
  `f66abadfaf8f8f4bb212a0268f358e467962d9158ae06e1b4dd307f4cef74535`.

The final bounded BA-06 runtime repair is PR #9569. It changes only
`AuditStore.list_records` materialization and its regression test. The live
coverage HTTP call retained durable-review and stale-review assertions and
fell from 64.88 seconds to 17.55 seconds. This is runtime-friction evidence,
not a benchmark claim.

## Exact-head, focused, and hosted receipts

| Slice | Exact reviewed head | Merge | Focused receipt | Hosted receipt | Classification |
| --- | --- | --- | --- | --- | --- |
| BA-05 D1 append-only publication (#9559) | `ec8061c4517fb3926fefd1b1a2cb6cb3cdcf0088` | `6787f5b93a1023dd243f66d81b1e6a13fe6871a3` | 102 passed | run `35655668527`, accepted retry | implemented_and_verified (bounded) |
| BA-05 D2 route/accounting (#9558) | `78fb3c818fafe1ae0ea1522c96092e55b8460e3b` | `6982a2456aa98cabff6f5186d396d4f15eb72b48` | 113 passed, 2 skipped | run `35658827492` | implemented_and_verified (bounded) |
| BA-06 server-held publication (#9561) | `e2c84dd8ea75bd63a62005ecbc55ae7847d06328` | `208e8a3bf02f5b2c2b3eafe39fe776da07a9dcc7` | 149 passed | run `35667772524` | diagnostic_only |
| BA-06 external MCP context (#9565) | `2d486ed94bfc1d785e1fe576ca8232ab3bc7d880` | `ac2818befaf5e5ce1c783a02d3eed2759f27eb40` | 2 focused tests passed | run `35676387775`, attempt 2 | diagnostic_only |
| BA-06 runtime repair (#9569) | `4833a1eb06b60f04aca78003789bdf992361c8be` | `7521db3b5f1b7ab562971ffe1000383416e4eadc` | store/render/server: 181 passed; live call 17.55 s | PR run `35683653944`; changed coverage passed; merge gate `35685480912` | implemented_and_verified (bounded) |

For #9569, the hosted run had 31 successful required checks, an intentionally
skipped full coverage job, and a successful changed-coverage gate
(`106609257737`). The independent exact-head review carried
`gate-verdict: accepted`, `base-policy: ordinary-cas`, and the reconciled
metadata digest. A post-merge workflow was still running when this report was
prepared; it is not silently promoted as merge-SHA evidence.

## Reproducible command and exit receipts

All commands below exited 0 unless explicitly marked otherwise.

| Command | Result | Evidence class |
| --- | --- | --- |
| `.venv/bin/python -m pytest tests/analysis_workbench/test_audit_github_rest.py tests/analysis_workbench/test_audit_github.py tests/analysis_workbench/test_audit_github_service.py -q` | 126 passed in 49.85 s | BA-05 fake/injected provider contract |
| `.venv/bin/python -m pytest tests/analysis_workbench/test_audit_codex_app_server.py::test_d2_strict_provider_ceiling_requires_verified_capability tests/analysis_workbench/test_audit_codex_app_server.py::test_d2_offline_mode_refuses_provider_work_before_transport tests/analysis_workbench/test_audit_codex_app_server.py::test_d2_strict_ceiling_rejects_a_reservation_above_verified_cap tests/analysis_workbench/test_audit_codex_app_server.py::test_d2_local_accounting_records_provider_overspend_without_retry -q` | 4 passed in 2.97 s | accounting gate |
| `.venv/bin/python -m pytest tests/analysis_workbench/test_audit_store.py tests/render/test_audit_workbench_live_coverage.py tests/render/test_audit_workbench_server.py tests/render/test_audit_workbench.py -q` | 181 passed; live call 17.55 s | BA-06 runtime repair |
| `.venv/bin/python -m pytest tests/analysis_workbench/test_audit_service_review.py -q` | 6 passed | context-CAS queue repair (#9568) |
| `uv run python scripts/dev/pr_ready_check.sh` with final base and two workers | exit 120 after 967 passed | host storage exhausted (`sqlite3.OperationalError: database or disk is full`); friction #9532, not changed-code evidence |

The first three receipts are the acceptance proof. The storage-limited local
readiness attempt is retained to make the environment limitation explicit and
is not treated as a passing full-suite result.

## Six BA package dispositions

| Package | Issue | Current state | Disposition |
| --- | --- | --- | --- |
| BA-01 campaign accounting/detectors | #9485 | closed | implemented_but_unproven: focused contracts exist; no second current-schema campaign receipt |
| BA-02 queue/ranking/control | #9486 | closed | implemented_but_unproven: queue and control tests exist; full coverage protocol remains incomplete |
| BA-03 records/annotations/findings | #9484 | closed | implemented_and_verified for canonical record/annotation/finding contracts; full retained-media workflow remains unproven |
| BA-04 protocol/coverage/health | #9487 | closed | implemented_but_unproven: hand-computed and service coverage tests exist; no complete real campaign health receipt |
| BA-05 service/Codex/MCP/GitHub | #9488 | open (`state:running`) | accepted bounded D1/D2 contracts; live provider, physical cap, and full recovery remain gated |
| BA-06 workbench integration/end-to-end | #9489 | open (`state:blocked-dependency`, reopened after accidental merge-time close) | diagnostic_only / missing: runtime and publication slices exist; real-data browser workflow and end-to-end recovery are not accepted |

## SREV and prerequisite dispositions

SREV #9285 (synchronized panels), #9287 (annotation speeds/references), and
#9288 (optional planner/pedestrian diagnostics) have contract/test integration
and remain `implemented_but_unproven` for real retained media and source
coverage. #9296 (bounded loop) and #9299 (session UI) remain
`implemented_but_unproven` for admitted-source/live execution. Prerequisite
#9417 (source-admission enforcement) remains an evidence dependency:
`implemented_but_unproven`, not a waiver for a string-only or self-authorized
receipt. The issues are closed in GitHub history where noted, but closure is
not reinterpreted as full epic acceptance.

## Thirty-nine decision matrix

The matrix preserves the plan's accepted numbering. `implemented_and_verified`
means the narrow contract has direct focused evidence; it does not imply the
complete BA-06 workflow. `implemented_but_unproven` means code/tests exist but
the required live or campaign evidence is absent.

| # | Accepted requirement | Status | Evidence or gate |
| ---: | --- | --- | --- |
| 1 | Correctness, behaviour understanding, and interesting examples | implemented_but_unproven | BA packages provide separate paths; no complete end-to-end campaign receipt |
| 2 | Dynamic episode, comparison, cell, or group scope | implemented_but_unproven | queue/packet contracts; full UI route not accepted |
| 3 | Cheap whole-campaign checks and selective expensive analysis | implemented_but_unproven | scan/detector contracts; no second campaign proof |
| 4 | All anomaly dimensions subject to source availability | implemented_but_unproven | twelve-family registry exists; unavailable telemetry still needs campaign evidence |
| 5 | Transparent priority ordering | implemented_but_unproven | ranking tests; no full coverage-aware acceptance |
| 6 | Flag/explain first; grading later | implemented_and_verified | typed suspicion/observation/evidence/confidence fields and service tests |
| 7 | Small annotation taxonomy with optional detail | implemented_and_verified | annotation contract tests |
| 8 | Separate observations, hypotheses, evidence, confidence | implemented_and_verified | finding/annotation schema tests |
| 9 | Rich and fast annotations | implemented_but_unproven | record contracts exist; full synchronized editor workflow absent |
| 10 | Scene/video/summary/metrics/annotations together | diagnostic_only | loopback retained-trace route only; full media synchronization absent |
| 11 | Toggleable metric/event channels | implemented_but_unproven | panel contracts; no complete browser receipt |
| 12 | Cross-planner comparison and Codex/MCP discussion | diagnostic_only | fake provider and MCP/loopback tests; no live provider receipt |
| 13 | Suspicious successes and unusual combinations | implemented_but_unproven | detector/queue contracts; no campaign-level result |
| 14 | Formal audit-coverage rules | implemented_and_verified | protocol and hand-computed tiny-strata tests |
| 15 | Deterministic rules and statistical discovery | implemented_but_unproven | detector/statistical adapters exist; no second campaign proof |
| 16 | Bounded diagnostic reruns | diagnostic_only | native diagnostic contracts and cancellation tests; admitted native source not proven |
| 17 | Portable records plus rebuildable SQLite index | implemented_and_verified | canonical store tests and projection materialization repair |
| 18 | Queue, annotations, findings, health, bugs, and examples | implemented_but_unproven | service/store ownership exists; full knowledge loop absent |
| 19 | Visible score components and evidence channels | implemented_and_verified | queue reason/component contract tests |
| 20 | Human-approved detector evolution before adaptive/learned policy | authorized_later_extension | proposal path is allowed; adaptive ranking/classifiers remain later work |
| 21 | One-click, quick, and full annotation modes | implemented_but_unproven | contracts exist; complete UI proof absent |
| 22 | Timestamp/spatial/actor/object/goal references | implemented_but_unproven | reference schema tests; source/media coverage incomplete |
| 23 | Toggleable overlays and numbered references | implemented_but_unproven | rendering contracts; browser acceptance absent |
| 24 | Findings group multiple episode annotations | implemented_and_verified | finding aggregation and persistence tests |
| 25 | Explained similarity modes and suggestions | implemented_but_unproven | similarity contract exists; no complete related-case campaign receipt |
| 26 | Redundancy lowers priority without false review credit | implemented_but_unproven | queue tests; full protocol integration absent |
| 27 | Coverage-aware ordinary/control sampling | implemented_but_unproven | control-selection contracts; no complete campaign receipt |
| 28 | Release-specific protocol completion and exceptions | missing | no accepted campaign/release completion receipt |
| 29 | Autonomous analysis, annotation, issues, diagnostics from V1 | diagnostic_only | service and fake-provider routes; live provider and native source unproven |
| 30 | Agent access to complete scoped data/source/config | blocked_external | path/context guards exist; complete admitted live source is not available |
| 31 | Proactive evidence, related cases, hypotheses | implemented_but_unproven | service contracts; no real Codex receipt |
| 32 | Findings, not isolated episodes, drive GitHub linkage | implemented_and_verified (bounded) | local outbox, finding marker, fake-provider exact retry tests |
| 33 | Any current-schema campaign | blocked_external | current-schema contracts exist; historical adapters and second campaign evidence remain absent |
| 34 | Lazy materialization/regeneration of missing material | implemented_but_unproven | materialization contracts; native/source assets not accepted |
| 35 | Twelve detector families plus advisory statistics | implemented_but_unproven | registry/tests exist; full campaign execution absent |
| 36 | Versioned JSON/NDJSON canonical; SQLite projection | implemented_and_verified | store schema and projection tests; list-record runtime repair hosted-green |
| 37 | Full agreed audit UI in V1 | missing | no one-route real-data browser acceptance |
| 38 | Agent-centric API/MCP first-class | diagnostic_only | stdio/MCP dispatcher tests and loopback route; no live external provider |
| 39 | Fixed and active Next Best Review | implemented_but_unproven | queue/Next tests; no full active-policy campaign receipt |

## Explicit unresolved gates

### GitHub CAS and exactly-once delivery

GitHub's ordinary issue API does not provide the required cross-system atomic
CAS between the local journal, a finding revision, and a remote issue. The
accepted boundary is one immutable issue snapshot plus append-only comments,
each carrying an exact request marker. Complete pagination and readback,
duplicate-marker detection, timeout reconciliation, local outbox recovery, and
the canonical finding-link CAS are mandatory before a live result can be
accepted. A read-then-`PATCH` body update is explicitly not CAS. No claim of
exactly-once remote delivery is made; an ambiguous outcome blocks unsafe
automatic retry. The current tests use fakes/injected transports and do not
mutate GitHub.

### Provider-cap and Codex route

`offline` is provider-free and read-only. `local_accounting` reserves finite
local budgets and records observed usage/overspend, but cannot enforce a
physical provider limit. `strict_provider_ceiling` refuses admission without a
verified finite provider compute ceiling and exact route/provider/model. The
tested App Server exposes token telemetry but no measured provider compute
quantity; therefore strict live acceptance is blocked and no premium/fallback
route is used to bypass the gate.

### Real data, browser, and native diagnostics

No accepted receipt proves the complete real-data/browser workflow, live Codex
conversation, native source-bound rerun, all media recovery, or cancellation /
reconnect across an external provider. The retained trace and loopback route
are diagnostic-only. The optional `imageio_ffmpeg` local gap (#9560) and the
host filesystem exhaustion (#9532) are recorded limitations, not successful
fallbacks.

## Decision and next action

BA-05's bounded D1/D2 service slice is accepted and merged. BA-06 may now
resume, but only as separately scoped work against the gates above. #9488,
#9489, and epic #9483 remain open; no issue, report, label, or merge receipt in
this packet declares the epic complete.

Machine-readable companion:
[`benchmark_auditor_v1_acceptance_2026-09-22.json`](./benchmark_auditor_v1_acceptance_2026-09-22.json).

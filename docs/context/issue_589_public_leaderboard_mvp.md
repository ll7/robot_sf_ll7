# Issue 589 Public Leaderboard MVP Boundary

## Goal

Decide whether Robot SF should pursue a public social-navigation planner leaderboard now, and define
the smallest credible MVP boundary if it becomes valuable later.

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/589>
Next actionable follow-up: <https://github.com/ll7/robot_sf_ll7/issues/1136>

## 2026-05-10 Backlog Status

Issue #589 was reopened as a future planned idea after the original MVP-boundary PR merged. Keep it
as a parent/backlog concept, not a direct implementation request.

The next actionable slice is #1136: draft the external planner submission manifest schema,
validator, and maintainer-reviewed smoke path. That issue is intentionally narrower than a
leaderboard website, public upload endpoint, or execution of arbitrary user code.

## Current Decision

Do not implement a public upload, benchmark-execution, or website leaderboard in the current
benchmark phase.

The repository has the building blocks for public evidence:

- benchmark release protocol: `docs/benchmark_release_protocol.md`
- release reproducibility contract: `docs/benchmark_release_reproducibility.md`
- DOI-ready artifact publication: `docs/benchmark_artifact_publication.md`
- camera-ready campaign workflow: `docs/benchmark_camera_ready.md`
- fail-closed fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`
- experimental planner guardrails: `docs/benchmark_experimental_planners.md`

Those surfaces are not yet a submission service. Treat the leaderboard as a downstream design
option that depends on a stable benchmark-release contract and a trustworthy external-planner
execution boundary.

## Plausible MVP

The narrowest future MVP should be repository-native and asynchronous:

1. Submission is a GitHub PR, not direct public upload.
2. The submission includes a planner manifest with:
   - planner key and display name,
   - source repository and commit SHA,
   - license,
   - dependency/install command or container reference,
   - expected execution mode (`native`, `adapter`, or unavailable),
   - required model/checkpoint artifact references and checksums,
   - declared hardware/runtime requirements,
   - maintainer contact.
3. CI or a maintainer-triggered workflow runs a smoke/preflight release manifest first.
4. Full benchmark execution is manual or approval-gated until runtime isolation is proven.
5. Results are published only from a benchmark-valid release bundle, not ad hoc local artifacts.
6. Fallback, degraded, partial-failure, and not-available outcomes are displayed as non-successes,
   following the benchmark fallback policy.
7. The public page is generated from accepted release bundles and manifests; it does not accept
   arbitrary runtime uploads directly.

This MVP is intentionally less ambitious than a hosted upload portal. It keeps review, provenance,
and failure semantics inside the existing repository workflow.

## Required Prerequisites

Do not open implementation work until these are true:

- A benchmark release tag and publication bundle are available for at least one current paper-facing
  matrix.
- The planner submission manifest schema is drafted and reviewed.
- External planner execution is sandboxed or approval-gated with documented resource limits.
- Dependency failures fail closed with explicit `not_available` or `failed` status.
- Runtime mode and readiness status appear in the published result table.
- Governance exists for licenses, model artifacts, and result removal/correction.
- A smoke manifest can validate a submitted planner without consuming a full benchmark budget.

## Non-Goals

- No public upload endpoint now.
- No website or deployment stack now.
- No automatic execution of arbitrary user code.
- No benchmark success claim for fallback-only or degraded execution.
- No inclusion of testing-only planners without their existing promotion evidence.

## Follow-Up Boundary

If the team wants to continue later, create a narrower issue for the submission manifest schema
first. That issue should not build a website; it should define the manifest, validation rules, and
one maintainer-reviewed smoke path.

Only after that schema exists should a separate issue consider a static leaderboard page generated
from accepted release bundles.

As of 2026-05-10, that narrower schema issue is #1136.

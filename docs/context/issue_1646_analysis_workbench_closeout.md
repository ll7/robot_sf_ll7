# Issue #1646 Closeout Audit — Analysis & Visualization Workbench Epic

Issue: [#1646](https://github.com/ll7/robot_sf_ll7/issues/1646)
Date: 2026-07-06
Status: **closeout audit** — maps each epic acceptance criterion to merged evidence and records
what is production-ready, what stays exploratory, and which benchmark claims this work does **not**
establish. Diagnostic/qualitative-only; not benchmark or paper-facing evidence.

## Why this note exists

Epic #1646 coordinated a richer analysis and visualization workbench for simulation runs. It was
intentionally de-emphasized and kept blocked for direct UI implementation, with all work routed
through narrow child issues (schema → timeline fixture → annotation → report → failure-case pack).
As of this audit **every routed child is closed** and the epic's own final acceptance criterion asks
for a durable closeout record. This note is that record: acceptance criterion 7 ("the final epic
closeout records what is production-ready, what remains exploratory, and which benchmark claims are
not established by this work").

This note changes no schema and no behavior. It consolidates the merged child evidence into one
criterion→evidence surface so the maintainer can make the close-vs-keep-open call with the full map
in front of them.

## Acceptance criterion → evidence map

| # | Acceptance criterion (epic body) | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Child issues created or explicitly deferred for schema, internal-state exposure, frame inspector, annotations, agent discussion, report generation, validation fixtures | **Met** | Children created and closed: #1689 (export schema, PR #1698), #1859 (timeline fixture, PR #1926), #1962 (annotation fixture/schema, PR #1968), #1996 (annotation-aware report, PR #1998), #2159 (research-v1 failure-case pack), #3223 (renderer-neutral timeline artifact, PR #3225). Broad browser/AI-discussion children explicitly deferred (see below). |
| 2 | Minimal end-to-end path from a run/fixture to a frame-indexed timeline with visible state, internal state, and annotations | **Met** | `robot_sf/analysis_workbench/simulation_trace_export.py` → `simulation_timeline.py` (`simulation_timeline.v1`) → `trace_annotation.py` (`trace_annotation_set.v1`). Tracked fixtures under `tests/fixtures/analysis_workbench/`. |
| 3 | At least one viewer or report surface inspects a selected frame/event sequence and cites the artifact path plus frame/event identifiers | **Met** | `scripts/tools/render_trace_report.py --annotations` cites source fixture paths, annotation-set id, frame ranges, event ids, entity refs. Static Three.js trace viewer validated in #2038 (PR #2044) with a browser pixel smoke. |
| 4 | Agent-assisted output distinguishes observed evidence from hypothesis and links back to source frames/events | **Met** | `scripts/analysis/agent_discussion_sequence_issue_1646.py` (`generate_agent_discussion`) summarizes a selected trace+annotation sequence while citing source artifact paths and frame/event ids; tested in `tests/analysis_workbench/test_agent_discussion_sequence.py`. `run_annotation.v1` (`robot_sf/analysis/run_annotation.py`, PR #3598) keeps `observation`/`hypothesis` separable by contract; `render_trace_report.py` preserves `observed`/`hypothesis`/`commentary` as distinct evidence types. A *broad interactive/browser* discussion UX remains deferred. |
| 5 | Documentation explains how to generate, inspect, annotate, and reference a run analysis artifact | **Met** | `docs/debug_visualization.md` documents trace export, `render_trace_report.py`, and `trace_annotation_set.v1` embedding with runnable commands. Context pack: `docs/context_packs/visualization_workbench.yaml`. |
| 6 | Artifact policy enforced — durable small evidence may be tracked under `docs/context/evidence/`; local `output/` stays disposable unless promoted | **Met** | Loaders fail closed on generated `output/` trace/annotation dependencies; fixtures are small and tracked; evidence promoted under `docs/context/evidence/` per note references. |
| 7 | Final epic closeout records production-ready vs exploratory scope and un-established benchmark claims | **Met by this note** | This closeout audit. |

## Production-ready (schema/data/report, qualitative-only)

- `simulation_trace_export.v1`, `simulation_timeline.v1`, `trace_annotation_set.v1`,
  `trace_failure_predicates.v1`, `real_trace_validation_contract.v1` schemas + typed loaders in
  `robot_sf/analysis_workbench/`.
- `run_annotation.v1` annotation model in `robot_sf/analysis/run_annotation.py`.
- `scripts/tools/render_trace_report.py` annotation-aware Markdown report renderer.
- `scripts/analysis/agent_discussion_sequence_issue_1646.py` agent-discussion-sequence generator
  (observed-vs-hypothesis, artifact-cited).
- Tracked smoke fixtures under `tests/fixtures/analysis_workbench/` and tests under
  `tests/analysis_workbench/` + `tests/analysis/test_run_annotation.py`.

## Exploratory / intentionally deferred (not delivered by this epic)

- Broad browser / Three.js interactive playback and inspector UI (child #1645, closed). Only the
  static, fixture-driven viewer smoke (#2038) landed; a full interactive workbench was never built
  and stays out of scope.
- An *interactive* agent-discussion UX (live debate over a sequence). The batch generator and
  observed-vs-hypothesis data contract landed (see criterion 4); the interactive UX did not.
- Frame-by-frame scrubbing inspector UI with synchronized panels.

## Benchmark-claim boundary (explicit)

This epic and all its children are **qualitative / diagnostic-only** (`analysis_workbench_qualitative_only`).
Nothing here establishes benchmark, metric, schema-of-record, or paper-facing results. Trace review,
annotations, and failure-case packs are for mechanism inspection and falsification, not for claiming
planner/policy performance. Fallback or degraded execution observed in any trace is a caveat, never a
success condition (see `docs/context/issue_691_benchmark_fallback_policy.md`).

## Closure recommendation

The epic's defined acceptance criteria (1–7) are all met by merged children plus this closeout, and
every routed child issue — including the UI child #1645 — is closed. The only remaining named work
(broad browser/Three.js UI and interactive agent-discussion UX) was **explicitly deferred**, not
promised as epic deliverables. On that basis the epic's scope is complete and it can be closed.

Residual note for the maintainer: earlier triage kept this open as a "coordination umbrella." That
was under a state with open children; none remain open now. If future workbench UI work is desired
it should be opened as a fresh, separately-scoped issue rather than reopening this epic.

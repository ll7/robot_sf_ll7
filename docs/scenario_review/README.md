# Scenario review

Executable contracts for reviewing retained traces and scenarios
([review contracts](./review_contracts.md)): versioned review bundles,
visualization specs, component request/result envelopes with capability
descriptors, and experiment recipes. Diagnostic tooling only — no
scientific admission, benchmark results, or simulator/planner changes.

The offline [Benchmark Auditor campaign scan](./audit_scan.md) accounts for
recorded campaign rows and emits deterministic BA-03 detector signals without
rerunning simulations.

The bounded [BA-05 source-first materialization leaf](./audit_materialize.md)
verifies historical recordings and lazily renders retained trace or replay
state without starting a simulation.

The contracts are consumed by offline components; the first consumer is the
[SREV-15 review workbench](./review_contracts.md#review-workbench-srev-15)
(`python -m robot_sf.render.review_workbench`), which renders a local,
network-free artifact/provenance view plus a presentation plan.

The synchronized SREV-16 scene/video/metric/event extension is documented in
[`review_panels.md`](./review_panels.md) and runs with
`python -m robot_sf.render.review_panels`.

The SREV-17 offline annotation and storyboard editor is documented in
[`review_editor.md`](./review_editor.md) and runs with
`python -m robot_sf.render.review_editor`.

The bounded BA-06 fixture-backed queue-to-finding UI slice is documented in
[`audit_workbench_fixture.md`](./audit_workbench_fixture.md). It is diagnostic
UI-wiring evidence only; final integration uses the existing SREV-15 launch
path and BA-05's accepted service facade.

The [local live Benchmark Auditor launch](./audit_workbench_live.md) is an
in-progress integration slice that opens one campaign, queue, server-held
service session, and SREV-15 browser mount. Its documented limits still apply;
it is not the completed BA-06 workflow.

The Benchmark Auditor release-bound coverage evaluator is documented in
[`audit_coverage.md`](./audit_coverage.md). It consumes BA-01 scan results and
BA-03 review receipts, emits BA-02-compatible coverage deficits, and reports
operational health without making benchmark-validity or statistical claims.

The optional Scenario Review (SREV-18) recorded planner, control, pedestrian, and failure
diagnostic extension is documented in
[`review_diagnostics.md`](./review_diagnostics.md) and runs with
`python -m robot_sf.render.review_diagnostics`. It consumes retained trace
rows only; it does not instrument a planner, run a simulator, or make a
benchmark/scientific claim.

The SREV-10 source-backed review encoder is documented in
[review_encode.md](./review_encode.md) and runs with
python -m robot_sf.render.review_encode. It emits diagnostic-only MP4,
receipt, and presentation-time-map artifacts from a verified frame sequence
or source clip; it does not produce benchmark or scientific evidence.

## Leaf fast-lane registration convention

Scenario-review leaves that add a deterministic test file must also register
it for fast continuous-integration (CI) shards, or the changed-coverage gate
fails with `missing-fast-registration`.

- Scope: a single `tests/conftest.py` `_FAST_FILES` entry plus an SREV comment
  naming the leaf issue. No other central-file edit rides with the leaf PR.
- Disclosure: name the entry in the PR body as an allowlist deviation; leaf
  issue allowlists do not need to list `tests/conftest.py` once this
  convention is followed.
- Rationale: the changed-coverage gate only counts coverage from fast shards,
  so an unregistered deterministic test file reliably costs one full CI cycle
  plus a fix-push cycle (issue #9414; precedents PR #9404, PR #9413, PR #9455).
- Local check before push: `python scripts/dev/check_fast_lane_routing.py`
  must report zero findings for the leaf module.

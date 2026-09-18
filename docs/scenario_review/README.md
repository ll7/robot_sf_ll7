# Scenario review

Executable contracts for reviewing retained traces and scenarios
([review contracts](./review_contracts.md)): versioned review bundles,
visualization specs, component request/result envelopes with capability
descriptors, and experiment recipes. Diagnostic tooling only — no
scientific admission, benchmark results, or simulator/planner changes.

The offline [Benchmark Auditor campaign scan](./audit_scan.md) accounts for
recorded campaign rows and emits deterministic BA-03 detector signals without
rerunning simulations.

The contracts are consumed by offline components; the first consumer is the
[SREV-15 review workbench](./review_contracts.md#review-workbench-srev-15)
(`python -m robot_sf.render.review_workbench`), which renders a local,
network-free artifact/provenance view plus a presentation plan.

The synchronized SREV-16 scene/video/metric/event extension is documented in
[`review_panels.md`](./review_panels.md) and runs with
`python -m robot_sf.render.review_panels`.

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

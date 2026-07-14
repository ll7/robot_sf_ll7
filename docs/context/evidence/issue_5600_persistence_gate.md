<!-- AI-GENERATED: validation-contract evidence; NEEDS-REVIEW: maintainer verification before reuse. -->
# Issue #5600 — Persistence Promotion Gate (compact evidence)

Status: implemented (CPU-only, no campaigns). Evidence tier: schema + fail-closed
unit contract + end-to-end candidate runner conformance. No benchmark, metric,
model-provenance, or paper-facing claim.

## What changed (slice 1: PR #5606)

The stage-1 generation pipeline (#4932) produces catalog *hypotheses* but does not
gate them on whether the discovered critical event actually reproduces and persists.
Slice 1 added the missing evidence record `generated_scenario_persistence.v1` and a
fail-closed promotion gate.

Files (all under allowed paths):

- `robot_sf/benchmark/schemas/generated_scenario_persistence.v1.json` — versioned
  schema with source episode / generated scenario / planner / seed / config / commit
  hashes and the three independent status blocks plus promotion verdict.
- `robot_sf/benchmark/scenario_generation/persistence_gate.py` — schema validator and
  three independent checks assembled by `compute_persistence_record` into a
  fail-closed `promotion` verdict. Exposed from the package `__init__`.
- `scripts/tools/validate_generated_scenario_persistence.py` — CPU-only CLI that
  validates a prebuilt record against the schema and fail-closed invariants.
- `configs/analysis/issue_5600_persistence_gate.yaml` — **frozen** preregistered
  timing/speed grid, tolerances, and promotion threshold (frozen before any smoke run).
- `tests/benchmark/test_scenario_persistence_gate.py` — 15 unit tests covering
  positive, negative, replay divergence, deliberately non-persistent, missing-cell,
  unknown-event, unfrozen-config, schema-valid, checksum-identical, and two-candidate
  promote+reject smoke fixtures.

## What changed (slice 2: candidate runner and conformance evidence)

Slice 2 wires the gate into an end-to-end candidate runner that takes episode traces
or catalog entries and produces `generated_scenario_persistence.v1` conformance records
with published promote/reject evidence.

Files (all under allowed paths):

- `robot_sf/benchmark/scenario_generation/candidate_runner.py` — orchestration module
  that runs `extract_critical_segment`, `assess_exact_replay`,
  `assess_critical_event_reproduction`, and `evaluate_perturbation_grid` for each
  candidate, assembling a schema-valid persistence record. Exposed from the package
  `__init__`.
- `scripts/tools/run_persistence_candidate_smoke.py` — CLI runner that accepts episode
  JSONL, catalog-entry JSONL, or `--synth` for synthetic candidates, producing
  individual record JSON files, batch JSONL, and a summary JSON.  Uses
  `run_candidate_persistence_smoke` internally.
- `tests/benchmark/test_candidate_runner.py` — 10 tests covering critical event
  extraction, episode-to-record pipeline, catalog-entry-to-record pipeline, perturbation
  verdict functions, and batch smoke with config enforcement.
- Package `__init__.py` updated to export `build_persistence_from_episode_trace`,
  `build_persistence_from_catalog_entry`, `run_candidate_persistence_smoke`,
  `get_critical_event_from_frames`, and `build_cell_verdict_from_trace_replay`.

## The three statuses stay separate

- `exact_replay`: byte/config-equivalent replay of the source episode (digest match).
- `critical_event_reproduced`: event of `event_type` recurs under the source planner
  within declared time/location tolerances.
- `perturbation_persistence`: the event survives a preregistered timing/speed grid.

A `pass` in one does **not** imply a `pass` in another. A candidate is promoted only
when all three required statuses are `pass` and there are no missing cells.

## Conformance evidence (published)

A two-candidate smoke run via `--synth` demonstrates both verdict paths:

```
candidates: 2  promoted: 1  rejected: 1
PROMOTE generated-54390a9cf13d98e4: all three independent status checks passed
REJECT generated-0d1ab5599bea6d7e: perturbation_cell:-0.25:-0.2:fail; ...
exit=2
```

The promoted candidate has persistence_rate=1.0 with all three checks passing.
The rejected candidate fails the perturbation grid (all 9 cells fail).

## Acceptance-criteria coverage

- [x] Versioned schema and validator for `generated_scenario_persistence.v1`.
- [x] Exact replay, event reproduction, and perturbation persistence are independently reported.
- [x] Perturbation ranges, tolerances, and promotion threshold are frozen before the real smoke run (`configs/analysis/issue_5600_persistence_gate.yaml`, `frozen: true`).
- [x] Positive, negative, divergence, and deliberately non-persistent fixtures covered.
- [x] A real two-candidate smoke demonstrates both promotion and rejection paths
  (candidate runner + CLI; validated by both `test_candidate_runner` and
  `test_scenario_persistence_gate`).
- [x] Identical inputs produce checksum-identical output (test
  `test_identical_inputs_produce_checksum_identical_output`).
- [x] Promotion fails closed on missing trace fields, replay divergence, or unfrozen
  configuration.

## Validation commands and output

```
uv run ruff check robot_sf/benchmark/scenario_generation scripts/tools tests/benchmark
# All checks passed!
uv run pytest -q tests/benchmark -k 'scenario_generation and (replay or persistence or candidate)'
# 25 passed
uv run python scripts/tools/run_persistence_candidate_smoke.py --synth
# candidates: 2  promoted: 1  rejected: 1
# exit=2 (one rejection is expected)
uv run python scripts/tools/validate_generated_scenario_persistence.py --batch output/*.json
# PROMOTE generated-... :: all three independent status checks passed
# REJECT generated-... :: perturbation_cell:....:fail ...
git diff --check   # clean
```

## Stop-rule note

The gate and runner are wired end-to-end.  The negative-conformance path
(promote none on failure) was demonstrated by the synthetic reject candidate
and tested in `test_candidate_runner`.  Candidates are promoted only when
all three required statuses pass and no perturbation cell fails. No threshold
loosening is possible because ranges are frozen.

## Claim boundary

Generated scenario hypotheses only. This is not benchmark evidence and does not
reimplement the coverage-constrained portfolio selector (#5600 depends on #4932's
pipeline, distinct from #5442 counterfactual branches).
# Issue #5600 — Persistence Promotion Gate (compact evidence)

Status: implemented (CPU-only, no campaigns). Evidence tier: schema + fail-closed
unit contract. No benchmark, metric, model-provenance, or paper-facing claim.

## What changed

The stage-1 generation pipeline (#4932) produces catalog *hypotheses* but does not
gate them on whether the discovered critical event actually reproduces and persists.
This slice adds the missing evidence record `generated_scenario_persistence.v1` and a
fail-closed promotion gate.

Files (all under allowed paths):

- `robot_sf/benchmark/schemas/generated_scenario_persistence.v1.json` — versioned
  schema with source episode / generated scenario / planner / seed / config / commit
  hashes and the three independent status blocks plus promotion verdict.
- `robot_sf/benchmark/scenario_generation/persistence_gate.py` — schema validator and
  three independent checks (`assess_exact_replay`, `assess_critical_event_reproduction`,
  `evaluate_perturbation_grid`) assembled by `compute_persistence_record` into a
  fail-closed `promotion` verdict. Exposed from the package `__init__`.
- `scripts/tools/validate_generated_scenario_persistence.py` — CPU-only CLI that
  validates a prebuilt record against the schema and fail-closed invariants.
- `configs/analysis/issue_5600_persistence_gate.yaml` — **frozen** preregistered
  timing/speed grid, tolerances, and promotion threshold (frozen before any smoke run;
  not tuned after observing candidates).
- `tests/benchmark/test_scenario_persistence_gate.py` — positive, negative, replay
  divergence, deliberately non-persistent, missing-cell, unknown-event, unfrozen-config,
  schema-valid, checksum-identical, and two-candidate promote+reject smoke fixtures.

## The three statuses stay separate

- `exact_replay`: byte/config-equivalent replay of the source episode (digest match).
- `critical_event_reproduced`: event of `event_type` recurs under the source planner
  within declared time/location tolerances.
- `perturbation_persistence`: the event survives a preregistered timing/speed grid.

A `pass` in one does **not** imply a `pass` in another. A candidate is promoted only
when all three required statuses are `pass` and there are no missing cells.

## Acceptance-criteria coverage

- [x] Versioned schema and validator for `generated_scenario_persistence.v1`.
- [x] Exact replay, event reproduction, and perturbation persistence are independently reported.
- [x] Perturbation ranges, tolerances, and promotion threshold are frozen before the real smoke run (`configs/analysis/issue_5600_persistence_gate.yaml`, `frozen: true`).
- [x] Positive, negative, divergence, and deliberately non-persistent fixtures covered.
- [x] A real two-candidate smoke demonstrates both promotion and rejection paths (`scripts/tools/validate_generated_scenario_persistence.py --batch`).
- [x] Identical inputs produce checksum-identical output (test `test_identical_inputs_produce_checksum_identical_output`).
- [x] Promotion fails closed on missing trace fields, replay divergence, or unfrozen configuration.

## Validation commands and output

```
uv run ruff check robot_sf/benchmark/scenario_generation scripts/tools tests/benchmark
# All checks passed!
uv run pytest -q tests/benchmark -k 'scenario_generation and (replay or persistence)'
# 20 passed
uv run python scripts/tools/validate_generated_scenario_persistence.py --help
uv run python scripts/tools/validate_generated_scenario_persistence.py --batch /tmp/promote.json /tmp/reject.json
# PROMOTE /tmp/promote.json :: all three independent status checks passed
# REJECT /tmp/reject.json :: perturbation_cell:...:fail ...
# exit=2
git diff --check   # clean
```

## Stop-rule note

This slice implements the gate and its contract. The negative-conformance path
(publish a report and promote none) is exercised by the reject fixtures but is wired
to the pipeline's real smoke run only when a real candidate set exists; no real
candidate replay/perturbation executions were performed here (CPU-only, no campaigns).
If a real smoke run shows no candidate reproduces its critical event or passes the
frozen gate, the gate already emits `reject` with machine-readable exclusion reasons
and promotes none — no threshold loosening is possible because ranges are frozen.

## Claim boundary

Generated scenario hypotheses only. This is not benchmark evidence and does not
reimplement the coverage-constrained portfolio selector (#5600 depends on #4932's
pipeline, distinct from #5442 counterfactual branches).

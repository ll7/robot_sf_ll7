<!-- AI-GENERATED: validation-contract evidence; NEEDS-REVIEW: maintainer verification before reuse. -->
# Issue #5600 — Persistence Promotion Gate (compact evidence)

Status: implemented + wired (CPU-only, no campaigns). Evidence tier: schema +
fail-closed unit contract + real pipeline-artifact conformance. No benchmark,
metric, model-provenance, or paper-facing claim.

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

## What changed (slice 2: this PR — wiring + conformance evidence)

Slice 1 left #5600 open because the gate had no committed caller: the
`generated_scenario_persistence.v1` record was only ever assembled from hand-built
dictionaries in unit tests. Slice 2 wires the gate into a real pipeline artifact: it
runs the stage-1 generation pipeline (#4932), reads the produced catalog + episode
traces, evaluates the three independent checks for each candidate, and emits
schema-valid persistence records as conformance evidence. This is the "real candidate
run + publish negative/positive conformance evidence" item named in the issue thread.

Files (all under allowed paths):

- `scripts/tools/evaluate_pipeline_persistence_gate.py` — **new, committed** wiring CLI.
  Loads a pipeline `run_manifest.json` (or a single candidate entry + episodes JSONL),
  derives `exact_replay` from the trace, evaluates `critical_event_reproduced` from the
  catalog entry's criticality block, and runs the preregistered perturbation grid in
  CPU-only mode (timestamp shift + position interpolation). Emits one
  `generated_scenario_persistence.v1` JSON per candidate plus a promote/reject summary.
- `tests/benchmark/test_pipeline_persistence_gate_wiring.py` — **new**. Runs the real
  `run_generation_pipeline` with a deterministic fake batch runner, then wires the
  produced catalog + episodes through the tool and asserts one **promote** and one
  **reject** verdict, both schema-valid. No simulations: the fake runner supplies the
  step traces, satisfying the CPU-only constraint.

The tool also accepts `--replay-results PATH` to skip CPU-only heuristics and consume
pre-computed replay/perturbation verdicts from a simulation-capable runner (CARLA),
preserving the generated-hypothesis boundary.

## The three statuses stay separate

- `exact_replay`: byte/config-equivalent replay of the source episode (digest match).
- `critical_event_reproduced`: event of `event_type` recurs under the source planner
  within declared time/location tolerances.
- `perturbation_persistence`: the event survives a preregistered timing/speed grid.

A `pass` in one does **not** imply a `pass` in another. A candidate is promoted only
when all three required statuses are `pass` and there are no missing cells.

## Conformance evidence (published, real pipeline artifact)

The wiring test (`test_pipeline_persistence_gate_wiring.py`) executes the actual
generation pipeline and produces two records from a real catalog + episode-trace pair:

- Candidate A (near-static robot/pedestrian pair): `exact_replay=pass`,
  `critical_event_reproduced=pass`, `perturbation_persistence` rate = 1.0 →
  **PROMOTE** ("all three independent status checks passed").
- Candidate B (moving robot/pedestrian pair; min clearance 0.2 m at the critical frame
  grows under ±0.25 s / ±0.2 m/s perturbation): `exact_replay=pass`,
  `critical_event_reproduced=pass`, but perturbation cells fail → **REJECT**
  ("perturbation_cell:...:fail ...").

The promote verdict only fires when all three independent checks pass; candidate B is
rejected purely on perturbation persistence despite its replay and event checks
passing — exercising the fail-closed separation end-to-end.

## Acceptance-criteria coverage

- [x] Versioned schema and validator for `generated_scenario_persistence.v1`.
- [x] Exact replay, event reproduction, and perturbation persistence are independently reported.
- [x] Perturbation ranges, tolerances, and promotion threshold are frozen before the real smoke run (`configs/analysis/issue_5600_persistence_gate.yaml`, `frozen: true`).
- [x] Positive, negative, divergence, and deliberately non-persistent fixtures covered.
- [x] A two-candidate smoke demonstrates both promotion and rejection paths **from a real
  pipeline artifact** (`test_pipeline_persistence_gate_wiring.py`), not only synthetic dicts.
- [x] Identical inputs produce checksum-identical output (test
  `test_identical_inputs_produce_checksum_identical_output`).
- [x] Promotion fails closed on missing trace fields, replay divergence, or unfrozen
  configuration.

## Validation commands and output

```
uv run ruff check robot_sf/benchmark/scenario_generation scripts/tools tests/benchmark
# All checks passed!
uv run pytest -q tests/benchmark -k 'scenario_generation and (replay or persistence)'
# 17 passed (15 gate unit tests + 2 wiring tests)
uv run python scripts/tools/validate_generated_scenario_persistence.py --help
uv run python scripts/tools/evaluate_pipeline_persistence_gate.py \
  --manifest <run_manifest.json> \
  --config configs/analysis/issue_5600_persistence_gate.yaml \
  --output <output_dir> --commit-hash <h> --config-hash <c>
# PROMOTE <output_dir>/generated-....persistence.json :: all three independent status checks passed
# REJECT <output_dir>/generated-....persistence.json :: perturbation_cell:...:fail ...
# Summary: 1 promoted, 1 rejected, 2 total
git diff --check   # clean
```

## Stop-rule note

The gate is now wired through a real pipeline run and emits promote/reject evidence.
The CPU-only perturbation mode proves the wiring end-to-end but does not substitute
for a real simulator replay; production use should pass `--replay-results` from a
CARLA-capable runner. The negative-conformance path (promote none) is inherent: any
`fail`/`unknown` required status or missing cell fails closed. No threshold loosening
is possible because ranges are frozen.

## Claim boundary

Generated scenario hypotheses only. This is not benchmark evidence and does not
reimplement the coverage-constrained portfolio selector (#5600 depends on #4932's
pipeline, distinct from #5442 counterfactual branches).

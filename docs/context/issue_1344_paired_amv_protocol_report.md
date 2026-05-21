# Issue #1344 Paired AMV Primary Protocol Report

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1344>

## Goal

Issue #1344 asked for a rerunnable paired nominal/stress AMV protocol that separates routine
shared-space competence from stress robustness. The maintainer decision was to run primary planner
rows first and defer all-runnable planner expansion until this first report proves the runtime and
artifact path.

## Protocol

This branch adds two versioned benchmark configs:

- `configs/benchmarks/issue_1344_paired_nominal_v1_primary.yaml`
- `configs/benchmarks/issue_1344_paired_stress_primary.yaml`

Both configs use:

- S3 `eval` seed policy from `configs/benchmarks/seed_sets_v1.yaml`: `[111, 112, 113]`,
- `paper_profile_version: paper-matrix-v1`, `paper_facing: false`,
- differential-drive kinematics,
- primary/core planners only: `goal`, `social_force`, `orca`,
- SNQI v3 weights/baseline with warn/fail diagnostics,
- AMV profile `amv-paper-v1` with coverage enforcement set to `warn`.

The stress side uses `configs/scenarios/classic_interactions_francis2023.yaml`; the nominal side
uses `configs/scenarios/nominal_v1.yaml`.

## Collision-Metric Fix

The first attempted nominal run exposed a fail-closed integrity blocker shared with issue #1318:
exact environment collision flags could report a collision while sampled collision metrics stayed at
zero. `robot_sf/benchmark/map_runner.py` now floors collision metrics from exact typed collision
flags before outcome-integrity validation, with regression coverage in
`tests/benchmark/test_map_runner_utils.py`.

Without this fix, primary rows were not reportable; with it, both primary campaigns completed.

## Results

Final campaign commit: `c16ae67b5fe2c605476152113d43e569828958a7`

| Surface | Campaign ID | Scenario matrix hash | Episodes | Successful rows | Runner completion |
| --- | --- | --- | ---: | ---: | --- |
| nominal | `issue_1344_nominal_primary_final` | `73acddfd12cf` | 36 | 3/3 | completed |
| stress | `issue_1344_stress_primary_final` | `8ac8ab9387f4` | 432 | 3/3 | completed |

| Planner | Nominal success | Stress success | Nominal collisions | Stress collisions | Nominal SNQI | Stress SNQI | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `goal` | 0.2500 | 0.0000 | 0.3333 | 0.2500 | -0.0967 | -0.1752 | Routine completion drops to zero under stress; collisions remain high in both surfaces. |
| `orca` | 0.2500 | 0.1667 | 0.0833 | 0.0764 | -0.2999 | -0.2466 | Best routine/stress balance among primary rows, but still low success and many near misses. |
| `social_force` | 0.0000 | 0.0000 | 0.0000 | 0.2500 | -1.0435 | -0.8591 | Avoids nominal collisions but fails completion; stress adds collisions without improving success. |

Both final campaigns had no runner warnings and `benchmark_success=true`. That status means the
configured rows completed and passed fail-closed campaign execution, not that the result is
paper-claim quality. Both campaigns report `amv_coverage_status=warn` because the required AMV
dimensions are completely unannotated in the source scenario metadata: the coverage summaries show
`Observed = -` and all required values missing for each required dimension. Both campaigns also
report `snqi_contract_status=fail`. This evidence should not be promoted into paper-facing claims
without a separate BenchmarkClaim/claim-scope review and an explicit decision about AMV annotation
coverage and SNQI positioning.

## Evidence

Compact evidence is tracked in:

- `docs/context/evidence/issue_1344_paired_amv_primary_2026-05-20/README.md`
- `docs/context/evidence/issue_1344_paired_amv_primary_2026-05-20/nominal_campaign_summary.json`
- `docs/context/evidence/issue_1344_paired_amv_primary_2026-05-20/stress_campaign_summary.json`
- `docs/context/evidence/issue_1344_paired_amv_primary_2026-05-20/manifest.sha256`

Raw campaign outputs remain under `output/benchmarks/issue_1344/` and are intentionally ignored.

## Validation Commands

```bash
uv run pytest tests/benchmark/test_map_runner_utils.py -q

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_nominal_v1_primary.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1344 \
  --campaign-id issue_1344_nominal_primary_final_preflight

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_stress_primary.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1344 \
  --campaign-id issue_1344_stress_primary_final_preflight

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_nominal_v1_primary.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1344 \
  --campaign-id issue_1344_nominal_primary_final \
  --log-level INFO

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1344_paired_stress_primary.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1344 \
  --campaign-id issue_1344_stress_primary_final \
  --log-level INFO
```

## Follow-Up Boundary

The primary-row protocol is now proven. All-runnable planner expansion, cross-kinematics expansion,
or paper-facing claim generation should be handled as follow-up work after deciding whether the
`amv_coverage_status=warn` and `snqi_contract_status=fail` boundaries are acceptable for the next
scope.

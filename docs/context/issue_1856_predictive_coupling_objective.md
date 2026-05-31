# Issue #1856 Predictive-v2 Coupling Objective

Related issues:

- Issue #1856: <https://github.com/ll7/robot_sf_ll7/issues/1856>
- Parent issue #1490: <https://github.com/ll7/robot_sf_ll7/issues/1490>
- Negative audit #1543: <https://github.com/ll7/robot_sf_ll7/issues/1543>
- Prior prerequisite #1427 / PR #1480: <https://github.com/ll7/robot_sf_ll7/pull/1480>

## Decision

Use a local, fail-closed closed-loop gate before spending on any renewed predictive-v2 Slurm
training matrix.

The revised hypothesis is planner-side, not model-row-side: keep the same checkpoint under test,
then require a phase-coupled planner objective to improve closed-loop success relative to the
`baseline_like` planner row before treating more ego/obstacle conditioning work as justified.

This note is proposal and preflight guidance only. It is not benchmark evidence and does not
promote predictive-v2.

## Evidence Boundary

The durable #1543 audit found the #1427 obstacle-feature prerequisite negative:

- baseline predictive success: `0.1304`
- obstacle-feature predictive success: `0.1014`
- hard-seed success: `0.0000` for both variants
- mean min distance: `2.1931` baseline vs `2.2105` obstacle-feature final eval
- best planner-grid row changed from `risk_aware_adaptive` to `baseline_like`

That pattern supports prediction-to-control coupling failure more than predictor-quality failure.
Do not use ADE/FDE, forecast loss, or clearance improvements alone as a go signal.

## Bounded Planner-Side Hypothesis

Config:
`configs/benchmarks/predictive_sweep_planner_grid_v2_coupling_gate.yaml`

The revised row is `phase_coupled_sequence_gate`. It touches planner-side objective and search
surfaces only:

- enables short sequence search so the objective can score commit/yield/recover behavior over more
  than one immediate command;
- increases progress-risk coupling while keeping hard clearance and TTC terms active;
- enables phase logic so clear states reward forward commitment, close states penalize unsafe
  forward motion, and clear-but-stalled states are rejected;
- keeps checkpoint/model selection out of the grid so this remains a coupling test, not another
  predictor-row expansion.

Hypothesis: if predictive-v2 is worth expanding, this row should convert at least some forecast or
clearance signal into higher closed-loop success versus `baseline_like` on the same checkpoint and
same local seed surface.

## Local Gate

Use the predictive success campaign with the #1856 grid and the optional closed-loop gate:

```bash
uv run python scripts/validation/run_predictive_success_campaign.py \
  --checkpoints <predictive_model.pt> \
  --planner-grid configs/benchmarks/predictive_sweep_planner_grid_v2_coupling_gate.yaml \
  --horizon 80 \
  --workers 1 \
  --bootstrap-samples 200 \
  --closed-loop-gate-baseline-variant baseline_like \
  --closed-loop-gate-min-global-success-delta 0.02 \
  --closed-loop-gate-min-hard-success-delta 0.0 \
  --closed-loop-gate-max-min-distance-regression 0.10 \
  --output-dir output/tmp/predictive_planner/campaigns/issue1856_coupling_gate
```

Gate interpretation:

- pass: `phase_coupled_sequence_gate` or another revised row improves global closed-loop success by
  at least `0.02` over `baseline_like` without losing more than `0.10 m` global mean min distance;
- fail: the best row is the baseline, only improves clearance, loses hard-suite success, or gives
  back more clearance than the configured bound;
- blocked: no local checkpoint is available; do not substitute forecast metrics for this gate.

The exact thresholds are deliberately small because this is a local smoke/preflight, not promotion
evidence. Passing the gate only justifies the next child issue; it does not justify predictive-v2
claims without same-seed closed-loop artifacts.

## Child Issue Routing

REST preflight on 2026-05-31 found the four-way expansion issues already open and blocked:

- Issue #1505 `state:blocked`: keep blocked until this local gate has a passing result or the data-row
  preflight is re-scoped behind the revised objective.
- Issue #1506 `state:blocked`: keep blocked; do not launch the old four-way Slurm matrix.
- Issue #1507 `state:blocked`: can be re-scoped to analyze forecast-to-control transfer only after the
  Issue #1856 gate records a concrete pass/fail local campaign.

## Proposed #1490 Update Text

```md
Issue #1856 proposes the next predictive-v2 step: keep the old four-way matrix blocked and first
test a planner-side coupling gate.

The revised hypothesis is not "train more rows." It is: using the same checkpoint, a
phase-coupled sequence-search planner objective should improve closed-loop success over the
`baseline_like` planner row before we spend on ego/obstacle four-way training.

Local preflight:

`configs/benchmarks/predictive_sweep_planner_grid_v2_coupling_gate.yaml`

Run `scripts/validation/run_predictive_success_campaign.py` with
`--closed-loop-gate-baseline-variant baseline_like`,
`--closed-loop-gate-min-global-success-delta 0.02`, and
`--closed-loop-gate-max-min-distance-regression 0.10`.

If the gate fails, keep #1505/#1506/#1507 blocked and revise the planner coupling/objective again.
If it passes, route the next child issue to preserve the local campaign summary and only then decide
whether a same-seed Slurm row is worth launching.

Boundary: this is proposal/preflight evidence only. Predictive-v2 still needs same-seed
closed-loop improvement with durable artifacts before any promotion or paper-facing claim.
```

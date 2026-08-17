# Predictive planner baseline diagnostic

This page documents the bounded implementation slice for [issue #7319](https://github.com/ll7/robot_sf_ll7/issues/7319).
It adds a predictive Gaussian human-cost option to the existing Model Predictive
Path Integral (MPPI) planner and records a same-seed smoke composition of the
existing MPPI reference, the adapted cost, and Robot SF's nonlinear model
predictive control (NMPC) plus control-barrier-function (CBF) filter.

The result is diagnostic-only. It does not run the simulator, establish safety,
rank planners, reproduce either cited source method, or provide evidence for
autonomous-micromobility transfer.

## Canonical smoke

```bash
uv run python scripts/validation/run_predictive_baseline_diagnostic.py \
  --config configs/benchmarks/issue_7319_predictive_baselines_smoke.yaml \
  --output output/diagnostics/issue_7319_predictive_baselines.json
```

The report is validated against
`robot_sf/benchmark/schemas/predictive_baseline_diagnostic.v1.json`. It records
the resolved configuration digest, fixed scenario/seed identity, action and
observation contracts, method-card digests, deterministic repeat commands, and
unavailable simulator metrics. The output directory is temporary and ignored.

## Implemented method boundary

The opt-in `PredictiveGaussianHumanCost` advances each pedestrian by the
observed velocity, aligns a Gaussian's longitudinal axis with that velocity,
and increases longitudinal spread with speed and prediction time. The exposed
formula is an explicit Robot SF adaptation of the predictive Gaussian
interaction-field idea described in [Mundane, 2026](https://arxiv.org/abs/2608.08323),
not a claim of exact parameter or implementation parity. The default MPPI
configuration keeps the cost disabled, and malformed nested configuration fails
closed.

The constrained lane composes the existing `NMPCSocialPlannerAdapter` with the
existing collision-cone `CbfSafetyFilterPlannerWrapper`. It copies no external
code. The cited multi-room MPC paper uses a sensor and map stack that is not
identical to Robot SF; its approach is therefore a design reference, not a
direct benchmark comparator. See [Gravina et al., 2026](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1812386/full).

## What remains before a campaign

The real-manifest follow-up is tracked in [issue #7340](https://github.com/ll7/robot_sf_ll7/issues/7340).
It must provide identical scenario/seed inputs, a declared uniform/random
baseline, simulator-grounded outcomes, uncertainty and failure traces, and
domain-aware approval before any larger comparison is treated as benchmark
evidence. Missing metrics remain unavailable rather than being filled with
zeros; fallback or degraded rows cannot count as success.

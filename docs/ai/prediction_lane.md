# Prediction Research Lane

This note is the agent-facing routing map for forecast and prediction work under issue #2835.
It keeps the lane discoverable without turning diagnostic forecast results into benchmark,
safety, dissertation, or paper-facing claims.

## Current Boundary

Prediction work in this repository is a staged research lane for local navigation, not a default
planner upgrade. Forecast artifacts, metrics, risk channels, and learned predictors remain gated by
provenance, same-seed closed-loop evidence, and explicit fallback/degraded status.

Use this vocabulary:

- `diagnostic-only`: executable or analytical evidence that checks a mechanism, fixture, schema, or
  proxy metric but does not support a benchmark or safety claim.
- `benchmark-eligible`: durable, versioned evidence with valid denominators, scenario/seed lineage,
  observation tier, fallback/degraded status, and a reproducible command.
- `paper-facing`: benchmark-eligible evidence that also passes the relevant paper/release contract
  and has conservative claim wording.
- `blocked`: work is unavailable because a required artifact, split, metric, gate, environment, or
  maintainer decision is missing.
- `revise`: a mechanism ran but did not justify the next stronger step.
- `continue`: the evidence supports the next scoped step, not automatic promotion.
- `stop`: the evidence argues against pursuing the path until a different hypothesis appears.

Fallback, degraded, oracle-only, proxy-only, stale, unavailable, or denominator-invalid rows must
stay explicit limitations.

## Dependency Order

Follow this order unless a later issue has an explicitly narrower diagnostic purpose:

1. Define and validate the artifact contract.
   `ForecastBatch.v1` is tracked by #2836 and summarized in
   `docs/context/issue_2836_forecast_batch_schema.md`. It records predictor identity,
   observation tier, coordinate frame, `dt_s`, horizons, scenario id, seed, fallback/degraded
   status, actor ids, masks, feature schema, optional samples/modes/occupancy, and oracle-state
   boundaries.
2. Build durable motion-rich fixtures and baseline ladder.
   Use #2774 for non-corridor trace fixtures, #2758 for signal/goal-aware baselines, and #2781 for
   interaction-aware comparison. Existing results are diagnostic-only.
3. Add observation-tier adapters and dataset/split provenance.
   Issues #2838 and #2839 should prevent oracle, perfect-state, perception-like, and
   planner-observation inputs from being mixed silently.
4. Add metric and calibration reports.
   Issues #2840 and #2846 generalize probabilistic forecast metrics and separate actor-class
   denominators. Issue #2837 compares horizon and output timestep tradeoffs, #2841 handles
   reliability/calibration, and #2842 explores conformal or reachable-set uncertainty.
5. Test planner coupling before training-heavy expansion.
   Issue #2843 defines the closed-loop coupling gate. The latest gate recommendation is `revise`,
   not `continue`, because diagnostic forecast improvement did not translate into a passing
   same-seed closed-loop gate.
6. Add local-policy risk-channel diagnostics only as opt-in behavior.
   Issue #2759 connects forecast risk to `PolicyStackV1` scoring with
   `forecast_risk_weight=0.0` by default. The evidence bundle is diagnostic-only and not a safety
   or live benchmark claim.
7. Run transferability and learned-model work only after the prerequisites are ready.
   Issue #2847 should stratify noise, latency, dropout, occlusion, map family, density,
   pedestrian-model shift, actor type, and observation tier. #2844 and #2845 stay blocked until
   schema, data, metrics, and closed-loop gate evidence justify learned or heavy-model work.

## Active Issue Map

| Issue | State on 2026-06-15 | Role | Routing note |
| --- | --- | --- | --- |
| #2835 | open | Epic | Parent coordination surface for the lane. |
| #2836 | closed | Forecast artifact schema | `ForecastBatch.v1`; prerequisite for durable forecast artifacts. |
| #2774 | closed | Motion-rich fixtures | Diagnostic trace-family expansion beyond corridor fixtures. |
| #2758 | closed | Semantic baselines | Signal/goal-aware CV variants; diagnostic-only. |
| #2781 | closed | Interaction-aware baseline | Mixed result: likelihood proxy improved while 1s point accuracy worsened. |
| #2843 | closed | Closed-loop coupling gate | Latest recommendation: `revise` before learned predictor training. |
| #2759 | closed | Forecast risk scoring | Opt-in diagnostic risk channel for `PolicyStackV1`; default remains off. |
| #2727 | closed | Fast dynamic actor fixture | Enables future actor-class forecast denominator work. |
| #2840 | closed | Probabilistic forecast metrics | Metric surface for deterministic/probabilistic forecast comparison. |
| #2846 | closed | Fast dynamic actor forecast metrics | Separates pedestrian and fast-agent denominators. |
| #1490 | open, blocked | Predictive planner v2 comparison | Do not repeat the old four-way expansion until revised gate evidence exists. |
| #2837 | open | Horizon and timestep ablation | Analysis-only report for forecast horizon/output-step presets. |
| #2838 | open | Observation-level adapters | Required before deployable/oracle observation tiers can be compared safely. |
| #2839 | open | Dataset recorder and split manifest | Required before learned predictor training or durable split comparisons. |
| #2841 | open | Calibration and reliability | Metric/report surface for probabilistic forecast quality. |
| #2842 | open | Conformal/reachable-set pilot | Diagnostic uncertainty pilot; no planner/safety claim by itself. |
| #2844 | open, blocked | Learned probabilistic graph predictor | Blocked until schema/data/metrics/coupling prerequisites unblock. |
| #2845 | open, blocked | Transformer/diffusion study | Blocked analysis until lighter prerequisites and bounded data exist. |
| #2847 | open | Transferability stress matrix | Should run after observation tier, metrics, and fixture/split surfaces are usable. |
| #2848 | open | This routing doc | Keep this map current when lane gates or issue states change materially. |

## Gate Conditions

Do not start learned predictor training or heavy-model evaluation unless all of these are true:

- Forecast artifacts use `ForecastBatch.v1` or a documented compatible adapter.
- Input provenance names observation tier, feature schema, frame, history, masks, scenario id, and
  seed.
- Dataset/split provenance prevents scenario/seed leakage and records durable source traces.
- Baseline ladder includes constant-velocity plus the available semantic/interaction baselines.
- Forecast metrics include horizon, `dt_s`, actor-class denominator, fallback/degraded status, and
  calibration or uncertainty fields when applicable.
- Closed-loop coupling gate reports `continue` on same-seed evidence with mechanism activation,
  success/progress non-regression, false-positive accounting, and runtime caveats.

If any condition is missing, route the follow-up as `blocked`, `diagnostic-only`, or `revise`
instead of treating it as benchmark evidence.

## Validation Commands

Use the smallest command that matches the changed surface:

- Schema and artifact contract:
  `uv run pytest tests/benchmark/test_forecast_batch.py`
- Forecast-risk scoring diagnostic:
  `uv run pytest tests/planner/test_policy_stack_v1.py tests/validation/test_forecast_risk_policy_stack.py`
  and `uv run python scripts/validation/validate_forecast_risk_policy_stack.py --out-dir <dir>`
- Closed-loop coupling gate:
  `uv run pytest tests/validation/test_closed_loop_forecast_coupling_gate.py` and
  `uv run python scripts/validation/validate_closed_loop_forecast_coupling_gate.py --out-dir <dir>`
- Learned-prediction readiness:
  `uv run pytest tests/validation/test_validate_learned_prediction_readiness.py` and
  `uv run python scripts/validation/validate_learned_prediction_readiness.py`

For docs-only updates to this lane, validate referenced paths and issue numbers, inspect the diff
for overclaiming, and run a cheap docs/path check if one is available. Full PR readiness is optional
for docs-only changes unless the PR touches code, schema, metrics, or validation scripts.

## Non-Goals

- Do not present diagnostic forecast comparisons as paper-facing results.
- Do not promote oracle-state or perfect-state rows as deployable comparisons.
- Do not make forecast-risk scoring default behavior without a separate benchmark contract.
- Do not train learned predictors from worktree-local `output/` data without a durable manifest or
  artifact registry entry.
- Do not hide unavailable transfer-matrix cells by dropping rows silently.

# Issue #3558 — `stream_gap` uncertainty-gate threshold calibration sweep

**Status:** the deferred threshold sweep is built and run; it feeds the merged
`calibrate_stream_gap_gate` decision layer. Diagnostic-tier result: a **safe operating region
exists**, but only at gate thresholds permissive enough to *retain* the degraded agent — every
threshold that actually drops it (including the **0.5 production default**) is `less_safe`,
confirming the #3471 finding.

## Claim boundary (read first)

- **What landed:** (1) `scripts/validation/run_stream_gap_gate_threshold_sweep_issue_3558.py`, which
  reuses the #3471 episode harness (extended with optional per-run gate-threshold overrides) to roll
  the `uncertain_dropped` mode across a grid of `uncertainty_min_existence_probability` thresholds and
  feed the per-setting safety aggregates + conservative-retention baseline into the pure
  `robot_sf.planner.stream_gap_gate_calibration.calibrate_stream_gap_gate` decision layer.
- **Evidence tier:** `diagnostic` — the real `stream_gap` planner + `ScenarioBelief` gate on one
  synthetic controlled crossing scenario. **Not** the full benchmark environment, **not** paper-grade,
  no trained-policy or traffic-realism claim.
- **Active axis:** the #3471 scenario degrades only the corridor agent's *existence* confidence
  (to 0.2), so the sweep exercises `uncertainty_min_existence_probability` only. The other three gate
  thresholds are recorded and passed through but are not exercised by this single-source degradation.
- **Out of scope:** changing the production default (a separate decision); benchmark-env promotion
  (the #3471/#3556 follow-up).

## Result (`report.json`, 12 seeds 101–112, `max_steps=120`)

The gate drops an agent when `existence < min_existence_probability` (strict); the degraded agent
sits at existence 0.2.

| existence threshold | drops agent? | classification | unsafe-commit rate | collision rate | worst sep (m) |
| --- | --- | --- | --- | --- | --- |
| baseline (retention, gate off) | — | — | 0.833 | 0.417 | −0.169 |
| 0.1 | no | `at_least_as_safe` | 0.833 | 0.417 | −0.169 |
| 0.2 | no | `at_least_as_safe` | 0.833 | 0.417 | −0.169 |
| 0.3 | yes | `less_safe` | 1.000 | 0.917 | −0.621 |
| 0.5 (**default**) | yes | `less_safe` | 1.000 | 0.917 | −0.621 |
| 0.7 | yes | `less_safe` | 1.000 | 0.917 | −0.621 |

- **Conclusion:** `safe_region_exists`; safe region = `{0.1, 0.2}`; recommended setting = `0.1`.
- **Interpretation:** the only gate settings that are at least as safe as conservative retention are
  those permissive enough *not to drop* the (correctly present) degraded corridor agent. Every setting
  that actually drops it is strictly less safe — so on this scenario the existence gate is "safe"
  exactly when it is inert, and the **0.5 production default is unsafe**, reproducing #3471. This does
  not recommend a *new* dropping default; it confirms conservative retention should stand for
  safety-relevant use at this scenario's degradation level.
- **Scenario note:** even the retention baseline collides in ~42% of seeds — this controlled crossing
  is genuinely contested, so the signal is the *relative* contrast (dropping raises collisions from
  0.417 → 0.917), not an absolute safety claim.

## Reproduce

```bash
uv run python scripts/validation/run_stream_gap_gate_threshold_sweep_issue_3558.py \
  --output-json output/issue_3558_gate_threshold_sweep.json
```

Tests: `tests/validation/test_run_stream_gap_gate_threshold_sweep_issue_3558.py` (and the unchanged
#3471 harness tests, which still pass with the additive `gate_thresholds` parameter).

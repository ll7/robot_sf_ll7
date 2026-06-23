# Issue #2444 AMMV Divergence Classification Evidence

Date: 2026-06-23

- claim_boundary: `diagnostic_only`
- evidence_tier: `stress` (a real, more-sensitive diagnostic slice was executed and produced
  nonzero divergence)
- result_classification: `nonzero_divergence_found`
- git HEAD: `fedee58afe8c5834f4e6ccb1179cdc00a8354606`
- deterministic seeds: `42` (close-front probe), `3202` (anticipatory crossing probe)

## Goal

Issue #2444 asked us to find or generate an AMMV / default Social Force trace pair with **nonzero
mechanism divergence**, OR honestly classify the AMMV mechanism as inactive under the tested
settings.

## Issue #2434 Negative Baseline (Cited)

`docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/` recorded
`max_per_frame_abs_delta = 0.0` and `max_episode_metric_abs_delta = 0.0` across 15 matched
default-vs-AMMV episode pairs. Those pairs ran through the **differential-drive benchmark adapter**
and are therefore rendering fixtures, NOT behavioral-difference evidence.

## More-Sensitive Slice Run Here

This bundle ran a deterministic, **direct `SocialForcePlanner` mechanism probe** that bypasses the
benchmark adapter and exposes the AMMV force term directly (reusing
`scripts/tools/run_ammv_social_force_pair_diagnostic.py::_run_mechanism_probe`). The control arm is
the **same AMMV-aware config with `ammv_aware_enabled` toggled off** (all other parameters
identical), so the paired delta isolates the AMMV interaction term rather than conflating it with
unrelated config differences. Two probes were exercised:

| probe (scenario_id) | seed | frames | max_ammv_force_delta (N) | max_robot_state_delta | mechanism_activation_observed | outcome_changed |
| --- | --- | --- | --- | --- | --- | --- |
| `issue_2168_close_front_agent_probe` | 42 | 20 | 2.641802 | 0.214876 | true | true |
| `issue_3202_anticipatory_crossing_probe` | 3202 | 24 | 2.642146 | 0.783477 | true | true |

`max_pedestrian_state_delta` is `null` by construction: the direct robot-planner probe does not own
pedestrian trajectories.

## Result

`nonzero_divergence_found`. On the direct-planner surface the AMMV term **activates** (force
magnitude ~2.64 N, intrusion count >= 1) and, with **only `ammv_aware_enabled` toggled** (everything
else identical), produces a same-seed behavioral difference (nonzero robot-state and selected-action
deltas of 0.21 m and 0.78 m). Because the comparison isolates the AMMV term, the delta is
attributable to AMMV and not to incidental config differences. The #2434 zero-delta result is
therefore an **adapter-mode artifact** (the differential-drive benchmark adapter washes out the
term), not evidence that the AMMV mechanism is globally inactive.

## Zero-Divergence Guard (Acceptance Criterion)

`classify_divergence` / `build_selection_block` only treat a pair as behavioral evidence when
`max_ammv_force_delta > 0` AND a paired delta is strictly nonzero. An identical (zero-force,
zero-delta) pair -- exactly the #2434 case -- is rejected and labelled
`ammv_inactive_under_tested_settings`. This stops future #2159 / #2227 mechanism panels from using
identical traces as behavioral evidence. The guard is covered by
`tests/analysis/test_ammv_divergence_classification_issue_2444.py::test_zero_divergence_guard_rejects_identical_pair_as_behavioral_evidence`.

## Reproduce

```bash
uv run python scripts/analysis/run_ammv_divergence_classification_issue_2444.py
python -m json.tool output/issue_2444_ammv_divergence/ammv_divergence_classification.json
uv run python -m pytest tests/ -k "ammv or divergence or 2444" -q
```

## Limitations

- Diagnostic-only: a nonzero divergence proves the AMMV term is active and behaviorally
  non-identical **at the planner level**, NOT a benchmark advantage.
- Direct robot-planner probe; pedestrian dynamics are simulator-owned and not measured here.
- Two-probe slice; not a benchmark matrix or parameter sweep.

# Issue #3471 — Episode-level ScenarioBelief uncertainty → planner safety

**Evidence status:** `diagnostic` — controlled-scenario evidence, **not** the full benchmark
environment and **not** paper-grade.

## Claim boundary (read first)

- **Claim:** in a controlled crossing scenario with the real `stream_gap` planner, **dropping**
  uncertain agents from the planner's reasoning produces *more unsafe commitment* than **retaining**
  them under conservative (gate-off) handling.
- **Not claimed:** absolute safety rates, trained-policy or perception-model behavior, traffic-light
  realism, or any full-benchmark result. The scenario is a deliberately hard, single-crossing
  diagnostic; the result is the **relative contrast** between belief modes, not an absolute number.
- **Uncertainty:** ~0.6. The dominant caveat is that the oracle baseline itself collides ~42% of the
  time (the crossing is hard and the wait-for-gap planner + coarse unicycle integration do not fully
  avoid contact). The contrast is robust *given* that limitation.

## What this is

PR #3450 (closing #2546) was an explicit **single synthetic step** diagnostic and named this
episode-level follow-up as the next bounded step. This rolls a controlled crossing scenario over time
with the real `StreamGapPlannerAdapter` driving the robot, under three belief modes that share an
identical ground truth and differ only in what the planner is allowed to trust:

| Mode | Belief | Planner uncertainty gate |
| --- | --- | --- |
| `oracle` | certain | off (reacts to every agent) |
| `uncertain_retained` | corridor agent existence-degraded | **off** → fail-closed keeps it (conservative) |
| `uncertain_dropped` | same degraded belief | **on** → drops the low-confidence agent |

Safety is scored against the **true** simulated pedestrian position, which the planner never sees in
the dropped mode.

## Result (12 seeds, default config)

| Mode | collision rate | worst min-sep (m) | mean min-sep (m) | unsafe-commit steps |
| --- | --- | --- | --- | --- |
| `oracle` | 0.42 | -0.17 | +0.34 | 45 |
| `uncertain_retained` | 0.42 | -0.17 | +0.34 | 45 |
| `uncertain_dropped` | **0.92** | **-0.62** | **-0.35** | **246** |

**Decision: `revise`.** Two findings:

1. **Representational vs actual safety (clean separation).** `uncertain_retained` is *identical* to
   `oracle` — merely *having* degraded uncertainty changes nothing when the planner keeps the agent.
   So the effect is not representational.
2. **Dropping is the harmful action.** Turning the planner's uncertainty gate on (dropping the
   low-confidence corridor agent) raises collisions 0.42 → 0.92 and unsafe-commit steps 45 → 246. The
   uncertainty-dropping default should be **revised/blocked** for safety-relevant use; conservative
   retention should be preferred until a calibrated perception model justifies dropping.

## Reproduce

```bash
uv run python scripts/validation/run_scenario_belief_episode_safety_issue_3471.py \
  --config configs/benchmarks/scenario_belief_episode_safety_issue_3471.yaml \
  --output-json docs/context/evidence/issue_3471_scenario_belief_episode_safety_2026-06-24/report.json
uv run python -m pytest tests/validation/test_run_scenario_belief_episode_safety_issue_3471.py -q
```

- Config / predeclared matrix: `configs/benchmarks/scenario_belief_episode_safety_issue_3471.yaml`
- Runner: `scripts/validation/run_scenario_belief_episode_safety_issue_3471.py`
- Report: `report.json` (this directory). `runtime_sec` fields are wall-clock and non-reproducible;
  all safety metrics are deterministic per seed.

## Seed budget

12 seeds (101–112), perturbing the crosser's start phase and speed. This is a **bounded diagnostic
matrix**, not a paper-grade seed-sufficiency budget; the retained-vs-dropped contrast is consistent
in direction across the matrix. A paper-facing claim would require the repository seed-sufficiency
procedure and the full benchmark environment.

## Limitations / next steps

- Controlled scripted scenario + coarse unicycle integration, not the full benchmark env/runner.
- Oracle baseline collides ~42%; a softer crossing or a better-tuned planner gate would give a
  cleaner absolute baseline (tuning did not trivially improve it — see PR discussion).
- Single uncertainty source (existence-degradation). Visibility/occlusion/tracking-noise sources
  exist in `scenario_belief.py` and are natural extensions.

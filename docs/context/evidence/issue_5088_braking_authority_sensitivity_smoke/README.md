# Issue #5088 Braking-Authority Sensitivity Smoke

- Status: `signal_activated`
- Evidence tier: `targeted-smoke`
- Claim boundary: targeted-smoke diagnostic evidence only; this comparison does not calibrate braking, establish safety, or support paper/dissertation claims
- Run commit: `e4e5d8b6fcd92c3688f8c05a5d1a64addfad278f`
- Config: `configs/benchmarks/issue_5088_braking_authority_sensitivity_smoke.yaml`
- Scenario: `configs/scenarios/classic_interactions.yaml` (`classic_cross_trap_high`)
- Seeds: `101, 102, 103`
- Planner: `social_force`

## Result

Metric-sensitivity signal activated: `True`.
Activated metrics: `min_clearance, time_to_collision_min`.
Changed-seed counts: `near_misses=0/3`, `min_clearance=1/3`, `time_to_collision_min=1/2`.

| Arm | Braking (m/s^2) | Stop distance at max speed (m) | Mode | Readiness | Near misses | Min clearance (m) | Min TTC (s) | TTC valid |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `weak_braking` | 0.250 | 8.000 | adapter | adapter | 1.666667 | 0.294697 | 4.011251 | 2/3 |
| `strong_braking` | 2.000 | 1.000 | adapter | adapter | 1.666667 | 0.365671 | 4.026497 | 2/3 |

## Reproduce

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python scripts/tools/run_braking_authority_sensitivity_smoke.py --config configs/benchmarks/issue_5088_braking_authority_sensitivity_smoke.yaml --output '<fresh-artifact-dir>'
```

Replace `<fresh-artifact-dir>` with an empty local scratch directory. Raw episode JSONL remains disposable and untracked; `report.json` plus this README are the tracked compact evidence. Fallback/degraded execution is excluded, and this smoke makes no calibrated safety or paper-facing claim.

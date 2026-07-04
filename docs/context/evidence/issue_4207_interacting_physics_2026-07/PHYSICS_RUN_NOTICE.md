# Physics-Run Provenance Notice

This evidence packet **is** the product of an actual CPU robot/pedestrian physics simulation. It is
the empirical, diagnostic-tier counterpart to the synthetic tooling-validation packet at
`docs/context/evidence/issue_4207_interacting_smoke_2026-07/` (issue #4327, following #4207 / #4315
/ #4308).

## How it was produced

Run from the repository root (CPU-only, no Slurm/GPU, no retraining):

```bash
SDL_VIDEODRIVER=dummy MPLBACKEND=Agg DISPLAY= \
  python scripts/benchmark/run_certification_transfer_issue_4207.py \
    --config configs/benchmarks/issue_4207_interacting_physics_probe.yaml \
    --gate-spec configs/benchmarks/release_gates/issue_4207_interacting_smoke_gates.yaml \
    --output-dir docs/context/evidence/issue_4207_interacting_physics_2026-07 \
    --generated-at 2026-07-04T00:00:00+00:00
```

- Probe config: `configs/benchmarks/issue_4207_interacting_physics_probe.yaml` (physics; `horizon: 400`)
- Gate spec: `configs/benchmarks/release_gates/issue_4207_interacting_smoke_gates.yaml` (reused from #4315)
- Scenario matrix: `configs/scenarios/single/issue_4207_interacting_smoke.yaml` (reused from #4315)
- Matrix: 4 arms × 2 pedestrian models × 3 seeds `[111, 112, 113]` × 1 scenario = 24 episodes; ~25 s wall on CPU.

The recorded metric values in `summary.json` / the CSVs are **measurements** from these episodes,
not fixture design choices.

## What it proves (diagnostic tier)

- **Near-field contact is real, from simulation.** The route-following `goal` baseline reaches the
  blind corner and makes near-field contact with the crossing pedestrian:
  packet-level `max_robot_ped_within_5m_frac = 0.10558655435990806`,
  `min_clearance_m = -0.024278621748445417`, and `physics_near_field_confirmed = true`. This
  confirms the issue #4207 acceptance criterion — nonzero near-field interaction
  (`robot_ped_within_5m_frac > 0`) from physics, not from fixture design.
  `model_sensitivity_exercised = true` is therefore backed by a real interacting cell.
- **The interacting geometry needs a long-enough horizon.** At the smoke config's `horizon: 60`
  (6 s) every real cell stayed `non_interacting` (robot ~24 m away): the robot cannot traverse the
  L-route in 6 s. `horizon: 400` (matching the scenario's `max_episode_steps`) is required for
  contact. This is the concrete geometry/spawn finding the #4315 synthetic slice deferred.

## What it does NOT prove

- **No trained-planner comparison.** No policy checkpoint is attached (`algo_config` omitted), so the
  learned arms (`ppo`, `guarded_ppo`, `prediction_planner`) run in goal/sampling **fallback** mode.
  Under fallback, their direct-to-goal navigation never traverses the corridor: all three stay
  `non_interacting` (~20–24 m from the pedestrian). Their `pass`/`fail` gate statuses are therefore
  vacuous with respect to certification and must not be read as planner safety. Attaching real arm
  checkpoints and re-running is the tracked follow-up.
- **No model-transfer flip reproduces under physics.** For every arm the `social_force_default` and
  `hsfm_total_force_v1` cells produced byte-identical metrics, so `flip_cases = 0`. The synthetic
  fixture's fabricated `ppo` `fragile_pass_to_fail` flip does **not** appear in the real run. Model
  sensitivity is *exercised* (the near field is entered) but did not produce a gate-decision
  divergence here.
- No deployment, real-world safety, benchmark-strength, or paper/dissertation claim follows. The
  gates are provisional reporting thresholds, not certification approval.

## Relationship to the synthetic packet

`docs/context/evidence/issue_4207_interacting_smoke_2026-07/` remains **tooling-validation only**
(it exercises the interaction-validity guard's positive path with a synthetic fixture). This packet
supersedes it as the *empirical* diagnostic reference for the certification-transfer surface; the
synthetic packet is unchanged.

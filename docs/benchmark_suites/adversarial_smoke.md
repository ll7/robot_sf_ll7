# Adversarial Smoke Suite

```yaml
suite_id: adversarial_smoke
benchmark_track: development_stress_test
status: runnable_with_surface_specific_limits
```

## Purpose

Run bounded adversarial route/search probes that produce explicit candidate status, failure, and
invalid-candidate accounting. This suite is for development stress and falsification-oriented
triage, not nominal benchmark aggregation.

## Scenarios And Seeds

Two current anchors exist:

- Route/start-state generation prototype:
  - Config: `configs/adversarial_routes/default.yaml`
  - Source scenario file: `configs/scenarios/classic_interactions.yaml`
  - Scenario ID: `classic_head_on_corridor_low`
  - Seed: `123`
  - Trial count: `20`
- Crossing/TTC smoke evidence:
  - Search space: `configs/adversarial/crossing_ttc_space.yaml`
  - Launcher: `SLURM/Auxme/adversarial_smoke_1501.sl`
  - Seed: `42`
  - Budget: `32` candidates per sampler
  - Samplers: `random`, `optuna`

## Eligible Planners

The route-generation prototype uses `ClassicGlobalPlanner` / `theta_star_v2` for route feasibility.
The crossing/TTC smoke mapped the frozen `classic_global_theta_star` row to the current `goal`
benchmark policy and ran `orca` directly; guided route search was recorded as `not_available`.

## Metrics

Valid trials, failed trials, invalid candidates, simulation errors, not-available rows, objective
score, objective components, valid non-failure count, valid failure count, archived failure count,
cluster count, sampler comparison, and row-status summary.

## Canonical Commands

Route/start-state generation prototype:

```bash
uv run python scripts/tools/generate_adversarial_routes.py \
  --config configs/adversarial_routes/default.yaml
```

Crossing/TTC SLURM smoke launcher:

```bash
ADVERSARIAL_SMOKE_LABEL=manual-crossing-ttc \
ADVERSARIAL_SMOKE_BUDGET=32 \
ADVERSARIAL_SMOKE_SEED=42 \
scripts/dev/sbatch_use_max_time.sh --time 04:00:00 --partition a30 --qos a30-gpu \
  SLURM/Auxme/adversarial_smoke_1501.sl
```

## Expected Runtime

The route/start-state prototype is a local smoke. The crossing/TTC smoke is a SLURM/Auxme run; the
recorded Issue #1501 job completed in about two minutes after scheduling, but queue time and
partition availability vary.

## Claim Boundary

Adversarial smoke is development stress evidence. It does not certify a scenario, prove planner
robustness, or create paper-facing benchmark rows. Generated cases need explicit promotion review
before entering durable scenario configs or aggregate benchmarks.

## Caveats

Invalid candidates, simulation errors, not-available rows, fallback rows, and degraded rows do not
count as success evidence. High invalid-candidate counts are generator-quality signals, not
successful stress evidence.

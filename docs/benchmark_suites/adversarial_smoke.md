# Adversarial Smoke Suite

```yaml
suite_id: adversarial_smoke
benchmark_track: development_stress_test
status: runnable_with_surface_specific_limits
```

## Purpose

Run the existing adversarial pedestrian smoke lanes with explicit seeds,
candidate status, invalid-candidate accounting, and artifact boundaries. This
suite is falsification-oriented development triage, not nominal benchmark
aggregation.

Issue #4360 uses this page as the current inventory of dispatchable adversarial
pedestrian machinery. The separately implemented bounded residual-control
policy interface is documented in
[the reactive residual-adversary design note](../context/issue_4360_reactive_adversary_design.md).
It is a capability-only runtime slice, not a manifest/search smoke lane, and it
does not add a stress-case metric or any benchmark, planner-ranking, safety, or
paper-facing claim.

## Current Hooks

| Surface | Owner | What it covers | Claim boundary |
| --- | --- | --- | --- |
| Manifest generation | `scripts/tools/generate_adversarial_scenario_manifests.py`, `robot_sf/adversarial/scenario_manifest.py` | Deterministic `adversarial_scenario_manifest.v1` candidates from `SearchSpaceConfig` | Generated-only validator evidence; no planner weakness claim. |
| Manifest planner smoke | `scripts/tools/run_adversarial_manifest_smoke.py`, `robot_sf/adversarial/materialize.py` | Materializes valid manifests and runs tiny planner smoke through `run_batch` | Smoke-only development stress evidence. |
| Search runner API | `robot_sf/adversarial/search.py`, `robot_sf/adversarial/config.py` | Python API for bounded random/coordinate search with evaluator injection | Runner plumbing evidence; generated bundles under `output/` are not durable claims. |
| Route/start-state generation | `scripts/tools/generate_adversarial_routes.py`, `robot_sf/nav/adversarial_route_generation.py` | Head-on corridor route optimization prototype | Route-generation smoke, not a benchmark row. |
| Failure archive | `scripts/tools/curate_adversarial_failure_archive.py`, `robot_sf/adversarial/archive.py` | Compact `adversarial_failure_archive.v1` replay pointers for selected failures | Archive curation only; raw bundles stay local unless separately promoted. |
| Rerun readiness | `scripts/validation/check_failure_archive_rerun_readiness.py`, `scripts/adversarial/produce_rerun_closure_packet.py` | Fail-closed checks for disjoint certified archive reruns | Readiness or diagnostic blocker packet, not model-quality evidence. |
| Residual adversary smoke | `robot_sf/ped_npc/residual_adversary.py`, `tests/adversarial/test_residual_adversary.py`, `tests/sim/test_residual_adversary_wiring.py` | Deterministic bounded residual-control reactive adversary: bound pipeline, macro-action cadence, opt-in gating, perturb-not-replace, simulator wiring | Capability-only smoke evidence; no benchmark, metric, or safety claim. |

## Scenarios And Seeds

The reproducibility manifest for this suite is
`configs/adversarial/issue_4360_dispatchable_repro_seeds.yaml`.

| Lane | Scenario surface | Seed | Budget | Entrypoint |
| --- | --- | --- | --- | --- |
| `crossing_ttc_manifest_generation` | `configs/scenarios/templates/crossing_ttc.yaml` with `configs/adversarial/crossing_ttc_space.yaml` | `42` | `16` generated manifests | `scripts/tools/generate_adversarial_scenario_manifests.py` |
| `crossing_ttc_planner_smoke` | same crossing/TTC template and search space | `42` | `4` manifests for tiny smoke | `scripts/tools/run_adversarial_manifest_smoke.py` |
| `head_on_route_generation` | `configs/scenarios/classic_interactions.yaml`, scenario `classic_head_on_corridor_low` | `123` | `20` trials | `scripts/tools/generate_adversarial_routes.py` |

The frozen broader comparison manifest is
`configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml`. Its
random and Optuna crossing/TTC lanes use global seed `42`; guided route search
uses seed `123`. It remains a development-stress contract unless a later
promotion review records stronger evidence.

## Eligible Planners

The manifest planner smoke defaults to `goal` and `social_force` because those
are lightweight local planner rows used by the current smoke tool. The frozen
comparison manifest also names `classic_global_theta_star` and `orca` rows with
availability caveats. Any planner-specific fallback, degraded execution,
missing checkpoint, or unavailable adapter must be reported as an exclusion or
limitation, not as successful adversarial evidence.

## Metrics

Current lanes report candidate and smoke accounting only:

- generated, valid, invalid, and degenerate candidates;
- duplicate normalized control hashes and rejection reasons;
- failed evaluations and simulation errors;
- row status such as `success`, `valid_behavioral_failure`, `fallback`,
  `degraded`, and `not_available`;
- archive counts and rerun-readiness blocker counts when using archive tools.

The residual controller also exposes a versioned
`residual_adversary_behavior.v1` summary for diagnostic runtime accounting:
steps, macro-actions, targeted-row fraction, applied residual norm
mean/max/integral, nonzero fraction, proposal-to-applied adjustment fraction,
and finite/bound-safe status. It is resettable, returned through the simulator's
read-only `residual_adversary_behavior_summary` property after lazy allocation,
and is not an episode-schema field or benchmark metric. Invalid, non-finite, or
bound-unsafe execution remains diagnostic/blocker evidence only.

New stress-case metrics are out of scope for this dispatchable slice.

## Canonical Commands

Generate validator-backed crossing/TTC manifests without planner execution:

```bash
uv run python scripts/tools/generate_adversarial_scenario_manifests.py \
  --search-space configs/adversarial/crossing_ttc_space.yaml \
  --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
  --count 16 \
  --seed 42 \
  --output-dir output/adversarial/issue4360_manifest_generation_smoke
```

Run the tiny manifest-to-planner smoke locally:

```bash
uv run python scripts/tools/run_adversarial_manifest_smoke.py \
  --search-space configs/adversarial/crossing_ttc_space.yaml \
  --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
  --count 4 \
  --seed 42 \
  --planners goal social_force \
  --output-dir output/adversarial/issue4360_planner_smoke
```

Run the route/start-state prototype:

```bash
uv run python scripts/tools/generate_adversarial_routes.py \
  --config configs/adversarial_routes/default.yaml
```

Run the residual adversary deterministic CPU smoke:

```bash
uv run pytest tests/adversarial/test_residual_adversary.py \
  tests/sim/test_residual_adversary_wiring.py -q
```

This proves the bounded residual-control reactive adversary's runtime wiring,
bound enforcement, deterministic scripted-policy reproducibility, and the
diagnostic `residual_adversary_behavior.v1` summary. The reproducibility test
loads `configs/adversarial/issue_4360_residual_adversary.yaml`, uses fixed seed
`42`, and runs `20` fixed `0.1 s` steps; repeated runs compare the canonical
summary values. It is capability-only smoke evidence — not a benchmark, metric,
or safety claim.

The historical crossing/TTC SLURM smoke launcher remains available, but this
issue #4360 slice does not submit compute:

```bash
ADVERSARIAL_SMOKE_LABEL=manual-crossing-ttc \
ADVERSARIAL_SMOKE_BUDGET=32 \
ADVERSARIAL_SMOKE_SEED=42 \
scripts/dev/sbatch_use_max_time.sh --time 04:00:00 --partition a30 --qos a30-gpu \
  SLURM/Auxme/adversarial_smoke_1501.sl
```

## Expected Runtime

Manifest generation should be local and quick. The tiny manifest planner smoke
is intended as a local wiring check, but runtime depends on planner availability
and simulator startup. The SLURM launcher is queue-dependent and outside this
PR's authorization.

## Claim Boundary

Adversarial smoke is development stress evidence. It does not certify a
scenario, prove planner robustness, create a paper-facing result, or add a
benchmark row. Generated cases need explicit promotion review before entering
durable scenario configs, aggregate benchmarks, claim maps, or dissertation
text.

## Caveats

Invalid candidates, simulation errors, not-available rows, fallback rows, and
degraded rows do not count as success evidence. High invalid-candidate counts
are generator-quality signals, not successful stress evidence. Outputs under
`output/` are local and disposable unless a later change promotes a compact
tracked evidence bundle or external artifact pointer.

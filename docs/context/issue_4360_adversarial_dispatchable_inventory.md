# Issue #4360 Adversarial Dispatchable Inventory

Issue: <https://github.com/ll7/robot_sf_ll7/issues/4360>

## Scope

The accepted 2026-07-04 maintainer rescope split issue #4360. This note covers
only the dispatchable half: inventory existing adversarial pedestrian hooks,
repeatable seeds and configs, runner assumptions, and how-to-run docs. The
post-release half remains parked: bounded adversary policy-interface redesign
and new stress-case metrics.

## Current Inventory

The canonical runbook is
[`docs/benchmark_suites/adversarial_smoke.md`](../benchmark_suites/adversarial_smoke.md).
The reproducibility manifest is
[`configs/adversarial/issue_4360_dispatchable_repro_seeds.yaml`](../../configs/adversarial/issue_4360_dispatchable_repro_seeds.yaml).

Existing owner surfaces are:

- `robot_sf/adversarial/scenario_manifest.py` and
  `scripts/tools/generate_adversarial_scenario_manifests.py` for generated-only
  `adversarial_scenario_manifest.v1` candidates.
- `scripts/tools/run_adversarial_manifest_smoke.py` for tiny local
  manifest-to-planner smoke checks.
- `robot_sf/adversarial/search.py` and `robot_sf/adversarial/config.py` for the
  bounded Python search runner API.
- `scripts/tools/generate_adversarial_routes.py` and
  `robot_sf/nav/adversarial_route_generation.py` for the head-on
  route/start-state generation prototype.
- `robot_sf/adversarial/archive.py`,
  `scripts/tools/curate_adversarial_failure_archive.py`,
  `scripts/validation/check_failure_archive_rerun_readiness.py`, and
  `scripts/adversarial/produce_rerun_closure_packet.py` for failure-archive
  curation and rerun-readiness checks.

## Seeds And Reproducibility

Three current lanes are dispatchable without adding new semantics:

| Lane | Seed | Budget | Output boundary |
| --- | --- | --- | --- |
| crossing/TTC manifest generation | `42` | `16` manifests | generated-only candidate YAML and `summary.json` under `output/` |
| crossing/TTC local planner smoke | `42` | `4` manifests for `goal` and `social_force` | smoke-only run bundle under `output/` |
| head-on route/start-state generation | `123` | `20` trials | route-generation smoke output under `output/` |

The frozen broader campaign manifest remains
`configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml`; it is
not promoted here.

## Runner Assumptions

- Search-space and manifest validators classify invalid or degenerate candidates
  before planner interpretation.
- Planner smoke rows must preserve fail-closed row statuses. Fallback, degraded,
  not-available, simulation-error, invalid, or degenerate rows are caveats or
  exclusions.
- Generated cases under `output/` are disposable local artifacts until a later
  PR records a compact tracked evidence bundle or an external artifact pointer.
- The current docs do not claim planner weakness, adversarial coverage,
  benchmark ranking, release evidence, or paper/dissertation support.

## Validation

This note is documentation and reproducibility inventory only. Appropriate
validation is path/link checking, YAML parsing, and focused help/import checks
for the named commands. Running a full benchmark campaign, submitting SLURM/GPU
jobs, or editing paper/dissertation claims is outside this slice.

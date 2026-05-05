# Issue #1001: Architecture Seam Audit

## Goal

Issue #1001 turns the current hotspot scan into PR-sized refactor work and proves the plan with one
no-behavior-change extraction. The scope is architecture hygiene only: do not change benchmark
semantics, fallback/degraded-mode interpretation, training behavior, or planner quality claims.

## Current Hotspots

Measured with:

```bash
rtk uv run python scripts/dev/complexity_runtime_baseline.py robot_sf/planner/socnav.py robot_sf/benchmark/camera_ready_campaign.py robot_sf/benchmark/map_runner.py scripts/training/train_ppo.py robot_sf/benchmark/cli.py robot_sf/benchmark/metrics.py --top 8
```

Largest modules:

| path | code lines | total lines |
| --- | ---: | ---: |
| `robot_sf/planner/socnav.py` | 4379 | 4896 |
| `robot_sf/benchmark/camera_ready_campaign.py` | 3357 | 3645 |
| `robot_sf/benchmark/map_runner.py` | 2858 | 3131 |
| `scripts/training/train_ppo.py` | 2740 | 3072 |
| `robot_sf/benchmark/cli.py` | 1948 | 2171 |
| `robot_sf/benchmark/metrics.py` | 1941 | 2481 |

Longest functions:

| path | function | lines |
| --- | --- | ---: |
| `robot_sf/benchmark/camera_ready_campaign.py` | `run_campaign` | 1003 |
| `robot_sf/benchmark/map_runner.py` | `_build_policy` | 979 |
| `robot_sf/benchmark/map_runner.py` | `run_map_batch` | 338 |
| `robot_sf/benchmark/cli.py` | `_attach_core_subcommands` | 310 |
| `robot_sf/benchmark/camera_ready_campaign.py` | `prepare_campaign_preflight` | 280 |
| `robot_sf/benchmark/map_runner.py` | `_run_map_episode` | 271 |

## Ownership Boundaries

Benchmark orchestration:

- Owns scenario expansion, readiness/profile gates, resume identity, episode writing, and artifact
  root policy.
- Current primary files: `robot_sf/benchmark/map_runner.py`,
  `robot_sf/benchmark/runner.py`, `robot_sf/benchmark/camera_ready_campaign.py`, and
  `robot_sf/benchmark/cli.py`.
- Should not own individual planner adapter construction details beyond invoking a stable factory.

Policy construction and command contracts:

- Owns algorithm key resolution, planner adapter construction, action-space/projection metadata,
  availability status, and kinematics feasibility counters.
- Current primary files: `robot_sf/benchmark/map_runner.py`,
  `robot_sf/benchmark/algorithm_readiness.py`, `robot_sf/benchmark/algorithm_metadata.py`, and now
  `robot_sf/benchmark/planner_command_contract.py`.
- This boundary must preserve explicit native/adapter/fallback/degraded semantics; hidden fallback
  success is benchmark drift.

Metric computation:

- Owns per-episode metric calculations, post-processing, and schema-compatible payloads.
- Current primary files: `robot_sf/benchmark/metrics.py`,
  `robot_sf/benchmark/termination_reason.py`, and tests under `tests/test_metrics.py`,
  `tests/unit/benchmark/`, and `tests/benchmark/`.
- Should not depend on planner internals except through episode records and termination/status
  fields.

Training entry points:

- Own config loading, environment construction, model initialization, checkpointing, and evaluation
  hooks for training workflows.
- Current primary files: `scripts/training/train_ppo.py`, `robot_sf/training/`,
  `configs/training/`, and model/provenance docs.
- Should call reusable library helpers rather than embedding benchmark or scenario-loader policy.

## Top Refactor Candidates

1. Split map-runner policy construction out of `_build_policy`.

   Candidate shape: a `robot_sf/benchmark/policy_factory/` or `policy_builders.py` registry with
   one builder per planner family. Keep metadata enrichment and readiness profile handling
   centralized, but move per-planner construction branches behind named functions.

   Proof path: preserve `tests/benchmark/test_map_runner_utils.py`, add builder-level tests for one
   migrated planner at a time, and run `BASE_REF=origin/main rtk scripts/dev/pr_ready_check.sh`.

2. Split camera-ready campaign execution into phase objects or modules.

   Candidate shape: keep `run_campaign` as orchestration, but move preflight, run scheduling,
   report assembly, and artifact publication into narrow helpers with explicit input/output
   dataclasses.

   Proof path: preserve `tests/benchmark/test_camera_ready_campaign.py` and use existing campaign
   config fixtures; do not change campaign table or metadata schemas in the same PR.

3. Split `scripts/training/train_ppo.py` into reusable training setup helpers.

   Candidate shape: extract config normalization, environment/vector-env construction, checkpoint
   metadata, and evaluation-hook wiring into `robot_sf/training/` modules. Leave the script as a
   CLI facade.

   Proof path: targeted training-config tests first, then the canonical smoke command for the
   touched training path. Do not change PPO defaults or checkpoint provenance in a seam-only PR.

## Executed Extraction

Extracted planner command-space and kinematics-feasibility helpers from the `map_runner.py`
hotspot into `robot_sf/benchmark/planner_command_contract.py`:

- `default_robot_command_space`
- `init_feasibility_metadata`
- `project_with_feasibility`
- `planner_kinematics_compatibility`

`map_runner.py` keeps private aliases for compatibility with existing tests and internal call sites,
but the command-contract seam is now independently testable.

This is intentionally not the full policy-factory split. It is a small, reversible extraction that
proves the map-runner hotspot can lose a coherent ownership slice without behavior changes.

Post-extraction measurement:

```bash
rtk uv run python scripts/dev/complexity_runtime_baseline.py robot_sf/benchmark/map_runner.py robot_sf/benchmark/planner_command_contract.py --top 5
```

Result: `map_runner.py` drops to `2771` code lines, and the extracted command-contract module is
`104` code lines.

## Validation

Focused lint:

```bash
rtk uv run ruff check robot_sf/benchmark/planner_command_contract.py robot_sf/benchmark/map_runner.py tests/benchmark/test_planner_command_contract.py tests/benchmark/test_map_runner_utils.py
```

Result: `All checks passed!`

Focused tests:

```bash
rtk uv run pytest tests/benchmark/test_planner_command_contract.py tests/benchmark/test_map_runner_utils.py -q
```

Result: `78 passed`.

## Follow-Up Boundary

- This issue should close with the audit plus the command-contract extraction.
- Do not fold broader `_build_policy` registry work into this branch; create a follow-up issue for
  the first policy-builder migration.
- Do not change planner fallback/degraded semantics in architecture cleanup PRs. Those require a
  dedicated benchmark-semantics issue and representative proof.

# Open Issues PR Split Strategy (2026-05-13)

## Goal

Split the current broad open-issues implementation pass into reviewable branches and PRs instead of
merging the mixed local state directly.

This note is the handoff surface for extracting the current WIP snapshot into scoped PRs. It should
be read together with:

- `docs/context/open_issues_maintainer_input_triage.md`
- `docs/context/open_issues_implementation_status_2026-05-12.md`
- `docs/dev_guide.md`
- `docs/context/README.md`

## Current decision

Do not merge the bulk local branch as one PR.

Use the current worktree only as a WIP source snapshot, then create clean issue branches from latest
`origin/main` and restore only the files needed for each scoped PR. This keeps review size small,
reduces semantic coupling, and avoids landing partially validated issue work as a single large
change.

## Safety snapshot workflow

1. Add this strategy note to the current mixed worktree.
2. Create a WIP branch from the current checkout.
3. Commit the full mixed state as a safety snapshot.
4. Extract clean PR branches from `origin/main` by restoring selected paths from the WIP snapshot.
5. Validate and open draft PRs one at a time.

Recommended snapshot branch:

```bash
git switch -c wip/open-issues-bulk-snapshot-2026-05-13
git add -A
git commit -m "wip: snapshot open issue implementation pass"
```

## PR stack

### PR 1: Crowd simulation and environment constructor cleanup

Branch: `issue-1150-crowd-sim-env`

Issues: `#1150`, `#1141`, `#1146`, `#1148`

Scope:

- add `robot_sf/gym_env/crowd_sim_env.py`
- add `make_crowd_sim_env(...)` in `robot_sf/gym_env/environment_factory.py`
- remove `EmptyRobotEnv` and `SimpleRobotEnv`
- make maintained environment constructors safe to instantiate without sharing mutable config
  defaults
- keep fast no-robot stepping lean
- support optional rendering and recording
- ignore robot-policy inputs with a warning for the no-robot crowd surface

Primary validation:

```bash
uv run pytest \
  tests/test_crowd_sim_env_contract.py \
  tests/test_environment_factory_signatures.py \
  -q
```

### PR 2: Small correctness fixes

Branch: `issue-1142-1144-1147-small-fixes`

Issues: `#1142`, `#1144`, `#1147`

Scope:

- encode video at the configured target FPS when available
- document and test `WheelSpeedState` as angular wheel speeds in rad/s
- make `scripts/classic_benchmark_full.py` fail closed with an actionable error when unavailable
- optionally include low-risk docstring cleanup touched by the same files

Primary validation:

```bash
uv run pytest \
  tests/visuals/test_sim_view_coverage_paths.py \
  tests/differential_drive_test.py \
  tests/benchmark_full/test_classic_benchmark_full_cli.py \
  -q
```

### PR 3: Sensor history contract

Branch: `issue-1143-1149-sensor-history-contract`

Issues: `#1143`, `#1149`

Scope:

- document oldest-to-newest temporal ordering in stacked sensor history
- centralize append semantics in `robot_sf/sensor/sensor_fusion.py`
- update image sensor fusion to use the same helper

Primary validation:

```bash
uv run pytest tests/test_sensor_fusion_stack.py -q
```

### PR 4: Dynamic pedestrian occlusion

Branch: `issue-1124-dynamic-ped-occlusion`

Issue: `#1124`

Scope:

- add opt-in dynamic pedestrian occlusion to SocNav observations
- expose `observation_visibility.dynamic_occlusion`
- preserve ground-truth separation and existing static-occlusion behavior

Primary validation:

```bash
uv run pytest \
  tests/test_socnav_dynamic_occlusion.py \
  tests/test_socnav_observation.py \
  tests/training/test_scenario_loader.py \
  -q
```

### PR 5: Predictive obstacle feature schema

Branch: `issue-1138-predictive-obstacle-features-v1`

Issue: `#1138`

Scope:

- add `predictive_obstacle_features_v1`
- expose deterministic nearest-local-obstacle features
- keep unavailable obstacle data explicit with a sentinel and validity mask

Primary validation:

```bash
uv run pytest tests/planner/test_obstacle_features.py -q
```

### PR 6: Multi-AMV episode extension

Branch: `issue-1128-multi-amv-episode-extension`

Issue: `#1128`

Scope:

- add namespaced `multi_amv` episode extension
- fail closed for single-robot or missing metric inputs
- update smoke runner output contract

Primary validation:

```bash
uv run pytest tests/benchmark/test_multi_amv.py -q
```

### PR 7: CARLA oracle/replay parity adapter

Branch: `issue-1110-carla-parity-adapter`

Issue: `#1110`

Scope:

- add lightweight CARLA parity comparison utilities
- add CLI for oracle-vs-replay metric comparison
- keep degraded or missing inputs explicit, not benchmark-success evidence

Primary validation:

```bash
uv run pytest tests/carla_bridge/test_parity.py tests/carla_bridge/test_parity_cli.py -q
```

### PR 8: Manual control MVP foundations

Branch: `issue-1151-manual-control-foundations`

Issues: `#1151`; references `#1152`, `#1153`, `#1154`

Scope:

- add keyboard-hold, fixed-map differential-drive manual input mapper
- add attempt/session state, JSONL recording, manifest, replay grouping, baseline comparison, and
  demonstration export helpers
- keep web-game work as schema-aligned future work, not an implemented dependency

Primary validation:

```bash
uv run pytest tests/test_manual_control_*.py -q
```

### PR 9: Docstring cleanup pass

Branch: `issue-1145-docstring-cleanup-pass`

Issue: `#1145`

Scope:

- include remaining docstring-only cleanup only if it is still useful after PRs 1-8
- avoid mixing docstring cleanup into semantic PRs unless the touched file already belongs to that
  PR

Primary validation:

```bash
uv run ruff check robot_sf tests scripts
```

## Merge order

Open and merge low-risk, dependency-light PRs first:

1. PR 2: small correctness fixes
2. PR 3: sensor history contract
3. PR 1: crowd simulation environment
4. PR 4: dynamic pedestrian occlusion
5. PR 5: predictive obstacle features
6. PR 6: multi-AMV episode extension
7. PR 7: CARLA parity adapter
8. PR 8: manual control foundations
9. PR 9: remaining docstrings if still needed

## Extraction pattern

For each branch:

```bash
git fetch origin main
git worktree add ../<branch-name> origin/main -b <branch-name>
cd ../<branch-name>
git restore --source wip/open-issues-bulk-snapshot-2026-05-13 -- <scoped paths>
```

Then run the PR-specific validation command, commit, push, and open a draft PR.

Before creating a non-draft PR, follow the repository PR readiness rule:

```bash
git fetch origin main
git merge origin/main
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Blocked or no-code issues

Do not create implementation PRs from the current snapshot for issues that still need external
evidence or assets:

- `#1119`: Docker reproduction needs a Docker-capable host.
- `#1111`: CARLA T1 needs a CARLA-capable host.
- `#1134`: SocNavBench ETH needs official assets.
- `#1126`: SDD importer needs official staged data.
- `#1108`: BC/PPO experiment requires actual execution artifacts.
- `#1154`: web-game collection depends on schema validation and a separate UI surface.

Keep those issues as explicit blockers or planning comments until their external proof inputs
exist.

## Validation status

No validation has been run for the mixed snapshot at the time this note was created. Each extracted
PR must carry its own validation evidence before review.

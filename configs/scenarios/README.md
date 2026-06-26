# Scenario Layout

This directory supports a mix of **per-scenario**, **per-archetype**, and
**manifest** files.

## Layout

- `classic_interactions.yaml`, `francis2023.yaml`: manifest entry points for the two suites.
- `classic_interactions_francis2023.yaml`: combined manifest for both suites.
- `nominal_v1.yaml`: compact nominal shared-space calibration matrix for routine, low-stress
  planner checks. It is intentionally separate from stress, adversarial, and camera-ready evidence.
- `confirmation_v1.yaml`: compact confirmation matrix with issue-596 atomic and sparse
  interaction archetypes for non-paper robustness checks.
- `single/`: one scenario per file (manual fine-tuning and small edits).
- `archetypes/`: groups of related scenarios (may contain one or many entries).
- `sets/`: manifest files that include other scenario files.
- `perturbations/`: perturbation manifests for validity preflight before paired criticality pilots.

`sets/vulnerable_user_proxy_pack_v0_deferred_issue3654.yaml` is a deferred vulnerable-user proxy
scaffold. It is disabled from default benchmark campaigns and records only synthetic proxy intent,
not real-world user-group evidence or paper-facing results.

## Plausibility tracking

Scenario metadata includes a `plausibility` block used to record verification
status and interaction metrics:

```yaml
metadata:
  plausibility:
    status: unverified
    verified_on: null
    verified_by: null
    method: null
    notes: null
    metrics_updated_on: null
    metrics:
      min_distance: null
      mean_distance: null
      robot_ped_within_5m_frac: null
      ped_force_mean: null
      force_q95: null
```

## Optional Rollover Stability Instrumentation

Scenarios may opt in to a three-wheeled vehicle (TWV) rollover-stability proxy through
`metadata.rollover_stability`. The flag is disabled by default, so existing benchmark evidence and
campaign tables are unchanged unless a scenario explicitly enables it.

```yaml
metadata:
  rollover_stability:
    enabled: true
    t_w: 0.8     # rear track width in meters
    L: 1.2       # wheelbase in meters
    h_c: 0.6     # center-of-gravity height in meters
    a: 0.5       # center-of-gravity distance from the front axle in meters
```

When enabled, metric rows may include `rollover_critical_count`,
`rollover_min_stability_margin`, `rollover_lateral_accel_abs_max`, and `rollover_event`.
`rollover_event: ROLLOVER_CRITICAL` marks timesteps where the lateral-acceleration proxy reaches
the configured rollover threshold. This is a bounded instrumentation check, not a full vehicle
dynamics model.

## Draft authoring workflow

For a small reviewable starting point, generate a draft YAML from the v1 authoring template and
validate it before handing the file to training or benchmark commands:

```bash
uv run python scripts/tools/create_scenario.py \
  --template bottleneck \
  --name draft_bottleneck_review \
  --output configs/scenarios/single/draft_bottleneck_review.yaml

uv run python scripts/tools/validate_scenario.py \
  configs/scenarios/single/draft_bottleneck_review.yaml
```

The generator currently supports the conservative `bottleneck` template. It writes deterministic
YAML with a `map_id`, `simulation_config`, empty `robot_config`, `metadata.authoring`, and explicit
`seeds`. The validator is intentionally stricter than the general scenario loader for authored
drafts: it requires a scenario name, exactly one map reference, required config/metadata sections,
non-empty integer seeds, and then reuses the repository scenario loader plus map/config builder to
catch map and config errors.

Passing this authoring validator only means the draft is structurally reviewable and loadable. It is
not scenario certification, benchmark promotion, or benchmark evidence.

Example reproducible bottleneck draft generation:

```bash
uv run python scripts/tools/create_scenario.py \
  --template bottleneck \
  --name draft_bottleneck_issue_3027 \
  --density med \
  --flow cross \
  --obstacle maze \
  --groups 0.25 \
  --speed-var high \
  --goal-topology circulate \
  --robot-context ahead \
  --seeds 101 102 \
  --output configs/scenarios/single/draft_bottleneck_issue_3027.yaml
```

Resulting YAML includes:
- `metadata.generation_profile` with `schema_version`, deterministic `seed_signature`, and
  normalized `parameters`.
- `metadata.initial_difficulty` with schema+score/band metadata for initial intent review.
- `metadata.authoring.benchmark_evidence: false` and `metadata.authoring.status: draft`.

The explicit `--flow/--obstacle/--goal-topology/...` flags are for reproducible pre-registration and
later diff review.

Traceability: related issue `#3027`; implementing PR `#3095`; proof artifact
`tests/tools/test_scenario_authoring.py`; canonical doc `docs/scenario_contracts.md`.

## Scenario Contracts

Versioned scenario-intent contracts live under `configs/scenarios/contracts/`.
`scenario_contract.v1` captures
authored assumptions such as ODD, actors, invariants, observables, termination semantics, and
provenance before a scenario is executed. It is a governance layer, not a benchmark result and not a
replacement for `scenario_cert.v1` feasibility checks.

See `docs/scenario_contracts.md` for the schema, loader API, and the boundary between intent
contracts, certification, and executed benchmark evidence.

## Manifest (include) files

Manifest files use `includes` (or `include` / `scenario_files`) to combine
per-scenario and per-archetype YAMLs into a single scenario list.

```yaml
# configs/scenarios/sets/classic_cross_trap_subset.yaml
schema_version: robot_sf.scenario_matrix.v1
includes:
  - ../archetypes/classic_cross_trap.yaml
select_scenarios:
  - classic_cross_trap_low
  - classic_cross_trap_high
map_search_paths:
  - ../../../maps/svg_maps
```

The loader expands includes relative to the manifest file and then applies
`select_scenarios` as an explicit, case-insensitive subset filter. Selector
order controls the final scenario order. Duplicate names in the expanded pool
raise an error so subset manifests fail closed when the source data is
ambiguous.

The optional `schema_version` field is metadata for manifest validation. Use
`robot_sf.scenario_matrix.v1`; `robot_sf_bench validate-config --matrix <manifest>` rejects
unsupported values before a run starts.

If `map_file` paths in included scenarios are not resolvable, you can set
`map_search_paths` to help locate map files. Each manifest resolves its own
`map_search_paths` entries relative to that manifest file, and the loader logs
a warning with the attempted paths and suggestion.

## Map references

Scenarios can reference maps via either `map_file` (a path) or `map_id` (a
registry key). `map_id` entries are resolved through `maps/registry.yaml` and
rebased relative to the manifest root so scenarios stay portable.

```yaml
scenarios:
  - name: cross_trap_demo
    map_id: classic_cross_trap
```

If both `map_id` and `map_file` are provided, `map_id` takes precedence (with a
warning). To override the registry path, set `ROBOT_SF_MAP_REGISTRY`. You can
regenerate the registry with `scripts/tools/generate_map_registry.py`.

## Platform Semantics

Station-platform scenarios can declare opt-in semantic regions with
`platform_semantics`. The current implementation is scenario-side metadata only:
it validates region shape and intent, but no planner or metric consumes the
regions yet.

Supported region kinds:

- `hazard`: platform-edge or similar risk areas.
- `keep_clear`: train-door or circulation areas that should not be treated as
  ordinary waiting space.

Supported shapes:

- `polygon` with at least three `[x, y]` points.
- `bbox` with `[min_x, min_y, max_x, max_y]`.

Use `status: metadata_only` for provenance and review. Use
`status: require_consumers` only when planner/metric behavior must consume the
regions; `build_robot_config_from_scenario` fails closed for that status until
explicit consumers are implemented.

```yaml
scenarios:
  - name: station_platform_semantics_demo
    map_file: maps/platform.svg
    platform_semantics:
      status: metadata_only
      regions:
        - id: platform_edge
          kind: hazard
          shape: polygon
          points:
            - [0.0, 0.0]
            - [10.0, 0.0]
            - [10.0, 0.5]
        - id: train_door_keep_clear
          kind: keep_clear
          shape: bbox
          bounds: [4.0, 1.0, 6.0, 2.0]
```

## Usage

Point training/analysis configs at a manifest (or a legacy combined file):

```yaml
scenario_config: ../scenarios/sets/classic_cross_trap_subset.yaml
```

Example:

```bash
uv run python scripts/tools/policy_analysis_run.py \
  --training-config configs/training/benchmark_orca_classic_cross_trap_subset.yaml \
  --policy socnav_orca
```

## Benchmark coverage notes

- `archetypes/classic_urban_crossing.yaml` adds a corner-building urban crossing layout
  to complement open-plaza crossing maps. This increases benchmark relevance for
  real-world constrained intersection flows.
- `archetypes/classic_realworld_bottleneck.yaml` adds a narrow-hallway bottleneck
  derived from a real-world interaction layout to improve constrained-flow coverage.
- `sets/station_platform_candidate_pack_issue736.yaml` is an exploratory station-platform
  variant pack. Keep it out of the default classic matrix until a benchmark run shows
  distinct value beyond corridor, bottleneck, doorway, and group-crossing controls. For
  config-only diagnostic triage before running episodes, use
  `uv run python scripts/tools/scenario_coverage_entropy.py <matrix> --output-json <path>
  --output-markdown <path>`; the resulting entropy and novelty values are not benchmark-success
  or safety metrics.
- `sanity_v1.yaml` is a non-paper-facing nominal calibration manifest for issue #1083. It selects
  four low-ambiguity scenes from existing validated surfaces so `goal` and `orca` can be checked
  on easy deployment-like cases before hard-matrix failures are interpreted.
- `nominal_v1.yaml` is a small shared-space calibration matrix for issue #1273. It selects one
  straight open-space route, one gentle crossing, one low-density doorway, and one low-density
  bottleneck. Use it for local sanity checks and nominal-vs-stress interpretation; do not treat
  success on this matrix as stress robustness or safety evidence.

## Task bundles

Named task bundles live under `configs/bundles/` and expand into existing scenario manifests through
the same loader. Use `bundle:<name>` when a benchmark or training command should consume a reusable
scenario package without copying scenario lists:

```bash
uv run robot_sf_bench validate-config --matrix bundle:sanity-smoke-v1
```

See `configs/bundles/README.md` for the versioned schema and the durable-input boundary.

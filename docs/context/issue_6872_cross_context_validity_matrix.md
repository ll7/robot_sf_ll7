# Issue #6872: Cross-Context Validity / Revalidation Matrix

> Status: draft/candidate
> Schema: `cross_context_validity_matrix.v1`
> Machine-readable: [`configs/benchmarks/cross_context_validity_matrix_v1.yaml`](../../configs/benchmarks/cross_context_validity_matrix_v1.yaml)

## Goal

Create a public, versioned cross-context validity/revalidation matrix that
distinguishes **protocol portability** (can the measurement protocol be reused?)
from **transfer of empirical conclusions** (do rankings and metric values
transfer?) and fails closed against overclaiming.

## What This Is

This is a **research-maintenance synthesis artifact**. It is:

- A diagnostic/planning document that records what this repository can
  currently substantiate about cross-context validity.
- A contract that prevents overclaiming by requiring explicit evidence
  status for every context axis combination.
- A lookup for future benchmark campaigns to understand what
  revalidation is needed when changing context.

This is **not**:

- A new benchmark campaign or data collection.
- Paper evidence or benchmark proof.
- Sim-to-real or physical-platform validation.
- A claim that any empirical results transfer across contexts.

## Four Axes

The matrix evaluates validity along four independent context axes:

### 1. Site / Topology

Physical environment geometry and scenario topology. Determines spatial
interaction patterns, visibility, and constriction dynamics.

| Value | Provenance |
| --- | --- |
| `classic_corridor` | classic_head_on_corridor, francis2023_narrow_hallway |
| `intersection` | classic_t_intersection, classic_urban_crossing, francis2023_intersection_* |
| `bottleneck` | classic_bottleneck, classic_realworld_bottleneck |
| `doorway` | classic_doorway, francis2023_narrow_doorway |
| `open_plaza` | francis2023_crowd_navigation, francis2023_robot_crowding |
| `blind_corner` | francis2023_blind_corner |
| `elevator` | francis2023_entering_elevator, francis2023_exiting_elevator |
| `following` | francis2023_following_human, francis2023_leading_human, francis2023_accompanying_peer |
| `station_platform` | classic_station_platform |
| `merging` | classic_merging |
| `overtaking` | classic_overtaking, francis2023_pedestrian_overtaking, francis2023_robot_overtaking |

### 2. Social / Cultural Assumptions

Pedestrian behavior model and social-force parameterization. Determines
how synthetic pedestrians interact.

| Value | Provenance |
| --- | --- |
| `social_force_default` | Default model; baseline in #3207 fidelity study |
| `social_force_heterogeneous_speeds` | Tested as fidelity axis in #3207 |
| `hsfm_zanlungo` | Experimental; issue #4973 |
| `hsfm_ttc_predictive` | Opt-in prototype; issue #3481 |

### 3. Robot Embodiment

Robot kinematics and command space. Determines motion constraints.

| Value | Provenance |
| --- | --- |
| `differential_drive` | Default; baseline-safe benchmark config |
| `holonomic` | Cross-kinematics compatibility testing only |
| `bicycle_drive` | Cross-kinematics compatibility testing only |

### 4. Observation / Perception Mode

What the planner observes. Determines perception assumptions.

| Value | Provenance |
| --- | --- |
| `oracle_full_state` | Default; privileged sim state |
| `tracked_agents_no_noise` | Perfect tracking, no noise |
| `tracked_agents_with_noise` | Synthetic noise robustness |
| `lidar_2d` | Range sensor projection |
| `occluded_partial_state` | Partial state with occlusion |

## Evidence Status Values

Every cell in the matrix has an `evidence_status`:

| Status | Meaning |
| --- | --- |
| `covered` | Actively evidenced with local configs or bundles |
| `partially_covered` | Some evidence exists but incomplete |
| `requires_revalidation` | No direct evidence; must revalidate |
| `not_evidenced` | No evidence; no local artifact supports the cell |
| `unavailable` | Blocked by missing external data |

**Fail-closed rule**: Missing evidence defaults to `requires_revalidation`
or `not_evidenced`, never `covered` by assumption.

## Protocol Portability vs Result Portability

The matrix distinguishes two separate concerns:

- **`protocol_portability`**: Can the AMV measurement protocol (metric
  definitions, episode structure, aggregation rules) be applied in this
  context? This is about the measurement instrument, not the measured
  values.

- **`result_portability`**: Do empirical conclusions (planner rankings,
  metric values, comparisons) transfer to this context? This requires
  dedicated cross-context evidence that this repository does not
  currently assert for any cross-axis cell.

Every `result_portability` value is `not_portable` in the current
matrix. This is deliberate: protocol portability does not imply result
portability.

## Canonical Configurations

The matrix inventories the repository's active benchmark configurations
without implying unsupported coverage:

| Configuration | Kinematics | Observation | Status |
| --- | --- | --- | --- |
| `paper_experiment_matrix_v1` | differential_drive | oracle_full_state | covered |
| `paper_experiment_matrix_7planners_v1` | differential_drive | oracle_full_state | covered |
| `cross_kinematics_v1` | differential_drive, bicycle_drive, holonomic | oracle_full_state | partially_covered |
| `camera_ready_holonomic` | holonomic | oracle_full_state | partially_covered |
| `fidelity_sensitivity_full_fixed_scope` | differential_drive | oracle_full_state | partially_covered |

Naming a configuration here does not imply it covers all matrix cells.

## No-Overclaim Boundary

These statements must accompany any public reference to this matrix:

1. This matrix is a research-maintenance contract, not benchmark evidence.
2. Protocol portability does not imply result portability.
3. `covered` cells indicate the protocol definition applies; they do not
   indicate that empirical conclusions transfer.
4. All `result_portability` values are `not_portable` unless specific
   cross-context evidence exists, which this matrix does not assert.
5. `not_evidenced` and `requires_revalidation` cells fail closed: no
   claim may be made without fresh evidence.
6. This matrix does not establish simulator realism, sim-to-real
   transfer, or physical-platform validity.
7. Fallback or degraded execution is not success evidence.

## Related Issues

| Issue | Role |
| --- | --- |
| [#3207](https://github.com/ll7/robot_sf_ll7/issues/3207) | Simulator fidelity sensitivity evidence |
| [#6472](https://github.com/ll7/robot_sf_ll7/issues/6472) | Social compliance protocol |
| [#6473](https://github.com/ll7/robot_sf_ll7/issues/6473) | Observation track architecture |

## Validation

```bash
uv run pytest tests/docs/test_issue_6872_cross_context_validity_matrix.py -v
uv run ruff format --check tests/docs/test_issue_6872_cross_context_validity_matrix.py
uv run ruff check tests/docs/test_issue_6872_cross_context_validity_matrix.py
uv run python -c "import yaml; yaml.safe_load(open('configs/benchmarks/cross_context_validity_matrix_v1.yaml'))"
```

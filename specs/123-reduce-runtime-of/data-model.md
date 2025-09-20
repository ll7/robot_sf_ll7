# Data Model: Reproducibility Test Acceleration

**Scope**: Minimal entities required to reason about and validate deterministic episode sequencing in the accelerated reproducibility integration test.

## Entities

### EpisodeSequence
- **Description**: Ordered collection of episode identifiers generated during a benchmark run under a fixed scenario matrix + seed configuration.
- **Attributes**:
  - `episode_ids: List[str]` – Stable ordering, each unique.
  - `count: int` – Derived length (sanity: > 0).
  - `generation_seed: int` – Root seed used for planning expansion.
  - `scenario_matrix_hash: str` – Hash snapshot at planning time.
- **Invariants**:
  - Deterministic: same `generation_seed` + identical scenario matrix → identical `episode_ids` ordering.
  - No duplicates.

### ScenarioHashSnapshot
- **Description**: Minimal structural digest ensuring scenario configuration identity across runs.
- **Attributes**:
  - `hash_value: str` – Hex digest derived from canonical YAML serialization.
  - `scenario_count: int` – Number of scenarios (expected 1 for minimized test).
  - `episodes_planned: int` – Derived total episodes (e.g., 2 seeds → 2 episodes).
- **Invariants**:
  - Unchanged across repeated runs; difference signals config drift.

### PerformanceEnvelope (Logical / Not Persisted)
- **Description**: Target execution time thresholds for the test.
- **Attributes**:
  - `local_soft_target_sec: float` (2.0 default)
  - `ci_upper_bound_sec: float` (4.0 default)
  - `strict_env_var: str` (`STRICT_REPRO_TEST` – when set, convert soft timing breach to failure)

## Relationships
- `EpisodeSequence.scenario_matrix_hash` references `ScenarioHashSnapshot.hash_value`.
- `PerformanceEnvelope` guides assertions but not persisted.

## Validation Rules
| Rule | Entity | Condition |
|------|--------|-----------|
| R1 | EpisodeSequence | `count > 0` |
| R2 | EpisodeSequence | `len(set(episode_ids)) == count` |
| R3 | EpisodeSequence | Re-run sequence equality check passes |
| R4 | ScenarioHashSnapshot | Hash stable across runs |
| R5 | PerformanceEnvelope | Wall clock < `ci_upper_bound_sec` (fail or warn depending on mode) |

## Derived Data
- `count` derived from `len(episode_ids)`.
- `episodes_planned` derived from scenario seeds planned.

## Out of Scope
- Detailed metric value comparisons (collision rates, success rates) – explicitly excluded for performance.
- Effect size computations.
- Bootstrap confidence intervals.

---
Prepared: 2025-09-20

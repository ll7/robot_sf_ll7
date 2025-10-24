# Feature Specification: Per-Pedestrian Force Quantiles

**Feature Branch**: `146-per-pedestrian-force`  
**Created**: 2025-10-24  
**Status**: Draft  
**Input**: User description: "Implement per-pedestrian force magnitude quantiles (q50, q90, q95) that compute quantiles for each pedestrian individually before averaging, as distinct from the existing aggregated force quantiles"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Individual Pedestrian Force Distribution Analysis (Priority: P1)

Benchmark researchers need to understand the distribution of forces experienced by **individual pedestrians** to identify if specific pedestrians are being subjected to consistently high forces, rather than just looking at global force statistics that might mask individual experiences.

**Why this priority**: This is the core value proposition. The existing `force_quantiles()` mixes all forces together, losing granularity about individual pedestrian experiences. Per-pedestrian quantiles reveal whether high forces are concentrated on specific individuals or distributed evenly.

**Independent Test**: Can be fully tested by running `compute_all_metrics()` on an episode with 3+ pedestrians where one pedestrian experiences consistently high forces while others experience low forces. The per-ped quantiles should show high values for that individual, while aggregated quantiles would average them out.

**Acceptance Scenarios**:

1. **Given** an episode with 3 pedestrians where ped 0 experiences forces [10, 10, 10] and peds 1,2 experience [1, 1, 1], **When** computing per-ped force quantiles, **Then** the result should reflect that one pedestrian's median is ~10 while the overall per-ped median is ~(10+1+1)/3 = 4
2. **Given** an episode with a single pedestrian experiencing varying forces [1, 5, 10], **When** computing per-ped quantiles, **Then** the per-ped median should equal that pedestrian's individual median (5), not affected by non-existent other pedestrians

---

### User Story 2 - Sensitivity Analysis for SNQI Weights (Priority: P2)

SNQI (Social Navigation Quality Index) researchers need stable, discriminative force metrics to optimize composite weights. Per-pedestrian quantiles provide a different statistical perspective that may improve rank stability across scenario subsets.

**Why this priority**: Enables downstream SNQI optimization and sensitivity analysis, which is the stated goal of the parent issue. However, the metric must exist first (P1) before it can be integrated into SNQI.

**Independent Test**: Can be tested by running the SNQI sensitivity analysis script with both aggregated and per-ped force quantiles included as candidate metrics, verifying that per-ped metrics produce different rankings than aggregated ones.

**Acceptance Scenarios**:

1. **Given** a set of benchmark episodes from different density scenarios, **When** computing both aggregated and per-ped force quantiles, **Then** the per-ped metrics should show lower correlation with aggregated metrics (Pearson r < 0.95), indicating they capture different information
2. **Given** baseline episodes for SNQI normalization, **When** computing median and p95 values for per-ped quantiles, **Then** these baseline statistics should be stable (variance < 10% across repeated runs with different seeds)

---

### User Story 3 - Documentation and Edge Case Coverage (Priority: P3)

Future maintainers and users need clear documentation of the per-ped quantile formula, naming conventions, and edge case handling so they can correctly interpret results and debug issues.

**Why this priority**: Important for maintainability but doesn't block initial implementation or testing. Can be refined after P1 implementation is validated.

**Independent Test**: Can be tested by a code reviewer reading the docstring and metrics_spec.md, then correctly predicting the output for edge cases (no peds, single ped, ped appearing mid-episode) without running code.

**Acceptance Scenarios**:

1. **Given** the updated `metrics_spec.md` documentation, **When** a user reads the per-ped force quantile definition, **Then** they should understand it computes "per-pedestrian quantiles first, then averages" vs "all forces flattened then quantile"
2. **Given** an episode where a pedestrian appears only in the last 2 timesteps (T_j subset), **When** computing that pedestrian's quantile, **Then** the documentation should clearly state quantiles are computed only over timesteps where the pedestrian is present

---

### Edge Cases

- **No pedestrians entire episode (K=0)**: Return `NaN` for all per-ped quantile keys, consistent with existing `force_quantiles()` behavior
- **Single pedestrian (K=1)**: Per-ped quantile should equal that pedestrian's individual quantile; mean-across-peds becomes identity operation
- **Pedestrian with single timestep**: If a pedestrian has only 1 force sample, quantiles collapse to that value (q50=q90=q95=that magnitude)
- **Pedestrian appears/disappears mid-episode**: Only include force magnitudes from timesteps where pedestrian is present (requires filtering NaN or zero-padded entries if applicable)
- **All forces for a pedestrian are identical**: Quantiles should all equal that value (degenerate distribution)
- **Force array contains NaN values**: Apply `np.nan_to_num(..., copy=False)` before computing magnitudes, consistent with other metrics

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST implement a new metric function `per_ped_force_quantiles(data: EpisodeData, qs: Iterable[float] = (0.5, 0.9, 0.95))` that computes force quantiles per-pedestrian before averaging
- **FR-002**: The function MUST return a dict with keys following the naming convention: `ped_force_q50`, `ped_force_q90`, `ped_force_q95` (distinct from existing `force_q50`, etc.)
- **FR-003**: For each pedestrian k, the system MUST compute force magnitudes `M_k = { ||F_{k,t}||₂ for t in T_k }` where T_k is the set of timesteps where pedestrian k is present
- **FR-004**: The system MUST compute quantiles `Q_k(q) = Quantile_q(M_k)` for each pedestrian individually using `np.quantile()` or `np.nanquantile()`
- **FR-005**: The system MUST return episode-level values as `mean_k Q_k(q)` - the mean of per-pedestrian quantiles across all pedestrians
- **FR-006**: The function MUST return `NaN` for all quantile keys when there are no pedestrians (K=0), consistent with existing `force_quantiles()` behavior
- **FR-007**: The function MUST handle single-sample pedestrians gracefully (quantiles collapse to that single magnitude value)
- **FR-008**: The system MUST be vectorized using NumPy operations to handle episodes with large T and K efficiently (O(T×K) not O(T×K×Q))
- **FR-009**: The new metric keys MUST be added to `METRIC_NAMES` list in `metrics.py`
- **FR-010**: The new function MUST be called from `compute_all_metrics()` and results merged into the returned dict
- **FR-011**: Documentation in `docs/dev/issues/social-navigation-benchmark/metrics_spec.md` MUST include formal mathematical definition with symbols
- **FR-012**: Unit tests MUST cover: no pedestrians, single pedestrian, multiple pedestrians with varying force patterns, edge case of all-identical forces per pedestrian

### Key Entities

- **EpisodeData**: Existing container with `ped_forces: (T,K,2)` array representing social forces on K pedestrians over T timesteps; each force is a 2D vector
- **Per-Pedestrian Force Magnitudes**: For pedestrian k, the time series `M_k = [||F_{k,0}||, ||F_{k,1}||, ..., ||F_{k,T-1}||]` computed via `np.linalg.norm(data.ped_forces[:, k, :], axis=1)`
- **Per-Pedestrian Quantile**: Scalar value `Q_k(q)` representing the q-th quantile of pedestrian k's force magnitude distribution
- **Episode-Level Per-Ped Quantile**: Scalar mean `(1/K) Σ_k Q_k(q)` representing the average quantile experience across all pedestrians

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: [Measurable metric, e.g., "Users can complete account creation in under 2 minutes"]
- **SC-002**: [Measurable metric, e.g., "System handles 1000 concurrent users without degradation"]
- **SC-003**: [User satisfaction metric, e.g., "90% of users successfully complete primary task on first attempt"]
- **SC-004**: [Business metric, e.g., "Reduce support tickets related to [X] by 50%"]

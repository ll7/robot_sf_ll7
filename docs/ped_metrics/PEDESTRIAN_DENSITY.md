<!--
Purpose: Canonical reference for pedestrian density configuration in Robot SF.
Describes units, recommended ranges, canonical benchmark values, mapping to difficulty, reproducibility guidance, and how tests enforce advisory warnings.
-->

# Pedestrian Density Reference

> Definitive guide to configuring, interpreting, and validating pedestrian density in Robot SF simulations.

## Table of Contents
- [Definition & Units](#definition--units)
- [Canonical Benchmark Densities](#canonical-benchmark-densities)
- [Recommended Operating Range](#recommended-operating-range)
- [Why Not a Hard Restriction?](#why-not-a-hard-restriction)
- [Difficulty Mapping (`SimulationSettings`)](#difficulty-mapping-simulationsettings)
- [Per-Scenario Override (`ped_density`)](#per-scenario-override-ped_density)
- [Choosing a Density](#choosing-a-density)
- [Reproducibility & Benchmark Integrity](#reproducibility--benchmark-integrity)
- [Interaction With Pedestrian Spawning](#interaction-with-pedestrian-spawning)
- [Performance Considerations](#performance-considerations)
- [Testing & Enforcement Policy](#testing--enforcement-policy)
- [Common Pitfalls](#common-pitfalls)
- [Examples](#examples)
- [FAQ](#faq)

## Definition & Units
Pedestrian density is expressed in **pedestrians per square meter (peds/m²)** and governs how many NPC pedestrians are spawned relative to traversable pedestrian zones (sidewalk polygons / spawn zones) on a map.

The effective number of pedestrians is computed approximately as:

```
num_peds ≈ ceil(total_pedestrian_area_m2 * ped_density)
```

The area depends on spawn zone geometry contained in the loaded SVG map.

## Canonical Benchmark Densities
The benchmark uses a *triad* of densities chosen to span light → moderate → moderately heavy flow while preserving interaction diversity without systematic congestion collapse:

| Label  | Density (peds/m²) | Rationale |
|--------|-------------------|-----------|
| low    | 0.02              | Sparse encounters; isolated avoidance maneuvers dominate. |
| medium | 0.05              | Frequent interactions; mixed crossing & lateral negotiation. |
| high   | 0.08              | Near‑crowd conditions without pathological gridlock on small archetype maps. |

These values appear in `configs/scenarios/classic_interactions.yaml` via the `ped_density` field per scenario.

## Recommended Operating Range
Empirically stable range: **0.02 ≤ ped_density ≤ 0.08**.

Below 0.02 the simulation becomes interaction‑sparse and metrics such as near misses or comfort variance lose statistical power. Above ~0.08 (on compact maps) collision risk and local deadlocks rise sharply; runs also become more seed‑sensitive.

## Why Not a Hard Restriction?
Some research workflows require stress‑testing (e.g., high density degradation curves) or ultra-sparse baselines. Therefore Robot SF:

* Emits a **warning** (not an error) in the classic interactions matrix test if a positive density falls outside the advisory range.
* Documents the canonical triad for reproducibility, while allowing exploratory values for experimentation.

## Difficulty Mapping (`SimulationSettings`)
`SimulationSettings.ped_density_by_difficulty` provides a *difficulty ladder* used when environments rely on a single integer `difficulty` index.

Default list (see `robot_sf/sim/sim_config.py`):
```python
ped_density_by_difficulty = [0.01, 0.02, 0.04, 0.08]
```
`peds_per_area_m2` property returns `ped_density_by_difficulty[difficulty]`.

Notes:
* `0.01` is intentionally below the canonical benchmark floor (0.02) to support ultra‑light pedagogical demos.
* You may supply your own list (e.g., `[0.02, 0.05, 0.08]`) before environment creation for tighter benchmarking semantics.

## Per-Scenario Override (`ped_density`)
Scenario YAML files (e.g., `classic_interactions.yaml`) can set `simulation_config.ped_density` directly. This bypasses the difficulty ladder and is the preferred method for explicit benchmark design.

## Choosing a Density
Consider these guidelines:
* **Behavior evaluation (standard)**: Use the triad {0.02, 0.05, 0.08}.
* **Curriculum training**: Start at 0.01 or 0.02 and gradually anneal toward 0.05–0.08.
* **Robustness stress test**: Extend upward incrementally (0.10, 0.12 ...) and inspect collision / near-miss metrics; document deviations.
* **Sparse baseline**: 0.01 when measuring fundamental navigation (path tracking) unaffected by frequent interactions.

## Reproducibility & Benchmark Integrity
When publishing results:
1. Report the exact density values used.
2. If deviating from the triad, justify (e.g., “stress test at 0.10 to probe saturation”).
3. Fix seeds or record the episode manifest (JSONL) to enable re‑sampling of identical population layouts.

## Interaction With Pedestrian Spawning
The density feeds into population sizing in `ped_population` logic. Two spawn strategies (area proportional vs. per-zone) both rely on `peds_per_area_m2`.

Edge effects:
* Very small spawn areas quantize pedestrian counts (ceil) introducing discrete jumps.
* Maps without spawn zones default to zero pedestrians regardless of density.

## Performance Considerations
Higher densities increase:
* Pairwise interaction computations (O(N) to O(N log N) depending on internal spatial structure).
* Risk of slower episode completion due to congestion.
Track performance using the provided performance smoke tests (`scripts/validation/performance_smoke_test.py`).

## Testing & Enforcement Policy
The test `tests/test_classic_interactions_matrix.py` enforces:
* Density must be positive.
* If outside `[0.02, 0.08]` a **warning** is emitted (not a failure).
* `RECOMMENDED_DENSITIES = {0.02, 0.05, 0.08}` and `RECOMMENDED_RANGE = (0.02, 0.08)` are documented for clarity.

Rationale: preserve canonical benchmark comparability while enabling exploratory science.

## Common Pitfalls
| Issue | Cause | Mitigation |
|-------|-------|------------|
| Unexpected zero pedestrians | Missing or malformed spawn zones in SVG | Validate SVG with `examples/svg_map_example.py --strict` |
| Large variance across seeds | Density near saturation (≥0.09) | Reduce density or increase map area |
| Long runtimes / timeouts | Extremely high density | Use performance smoke test; lower density |
| Metrics unstable (near misses ~0) | Density too low (<0.02) | Increase to at least 0.02 |

## Examples
### Override density via scenario YAML
```yaml
scenarios:
  - name: custom_dense_crossing
    map_file: maps/svg_maps/classic_crossing.svg
    simulation_config:
      max_episode_steps: 500
      ped_density: 0.10   # exploratory (warning expected)
    robot_config: {}
    metadata:
      archetype: crossing
      density: experimental_high
      flow: bi
      groups: 0.0
    seeds: [1,2,3]
```

### Programmatic difficulty ladder replacement
```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.sim.sim_config import SimulationSettings

custom_settings = SimulationSettings(
    difficulty=1,  # will refer to 0.05 below
    ped_density_by_difficulty=[0.02, 0.05, 0.08]
)
env = make_robot_env(sim_config=custom_settings)
print(env.sim_config.peds_per_area_m2)  # 0.05
```

## FAQ
**Q: Where is the density actually used?**  
In population sizing inside `robot_sf/ped_npc/ped_population.py` (ceil of area × density) and downstream in physics interactions.

**Q: Can I set different densities for different spawn zones?**  
Not yet; current model applies a uniform global density. You can approximate heterogeneity by editing spawn zone sizes or temporarily forking the population logic.

**Q: Why does the default difficulty ladder include 0.01?**  
To support tutorial / low-interaction demos without modifying benchmark YAMLs.

**Q: Do videos or visual rendering depend on density?**  
Only indirectly—more pedestrians means more rendering load, potentially lowering fps if uncapped.

---
Last updated: 2025-09-23.

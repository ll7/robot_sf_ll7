# Issue #3206 Pedestrian Archetype Reporting Packet

- Status: `composition_report_only`
- Config: `configs/research/pedestrian_archetypes_v1.yaml`
- Population size per composition: `30`
- Claim boundary: No benchmark, realism, or planner-ranking claim. This packet only records deterministic composition assumptions for the shipped speed-archetype MVP.

## Composition Reports

| Composition | Archetypes | Speed factor range | Assignment digest |
|---|---:|---|---|
| `homogeneous_standard` | 1 | 1-1 | `d445fb25af0e` |
| `mixed_balanced` | 3 | 0.7-1.4 | `50feea1c23ac` |
| `rush_hour` | 3 | 0.7-1.4 | `6b37896db37d` |

This packet records deterministic population-composition assumptions only.
It is not a homogeneous-vs-heterogeneous benchmark smoke and does not report metric deltas.

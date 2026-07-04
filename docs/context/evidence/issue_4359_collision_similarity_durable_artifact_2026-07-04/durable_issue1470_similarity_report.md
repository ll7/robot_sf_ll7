# Collision Scenario Similarity Report

- Schema: `collision_scenario_similarity.v1`
- Selected records: 2 / 6

## Groups

| group | size | representative | members |
| --- | ---: | --- | --- |
| group-1 | 1 | `classic_doorway_low--204--2040272edb778cb5` | `classic_doorway_low--204--2040272edb778cb5` |
| group-2 | 1 | `classic_head_on_corridor_low--202--93c37d658b16890f` | `classic_head_on_corridor_low--202--93c37d658b16890f` |

## Nearest Neighbors

| record | neighbors |
| --- | --- |
| `classic_head_on_corridor_low--202--93c37d658b16890f` | `classic_doorway_low--204--2040272edb778cb5` (0.800) |
| `classic_doorway_low--204--2040272edb778cb5` | `classic_head_on_corridor_low--202--93c37d658b16890f` (0.800) |

## Validation Context

- External labels: available (2 selected records with labels; 1 positive).
- Trajectory fields: unavailable (0 selected records).

## Limitations

- Scenario similarity is an analysis aid, not benchmark evidence by itself.
- Distances depend on logged descriptor fields and do not validate external labels.
- Missing trajectory-level fields are excluded rather than imputed.
- External labels and trajectory fields, when present, are descriptive validation context only.

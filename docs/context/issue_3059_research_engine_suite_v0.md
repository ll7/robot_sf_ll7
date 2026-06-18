# Issue #3059 Research-Engine Scenario Suite V0

Issue: [#3059](https://github.com/ll7/robot_sf_ll7/issues/3059)
Status: proposal manifest; not benchmark evidence.

Artifacts:
- Manifest:
  [`configs/benchmarks/issue_3059_research_engine_suite_v0.yaml`](../../configs/benchmarks/issue_3059_research_engine_suite_v0.yaml)
- Contract test:
  [`tests/benchmark/test_issue_3059_research_engine_suite.py`](../../tests/benchmark/test_issue_3059_research_engine_suite.py)
- Catalog entry: [`docs/context/catalog.yaml`](catalog.yaml)

## Purpose

`configs/benchmarks/issue_3059_research_engine_suite_v0.yaml` defines a compact six-family
scenario suite proposal for the Robot SF research-engine roadmap. The suite is intended to anchor
future seed-sufficiency, behavior-model, reporting, and robustness work without forcing each child
issue to invent a different scenario vocabulary.

## Claim Boundary

This is launch infrastructure only. It does not establish benchmark results, planner rankings,
paper-facing claims, calibrated social validity, or evidence that any planner improves navigation.
Fallback, degraded, unavailable, failed, partial, and missing-provenance rows remain visible caveats
and must not be counted as benchmark-strength or suite-strengthening evidence.

## Suite Families

| Family | Scenario IDs | Intended question |
| --- | --- | --- |
| `frame_consistency_sanity` | `empty_map_8_directions_east`, `empty_map_8_directions_north`, `goal_behind_robot` | Basic goal-direction and frame handling before social or topology stressors. |
| `static_obstacle_detour` | `single_obstacle_circle`, `line_wall_detour`, `narrow_passage` | Static-obstacle detour behavior without hidden fallback. |
| `topology_and_local_minima` | `symmetry_ambiguous_choice`, `corner_90_turn`, `u_trap_local_minimum` | Route-progress and local-minimum behavior with explicit diagnostic needs. |
| `paired_pedestrian_interactions` | `single_ped_crossing_orthogonal`, `head_on_interaction`, `overtaking_interaction` | Same-seed single-pedestrian crossing, head-on, and overtaking interactions. |
| `crowd_flow_and_density` | `classic_bottleneck_low`, `classic_group_crossing_high`, `dense_pedestrian_stress` | Crowd-flow diagnostics, denominators, and density-sensitive caveats. |
| `social_protocol_francis` | `francis2023_intersection_wait`, `francis2023_join_group`, `francis2023_narrow_hallway` | Francis-style waiting, joining, and hallway social-protocol diagnostics. |

## Seed Policy

The manifest freezes S5, S10, and S20 seed sets before execution. S5 is the first smoke or
diagnostic set. S10 and S20 are escalation sets for later scheduler issues when uncertainty remains
decision-relevant and endpoint definitions have not changed after protocol freeze.

## Next Proof Step

Bind one family at a time to executable configs and durable result-store rows. A follow-up should
prove row-status provenance, required metrics, and native or explicitly eligible adapter execution
before treating any family as benchmark-candidate evidence.

## Validation

Proposal validation:

```bash
uv run pytest tests/benchmark/test_issue_3059_research_engine_suite.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

<!-- AI-GENERATED (robot_sf#6641, 2026-08-02) - NEEDS-REVIEW -->

# Issue #6641 — Runtime Radius-Binding Canary (Benchmark 6600 Gate 1)

## Plain-language summary

A social-navigation benchmark only reports what it claims when the one selected
collision-envelope radius drives, consistently, every place that uses it. Issue
#6641 adds a small, fail-closed diagnostic — the **radius-binding canary** —
that loads one geometry-sensitive scenario, traces the selected robot and
pedestrian radius through five binding surfaces, and emits a machine-readable
go/no-go verdict per surface. It is `diagnostic-only` evidence: a pre-campaign
binding check, not a benchmark result, and it never changes the frozen
0.0.3.post1 metric semantics.

## Classification and claim boundary

- `schema_version`: `radius_binding_canary.v1`
- `evidence_status`: `diagnostic-only`
- `claim`: on at least one geometry-sensitive scenario, the selected
  collision-envelope radius propagates consistently to all five binding surfaces.
- `not a claim`: it is not benchmark evidence and not a production sweep; the
  feasibility-oracle rollout completion is deterministically stubbed, and the
  radius binding is proven via the oracle's geometric (certifier) margin.
- `frozen metric semantics`: the 0.0.3.post1 metric semantics are read
  (`human_collisions`) but never modified.

## The five binding surfaces and their probes

Each surface is probed through its real code path with a **differential test**:
vary the radius and check the surface's observable output moves with it. A radius
that is silently ignored leaves the observable unchanged, so the probe records a
fail and the overall verdict becomes `no-go`.

| Surface | Real code path probed | Pass criterion |
| --- | --- | --- |
| `simulator_collision_geometry` | `ContinuousOccupancy.is_obstacle_collision` (runtime component) over parsed map obstacle segments | collision flip radius == measured wall distance |
| `obstacle_pedestrian_contact` | `ContinuousOccupancy.is_pedestrian_collision` (contact envelope `robot_radius + ped_radius`) for two radius pairs | contact flip distance == `robot_radius + ped_radius` per pair |
| `feasibility_oracle` | `run_feasibility_oracle` over `make_envelope_scenario` variants at two radii | `\|clearance(a) - clearance(b)\| == \|radius_b - radius_a\|` |
| `metric_metadata_and_output_rows` | `human_collisions` (`EpisodeData.robot_radius`/`ped_radius`) + runner `_scenario_*_radius_m` resolver | same trajectory yields different collision counts when only radii change; resolver returns configured radii |
| `planner_inputs` | runner `_build_observation` → planner `Observation.robot.radius` / `agents[].radius` | observation carries the selected radii |

## Geometry-sensitive scenario

The default scenario is `configs/scenarios/canary_corridor.yaml`
(`atomic_corridor_test.svg`): a 4 m corridor whose route at `y = 9.7 m` sits
1.7 m from the nearest wall (`y = 8.0 m`). That 1.7 m wall distance is the
geometry-sensitive clearance the canary scans against. With the default selected
radii (robot 0.3 m, pedestrian 0.4 m) the verdict is `go` with all five surfaces
`pass` and a flip radius of 1.701 m (within the 1 mm scan step of the 1.7 m wall).

## Run it

```bash
uv run python scripts/benchmark/run_radius_binding_canary_issue_6641.py \
    --scenario configs/scenarios/canary_corridor.yaml \
    --robot-radius 0.3 --ped-radius 0.4 --out verdict.json
```

Exit code is 0 on `go` and non-zero (fail-closed) on any `no-go` surface.

## Owner surfaces

- Engine and probes: `robot_sf/benchmark/radius_binding_canary.py`
- CLI: `scripts/benchmark/run_radius_binding_canary_issue_6641.py`
- Tests: `tests/benchmark/test_radius_binding_canary_issue_6641.py`

## Caveats and limitations

- The feasibility-oracle rollout is deterministically stubbed; the radius
  binding is proven through the oracle's geometric (certifier) margin, not the
  completion rollout. This is labeled explicitly in the verdict.
- The canary probes the canonical `robot_config.radius` selection path. The
  repository documents a known historical divergence of the authoritative 1.0 m
  robot radius across some modules (deferred to issue #4856); that divergence is
  out of scope for this binding check and is not modified.

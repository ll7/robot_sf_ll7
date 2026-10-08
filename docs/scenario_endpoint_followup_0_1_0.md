# Endpoint follow-up evidence for 0.1.0

Diagnostic only. Refs #10225 and #10220. No waiver, certificate, nominal eligibility, evidence admission, or release approval is granted. The twelve findings remain unresolved for certification. Recommendations below are proposed dispositions, not enacted changes.

## Proposed disposition table

The [audited snapshot](scenario_hygiene/endpoint_footprint_0_1_0.json) contains full polygons, exact distances, effective radii and complete SHA256 fingerprints. All rows use the released classic/Francis matrix, with robot radius 1.0 m and pedestrian radius 0.4 m. Contact support is detected at distances ≤1.4 m, including tangency. Positive shortfall below is geometric support overlap, not a measured dynamic collision rate. Fingerprint prefixes identify the exact rows in the JSON.

| Scenario | Endpoint / actor support | Distance / shortfall (m) | Fingerprint prefix | Recommendation and evidence |
| --- | --- | --- | --- | --- |
| `classic_station_platform_medium` | `goal[0]` / `single p1` | 1.000000 / 0.400000 | `fdba9086285f` | Separate goal and p1 lane before nominal use; retain as station-interaction diagnostic meanwhile. Authored single-pedestrian trajectory is active. |
| `classic_head_on_corridor_low` | `goal[0]` / `ped_spawn 0` | 1.274021 / 0.125979 | `ce8bdb4bdfeb` | Separate active crowd spawn support and goal before nominal use; retain head-on diagnostic meanwhile. Crowd support is active at density 0.02; no dormant-population exemption. |
| `classic_head_on_corridor_medium` | `goal[0]` / `ped_spawn 0` | 1.274021 / 0.125979 | `6f1e4100aa72` | Separate active crowd spawn support and goal before nominal use; retain head-on diagnostic meanwhile. Crowd support is active at density 0.05; no dormant-population exemption. |
| `francis2023_frontal_approach` | `goal[0]` / `single h1` | 1.000000 / 0.400000 | `28063dcf993c` | Separate endpoint and h1 lane before nominal use; retain frontal-interaction diagnostic meanwhile. Authored single-pedestrian trajectory is active. |
| `francis2023_pedestrian_obstruction` | `goal[0]` / `single h1` | 1.000000 / 0.400000 | `495b0ab1f1e2` | Author must decide whether occupied-goal interaction is the intended task; require its explicit certification or repair before nominal use. Authored single-pedestrian trajectory is active. |
| `francis2023_robot_overtaking` | `goal[0]` / `single h1` | 1.000000 / 0.400000 | `42e531b4515d` | Separate endpoint and h1 lane before nominal use; retain overtaking diagnostic meanwhile. Authored single-pedestrian trajectory is active. |
| `francis2023_parallel_traffic` | `spawn[0]` / `crowd_route 0` | 0.500000 / 0.900000 | `a1e549ea70fe` | Repair lane/endpoint support separation with both radii and route jitter before nominal use; retain parallel-traffic diagnostic meanwhile. Crowd support is active at density 0.04; no dormant-population exemption. |
| `francis2023_parallel_traffic` | `spawn[0]` / `crowd_route 1` | 0.500000 / 0.900000 | `69d2004a0ef7` | Repair lane/endpoint support separation with both radii and route jitter before nominal use; retain parallel-traffic diagnostic meanwhile. Crowd support is active at density 0.04; no dormant-population exemption. |
| `francis2023_parallel_traffic` | `spawn[0]` / `ped_spawn 0` | 1.000000 / 0.400000 | `5385695a853e` | Repair lane/endpoint support separation with both radii and route jitter before nominal use; retain parallel-traffic diagnostic meanwhile. Crowd support is active at density 0.04; no dormant-population exemption. |
| `francis2023_parallel_traffic` | `spawn[0]` / `ped_spawn 1` | 1.000000 / 0.400000 | `fee33108aed3` | Repair lane/endpoint support separation with both radii and route jitter before nominal use; retain parallel-traffic diagnostic meanwhile. Crowd support is active at density 0.04; no dormant-population exemption. |
| `francis2023_parallel_traffic` | `goal[0]` / `crowd_route 0` | 0.500000 / 0.900000 | `0328e79df94a` | Repair lane/endpoint support separation with both radii and route jitter before nominal use; retain parallel-traffic diagnostic meanwhile. Crowd support is active at density 0.04; no dormant-population exemption. |
| `francis2023_parallel_traffic` | `goal[0]` / `crowd_route 1` | 0.500000 / 0.900000 | `b59d304e23be` | Repair lane/endpoint support separation with both radii and route jitter before nominal use; retain parallel-traffic diagnostic meanwhile. Crowd support is active at density 0.04; no dormant-population exemption. |

Reopen these recommendations after changed endpoint/lane geometry, contact/radius calibration, or an author ruling on the intended interaction contract. Such changes require regeneration and new fingerprints; historical 0.0.8 dispositions cannot authorize these rows.

## Fingerprint regression

`tests/validation/test_endpoint_footprint_snapshot.py` runs the production scenario loader and endpoint auditor under both policies. It compares the 53 historical contacts, 65 full-footprint contacts, twelve additions, per-scenario counts, identities, distances, polygons, radii, and fingerprints with the stored snapshot. Coordinate tuples are serialized to JSON arrays; list ordering is normalized by identity. Provenance and disposition prose are not regenerated or treated as permission.

On reviewed #10220 head, the new test and existing footprint/route-order tests pass: 6 tests. Replacing one stored fingerprint with zeros fails the new test with `Endpoint footprint evidence drift; regenerate and review dispositions`. On current main `77f61d9aaef935f39e848963d36ae25587041a5e`, the test fails because the auditor lacks the `endpoint_policy` argument. This is an unresolved prerequisite, not proof of stale evidence on main. The snapshot is copied byte-for-byte from #10220, pending dependency integration.

Test value: catches released geometry/radius drift or a stale manually edited fingerprint; the nearest synthetic radius tests do not bind the actual released matrix to its stored evidence. It exercises the real loader/audit output without a production seam. A static audit is cheaper than simulation. The snapshot currently matches the dependency, so the deliberate mutation supplies the drift-detection proof; there is no new runtime fix.

## Paired live-crowd comparison

Native runtime source: `40505d5379a94660dc29ce3711e4d3af8471cedb`. Both arms use this identical code; only route-following pedestrian navigators change `require_final_waypoint` from false (historical proximity rule) to true (ordered completion). Diff inspection confirms these are the entire route-completion semantics introduced by #10220. No robot completion policy changes. Seeds 1001–1010, four spawned workers, OMP/MKL/OpenBLAS threads one. Global Python/NumPy and runtime pedestrian RNGs are seeded for each arm; initial pedestrian trajectory frames agree in all pairs.

A stationary robot provides identical actions throughout each authored horizon, retaining native robot–pedestrian forces and obstacle forces. The simulator runs the full horizon rather than terminating on robot success/collision, so this measures pedestrian response under that fixed context. It does not establish equivalence with moving planners, other seeds, or longer horizons. Circular crossing is the geometrically exposed case; parallel traffic exercises the six endpoint findings and multiple crowd routes, while head-on medium exercises a distinct active counterflow crowd.

| Released scenario | Horizon / steps | Population | Group respawns old/new | Pedestrian respawns old/new | Equal trajectory digests |
| --- | --- | --- | --- | --- | --- |
| `francis2023_circular_crossing` | 40 s / 400 | 6 | 33/33 | 39/39 | 10/10 |
| `francis2023_parallel_traffic` | 40 s / 400 | 8 | 46/46 | 58/58 | 10/10 |
| `classic_head_on_corridor_medium` | 50 s / 500 | 4 | 42/42 | 53/53 | 10/10 |

All 30 paired trajectories and respawn counts match exactly. Zero premature group respawns were observed in either arm: the differing rule never activated during these sampled trajectories, despite geometric exposure in circular crossing. This negative result does not negate the deterministic loop/U-turn regression. Respawns count route groups and their member pedestrians after initialization; initial spawn/reset is excluded. Single-pedestrian goal switches are not route respawns.

Every seed’s old/new respawn counts, initial SHA256 and complete pedestrian trajectory SHA256 are preserved in [the paired JSON](scenario_hygiene/route_completion_live_crowd_0_1_0.json). Digests include initial and every post-step x/y/vx/vy frame in stable pedestrian row order, little-endian float64 with little-endian int64 shape prefixes. They are exact for the recorded Python/NumPy environment, not a cross-platform numerical tolerance claim. No fallback or degraded mode is used.

Reproduce after #10220 integration, from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python scripts/validation/compare_pedestrian_route_completion.py \
  --workers 4 --output /tmp/route_completion_live_crowd.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run pytest tests/validation/test_endpoint_footprint_snapshot.py -q
```

The JSON binds the source commit, runner SHA256, released manifest SHA256, environment versions, horizon, population and effective per-arm ordered-completion setting. Released maps and inherited manifests are bound by the source revision and are identical between arms. This follow-up changes diagnostic tooling/tests/docs only; no planner or benchmark runtime behavior changes, so a new empty-world behavior gate is not applicable. #10220’s own gate is separate evidence.

## Why the vendored proximity rule is unused

`robot_sf/sim/simulator.py` builds populations with `robot_sf.ped_npc.ped_population.populate_simulation`, which installs `robot_sf.ped_npc.ped_behavior.FollowRouteBehavior`. Those behaviors use `robot_sf.nav.navigation.RouteNavigator` and are stepped explicitly by `robot_sf.sim.simulator.Simulator.step_once` before physics.

The backend imported as `pysocialforce.Simulator` is the lightweight force/integration wrapper in `fast-pysf/pysocialforce/simulator.py`; it accepts existing state and groups and does not construct or step the vendored route behaviors. Vendored `pysocialforce.navigation.RouteNavigator.reached_destination` belongs to the separate `Simulator_v2`/vendored population path. It remains proximity-only for that standalone API. Updating it would change a separate backend API without benefiting robot_sf’s route completion, so this PR documents the ownership rather than changing vendored semantics.

## Delivery boundary

The follow-up targets main and is blocked until #10220 lands. The comparison and focused integration proof were run against its exact reviewed head, not claimed as passing on current main. No full readiness suite or hosted CI polling was run. Certification remains an author decision under #10225.

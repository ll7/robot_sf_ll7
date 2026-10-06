# Round 3 test value and counterproofs

AI-GENERATED/NEEDS-REVIEW. Tests exercise the real evaluator, route selector,
config builders and sensor outputs. No test-only production seam is added.
Expected preferences, stopping distance and arc sagitta are independently derived.

Each row answers the four test-value questions: protected behavior; credible
regression; nearest existing coverage gap; test-only production seam.

| Test (in `tests/planner/test_hybrid_feasibility_diagnostics.py`) | Protected behavior | Credible regression | Existing coverage gap | Seam |
| --- | --- | --- | --- | --- |
| `test_platform_speed_preference_does_not_saturate_at_comfort_cap` | At exactly 0.50 m surface gap, 0.60/0.80/1.20 m/s preferences are 0.30/0.40/0.5625 (last ramps), and slower can win | Normalize added speeds by comfort cap | Original candidate-extension test checks membership, not objective saturation | None |
| `test_platform_injected_speeds_preserve_nearest_pedestrian_braking_bound` | Added commands never exceed current-position braking bound 0.80 m/s | Bypass safety cap when bypassing comfort band | Original extension test deliberately permitted 1.20 m/s; prediction rejection alone misses current-position bound | None |
| `test_platform_checks_wall_stopping_distance_beyond_rollout_horizon` (static off/on) | A clear 0.20 s horizon cannot admit a command whose full 2.20 m stop hits wall 2 m away | Remove wall tail or tie it to physical-static flag | Existing swept tests cover first plant step, not stopping distance past horizon | None |
| `test_successor_validity_is_independent_and_preserves_legitimate_origin` | Successor validity works without static flag and propagates into hybrid route guide | Couple it back to static geometry or omit guide config | Round-2 terminal test tested only static flag and hybrid's direct selector | None |
| `test_grid_route_successor_validity_survives_config_builder` | Route guide keeps current goal when validity is false, accepts true origin | Ignore bit in grid route or discard flag in builder | Hybrid terminal test never exercises grid builder/selector | None |
| `test_physical_flag_keeps_explicitly_valid_origin_waypoint` | Valid origin remains a successor with physical flag on | Reinstate world-coordinate sentinel heuristic | Round-2 terminal witness covers absent zero, not real origin | None |
| `test_sensor_emits_explicit_validity_only_when_opted_in` | Optional float32 validity distinguishes None from real origin; default keys/spaces unchanged | Encode absence only as zeros or emit field by default | Existing sensor tests cover positions, not missing-versus-origin bit or opt-in space | None |
| `test_debug_unknown_rejection_reason_is_observable_without_crashing` | New reasons become `unknown:<reason>` with null threshold | Raise on unfamiliar rejection reason | Existing debug test covers known dynamic/angular reasons | None |
| `test_physical_sweep_conservatively_covers_grazing_turn_arc` | A 0.10 mm arc overlap is rejected despite clear midpoint chord | Omit conservative turning padding | First-step swept witness covers straight between-sample contact | None |
| `test_goal_validity_survives_real_observation_flattening` (hybrid/grid) | Optional bit survives production flattener into both selectors, valid origin remains usable | Drop unknown goal fields in shared reader | Nested selector/sensor tests skipped the occupancy-grid flattened path | None |
| `test_map_observation_bridge_preserves_optional_successor_validity` | Optional validity survives the map bridge; absent field adds no default key | Mirror only current/next in bridge | Existing bridge tests predate optional successor metadata | None |
| Existing terminal-goal witness (updated) | Explicit absent successor keeps terminal goal under its independent flag | Restore implicit coupling | Same prior test used coordinate-zero assumption | None |
| Existing candidate-extension witness (updated) | Legacy set retained, added speeds respect braking cap, old predicted exclusion remains | Restore cap bypass or remove legacy candidates | Same prior test allowed unsafe added maximum | None |

The eleven new test functions (thirteen parametrized cases) fail using the exact
`14c1adf4` occupancy reader, grid-route, hybrid, sensor and map bridge source bytes in fresh Python processes.
Other dependencies and test fixtures remain identical. The failure reasons are:

```text
comfort-cap normalization saturates added speeds (1.0 versus 0.30/0.40/0.5625)
injected speed exceeds physical braking cap (1.20 > 0.80)
finite horizon misses wall inside full stopping distance (both static variants)
absent successor selected
opt-in validity bit missing at sensor
ValueError: Unmapped evaluator constraint: future_physical_rule
midpoint chord misses grazing arc contact
route guide selects absent successor
world-origin guard discards legitimate successor
flattened observation loses successor validity (hybrid and grid)
map observation bridge drops opt-in successor validity
```

Reproduce the exact-head counterproof (source hashes and failure log are written
under the requested output folder):

```sh
python -m scripts.validation.prove_hybdiag_counterexamples --output output/hybdiag-counterproof
```

The fixed-source command is:

```sh
pytest tests/planner/test_hybrid_feasibility_diagnostics.py -n 8 -q
```

It passes all 23 cases. The source/test hashes and counterproof revision are
recorded in `counterproof.json`; default action/observation witnesses are in
`default-identity.json`. These unit fixtures perform no seeded environment reset.

The three forwarding cases additionally fail at intermediate `d333c08ab`, where
the sensor and nested guards already work, isolating the consumer forwarding
defect. Reproduce with `--base-ref d333c08abe8c32d555c29dd3def1e5cce8c35f91
--group flattened`; hashes and intended reasons are in `plumbing-counterproof.json`.

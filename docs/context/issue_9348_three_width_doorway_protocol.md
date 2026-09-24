# Three-width doorway comparison: preregistered geometry and launch gates

Source: ll7/robot_sf_ll7#9348 and the merged ll7/diss#2669 protocol at
`docs/context/research/2026-09-21_doorway_width_protocol_2669.md`. This records the application
configuration for the 0.0.8 campaign; it reports no comparison result.

## Geometry and source

The historical SVG at `maps/svg_maps/francis2023/francis2023_narrow_doorway.svg`
has a 2.0 m free opening centred on y=5 m and a 1.0 m wall depth. It is an
unchanged reference, not a width-comparison cell. The effective collision
radius is 1.0 m from `robot_sf/common/robot_defaults.py`, audited in
`docs/context/issue_6645_narrow_doorway_radius_binding.md`. The grid/planner
clearance has separate conventions; the grid feasibility oracle is therefore
an additional admission gate, not a substitute for continuous clearance.
At source `120c870d80daba4a06389f9df3c469a2f467da94`, the reference
scenario SHA-256 is
`d7b52f76128f366a74801c2441cb47e3f4af588c64566fb044eb3e2c54e43671`
and the SVG SHA-256 is
`7538ed173d462a5107afc1a1e43b5b2e6d2bc5c9604035cdec9a551e20a8b15e`.

The application manifest at
`configs/benchmarks/issue_9348_three_width_doorway_v1.yaml` freezes these
levels before any result run:

| Width (m) | Width / collision diameter | Continuous clearance (m) |
| ---: | ---: | ---: |
| 2.2 | 1.1 | 0.2 |
| 2.8 | 1.4 | 0.8 |
| 3.6 | 1.8 | 1.6 |

These are the exact levels frozen by diss#2669. On 2026-09-24, the 2.2 m
and 2.8 m variants had positive continuous clearance but the conservative
nominal-radius grid oracle reported no route. The 3.6 m variant had a grid
route. The distinction was observed under
`scripts/validation/run_issue_9348_three_width_doorway_preflight.py`.
The 3.6 m oracle rollout ended with collision/time truncation. These are
preflight findings, not confirmation results. The grid result must stay
visible as a planner/grid feasibility layer; it does not change the frozen
continuous-geometry width labels or authorize counting a grid failure as an
ordinary planner failure. The application preflight reports this check
separately from source/map/radius validity. Its `go` field permits only
diagnostic preflight continuation, never confirmation dispatch.

The #6644 generator moves only the two doorway wall ends symmetrically about
y=5 m. The application tests check that no other SVG element or scenario
condition changes. The same explicit pedestrian h1, pedestrian settings,
robot settings, route, starts, goals, map bounds and 1.0 m wall depth remain.
The historical map stays byte-identical.

## Frozen comparison and admission

The planner roster is `goal` and `social_force`, with exactly seed IDs
225, 226 and 227 under all three widths: 3 × 2 × 3 = 18 planned rows. The native
scenario horizon remains 400 steps. `robot_sf/sim/sim_config.py` defaults to
0.1 s per step, so the intended limit is 40 s; the campaign must record and
verify the effective step duration and all planner/checkpoint/config hashes.
No planner is retrained or tuned by width. The primary comparison is within
planner across width; success and typed collisions are separate endpoints.
Secondary measures are clearance, contact/near-miss exposure, time and
distance, and pedestrian delay or impairment. Time for failures is censored
at termination, not imputed as a successful arrival time. Execution errors,
fallback, degraded rows and unavailable cells are counted separately.

The three seed IDs do **not** yet prove paired realizations. Before campaign
submission, each width cell must record SHA-256 receipts for the initial
actor state and external RNG state, with matching receipts across all three
widths for each pair ID. `build_pair_receipt` hashes canonical reset actors
and the existing simulator counterfactual snapshot's NumPy, Python,
pedestrian-behaviour and residual-adversary RNG state;
`check_pair_receipts` fails closed on missing or unequal receipts. The shared
episode runner must invoke this directly after reset and before any planner
command, attach the receipts to each row, and check all 18 rows before width
comparison. Later closed-loop pedestrian paths may diverge naturally. Until
that hook is integrated and validated, the experiment is **not ready for
confirmation compute**; seed-only equality must not be called paired evidence.

Analysis will use paired resampling over complete realization IDs within
each planner, report uncertainty intervals and denominators for each endpoint,
and list missing/excluded pairs and failure cases. It will attach at least
one recorded trace per mechanism discussed. If paired custody or any grid
classification remains unresolved for an intended planner, the affected
cells remain diagnostic and no width effect is promoted. All results are
simulator internal, without real-world width or safety claims.

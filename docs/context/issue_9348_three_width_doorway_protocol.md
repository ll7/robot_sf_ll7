# Three-width doorway comparison: preregistered geometry and launch gates

Source: ll7/robot_sf_ll7#9348 and the merged ll7/diss#2669 protocol at
`docs/context/research/2026-09-21_doorway_width_protocol_2669.md`. This records the application
configuration for the 0.0.8 campaign; it reports no comparison result.

## H400 delivery plan

Build a serial, fresh-root launcher over the frozen 18 cells, reusing the
existing geometry preflight, episode runner and one `DoorwayPairingSession`.
The within-planner width contrast uses only complete, native planner/seed
pairs. A report must retain per-cell outcomes, failure and exclusion reasons,
per-cell metric denominators, raw paired differences, paired seed bootstrap
intervals, explicit degenerate binary-pair handling, and trace references.
Any missing or unequal reset receipt, changed source/config/asset digest, or
failed or unknown required preflight check stops the run. An expected unavailable
`slow_speed_tier` distributional metric may remain an explicit H1
fallback/degraded diagnostic after a successful, identity-stable rollout; it is
outside the confirmation matrix and never success evidence. Raw rows, source/config/asset hashes, the report
and checksums go to an explicit durable result root; the private operations
queue owns Slurm submission, retrieval and preservation. No H1 smoke is a
confirmation result. Validate with focused synthetic report tests and a
real H1 runner smoke, then the full PR gate after dependent source merges.

The three preserved H1 smokes each recorded exactly one actor-free oracle diagnostic:
`variant_id=gap_3p60__depth_1p00`, `reason=expected_distributional_metric_unavailable`,
and `marker=metrics.distributional_disruption.missing_data.slow_speed_tier.status=unavailable`.
To preserve that report visibility, H400 admission accepts either an empty
`oracle_expected_fallbacks` list or exactly this singleton. The diagnostic is
oracle metadata outside the 18 confirmation cells, so this narrow allowance
does not turn H1 output into H400 success evidence. Any other, multiple,
malformed or unknown diagnostic remains a fail-closed admission error.

The author [approved execution as part of release 0.0.8 on 2026-09-24](https://github.com/ll7/diss/issues/2669#issuecomment-5811968439).
The preregistration's earlier no-execution line applied to writing that protocol,
before this decision. Slurm submission remains subject to the frozen source,
paired-reset receipts and campaign preflight below; authorization alone does not
make any row valid comparison evidence.

## Geometry and source

The historical SVG at `maps/svg_maps/francis2023/francis2023_narrow_doorway.svg`
has a 2.0 m free opening centred on y=5 m and a 1.0 m wall depth. It is an
unchanged reference, not a width-comparison cell. The effective collision
radius is 1.0 m from `robot_sf/common/robot_defaults.py`, audited in
`docs/context/issue_6645_narrow_doorway_radius_binding.md`. The grid/planner
clearance has separate conventions; the grid feasibility oracle is therefore
a separate diagnostic, not a substitute for continuous clearance or a direct
test of either executed policy.
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
separately from source/map/radius validity. Its `go` field requires positive
continuous clearance, available execution, known geometry, and stable runtime
input identity. A known no-route remains a diagnostic and does not make the
width geometrically invalid. A completed 3.6 m rollout with the expected
unavailable `slow_speed_tier` distributional metric is marked fallback/degraded
for H1 diagnostics while remaining executable and identity bound; it does not
authorize confirmation evidence. Unknown geometry, execution failure, or
unstable binding keeps `go` false.

The manifest also records SHA-256 digests for the historical scenario and SVG. Loading the
manifest fails closed if either byte stream changes, so the generator can only create disposable
variants from the reviewed baseline.

The distinction is source-backed. The oracle calls
`robot_sf/scenario_certification/v1.py::_plan_inflated_shortest_path`, which
uses `ClassicGlobalPlanner` A* at the nominal 1.0 m radius with inflation
fallback disabled. The campaign path builds each generated scenario through
`robot_sf/benchmark/map_runner/map_runner_env.py::build_env_config`; the
effective `use_planner` is false at 2.2, 2.8 and 3.6 m (focused headless
config probe, 2026-09-24). Consequently
`robot_sf/gym_env/base_env.py::attach_planner_to_map` returns before attaching
the classic grid planner. The `goal` policy in
`robot_sf/benchmark/map_runner/map_runner.py::_goal_policy` steers toward the
observed goal. The `social_force` adapter in
`robot_sf/planner/socnav_social_force.py::plan_velocity_world` combines goal,
pedestrian and occupancy-grid obstacle forces; its grid is a 0.2 m observation
grid, not an inflated A* route search. Thus the 2.2/2.8 m oracle no-route is
not an automatic executed-planner admission failure. It remains a flagged
grid/certification discrepancy. Any affected row requires normal runtime
and pairing gates before comparison; an oracle exclusion cannot be reported
as an ordinary planner outcome or erased by a successful policy rollout.

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
The preregistered planner construction supplies `algo_config_path: null` for
both planners and resolves each configuration to `{}`. The current map-runner
policy builders record the canonical empty-mapping runtime identity
`44136fa355b3678a` for both planners. The manifest, producer, and row validator
must agree on this runtime identity; any mismatch is rejected. No planner
configuration file may be copied into the campaign inputs or supplied to the
episode runner.
No planner is retrained or tuned by width. The primary comparison is within
planner across width; success and typed collisions are separate endpoints.
Secondary measures are clearance, contact/near-miss exposure, time and
distance, and pedestrian delay or impairment. Time for failures is censored
at termination, not imputed as a successful arrival time. Execution errors,
fallback, degraded rows and unavailable cells are counted separately. H1
execution/binding readiness may record the expected unavailable distributional
metric. Before dispatch, the H400 producer admits only an empty
`oracle_expected_fallbacks` list or the exact singleton recorded by the
preserved H1 smokes above, even when H1 `go` is true; every other, multiple,
malformed or unknown diagnostic is rejected. This admission exception keeps
the actor-free oracle diagnostic visible and does not authorize it as success
evidence. Planner rows remain fail closed: fallback, degraded, non-native,
adapter or incomplete traces are excluded from H400 evidence and cannot be
admitted through this oracle-only path.

The three seed IDs alone do **not** prove paired realizations. Before campaign
submission, each width cell records SHA-256 receipts for the initial
actor state and external RNG state, with matching receipts across all three
widths for each pair ID. `build_pair_receipt` hashes canonical reset actors
and the existing simulator counterfactual snapshot's NumPy, Python,
pedestrian-behaviour and residual-adversary RNG state;
`check_pair_receipts` fails closed on missing or unequal receipts. The shared
episode runner has an opt-in hook after reset and before the first planner
command, attaches the receipt to its row, and checks all 18 rows before width
comparison. An earlier one-step real-runner smoke on 2026-09-24 verified six
complete planner/seed pairs across three distinct map SHA-256 digests, with
equal actor, external RNG and non-width configuration digests within each pair.
That smoke predates the current hook implementation and is historical
diagnostic evidence, not current-head validation. At executable source head
`01bfd35216febb68bad0e328bc268d08d5a0f525` on 2026-09-27,
`scripts/validation/run_issue_9348_paired_reset_smoke.py` was rerun three
times across all 18 cells; each run returned six complete pairs. The compact
reports and generated variants are preserved at
`/home/luttkule/preservation-campaigns/issue9348_h1_smoke_01bfd35_1_20260927/`,
`/home/luttkule/preservation-campaigns/issue9348_h1_smoke_01bfd35_2_20260927/`,
and
`/home/luttkule/preservation-campaigns/issue9348_h1_smoke_01bfd35_3_20260927/`.
Each `smoke.json` has SHA-256
`1a46bb127527deaa9539d7416c7963ff8401e44646daf4153d52d1130fcb745c`; the
respective `SHA256SUMS` file hashes are
`c1a4a78a096c22297ab95ad54f8b3a59e19a2cc79bb5a2c48f9c3b62233d41f7`,
`bb84f3043da2d992dd2a77bbe40774d604f7948780894cb0070c0e451eb08716`, and
`e1b2b81c9c2aa4ecce922799441a926c7506b0fb056c9ae4e339629c20d8b384`.
All rows are diagnostic-only, with
`preflight_admission=diagnostic_only_preflight_ready_with_expected_fallback`
(`go=true`). Positive continuous clearance, available execution, known
geometry, stable loader-bound runtime identity, and the canonical empty-mapping
planner config hash `44136fa355b3678a` for both planners were present. The 2.2
m and 2.8 m conservative grid no-route results were known diagnostic
classifications. The 3.6 m nominal oracle completed with
`termination_reason=success` and stable binding, while the expected unavailable
`slow_speed_tier` distributional metric remained an explicit fallback/degraded
diagnostic. H1 readiness was true, but
`confirmation_oracle_fallbacks_clear=false` and `confirmation_ready` remained
false; the H400 producer refuses this preflight. No row is success evidence.
These smokes do not claim a width comparison or H400 result; they are diagnostic
custody only and must not be substituted for the H400 campaign.
Later closed-loop pedestrian paths may diverge naturally. The full H400
campaign must preserve the same receipts, asset hashes and source commit in
durable storage before a width effect is promoted.

Analysis will use paired resampling over complete realization IDs within
each planner, report uncertainty intervals and denominators for each endpoint,
and list missing/excluded pairs and failure cases. It will attach at least
one recorded trace per mechanism discussed. If paired custody or any grid
classification remains unresolved for an intended planner, the affected
cells remain diagnostic and no width effect is promoted. All results are
simulator internal, without real-world width or safety claims.

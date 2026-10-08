# Authored join and leave groups

[Issue #10028](https://github.com/ll7/robot_sf_ll7/issues/10028) reproduces on
main `dada635b40c32d72219229c807c9bd0fe215b3e9`: both scenarios start and finish
40 seconds of stationary-robot simulation with group sizes `[1, 1, 1]`.
The join controller targets another singleton and its 0.2 m waypoint threshold
is unreachable under social repulsion. The population builder unconditionally
assigns each authored single pedestrian a singleton group.

## Fix and authoring contract

An optional `single_pedestrians[].initial_group_id` assigns shared **runtime**
membership within the authored population. It does not connect to stochastic
crowd groups or to the metric-only `social_groups` o-space declarations. Without
this field, the historical singleton initialization remains unchanged.

`single_pedestrians[].join_distance_m` is an optional, positive finite distance
from the target group centroid. It requires `role: join`; absent values keep the
existing waypoint threshold. The join scenario authors h1/h2 as a group and
sets h3's join distance to 0.8 m, an interaction-distance assumption that allows
joining under the existing social-force law. It does not reduce pedestrian
repulsion or change collision radii. The leave scenario starts h1/h2/h3 together;
h1 leaves on the first behaviour step. Explicitly authored memberships restore
on episode reset, including restoring the joiner's original singleton.
The bound physics engine receives the restored membership and every role
transition before group forces are evaluated; legacy unlabelled populations
retain their existing path.

These two common scenario files are imported transitively by the release matrix.
**Running that matrix on main changes the pedestrian behaviour of its join and
leave rows.** No frozen artifact, released matrix file or freeze branch is
edited, and released numbers are not recomputed or admitted by this diagnostic.

## Development evidence

Runtime revision: `670fe8417f0f35883cb253ca6879e412f6d41339`.
Comparator: fresh main `dada635b40c32d72219229c807c9bd0fe215b3e9`.
Matrix: `configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml`.
Only development seeds 1001, 1002 and 1003 are used. The
[statistics CSV](evidence/issue_10028_authored_groups/group_statistics.csv) records
all 144 scenario/seed reset observations and the two named scenarios after 40 s.

| Scenario | Base reset / 40 s sizes | Fixed reset sizes | Fixed 40 s sizes | Grouped fraction at reset / 40 s |
| --- | --- | --- | --- | --- |
| join_group | `[1,1,1]` / `[1,1,1]` | `[1,2]` | `[3]` | 2/3 / 1 |
| leave_group | `[1,1,1]` / `[1,1,1]` | `[3]` | `[1,2]` | 1 / 2/3 |

On each dev seed, joining completes at 12.6 s. Scenarios with live multi-member
groups at reset increase from **18/48 to 20/48** (union across the three
seeds). Per-seed counts are 14 to 16 (1001), 17 to 19 (1002), and 11 to 13
(1003). Robot spawn jitter and pedestrian population streams are both seeded;
a repeated fixed run returns identical observation rows. Exactly **5/48** explicitly
request groups after the fix: the three classic_group_crossing density variants
and these two authored scenarios (base: three explicit scenarios). The remaining
46 scenarios' reset group statistics are identical across all three seeds.
This is a reset inventory plus stationary-robot transition diagnostic, not a
full interaction benchmark or planner-ranking result.

Reproduce on either checkout (copy only this diagnostic script onto the base):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/audit_authored_group_roles.py \
  --matrix configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml \
  --seeds 1001 1002 1003 --output output/validation/authored_group_roles.json
```

The raw JSONs and test logs are retained in the task handoff evidence; the small
CSV is the durable review artifact. Histogram columns map group size to live
group count; empty retired containers are excluded. Empty after-40-s histograms
mean those 46 scenarios were only inventoried at reset.

## Behaviour gate status

Pending: [#10000](https://github.com/ll7/robot_sf_ll7/issues/10000) requires an
all-arm/all-release-map empty-world sweep via Slurm, a comparison to the fixed
release baseline, classification of every new failure and a full adversarial
review. The task does not authorize Slurm jobs or subagents. A local diagnostic
run and actor-free preflight cannot substitute for those requirements. This PR
must remain blocked until that gate and domain review are supplied; no success
rate is used to tune the fix.

The local all-roster attempt used two dev seeds (1001/1002), four workers, and
`--suite both --no-step-trace` at the earlier runtime revision
`dea98a0249c7c1d2a703ab15349f0932f9405474`. Both suites passed
actor-free preflight. The main suite produced 96 prediction_planner rows and 17
goal rows before the local attempt was stopped, because this heavy execution
cannot satisfy the prescribed Slurm gate. The width suite was not executed.
The local run is **incomplete**, not a gate receipt.

The 113 available rows contain 84 successes, 27 wall-contact terminations and
2 horizon exhaustions. Every observed failure is listed in the
[partial failure table](evidence/issue_10028_authored_groups/empty_world_observed_failures.csv):
26 wall contacts without release comparison, 2 horizon exhaustions without
release comparison, and 1 contact in the explicitly declared classic_doorway_medium
infeasibility probe. The fixed-release comparator and causal traces are absent,
so whether any failure is **new** remains unclassified. These observed outcome
classes must not be promoted to root-cause classifications or gate acceptance.
The full gate remains blocking; the table prevents partial outcomes disappearing
from the handoff.

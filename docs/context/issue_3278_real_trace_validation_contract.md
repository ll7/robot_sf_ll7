# Issue #3278 Real Micromobility Trace Validation Contract

Issue: <https://github.com/ll7/robot_sf_ll7/issues/3278>

Status: `Current` (contract slice). Evidence tier: `blocked` (no real data staged).

## Scope

Issue #3278 wants real-world micromobility traces collected and mapped so that simulation-only
trace predicates (notably `late_evasive_reaction` and `oscillatory_local_control`) can eventually
be checked against measured behavior. Collecting that data is **blocked on external access,
licensing, and provenance acceptance**, so this slice does not collect, ingest, or copy any
external/private data and makes **no real-world validation claim**.

The agent-executable slice delivered here is a *validation-contract checker*: given a metadata-only
descriptor of a candidate dataset, it reports which predicates could be validated, which are blocked
by missing channels, and what data/provenance blockers remain. This lets a future dataset be wired
to the predicate contract the moment access is accepted, without touching private artifacts now.

## What was added

- `robot_sf/analysis_workbench/real_trace_validation_contract.py` — the checker
  (`check_real_trace_validation_contract`) plus loader and dataclasses.
- `robot_sf/analysis_workbench/schemas/real_trace_validation_contract.v1.json` — descriptor schema.
- `configs/benchmarks/issue_3278_real_trace_validation_contract_example.yaml` — a placeholder
  candidate descriptor (no real data; `access_status: blocked`, `provenance_status: pending`).
- `scripts/tools/check_real_trace_validation_contract.py` — CLI wrapper.
- `tests/analysis_workbench/test_real_trace_validation_contract.py` — complete / incompatible /
  missing-metadata fixture tests.

## Descriptor contract (`real_trace_validation_contract.v1`)

A descriptor declares, for a *candidate* dataset:

- `metadata`: `description`, `license`, `provenance_status`
  (`accepted`/`pending`/`unknown`), `access_status` (`available`/`pending`/`blocked`),
  `coordinate_frame`, `units`, optional `source_reference`.
- `available_channels`: dotted trace-field paths the dataset exposes
  (e.g. `robot.position`, `planner.selected_action.angular_velocity`).
- `available_event_labels`: behavior labels the dataset directly annotates (human-coded ground
  truth), mapped to canonical predicate IDs via a small alias table.
- `target_predicates` (optional): defaults to all canonical predicates.

## Minimum sensor channels and field mapping

Required channels per predicate are taken from the single source of truth,
`build_trace_failure_predicate_definitions` (in `trace_failure_predicates.py`), so this checker
never re-lists requirements. An optional `trace_predicate_matrix.v1`
([issue #2688](issue_2688_trace_predicate_matrix.md)) can be unioned in to also enforce the
rate-interpretation matrix fields. Representative mapping:

| Predicate | Minimum channels (mapped to Robot SF trace fields) |
| --- | --- |
| `late_evasive_reaction` | `robot.position`, `pedestrians.id`, `pedestrians.position`, `planner.selected_action.linear_velocity`, `planner.selected_action.angular_velocity` |
| `oscillatory_local_control` | `planner.selected_action.angular_velocity` |
| `clearance_critical_interaction` | `robot.position`, `pedestrians.id`, `pedestrians.position` |
| `occlusion_triggered_near_miss` | `robot.position`, `pedestrians.id`, `pedestrians.position`, `planner.occlusion_or_visibility` |
| `bottleneck_deadlock` | `planner.event` |
| `zero_motion_timeout_behavior` / `low_progress` | `robot.position`, `planner.event` |
| `collision` | `robot.position`/`radius`, `pedestrians.id`/`position`/`radius` |

For real micromobility data, a robot-side `planner.selected_action.*` command is a *proxy*: the
recorded agent's measured velocity/yaw-rate substitutes for the simulated planner action. Document
that substitution per dataset; it weakens any command/response claim.

## What can and cannot be validated

- **Channel-computable** (`late_evasive_reaction`, `oscillatory_local_control`,
  `clearance_critical_interaction`, `collision`): derivable from position/velocity/yaw channels.
  They can be *computed*, but most real datasets carry **no directly observed ground-truth label**
  for them, so they cannot be cross-validated against an observed label — the checker flags this as
  a `computable from channels but no directly observed ground-truth label` limitation.
- **Label-derived** (`bottleneck_deadlock`, `zero_motion_timeout_behavior`, `low_progress`,
  `occlusion_triggered_near_miss`): require an explicit annotated channel (`planner.event`,
  `planner.occlusion_or_visibility`). These are only as reliable as the dataset's labels and are
  blocked when the dataset does not provide them.

## Claim boundary

This is a **contract check only** (`evidence_boundary: contract_check_only_no_real_world_validation`).
It does not establish that any real dataset exists, is licensed, or validates any predicate. A
`contract_status: ready` report means *the declared metadata is complete and the declared channels
cover a predicate* — not that real-world validation occurred. Follow-up implementation issues to
collect/stage data are created only after external access and provenance/license are accepted
through `.agents/skills/data-staging-provenance/SKILL.md` — not from this slice.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/analysis_workbench/test_real_trace_validation_contract.py -q
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/check_real_trace_validation_contract.py
```

Expected: focused tests pass; the example descriptor reports `contract_status: blocked` with
explicit access/provenance blockers (external data not yet staged).

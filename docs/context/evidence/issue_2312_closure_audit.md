# Issue #2312 Closure Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2312>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/1472>
Context note: [`issue_2312_learned_risk_trace_manifest.md`](../issue_2312_learned_risk_trace_manifest.md)

## Purpose

Closure audit for #2312 (`data: materialize durable learned-risk training traces for #1472`).
This note maps each acceptance criterion to merged-PR evidence and to a locally reproduced
validation run, then records the closure decision. It asserts only data-preflight readiness; it
does not materialize trace bytes, run training, or promote any model/benchmark/paper claim.

## Authoritative scope

The issue body's `agent-exec-spec:v1` block (appended 2026-06-20) is the authoritative scoping and
overrides the older issue-body wording. It defines the **agent-executable slice** as the trace
manifest + baseline-artifact-URI contract + fixture validation, and marks trace-byte
materialization as `Blocked-until` the #1472 / #2441 SLURM runs. The 2026-07-05 maintainer state
comment frames the identical residual: "durable trace + baseline artifacts from #1472/#2441 SLURM
runs to replace `:pending` aliases and set `retrieval_status: available`".

Per the repository COMPLETE-FIRST rule, an issue whose only remaining work is a compute/SLURM run
counts as complete for the agent-executable contract.

## Acceptance criteria → evidence

| Acceptance criterion (agent-exec-spec) | Status | Evidence |
| --- | --- | --- |
| Trace manifest + baseline artifact URI contract defined + fixture-validated | Met | PR #3762 (`1368cf139`) — `robot_sf/training/learned_risk_trace_manifest.py`, tracked manifest `configs/training/learned_risk_trace_manifest_issue_2312.yaml`, tests `tests/training/test_learned_risk_trace_manifest.py` |
| Missing/unresolvable traces → fail-closed `artifact_retrieval_blocked`, never an implied training-ready state | Met | PR #3762 validator `scripts/validation/validate_learned_risk_trace_manifest.py` exits `3` with `artifact_retrieval_blocked`; reproduced below |

| Definition-of-Done item | Status | Evidence |
| --- | --- | --- |
| A durable trace manifest is recorded or the missing source is fail-closed | Met | Tracked manifest recorded in honest `blocked` state (PR #3762) |
| A concrete baseline artifact URI is recorded and resolvable | Partial — compute-gated | URI recorded (`wandb-artifact://.../hybrid_rule_v3_static_margin0_waypoint2_baseline:pending`); *resolvable* state requires the #1472 / #2441 SLURM run to replace `:pending` and set `retrieval_status: available` |
| Label availability and checksums are documented | Met | Contract enforces `label_availability` (`collision`, `near_miss`, `low_progress`) and per-URI SHA-256 `checksums`; validated by fixture tests |
| #1472 can be updated to ready or remains blocked with a precise reason | Met | PR #3772 (`720e54048`) `robot_sf/training/learned_risk_campaign_readiness.py` consumes `validate_trace_manifest`; PR #4549 (`11142e2ac`) `build_trace_manifest_status_packet()` + `--status-json` emit the `learned-risk-trace-status.v1` handoff packet |

### Contributing merged PRs

| PR | Commit | Merged | Contribution |
| --- | --- | --- | --- |
| #3762 | `1368cf139` | 2026-06-27 | Durable trace manifest + fail-closed validator (contract + fixtures) |
| #3772 | `720e54048` | 2026-06-27 | Fail-closed campaign-readiness gate that wires the manifest into #1472 readiness |
| #4549 | `11142e2ac` | 2026-07-05 | `learned-risk-trace-status.v1` status packet + `--status-json` CLI for #1472 handoff |

## Reproduced validation (2026-07-06, `origin/main` @ `6ef5836ec`)

```bash
scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/validation/validate_learned_risk_trace_manifest.py \
  --config configs/training/learned_risk_trace_manifest_issue_2312.yaml --status-json
# exit 3; training_readiness_decision = artifact_retrieval_blocked;
# next_action = materialize_durable_trace_and_baseline_artifacts

scripts/dev/run_worktree_shared_venv.sh -- python -m pytest \
  tests/training/test_learned_risk_trace_manifest.py \
  tests/training/test_learned_risk_launch_packet.py -q
# 24 passed
```

The status packet carries the fail-closed decision (`training_readiness_decision:
artifact_retrieval_blocked`, `training_ready: false`) with the seven concrete blockers (three
`:pending` trace/baseline aliases, three `pending` labels, `retrieval_status` not `available`).

## Closure decision

**Close #2312.** Every agent-executable acceptance criterion is met and reproducibly validated on
`origin/main`. The single residual — a *resolvable* baseline/trace artifact URI with
`retrieval_status: available` — is not agent-executable: it requires the #1472 / #2441 SLURM run to
materialize and upload versioned artifacts, which the fail-closed manifest, validator, campaign
gate, and status packet already track precisely. That residual is owned by #1472 (with #2441), so
closing #2312 does not lose the tracking trail.

### Residual (tracked by #1472 / #2441)

- Materialize durable learned-risk trace + baseline artifacts via SLURM; replace the `:pending`
  aliases and set `retrieval_status: available` so `validate_trace_manifest` decides
  `ready_for_training_handoff`. No further agent-executable step is available until those bytes
  exist.

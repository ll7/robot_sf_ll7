# Issue #2273 Learned-Risk Trace Manifest Preflight

Date: 2026-06-05
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2273>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/1472>

## Scope

This note checks whether the learned-risk trace inputs for Issue #1472 are durable and complete
enough for training. It does not train the learned-risk model, run offline diagnostics, generate
new traces, or integrate a planner.

Compact preflight status lives at
`docs/context/evidence/issue_2273_learned_risk_trace_preflight_2026-06-05/trace_status.csv`.

## Evidence Sources

- `configs/training/learned_risk_model_issue_1395_launch_packet.yaml`
- `docs/context/issue_1395_learned_risk_launch_packet.md`
- `docs/context/policy_search/SLURM/001_learned_risk_model_v1.md`
- `docs/context/evidence/issue_1395_learned_risk_launch_packet_2026-05-24/README.md`
- `docs/context/evidence/issue_1395_learned_risk_launch_packet_2026-05-24/trace_contract_fixture.jsonl`
- `docs/context/evidence/issue_1395_learned_risk_launch_packet_2026-05-24/baseline_summary_stub.json`

## Preflight Result

Decision: `insufficient_for_training`.

Confidence: `0.9`.

The #1395 launch-packet validator passes, but it validates the launch-packet shape and one tracked
fixture path. The fixture proves required fields and labels can be represented; it is not a durable
trace manifest for training.

Observed validator summary:

- `status`: `valid`
- `candidate_id`: `learned_risk_model_v1`
- `trace_contract.label_targets`: `collision`, `near_miss`, `low_progress`
- `trace_contract.trace_fixture_count`: `1`
- baseline candidate: `hybrid_rule_v3_static_margin0_waypoint2`
- baseline slices: `stress_slice`, `full_matrix`
- baseline seeds: `111`, `112`, `113`
- learned output role: `auxiliary_cost_only`

## Trace Status

| Surface | Status | Interpretation |
| --- | --- | --- |
| `trace_contract_fixture.jsonl` | `fixture_valid` | Two small tracked rows include required fields and labels, with checksum recorded in the launch packet. This is contract proof only. |
| `baseline_summary_stub.json` | `freeze_stub` | The frozen baseline candidate/slices/seeds are recorded, but the file says to use durable run artifacts before SLURM training. |
| Pending W&B baseline alias | `missing_concrete_uri` | The launch packet still references `wandb-artifact://...:pending`; this is not a concrete artifact pointer. |
| Training trace manifest | `contract_defined_blocked` | Issue #2312 added a tracked manifest (`configs/training/learned_risk_trace_manifest_issue_2312.yaml`) and a fail-closed validator (`scripts/validation/validate_learned_risk_trace_manifest.py`). The manifest contract (durable URIs, checksums, splits, labels) is now mechanically checkable; the decision stays `artifact_retrieval_blocked` until the #1472 / #2441 run materializes the artifacts. See [`issue_2312_learned_risk_trace_manifest.md`](issue_2312_learned_risk_trace_manifest.md). |
| Learned-risk checkpoint | `missing` | Expected after training, not available at this preflight stage. |

## Gate Recommendation

Offline learned-risk diagnostics may not proceed yet. Before Issue #1472 trains or evaluates the
model, the next issue or launch update must record:

- a concrete durable trace artifact URI,
- a trace manifest with row count, scenario ids, seeds, candidate id, required fields, labels, and
  SHA-256 checksums,
- a concrete baseline artifact URI replacing the pending alias,
- the exact training/evaluation command surface that consumes those artifacts,
- the hard-guard diagnostics that prove learned risk remains auxiliary.

If those remain missing, the learned-risk lane should stay in launch-packet/preflight status rather
than become training or planner evidence.

## Claim Boundary

This is data-preflight evidence only. It does not say the learned-risk idea is weak; it says the
current repository state has only a valid launch packet and fixtures, not durable trace inputs for
training.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/validate_learned_risk_launch_packet.py \
  --config configs/training/learned_risk_model_issue_1395_launch_packet.yaml --json
for path in \
  docs/context/issue_1395_learned_risk_launch_packet.md \
  docs/context/policy_search/SLURM/001_learned_risk_model_v1.md \
  docs/context/evidence/issue_1395_learned_risk_launch_packet_2026-05-24/trace_contract_fixture.jsonl \
  docs/context/evidence/issue_1395_learned_risk_launch_packet_2026-05-24/baseline_summary_stub.json; do
  test -f "$path"
done
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

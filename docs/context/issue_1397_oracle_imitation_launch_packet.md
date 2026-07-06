# Issue #1397 Oracle Imitation Launch Packet

Date: 2026-05-24

## Scope

This note records the pre-Slurm launch packet for the oracle-imitation dataset campaign.
It does not generate the full dataset, augment hard slices, train an imitation policy, or
submit Slurm jobs.

## Launch Packet

- Config: `configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml`
- Validator: `scripts/validation/validate_oracle_imitation_launch_packet.py`
- Split contract: `docs/context/policy_search/contracts/oracle_imitation_dataset_split.md`
- Source oracle candidate: `hybrid_rule_v3_static_margin0_waypoint2`
- Source report: `docs/context/policy_search/reports/2026-04-30_best_non_learning_local_policy_report.md`
- Evidence fixture:
  `docs/context/evidence/issue_1397_oracle_imitation_launch_packet_2026-05-24/dry_run_dataset_stub.json`

The source oracle is the best current non-learning local policy identified by the policy-search
report. It is still an experimental, timeout-limited candidate; the launch packet uses it as an
oracle source for a later dataset collection campaign, not as a promoted benchmark replacement.

## Contract Enforced

The new validator fails closed when:

- train/validation/evaluation seeds overlap,
- validation or evaluation seeds drift from `configs/benchmarks/seed_sets_v1.yaml`,
- train seeds overlap `paper_eval_s20`,
- hard-slice examples enter evaluation without predeclaration,
- relabeling is configured outside the train split,
- `generating_commit` is missing or not a Git SHA,
- local artifact files are missing or have checksum mismatches,
- artifact paths depend on worktree-local `output/`, or
- no durable artifact URI is present.

## Validation

Executed locally:

```bash
uv run python scripts/validation/validate_oracle_imitation_launch_packet.py \
  --config configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml --json
```

Result: `status=valid`, `dataset_id=issue_1397_oracle_imitation_v1`, six scenarios, and
twelve planned manifest episode identifiers across train/validation/evaluation.

Targeted tests:

```bash
uv run pytest -q tests/training/test_oracle_imitation_launch_packet.py
```

## Follow-Up Boundary

The later dataset-collection issue should use this branch/commit, config path, validator command,
source candidate, split manifest, hard-slice rules, and durable artifact URI policy. It should
replace the `:pending` W&B artifact aliases with concrete aliases or run ids before collection
starts, then record generated dataset checksums before any imitation training issue begins.

## Downstream closure

The dataset-collection issue #1470 ran to completion on Slurm (#2441, job `12911`) and its traces +
manifest + checksums are tracked at
[`evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17/`](evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17/).
See the [#1470 closure audit](evidence/issue_1470_closure_audit_2026-07-06.md) for the
criterion→evidence mapping. The residual durable trace-URI registry (`training_ready=true`) is owned
by #2655 and the imitation-training benchmark by #1496.

# Issue #3058 Research-Engine Gap Audit 2026-06-18

Issue: [#3058](https://github.com/ll7/robot_sf_ll7/issues/3058)

Status: current preflight audit for the research-engine control plane.

This note compares checked-in experiment cards, registry validation, live GitHub issue state,
artifact pointers, expected artifact classes, configs, commands, and context evidence before new
empirical campaigns start. It does not upgrade any benchmark, metric, model, or paper-facing claim.

Confidence: 0.82 for the routing rows below. The main uncertainty is that live GitHub bodies and
labels can change after the 2026-06-18 snapshot.

## Commands

- `uv run python scripts/tools/validate_experiment_registry.py experiments/registry.yaml`
- `uv run pytest tests/tools/test_validate_experiment_registry.py`
- `gh issue view` for #1108, #1236, #1151, #1219, #1470, #1475, #1488, #2441, #2446, #2777, #2915, #2916, #2924, and #3027
- Targeted reads of `experiments/*.yaml` cards and tracked context evidence referenced in the rows below

The registry validator currently verifies required fields, schema versions, controlled vocabulary,
and `paper_facing` local-output durability. It does not query live GitHub state, reject
`durable_reference_required: true` without a durable reference, reject placeholder paths, or enforce
artifact-class templates for exploratory and paper-candidate cards.

## Required Conflict Examples

| Example | Observed state | Classification | Route |
|---|---|---|---|
| #1108 | Card `experiments/issue_1108_bc_warm_start_ppo.yaml` is `status: closed`; live issue is closed but still has `state:blocked` and `state:needs-artifact-promotion`. Card notes classify the rescue trail as rerun-required and not a fresh-training base. | Closed issue with nonterminal live labels; terminal state is recoverability failure or rerun required, not active experiment evidence. | Explicit fail-closed terminal state. Do not reopen by default. If label hygiene is worth doing, handle it as issue-state cleanup, not a new empirical campaign. |
| #1236 | Card is `status: closed` with exploratory local output and no durable pointer; live issue is closed. | Planned optimizer-backed card is closed, but durable pilot evidence is not visible from the issue body alone. | Keep as terminal non-paper evidence unless a new follow-up asks for optimizer sampler evidence with durable replay proof. |
| #1151/#1219 | Card `experiments/issue_1151_manual_control_mvp_collection.yaml` is `status: completed`; #1151 and child #1219 are closed. Outputs remain local exploratory paths. | Original blocker is resolved; remaining outputs are local diagnostic artifacts, not durable benchmark evidence. | No new issue. Optional PR-backed wording may narrow `completed` to "implementation dependency completed; durable data collection not claimed." |
| #1475 | Card is `status: blocked_on_slurm`; live issue is open with both `state:ready` and `state:running`, while the body still describes not-submitted Slurm work. | Label/body/card disagreement. `state:running` is unsupported by the card/body snapshot; local machine is not SLURM-capable. | Link existing #1475. Treat as ready for a SLURM-capable worker but not locally runnable or benchmark evidence. Clean labels in a separate issue-state pass if desired. |
| #1470/#2441 | #1470 card is `blocked_on_issue_2441`; #2441 card is `blocked_on_slurm` even though tracked trace evidence exists under `docs/context/evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17/`. Live #1470 and #2441 remain open with artifact-promotion/blocking labels. | Strong stale-state split: trace collection evidence exists, but final dataset materialization and promotion are not complete. | Link #1470 and #2441, do not duplicate. Route a PR-backed card/body correction to separate completed trace collection from remaining dataset materialization/promotion. |

## Validator And Artifact Gaps

These are not fixed in this audit because they require a validator-policy decision and tests.

| Surface | Gap | Current validator behavior | Route |
|---|---|---|---|
| `experiments/issue_1475_orca_residual_bc_lineage_smoke.yaml` | Pending artifact aliases and local `output/slurm/...` paths with `durable_reference_required: true`. | Passes because the card is exploratory. | Follow-up [#3084](https://github.com/ll7/robot_sf_ll7/issues/3084) for durable-reference-required enforcement and status gating. |
| `experiments/issue_1108_bc_warm_start_ppo.yaml` | `paper_relevance: paper_candidate` with local output and expected checkpoint artifacts requiring durable references. | Passes because only `paper_facing` is enforced. | Follow-up [#3084](https://github.com/ll7/robot_sf_ll7/issues/3084); do not treat the card as paper evidence. |
| `experiments/issue_1470_oracle_imitation_dataset_collection.yaml` | Expected dataset manifest, NPZ, checksum, and leakage validation require durability but point at local `output/`. | Passes because the card is exploratory and blocked. | Keep blocked; link #2441 and future dataset materialization/promotion rather than creating a duplicate lane. |
| `experiments/issue_2128_heldout_family_transfer_pilot.yaml` | Benchmark-like artifacts lack explicit raw run metadata/log/config/checksum classes. | Passes; artifact class policy is not encoded. | Follow-up [#3084](https://github.com/ll7/robot_sf_ll7/issues/3084) can define whether class templates belong in this validator or a later schema pass. |
| `experiments/issue_2135_seed_sufficiency_followup.yaml` and `experiments/issue_1585_amv_actuation_provenance_lane.yaml` | Placeholder paths such as `<campaign-id>` or `<local-file-or-dir>` remain in registered cards. | Passes because fields are non-empty strings. | Follow-up [#3084](https://github.com/ll7/robot_sf_ll7/issues/3084) for placeholder/status policy; allow only for clearly blocked or proposal cards. |

## Existing Issues To Link Instead Of Duplicate

| Issue | Snapshot state | Route |
|---|---|---|
| #1470 | Open; `state:needs-artifact-promotion`, `evidence:blocked`, `resource:slurm`. | Link as parent oracle-imitation dataset lane. Keep open for dataset materialization/promotion. |
| #1475 | Open; `state:ready` and `state:running`, `evidence:launch-packet`, `resource:slurm`. | Link as ORCA-residual smoke lane; avoid a fresh ORCA-residual audit issue. |
| #1488 | Open; `state:blocked`, `evidence:proposal`, `resource:slurm`. | Link as existing adversarial-search umbrella/closeout mismatch. |
| #2441 | Open; `state:ready` plus `state:needs-artifact-promotion`, `evidence:blocked`, `resource:slurm`. | Link under #1470 as the concrete submit/finalize child; separate trace handoff from downstream dataset work. |
| #2446 | Open; `type:analysis`, `state:ready`, `evidence:analysis-only`, `resource:local`. | Link for actuation-feasibility ranking questions. |
| #2777 | Open; `state:ready`, `evidence:stress`. | Link for live planner observation-noise replay. |
| #2915 | Open; `type:analysis`, `state:ready`, `evidence:proposal`, `resource:local`. | Link for forecast-baseline comparison. |
| #2916 | Open; `type:benchmark`, `state:ready`, `evidence:proposal`, `resource:local`. | Link for same-seed forecast-risk closed-loop coupling. |
| #2924 | Open; `type:analysis`, `state:ready`, `evidence:proposal`, `resource:local`. | Link for counterfactual/mechanism pair evaluation. |
| #3027 | Open; `state:ready`. | Link for standardized scenario-generation toolkit work. |

## Stop Rules

- Closed issues are not reopened by default.
- Local `output/` paths, pending artifact aliases, and exploratory cards are not durable evidence.
- Fallback, degraded, or incomplete rows remain exclusions unless a later issue explicitly measures
  that mode.
- Follow-up work should be issue-specific: validator policy, label hygiene, card correction, or
  dataset materialization. Do not open broad duplicate research campaigns from this audit.

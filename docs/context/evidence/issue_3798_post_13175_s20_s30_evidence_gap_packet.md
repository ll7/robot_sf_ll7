# Job 13175 S20/S30 Evidence-Gap Packet

This packet summarizes retrieved S20 (20-seed) job artifacts for issue #1554, keeps S30 (30-seed) as unexecuted escalation scope, and records why issue #3798 does not promote paper or dissertation claims.

## Claim Boundary

- Status: `diagnostic_only`
- Boundary: diagnostic-only evidence-gap packet; no full benchmark campaign run, no Slurm/GPU submission, no paper/dissertation claim edits, and no S20/S30 claim promotion

## Next Slurm Go/No-Go

- claim_promotion: `no_go`
- s20_archive_readiness: `run_fail_closed_checker_before_claim`
- s30_submission_from_issue_3798: `not_authorized_here`
- s30_escalation_status: `defer_until_claim_owner_authorizes_escalation`
- compute_submission: `not_authorized`
- next_valid_step: `run archive-readiness checker and decide S30 only in a separately authorized lane`

## Retrieved Artifacts

- Artifact root: `/home/luttkule/git/robot_sf_ll7/output/issue1554-s20-h500-l40s-mem180/13175`
- File count: 56
- Total bytes: 185111675
- Missing required metadata files: none

## Campaign Metadata

- campaign_id: `issue1554_s20_h500_l40s_mem180_20260628`
- config_name: `paper_experiment_matrix_v1_scenario_horizons_h500_s20`
- started_at_utc: `2026-06-28T18:25:45.459248Z`
- finished_at_utc: `2026-06-28T20:03:01.128441Z`
- git_commit: `38f921fe374bc954ccc8932bfb055fc021c5b528`
- git_branch: `slurm-issue-1554-s20-h500-l40s-mem180-20260628`
- seed_set: `paper_eval_s20`
- resolved_seed_count: `20`
- scenario_matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- scenario_matrix_hash: `8609d0192098`

## Coverage Snapshot

- Planner rows: 9
- Episode rows in compact seed table: 8640
- Seed count: 20
- Seeds: `[111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]`
- Planners: `['goal', 'hybrid_rule_v3_fast_progress_static_escape', 'orca', 'ppo', 'prediction_planner', 'sacadrl', 'scenario_adaptive_hybrid_orca_v1', 'social_force', 'socnav_sampling']`

## Promotable Review Files

- `campaign_manifest.json` (20775 bytes, sha256 `0dda7ab427c798a2fb018c5c196be698efddc6934fd9c32e7c92ab806ebe44bb`): small metadata/report surface only; diagnostic evidence, not a claim upgrade
- `run_meta.json` (4469 bytes, sha256 `69b40d3e29d3617c4ac1a04abcd0c9819658b3ee41c196f6612fc5ec2a352c96`): small metadata/report surface only; diagnostic evidence, not a claim upgrade
- `reports/campaign_table.csv` (2745 bytes, sha256 `8382e741fdfae6e7835eaf2f6e8a4ec56d1d4e5ae17cdd070409cecf08841368`): small metadata/report surface only; diagnostic evidence, not a claim upgrade
- `reports/campaign_table_core.md` (742 bytes, sha256 `8dc9ab25142a0d8e6b796f14793caf2fd33463070729e6bb0ba773002a62b2f0`): small metadata/report surface only; diagnostic evidence, not a claim upgrade
- `reports/campaign_table_experimental.md` (1323 bytes, sha256 `54fe6c5cf3ba4e579b1e02ccdf118d2f839f5483974988b04a50032ca4f11383`): small metadata/report surface only; diagnostic evidence, not a claim upgrade
- `reports/snqi_diagnostics.md` (2495 bytes, sha256 `6a7fae8df44ecc2c2b36a5206e741cfb7ab8ec876e29b08f6e31f7545b6c5fc0`): small metadata/report surface only; diagnostic evidence, not a claim upgrade
- `reports/statistical_sufficiency.json` (421732 bytes, sha256 `f9ee5c46121dc39a8fa0e647240258e575a2dcad15e712b2695b60b713652385`): small metadata/report surface only; diagnostic evidence, not a claim upgrade

## Claim Blockers

- The retrieved job is S20 only; S30 remains an escalation path, not executed evidence.
- Artifacts live under ignored worktree output and are not yet a canonical campaign result store.
- Archive-readiness still depends on the fail-closed result-store checker before any claim promotion.
- No dissertation, paper, ranking, safety, or significance claim is edited by this packet.

## Validation Commands

- `uv run python scripts/validation/extract_s20_s30_evidence_gap_packet.py --packet-fixture docs/context/evidence/issue_3798_post_13175_s20_s30_evidence_gap_packet.json --markdown`
  - Print this diagnostic packet from tracked metadata without submitting Slurm, requiring ignored output, or editing claims.
- `uv run pytest tests/validation/test_extract_s20_s30_evidence_gap_packet.py`
  - Fixture proof for present, missing, and tracked packet metadata.
- `uv run python scripts/validation/check_s20_s30_archive_readiness.py --json`
  - Fail-closed archive-readiness check; remains the claim gate for a canonical result store.

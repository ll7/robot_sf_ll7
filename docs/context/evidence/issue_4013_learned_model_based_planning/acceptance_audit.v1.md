# Issue #4013 Acceptance Audit

This audit maps issue #4013 acceptance criteria to merged PR evidence and the current real-trajectory readiness gate. It is conservative: diagnostic smoke evidence is not treated as benchmark, navigation-quality, paper, or dissertation evidence.

- Closure status: `partial`.
- Claim boundary: Closure audit and integration report only. Diagnostic smoke evidence is not benchmark, navigation-quality, paper, or dissertation evidence.
- Next empirical action: Provide a reachable checksum-pinned ETH/BIWI staging root, run real-trajectory predictor training, then run representative evaluation versus cv_prediction_mpc and one model-free baseline.

## Criteria

| Criterion | Status | Evidence | Remaining work |
| --- | --- | --- | --- |
| Short-horizon predictor contract and learned backend exist. | `met` | Merged PRs #4474/#4629 established the learned-prediction MPC path.<br>training_manifest.v1.json records a learned short-horizon predictor artifact. | None. |
| Short-horizon predictor trains or fails closed with dataset blocker. | `met` | training evidence_tier='diagnostic-only'.<br>training loss improved=True.<br>real_trajectory_readiness.v1.json records the Phase 3 real-data blocker. | None. |
| Model-based action selection runs on a smoke scenario. | `met` | PR #4644 loaded a checkpoint-backed planner without fallback.<br>comparison status='diagnostic_ready'.<br>comparison paired_seed_count=1. | None. |
| Comparator smoke includes cv_prediction_mpc and a model-free baseline. | `met` | comparison roles=['cv_prediction_mpc', 'learned_prediction_mpc', 'model_free_baseline'].<br>PR #4655 produced the 3-arm diagnostic comparison report. | None. |
| Fallback/degraded rows are excluded or marked non-evidence. | `met` | fallback_degraded_rows={'excluded': 0, 'included_as_non_evidence': 0}. | None. |
| Claim boundary excludes large world-model and paper-grade claims. | `met` | comparison evidence_tier='diagnostic-only'.<br>comparison claim_boundary='diagnostic matched-scenario comparison; not paper-grade benchmark evidence'. | None. |
| Real pedestrian trajectory dataset is reachable and checksum-pinned. | `blocked` | readiness status='blocked_manifest_contract'.<br>readiness blockers=[{'code': 'staging.env_unresolved_for_validated', 'message': "availability 'validated' requires staging.staging_dir to resolve to a local directory; set ROBOT_SF_EXTERNAL_DATA_ROOT or use an output/ path.", 'severity': 'error'}].<br>manifest availability='validated'. | Stage ETH/BIWI or approved real trajectories under ROBOT_SF_EXTERNAL_DATA_ROOT and satisfy the pinned checksum preflight. |
| Real-trajectory predictor training has run on validated data. | `blocked` | No checked-in evidence records a real-trajectory training run.<br>readiness next_action='Stage ETH/BIWI or another approved real pedestrian trajectory dataset under $ROBOT_SF_EXTERNAL_DATA_ROOT, validate its checksum, update the manifest, then run real-trajectory training and representative evaluation.'. | After readiness reaches ready_for_real_trajectory_training, run the real-trajectory trainer and publish compact metrics/manifest evidence. |
| Representative evaluation compares learned predictor against cv_prediction_mpc and a model-free baseline. | `blocked` | Existing comparison_report.v1.json is diagnostic synthetic/checkpoint smoke evidence.<br>Issue thread after PR #4732 still requires real-trajectory training plus representative evaluation. | Run the representative real-trajectory evaluation against cv_prediction_mpc and one model-free baseline after real-data training. |

## Merged PR Evidence

- PR #4474: learned-prediction MPC comparator preflight.
- PR #4587: fail-closed diagnostic comparison report contract.
- PR #4629: seeded synthetic short-horizon predictor training.
- PR #4644: checkpoint-backed planner plan() smoke.
- PR #4655: 3-arm diagnostic comparison report.
- PR #4665: BYO real-trajectory manifest and readiness checker.
- PR #4679: official ETH CVL acquisition URL and instructions.
- PR #4700: real-trajectory trainer data path.
- PR #4704: validated manifest requires reachable staging directory.
- PR #4712: validated manifest requires pinned tree checksum.
- PR #4732: staging checksum surfaced in manifest checker output.

## Blockers Remaining

- Stage ETH/BIWI or approved real trajectories under ROBOT_SF_EXTERNAL_DATA_ROOT and satisfy the pinned checksum preflight.
- After readiness reaches ready_for_real_trajectory_training, run the real-trajectory trainer and publish compact metrics/manifest evidence.
- Run the representative real-trajectory evaluation against cv_prediction_mpc and one model-free baseline after real-data training.

## Intentional Non-Actions

- No raw external dataset staging or redistribution.
- No full benchmark campaign run.
- No Slurm/GPU submission.
- No paper/dissertation claim edits.

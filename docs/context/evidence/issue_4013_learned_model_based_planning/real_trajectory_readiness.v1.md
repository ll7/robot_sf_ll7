# Issue #4013 Real-Trajectory Readiness

Status: `blocked_real_trajectory_data_unavailable`

Claim boundary: real-trajectory readiness only. No raw data is staged, no full benchmark campaign is run, and no paper/dissertation claim is made.

## Dataset Gate

- Manifest: `configs/data/issue_4013_eth_biwi_real_trajectory_manifest.yaml`
- Dataset: `issue_4013_eth_biwi`
- Availability: `missing`
- Benchmark eligibility: `diagnostic_only`

## Blockers

- `real_trajectory.availability_not_validated`: Phase 3 requires a checksum-validated real trajectory dataset; manifest availability is is 'missing'.

## Acceptance Evidence

| Criterion | Evidence |
| --- | --- |
| short-horizon predictor trained | Met at diagnostic tier by PR #4629 using seeded synthetic data; Phase 3 real-trajectory retraining remains blocked until this report reaches ready_for_real_trajectory_training. |
| model-based action selection runs on a smoke scenario | Met at diagnostic tier by PR #4644 and PR #4655. |
| comparison against cv_prediction_mpc and one model-free baseline | Met at diagnostic tier by PR #4655 with paired seed count 1; representative real-trajectory evaluation remains blocked by dataset readiness. |
| fallback/degraded rows are non-evidence | Met by PR #4587 report contract and PR #4655 diagnostic report. |
| claim boundary excludes world-model and paper-grade claims | Met in existing #4013 design/evidence docs; this report preserves that boundary. |

Next action: Stage ETH/BIWI or another approved real pedestrian trajectory dataset under $ROBOT_SF_EXTERNAL_DATA_ROOT, validate its checksum, update the manifest, then run real-trajectory training and representative evaluation.

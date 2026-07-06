# Issue #4013 Real-Trajectory Readiness

Status: `ready_for_real_trajectory_training`

Claim boundary: real-trajectory readiness only. No raw data is staged, no full benchmark campaign is run, and no paper/dissertation claim is made.

## Dataset Gate

- Manifest: `configs/data/issue_4013_eth_biwi_real_trajectory_manifest.yaml`
- Dataset: `issue_4013_eth_biwi`
- Availability: `validated`
- Benchmark eligibility: `research_only`
- Acquisition URL: `https://ethz.ch/content/dam/ethz/special-interest/itet/cvl/vision-dam/documents/ewap_dataset_light.tgz`
- Acquisition fail-closed: `True`

## Blockers

- None.

## Acceptance Evidence

| Criterion | Evidence |
| --- | --- |
| short-horizon predictor trained | Met at diagnostic tier by PR #4629 using seeded synthetic data; Phase 3 real-trajectory retraining remains blocked until this report reaches ready_for_real_trajectory_training. |
| model-based action selection runs on a smoke scenario | Met at diagnostic tier by PR #4644 and PR #4655. |
| comparison against cv_prediction_mpc and one model-free baseline | Met at diagnostic tier by PR #4655 with paired seed count 1; representative real-trajectory evaluation remains blocked by dataset readiness. |
| fallback/degraded rows are non-evidence | Met by PR #4587 report contract and PR #4655 diagnostic report. |
| claim boundary excludes world-model and paper-grade claims | Met in existing #4013 design/evidence docs; this report preserves that boundary. |

Next action: Stage ETH/BIWI or another approved real pedestrian trajectory dataset under $ROBOT_SF_EXTERNAL_DATA_ROOT, validate its checksum, update the manifest, then run real-trajectory training and representative evaluation.

Acquisition instructions:
Obtain the ETH/BIWI walking pedestrians annotation archive from the official ETH Computer Vision Lab datasets page under its current terms. Extract the raw annotation files into staging_dir. Do not commit raw annotations or converted trajectory files. After staging, compute the aggregate SHA-256 tree checksum, set checksums.tree_sha256 and expected_tree_sha256, and change availability to validated before using the data for #4013 real-trajectory training or representative evaluation.

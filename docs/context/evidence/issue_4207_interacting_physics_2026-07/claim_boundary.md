# Claim Boundary

- Diagnostic certification-transfer probe only.
- Provisional gates are reporting thresholds, not certification approval.
- `not_evaluable` cells are fail-closed and never count as `pass`.
- `non_interacting` cells (robot never inside the 5 m pedestrian near field) cannot demonstrate certification robustness; a stable status over them is vacuous, because the social-force model (SFM) / headed social-force model (HSFM) swap was never exercised.
- Learned or predictive arms that run through fallback execution or without a resolved trained checkpoint/config are excluded from trained-planner comparison claims.
- Pass/fail flips are a result: model-assumption fragility in the certification decision.
- No full benchmark campaign, Slurm or GPU submission, retraining, deployment claim, real-world safety claim, or paper/dissertation claim promotion is included.

Claim boundary token: `diagnostic_certification_transfer_probe_no_deployment_claim`

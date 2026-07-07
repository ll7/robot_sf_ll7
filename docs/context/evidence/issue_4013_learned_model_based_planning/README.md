# Issue #4013 Learned-Prediction MPC Evidence

This directory records the durable evidence for issue #4013. The paired diagnostic
comparison has now been **run**: real matched-scenario/seed episode JSONL exists for
all three required arms — `learned_prediction_mpc`, `cv_prediction_mpc`, and a
model-free baseline (`goal`) — and the comparison report reaches `diagnostic_ready`
with no blockers.

Reproduce the full RUN (train checkpoint if missing → run all three arms → build report):

```bash
uv run python scripts/benchmark/run_issue_4013_model_based_comparison.py
```

Or build only the report from existing episode JSONL:

```bash
uv run python scripts/analysis/compare_model_based_planning_issue_4013.py \
  --config configs/analysis/issue_4013_model_based_planning_comparison.yaml
```

The report fails closed unless matched scenario/seed rows exist for all three
arms, fallback/degraded rows are excluded, and the claim boundary remains
diagnostic-only. It does not run a full benchmark campaign or make
paper/dissertation claims.

## Paired diagnostic comparison run (diagnostic-only)

- `comparison_report.v1.json` — full report contract (`schema_version
  issue_4013.model_based_planning_comparison.v1`), status `diagnostic_ready`,
  per-role summaries, closure criteria, world-model exclusions.
- `comparison_report.v1.md` — human-readable comparison table.

Observed local run (scenario `francis2023_blind_corner`, seed 4013, horizon 30,
`dt=0.1`, CPU): all three arms produced one non-fallback evidence episode each
(paired seed count 1), `algorithm_metadata.status=ok` (no fallback/degraded rows),
and the report reached `diagnostic_ready` with all five closure criteria met. The
model-based arm ran on the trained checkpoint with `evidence_tier=checkpoint_loaded`
(no fallback). In this short smoke horizon no arm reached the goal (success rate
0.0 for all three), which is a truthful smoke observation, not a navigation-quality
claim.

Claim boundary: single scenario / single seed diagnostic smoke. It proves the
model-based selection path runs end-to-end and is comparable, paired by
scenario/seed, against a constant-velocity predictor and a model-free baseline. It
is **not** benchmark, navigation-quality, or paper/dissertation evidence, and it is
not a large generative world model. Scaling seeds/scenarios into nominal benchmark
evidence is a separate campaign (excluded here).

## Acceptance audit

`acceptance_audit.v1.json` and `acceptance_audit.v1.md` consolidate the merged
PR evidence for issue #4013 against the current real-trajectory readiness gate.
The audit keeps closure status `partial`: diagnostic predictor training,
checkpoint-backed model-based action selection, the 3-arm comparator smoke,
fallback/degraded exclusion, and claim-boundary criteria are met at diagnostic
tier, but real-trajectory data staging, real-trajectory training, and
representative evaluation remain blocked.

Generate it with:

```bash
uv run python scripts/analysis/audit_issue_4013_acceptance.py --write
```

This audit performs no data staging, no benchmark campaign, no Slurm/GPU
submission, and no paper/dissertation claim edit.

## Trained short-horizon predictor (diagnostic)

The predictor can now be trained on CPU so the model-based arm loads real learned
weights instead of the zero-initialized untrained smoke model:

```bash
uv run python scripts/training/train_learned_short_horizon_predictor_issue_4013.py \
  --config configs/training/learned_short_horizon_predictor_issue_4013_smoke.yaml
```

This writes a local checkpoint plus manifest and metrics. The checkpoint remains
worktree-local; compact manifest and metrics from a local run are promoted here
as durable evidence:

- `training_manifest.v1.json` — schema, architecture, trainer/predictor config, claim boundary.
- `training_metrics.v1.json` — initial/final training loss.

Observed local run (seed 4013, 512 samples, 400 epochs, CPU): training loss fell
from `0.0987` to `0.00044` and the resulting checkpoint loads into
`LearnedShortHorizonPedestrianPredictor` with `evidence_tier=checkpoint_loaded`
(not `diagnostic_untrained_smoke`).

Claim boundary: the training task is a seeded synthetic robot-repulsion
learnability probe, **not** real ETH/UCY pedestrian data. The checkpoint is
`smoke evidence` that the predictor trains and loads without fallback; it is not
benchmark, navigation-quality, or paper/dissertation evidence. The paired smoke
comparison across all three arms has now been run (see the section above); scaling
to nominal benchmark evidence remains a separate campaign.

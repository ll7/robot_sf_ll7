# Issue #3254 Predictive Crossing-Conflict Negative Result

Issue: [#3254](https://github.com/ll7/robot_sf_ll7/issues/3254)

Evidence bundle:
[issue_3254_predictive_crossing_conflict_13042_2026-06-23](evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23/README.md)

Confirming-eval plateau bundle (durably pinned, issue
[#4879](https://github.com/ll7/robot_sf_ll7/issues/4879)):
[issue_3213_predictive_nontransfer_confirming_eval_2026-07-08](evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/README.md).
This `#3213` maneuver-authority sweep evaluated five planner-authority variants across five
predictive-planner checkpoints; closed-loop success stayed on a `0.0667`-`0.1` plateau (the
`0.08696` final-eval point above sits inside it) versus the `0.30` gate, confirming the
predictive-planner non-transfer finding across checkpoints.

## Summary

Slurm job `13042` completed the schema-fixed crossing-conflict predictive training run, but the
trained planner failed the final evaluation gate. The result is useful because it separates the
now-fixed launch/schema problem from the remaining planner-quality problem.

## What Worked

- The run used the `predictive_ego_v1` feature schema.
- The mixed weighted dataset was non-degenerate with `40446` samples.
- Training completed and passed the trajectory-quality gate.
- The best checkpoint reached validation ADE `0.04837` and FDE `0.09735`.
- W&B run id `3tu3tmee` records the training run.

## What Failed

The final evaluation wrote `69` episodes but failed the configured quality gate:

- success rate: `0.08696`
- required success rate: `0.3`
- collision rate: `0.24638`
- mean minimum distance: `2.28687`
- failed episodes: `63`

Minimum-distance evidence passed, but success did not. The run therefore cannot support planner
promotion, benchmark-strength evidence, or a paper-facing claim.

## Decision

Do not blindly resubmit the same #3254 configuration. The next useful step is a public-side
planner-tuning analysis or negative-result synthesis that explains why the learned trajectory
model can fit validation targets while the closed-loop planner still fails the success gate.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23/summary.json
cd docs/context/evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23 && sha256sum -c SHA256SUMS
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_3254_predictive_crossing_conflict_negative_result.md \
  --path docs/context/evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23/README.md \
  --path docs/context/evidence/issue_3254_predictive_crossing_conflict_13042_2026-06-23/summary.json \
  --path docs/context/evidence/README.md \
  --path docs/ai/prediction_lane.md
```

## Claim Boundary

This is an analysis-only negative training result. It is not benchmark-strength planner evidence,
not a safety claim, not model promotion, and not paper-facing evidence.


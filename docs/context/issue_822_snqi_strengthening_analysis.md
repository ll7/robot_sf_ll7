# Issue 822 SNQI Strengthening Analysis

Date: 2026-04-14

Related issues:
- `robot_sf_ll7#822` Strengthen SNQI as an operational multi-objective aggregation metric
- `robot_sf_ll7#635` SNQI v3 paper-facing contract delta and canonical field mapping

## Goal

Decide whether SNQI should remain in the main narrative as an operational multi-objective
aggregation metric for AMV evaluation, or be downgraded to appendix-level synthesis aid.

This note records the evidence-driven conclusion from replaying the frozen publication bundle
against the current SNQI assets.

## Frozen Evidence Set

Use the publication bundle rooted at:

- `output/benchmarks/publication/camera_ready_baseline_safe_issue592_baseline_safe_rvo2_20260401_135310_publication_bundle`

Combined episode set used for analysis:

- `output/ai/autoresearch/snqi_issue822/episodes.jsonl`

This combined JSONL file contains the three planner rows from the frozen publication bundle:

- `goal`
- `orca`
- `social_force`

Total episodes analyzed: `423`

## Analysis Commands

Issue 822 is now implemented in the shared benchmark diagnostics path. The canonical contract
command for future campaigns is:

```bash
uv run python scripts/tools/analyze_snqi_contract.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --weights configs/benchmarks/snqi_weights_camera_ready_v3.json \
  --baseline configs/benchmarks/snqi_baseline_camera_ready_v3.json
```

This emits the strengthened evidence package directly into:

- `reports/snqi_diagnostics.json`
- `reports/snqi_diagnostics.md`
- `reports/snqi_sensitivity.csv`

Those artifacts now include:

- planner ordering by mean SNQI
- component-to-SNQI Spearman correlations
- ablation-based weight sensitivity rows
- an explicit SNQI positioning recommendation with caveats

Implementation proof artifact produced in this worktree:

- `output/ai/autoresearch/snqi_issue822/strengthened_snqi_diagnostics.json`
- `output/ai/autoresearch/snqi_issue822/strengthened_snqi_diagnostics.md`
- `output/ai/autoresearch/snqi_issue822/strengthened_snqi_sensitivity.csv`

Important nuance:

- the strengthened evidence package can recommend keeping SNQI as an operational aggregate even on a
  slice where the legacy contract pass/fail proxy still reports `fail`
- in the available baseline-safe issue-592 slice, the recommendation stayed `strengthen`, the
  planner ordering was `orca > goal > social_force`, and all variable components aligned with the
  expected direction, but the rank-alignment proxy remained negative on that slice

Baseline v1 comparison:

```bash
uv run python scripts/snqi_sensitivity_analysis.py \
  --episodes output/ai/autoresearch/snqi_issue822/episodes.jsonl \
  --baseline output/benchmarks/publication/camera_ready_baseline_safe_issue592_baseline_safe_rvo2_20260401_135310_publication_bundle/payload/reports/snqi_diagnostics.json \
  --weights configs/benchmarks/snqi_weights_camera_ready_v1.json \
  --output output/ai/autoresearch/snqi_issue822/v1 \
  --skip-visualizations \
  --validate \
  --log-level WARNING
```

Candidate v3 comparison:

```bash
uv run python scripts/snqi_sensitivity_analysis.py \
  --episodes output/ai/autoresearch/snqi_issue822/episodes.jsonl \
  --baseline configs/benchmarks/snqi_baseline_camera_ready_v3.json \
  --weights configs/benchmarks/snqi_weights_camera_ready_v3.json \
  --output output/ai/autoresearch/snqi_issue822/v3 \
  --skip-visualizations \
  --validate \
  --log-level WARNING
```

## Baseline Finding

The frozen publication bundle, when evaluated with the current v1 SNQI contract, is not a strong
scientific endorsement for the metric:

- `contract_status`: `fail`
- `rank_alignment_spearman`: `-0.5`
- `outcome_separation`: `-0.14215231703049858`
- `dominant_component`: `jerk_penalty`
- `dominant_component_mean_abs`: `1.0677334363831885`

Interpretation:

- The v1 contract is too dominated by jerk on this slice.
- The score ordering is not aligned with the broader outcome ordering we want reviewers to see.
- This is a useful baseline, but not a good main-narrative contract.

## Candidate v3 Finding

Running the same frozen episodes through the v3 weights/baseline gives a materially better
scientific shape.

Per-episode Spearman correlations between SNQI and the component metrics:

- `success`: `0.274971`
- `time_to_goal_norm`: `-0.269373`
- `near_misses`: `-0.728556`
- `comfort_exposure`: `-0.534039`
- `force_exceed_events`: `-0.532678`
- `jerk_mean`: `-0.358685`

`collisions` is constant on this frozen slice, so its correlation is undefined here.

Sensitivity summary for v3:

- most sensitive weights:
  - `w_near`: `0.14179390412462756`
  - `w_jerk`: `0.05359370202011595`
  - `w_time`: `0.03808595620934895`
- least sensitive weights:
  - `w_force_exceed`: `0.014631333543477513`
  - `w_comfort`: `0.013239354811990444`
  - `w_collisions`: effectively `0.0` on this slice because collisions do not vary
- normalization impact:
  - `median_p90`: `0.98534684739752`
  - `median_max`: `0.9543945690239598`
  - `p25_p75`: `0.9345202948449413`

Planner-level ordering changed from:

- v1: `goal` > `orca` > `social_force`

to:

- v3: `orca` > `goal` > `social_force`

This is the more defensible ordering on the frozen publication bundle because it matches the
success signal better while still preserving the expected worst-case ordering for `social_force`.

## Conclusion

Recommendation: **strengthen SNQI**, but frame it explicitly as an **operational multi-objective
aggregation metric** for AMV evaluation, not as a universal objective utility.

Why:

- the v3 contract improves alignment on the frozen publication bundle,
- the planner ordering is more plausible under v3 than under v1,
- the weight sensitivity is interpretable rather than random,
- and the metric still remains bounded and reproducible.

Required caveat:

- collision signal is degenerate on this frozen slice, so the analysis should not claim that SNQI
  is validated by collision variation here;
- the paper should keep one sentence of caution that SNQI is a benchmark aggregate, not a ground
  truth social-utility scalar.

If the manuscript has to choose between a terse main-narrative treatment and appendix-only
placement, keep SNQI in the main narrative but include the caveat above.

## Validation Notes

The analysis artifacts were written under:

- `output/ai/autoresearch/snqi_issue822/v1`
- `output/ai/autoresearch/snqi_issue822/v3`

These are scratch outputs only. The durable conclusion is this note plus the linked contract note
in `docs/context/issue_635_snqi_v3_paper_contract.md`.

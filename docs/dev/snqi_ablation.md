# SNQI Component Ablation

This tool measures how removing each SNQI component (setting its weight to 0) affects algorithm rankings.

For paper-facing benchmark work, prefer `scripts/tools/analyze_snqi_contract.py` first. That report
now packages the same one-at-a-time ablation signal together with contract health, component
correlations, planner ordering, and positioning guidance. Use
`scripts/tools/analyze_snqi_calibration.py` when the question is broader than component ablation and
you need to compare the fixed v3 contract against local weight perturbations and alternative
normalization anchors. Use `robot_sf_bench snqi-ablate` when you need a standalone rank-shift table
outside the camera-ready contract flow.

What it does
- Computes a base ranking by mean SNQI per group (default: per algorithm).
- For each weight in the SNQI formula, recomputes the ranking with that weight set to 0.
- Reports per-group rank shifts (positive = moved down/worse since SNQI is higher-is-better).

CLI usage

```bash
robot_sf_bench snqi-ablate \
  --in results/episodes.jsonl \
  --out results/ablation.md \
  --snqi-weights model/snqi_canonical_weights_v1.json \
  --snqi-baseline results/baseline_stats.json \
  --format md
```

Options
- --group-by: grouping key (default: scenario_params.algo)
- --fallback-group-by: fallback key (default: scenario_id)
- --format: md | csv | json
- --top: limit analysis to top-N groups by the base SNQI ranking (optional)
- --summary-out: write a compact per-weight impact summary JSON (optional)

API usage

```python
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.ablation import compute_snqi_ablation, format_markdown

records = read_jsonl("results/episodes.jsonl")
weights = {...}
baseline = {...}
rows = compute_snqi_ablation(records, weights=weights, baseline=baseline)
print(format_markdown(rows))
"""

Advanced usage

Limit to top-5 groups and save a summary JSON:

```bash
robot_sf_bench snqi-ablate \
  --in results/episodes.jsonl \
  --out results/ablation_top5.md \
  --snqi-weights model/snqi_canonical_weights_v1.json \
  --snqi-baseline results/baseline_stats.json \
  --top 5 \
  --summary-out results/ablation_summary.json \
  --format md
```

Notes
- Only weights present in the provided weight mapping are ablated.
- Groups missing after ablation (e.g., no valid episodes) keep their base position.
- SNQI is computed with the canonical implementation used across tools.
 - The summary JSON aggregates per-weight rank-shift stats across included groups
   (changed count, mean absolute shift, max absolute shift, positive/negative counts).

Calibration robustness workflow

Use this workflow for issue-level follow-ups that ask whether the fixed v3 SNQI contract should stay
fixed, be demoted further, or seed a future v4 candidate. It does not change benchmark outputs; it
replays frozen episode records through alternative scoring contracts.

```bash
uv run python scripts/tools/analyze_snqi_calibration.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --weights configs/benchmarks/snqi_weights_camera_ready_v3.json \
  --baseline configs/benchmarks/snqi_baseline_camera_ready_v3.json \
  --epsilon 0.15 \
  --output-json output/ai/autoresearch/snqi_issue838/snqi_calibration_analysis.json \
  --output-md output/ai/autoresearch/snqi_issue838/snqi_calibration_analysis.md \
  --output-csv output/ai/autoresearch/snqi_issue838/snqi_calibration_variants.csv
```

What it compares:

- baseline `v3_fixed` weights and anchors,
- simplex-preserving +-epsilon perturbations around each v3 weight,
- `uniform_simplex` and `component_subset_no_jerk` exploratory variants,
- dataset-derived anchor variants: median/p95, median/p90, median/max, and p25/p75.

Anchor variants apply to terms passed through the canonical `normalize_metric` helper;
`time_to_goal_norm` remains an already-normalized episode metric in the current SNQI implementation.

What it reports:

- planner-rank and episode-rank correlation versus v3,
- planner order-change counts,
- component-to-SNQI correlation alignment,
- dominant component magnitude,
- a deterministic recommendation: `keep_v3_fixed`, `demote_snqi_further`, or
  `propose_candidate_v4_contract`.

Interpretation guardrail:

- Treat this as a calibration decision aid for future contracts.
- Do not retroactively change current paper-facing S3/S5 numbers from this analysis alone.
- If publication-bundle artifacts are not mounted locally, run the same command against the closest
  frozen camera-ready campaign and document that input limitation in the context note.

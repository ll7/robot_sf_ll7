# Issue #3063 Campaign Comparison Report

Status: current analysis-only report path for canonical campaign result stores.

Related issue: [#3063](https://github.com/ll7/robot_sf_ll7/issues/3063)

## Goal

Issue #3063 asked for an automated path that turns completed campaign outputs into comparison
tables, uncertainty summaries, visual summaries, and statistical hooks without hiding row-status
caveats.

The accepted surface is `scripts/tools/build_campaign_comparison_report.py`. It consumes the
canonical campaign result-store contract from `scripts/tools/campaign_result_store.py` rather than
parsing ad hoc campaign directories. That keeps the report aligned with required row-status and
artifact-provenance fields.

## Current Boundary

The generated report is `analysis_only`. It is useful for rapid research iteration and handoff, but
it is not benchmark-strength or paper-facing evidence by itself.

The report preserves these caveats:

- `native` and `adapter` rows enter the benchmark-valid denominator.
- `diagnostic_only`, `fallback`, `degraded`, `unavailable`, and `failed` rows are counted and shown
  as limitations instead of successful benchmark evidence.
- Confidence intervals are descriptive normal-approximation summaries.
- Statistical hooks are pairwise mean deltas with sample gates; they do not claim significance.

## Evidence

Tracked fixture input:

- `tests/fixtures/campaign_result_store/issue_3063_episode_rows.json`

Generated compact evidence:

- [evidence/issue_3063_campaign_comparison_report/README.md](evidence/issue_3063_campaign_comparison_report/README.md)
- [evidence/issue_3063_campaign_comparison_report/report.json](evidence/issue_3063_campaign_comparison_report/report.json)
- [evidence/issue_3063_campaign_comparison_report/report.md](evidence/issue_3063_campaign_comparison_report/report.md)

The fixture report has four rows: one `native`, one `adapter`, one `fallback`, and one `degraded`.
It demonstrates denominator/caveat propagation, core metric summaries, social-compliance metric
summaries, visual bars, and descriptive statistical hooks.

## Validation

Focused validation for the implementation:

```bash
uv run ruff check scripts/tools/build_campaign_comparison_report.py scripts/tools/campaign_result_store.py tests/tools/test_build_campaign_comparison_report.py tests/tools/test_campaign_result_store.py
uv run pytest tests/tools/test_build_campaign_comparison_report.py tests/tools/test_campaign_result_store.py -q
uv run python scripts/tools/build_campaign_comparison_report.py --result-store output/issue_3063_campaign_comparison_fixture/result-store --input-label tests/fixtures/campaign_result_store/issue_3063_episode_rows.json --output-json docs/context/evidence/issue_3063_campaign_comparison_report/report.json --output-md docs/context/evidence/issue_3063_campaign_comparison_report/report.md --min-sample 1
```

The transient result store used for evidence generation lived under
`output/issue_3063_campaign_comparison_fixture/result-store` and is disposable. The durable inputs
and compact report artifacts are tracked.

## Follow-Up Boundary

Future campaign manifests can feed this report once they publish a canonical result store. If a
new campaign still only has older camera-ready `reports/` directories, use
`scripts/tools/compare_camera_ready_campaigns.py` for two-campaign drift comparison or first promote
the run into a result store before using the #3063 report path.

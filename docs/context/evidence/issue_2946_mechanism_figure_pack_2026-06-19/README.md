# Issue #2946 Mechanism-Evidence Figure Pack
Generated: 2026-06-19T00:00:00+00:00

## Figures

- 01_panel_default_social_force: `01_panel_default_social_force.png`
  - title: Issue #2428 default Social Force trajectory panel
  - claim_boundary: Single-row, scenario-seeded diagnostic panel; does not imply general AMMV advantage.
  - sources: docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/panels/trajectory_panels/trajectory_panel_default_social_force_classic_head_on_corridor_low_other_classic_head_on_corridor_low--111--b05ccbf52ac7ab9b.png

- 02_panel_ammv_social_force: `02_panel_ammv_social_force.png`
  - title: Issue #2428 AMMV-aware Social Force trajectory panel
  - claim_boundary: Single-row, scenario-seeded diagnostic panel; demonstrates AMMV trace renderability only.
  - sources: docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/panels/trajectory_panels/trajectory_panel_ammv_social_force_classic_head_on_corridor_low_other_classic_head_on_corridor_low--111--5fde302d5414e476.png

- 03_seed_pair_delta_breakdown: `03_seed_pair_delta_breakdown.svg`
  - title: Issue #2432 per-pair frame delta summary
  - claim_boundary: Adapter-mode local head-on seed slice only; all rows in this slice are numerically identical at frame level.
  - sources: docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/candidate_pair_comparison.csv, docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json

- 04_scenario_sweep_delta_summary: `04_scenario_sweep_delta_summary.svg`
  - title: Issue #2434 multi-scenario deltas (classic families)
  - claim_boundary: Adapter-mode compact 5-scenario sweep; no per-scenario frame/metric delta > 0 was found in the recorded outputs.
  - sources: docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/candidate_pair_comparison.csv, docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json

- 05_signalized_row_type_counts: `05_signalized_row_type_counts.svg`
  - title: Signalized row-type eligibility and exclusion
  - claim_boundary: Compliance denominator semantics only; two compatible observables and two denominator-zero excluded rows per source.
  - sources: docs/context/evidence/issue_2753_signalized_crossing_metrics/summary.json, docs/context/evidence/issue_2799_signalized_runtime/summary.json

## Reproducibility

Run: `uv run python scripts/analysis/build_issue_2946_mechanism_figure_pack.py`
Manifest: `docs/context/evidence/issue_2946_mechanism_figure_pack_2026-06-19/figure_pack_manifest.json`

## Follow-up Boundaries

This pack closes the first diagnostic figure-pack request for Issue #2946. Broader mechanism-evidence claims remain routed to existing follow-up lanes Issue #2444, Issue #2754, and Issue #2924.

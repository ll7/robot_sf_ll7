# Issue #2946 Mechanism Figure Pack

Date: 2026-06-19

Status: diagnostic-only pack for evidence hygiene and lane triage.

## Objective

Produce the first compact mechanism-evidence figure bundle for Issue #2946 using only existing tracked inputs from mechanism and signalized evidence workstreams.

## Source issues and tracked inputs used

This figure pack traces explicitly through these issue artifacts:

- Issue #2159
- Issue #2227
- Issue #2428
- Issue #2430
- Issue #2432
- Issue #2434
- Issue #2753
- Issue #2799

Legacy lineage references are documented for continuity but were not used as direct tracked inputs because no durable evidence artifacts were available in this snapshot:

- Issue #2444
- Issue #2754
- Issue #2924
- Issue #2923 (schema contract issue only; no direct runtime/trace payload used for this pack)

These are not unowned follow-ups: Issue #2444, Issue #2754, and Issue #2924 remain the existing
lanes for nonzero AMMV/default divergence, signalized failure-case packaging, and counterfactual
scenario-pair runner work respectively.

## Figures included

This pack contains five figure files under `docs/context/evidence/issue_2946_mechanism_figure_pack_2026-06-19/figures/`:

1. `01_panel_default_social_force.png`
   - Source: `issue_2428` default panel PNG
   - Claim boundary: single paired trace panel example; not generalizable AMMV behavior evidence

2. `02_panel_ammv_social_force.png`
   - Source: `issue_2428` AMMV panel PNG
   - Claim boundary: single paired trace panel example; not generalizable AMMV behavior evidence

3. `03_seed_pair_delta_breakdown.svg`
   - Source: `issue_2432` CSV + summary
   - Claim boundary: adapter-mode Issue #2168 head-on slice (`seeds 111..113`) has zero per-frame deltas for all compared fields

4. `04_scenario_sweep_delta_summary.svg`
   - Source: `issue_2434` CSV + summary
   - Claim boundary: compact five-scenario classic sweep in adapter mode shows zero max deltas in tracked metrics for this slice

5. `05_signalized_row_type_counts.svg`
   - Source: `issue_2753` and `issue_2799` summaries
   - Claim boundary: denominator/exclusion semantics only; two compliant row types and two denominator-zero excluded row types per source

## Reproducibility and provenance

Output bundle and provenance are written by:

- `scripts/analysis/build_issue_2946_mechanism_figure_pack.py`

The generator writes:

- `docs/context/evidence/issue_2946_mechanism_figure_pack_2026-06-19/figure_pack_manifest.json`
- `docs/context/evidence/issue_2946_mechanism_figure_pack_2026-06-19/figure_pack_metadata.json`
- `docs/context/evidence/issue_2946_mechanism_figure_pack_2026-06-19/README.md`

Each output includes source SHA-256 provenance and per-figure source lists.

## Validation

```bash
uv run python scripts/analysis/build_issue_2946_mechanism_figure_pack.py
python -m json.tool docs/context/evidence/issue_2946_mechanism_figure_pack_2026-06-19/figure_pack_manifest.json
python -m json.tool docs/context/evidence/issue_2946_mechanism_figure_pack_2026-06-19/figure_pack_metadata.json
python -m json.tool docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json
python -m json.tool docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json
python -m json.tool docs/context/evidence/issue_2753_signalized_crossing_metrics/summary.json
python -m json.tool docs/context/evidence/issue_2799_signalized_runtime/summary.json
scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2946_mechanism_figure_pack.md --path docs/context/evidence/issue_2946_mechanism_figure_pack_2026-06-19/README.md
```

## Risks / claim boundaries

- These figures are intentionally diagnostic-only and derived from tracked compact artifacts.
- Issue #2432 and Issue #2434 remain adapter-mode slices where no behavioral divergence was found in tracked comparisons.
- Issue #2444, Issue #2754, and Issue #2924 are still relevant context issues, but they are not in the current tracked input set.

## Can #2946 close?

Yes. The issue contract requesting a first compact, provenance-first diagnostic figure pack is
complete. Broader mechanism-evidence claims, including behavioral divergence and benchmark-strength
mechanism closure, remain outside this pack and are routed to existing follow-up lanes Issue #2444,
Issue #2754, and Issue #2924.

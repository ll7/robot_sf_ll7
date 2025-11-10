# Orphan Scan Report

Generated: 2025-11-10 (Feature 242 - Reorganize Docs)

## Purpose
List markdown files under `docs/` not explicitly linked from `docs/README.md` to ensure discoverability.

## Methodology
- Listed all `.md` files under `/Users/lennart/git/robot_sf_ll7/docs/`
- Checked each against central index (`docs/README.md`)
- Identified files not explicitly mentioned

## Results

### Potential Orphans (not directly linked in main index)
These files may be reachable via subdirectory README files or are legacy/deprecated:

- `docs/baselines/social_force.md` - **ACTION**: Already linked via dev/baselines/README.md (no action needed)
- `docs/baseline_table.md` - Generated output file, not a guide (no action needed)
- `docs/curvature_metric.md` - Specialized metric doc (consider adding to "Benchmarking & Metrics" if important)
- `docs/distribution_plots.md` - Mentioned in benchmark.md cross-link (acceptable)
- `docs/fast_pysf_wrapper.md` - Technical deep-dive (consider linking under Architecture if needed)
- `docs/single_pedestrians.md` - Referenced in legacy detailed index (acceptable)
- `docs/scenario_thumbnails.md` - Referenced in legacy detailed index (acceptable)
- `docs/snqi_weight_cli_updates.md` - Historical update notes (acceptable as supplementary)
- `docs/feature_extractors/` - Subdirectory with README (reachable via exploration, acceptable)
- `docs/ped_metrics/` - Subdirectory listed in legacy index (acceptable)
- `docs/dev/` - Engineering notes and issue folders (intentionally not in main index; acceptable)
- `docs/templates/` - Design doc templates (intentionally not in main index; acceptable)
- `docs/refactoring/` - Fully linked via main index (no action)
- `docs/2x-speed-vissimstate-fix/` - Historical design note (acceptable as supplementary)
- `docs/205-complexity-refactoring/` - Historical refactoring (acceptable as supplementary)
- `docs/extract-pedestrian-action-helper/` - Tool docs (acceptable)

### Recommendations
1. **No critical orphans found** - all major guides are indexed.
2. Consider adding to main index if these become primary references:
   - `curvature_metric.md` (if widely used)
   - `fast_pysf_wrapper.md` (if important for integration)
3. Historical/supplementary docs intentionally not in main index (acceptable).

## Conclusion
Central index successfully covers all primary user-facing guides. Subdirectory README files and legacy detailed index provide discoverability for specialized/historical docs.

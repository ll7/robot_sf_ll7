# Issue #2182 Component Effect Synthesis

Status: current synthesis over Issue #2104/#2170/#2180 evidence.

Issue #2182 interprets the completed one-factor h500 run for parent Issue #2104. The durable
component-effect table is:

- `docs/context/evidence/issue_2182_component_synthesis_2026-06-03/component_effects.csv`

## Component Classification

| Component | Classification | Evidence |
| --- | --- | --- |
| Static escape only | neutral | `static_escape_only_minus_base`: success/collision/near-miss/avg-speed deltas all `0.000`; runtime `+27.690s`. |
| Static recenter only | supported | `static_recenter_only_minus_base`: success `+0.056`, avg-speed `+0.075`, no collision or near-miss penalty. |
| Recenter after static escape | supported | `escape_recenter_pair_minus_static_escape_only`: success `+0.111`, avg-speed `+0.116`, no collision or near-miss penalty. |
| Corridor-transit terms | neutral | `grouped_transit_minus_escape_recenter_pair`: success/collision/near-miss deltas all `0.000`. |
| Continuous static checks | trade-off | `continuous_checks_minus_grouped_static`: near-miss `-0.222`, but success `-0.111` and runtime `+11.542s`. |
| Scenario-adaptive ORCA selector | weaker | `selector_only_minus_grouped_static`: success `-0.056`, avg-speed `-0.057`, runtime `-10.289s`. |
| Speed/progress 2.4 | weaker | `speed_progress_2p4_minus_base`: success `-0.056`, near-miss `+0.111`. |

## Acceptance Mapping

| Issue #2104 criterion | Evidence state | Notes |
| --- | --- | --- |
| Freeze a manifest with candidate rows, toggled components, scenarios, planners, and seeds. | satisfied | Issue #2170 manifest and context note freeze the execution contract. |
| Keep one-factor and grouped ablations separate. | satisfied | Issue #2170 separates planned rows; Issue #2180 reports one-factor and grouped-comparator rows separately. |
| Use identical seeds and scenario configs across compared rows. | satisfied | Issue #2180 uses the manifest scenario slice with 18 rows per candidate and zero failed jobs. |
| Emit effect-size tables for success, collision, near miss, low-progress/timeout, and runtime. | mostly satisfied | Success, collision, near-miss, speed, and runtime are preserved in Issue #2180; low-progress/timeout remains available in raw candidate summaries but is not promoted in the compact table. |
| Emit at least one component-contribution figure. | satisfied by table artifact | `component_effects.csv` and the classified Markdown table are the durable figure-like artifact; a chart would not add information for seven rows. |
| Classify observed failures using the failure-mechanism vocabulary where enough evidence exists. | partially satisfied | The h500 run has zero failed jobs and no fallback rows. Detailed episode failure taxonomy is not synthesized here because this issue is about component effects, not trace-level mechanism review. |
| Store source paths, checksums, command, and commit metadata for generated tables/figures. | partially satisfied | Commands, commit, and source paths are recorded in Issue #2180 evidence. Checksums are not added for the small manually promoted CSV because it is tracked directly in git. |

## Conclusion

Issue #2104 has enough local diagnostic evidence to close as a first-pass component ablation. The
strongest actionable result is to keep recentering in future hybrid candidates and deprioritize
static escape alone, corridor-transit terms, selector-only, and speed/progress-2.4 as independent
improvement directions on this slice.

The remaining uncertainty is paper-facing causality: one-row and two-row success deltas over an
18-row slice should guide planner design, not serve as final benchmark claims. A narrower follow-up
would be to retest recentering on a broader scenario or seed slice if the project needs stronger
claim support.

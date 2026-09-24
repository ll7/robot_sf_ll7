# 0.0.8 source audit before the campaign

This audit identifies changes since the frozen 0.0.7 benchmark source that could affect old
episode metrics. It is a preparation record, not acceptance evidence for a 0.0.8 campaign.

## Compared sources and evidence

- Frozen 0.0.7 benchmark source: `07f7e8d43084de748915e1b1eb8b2a1603357c6e`.
- First audit base: `origin/main@120c870d80daba4a06389f9df3c469a2f467da94`.
- Command: `git diff --name-status 07f7e8d43084de748915e1b1eb8b2a1603357c6e..120c870d80daba4a06389f9df3c469a2f467da94 -- robot_sf/benchmark robot_sf/sim robot_sf/ped_npc configs/benchmarks configs/scenarios scripts/tools/run_benchmark_release.py`.
- The 0.0.7 publication archive is the independently preserved issue #9431 bundle, with SHA-256
  checksum `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`.
  Its 20,160 episode identities cover 14 arms, 48 scenarios, and seeds 111–140.

## Changes found before issue #9666 and #9667 merge

| Path | Change | Potential old-metric effect |
| --- | --- | --- |
| `robot_sf/benchmark/metric_layers.py` | Changes `failure_to_progress_rate` source attribution and requires a strict binary `route_complete` input. | Derived metric availability or provenance can change on malformed inputs; valid 0/1 release rows should retain the value, but that remains to be checked on the new campaign. |
| `robot_sf/benchmark/runner.py` | Lazily imports `run_map_batch` and passes episode, seed, and execution identity into an analysis trace. | The import refactor should preserve map execution; the trace payload changes. No direct edit to the episode metric formula was found. |
| `robot_sf/benchmark/analysis_trace.py` | Adds optional trace identity fields. | Trace metadata changes only; no direct episode metric calculation changed. |
| `robot_sf/benchmark/runtime_smoke_admission.py` | Refactors smoke admission logic. | Admission status can move; it is outside episode metric computation, but release gating must use the final source. |
| `robot_sf/benchmark/diagnostic_report.py` | Adds a read-only diagnostic report adapter. | No episode metric writer changed. |
| `configs/benchmarks/issue_9431_trace_worked_examples_v007.yaml` | Adds a diagnostic trace config after the release commit. | It is not the 0.0.7 release campaign config and cannot be treated as frozen release input. |

No pre-existing `robot_sf/benchmark/metrics.py`, simulator force-law, canonical campaign template,
or scenario-matrix file changed in this source range. The final audit must be repeated at the
exact 0.0.8 source commit after all feature PRs merge.

## Mainline refresh before the feature merges

The release worktree merged `origin/main@1e9715dc837a811ae2662bccad37920595f897cb`. A broader
source diff from 0.0.7 exposed additional runtime paths outside the first audit's scope:

| Path | Change | Equivalence risk |
| --- | --- | --- |
| `robot_sf/nav/map_config.py` | Moves plotting imports inside `plot()`. | No intended simulation change; confirm with the executable gate. |
| `robot_sf/sensor/pedestrian_tracking.py` | Adds track observation/lifecycle validation and reset epochs. | Tracking is opt-in through `observation_visibility.include_track_ids` (default false), and no release config override was found. A tracking-enabled path could still reject a track state; no equivalence is inferred without execution. |
| `robot_sf/sensor/socnav_observation.py` | Retains the latest tracking result as a side channel. | The declared public observation schema is unchanged, but the runner path must still be exercised. |
| `robot_sf/planner/scenario_belief_adapter.py` | Adds an identity-safe projection used by the scenario-belief gap-reference hook. | No gap-reference arm appears in the frozen 14-arm roster; confirm the new import is not selected by the release configs. |
| `robot_sf/nav/force_residual_intent_predictor.py` | Adds a force-residual predictor after the first source audit. | No direct reference was found in the selected campaign template; review indirect hooks and compare the actual roster's episode rows. |
| `robot_sf/nav/predictive_types.py` | Clarifies the unset-timestamp documentation. | No executable change was found in this file. |

This refresh widens the final source audit to `robot_sf/nav`, `robot_sf/planner`, and
`robot_sf/sensor`, as well as the benchmark/simulator paths. A small exact-config episode
comparison before the full campaign can detect runtime drift early; only the full 20,160-row
equivalence gate can establish the release claim.

## Equivalence gate prepared

`scripts/validation/check_release_metric_equivalence.py` compares the SHA-pinned 0.0.7 archive
with a candidate campaign by arm, scenario, and seed. It checks every predecessor field under
`metrics` and `metric_values`, plus outcome, status, and executed steps. New metric fields are
permitted; missing old fields, missing or extra identities, and changed finite values beyond
absolute tolerance `1e-12` fail the gate. Existing nonfinite unavailable sentinels must match
their class exactly; they are never counted as finite measurements. It also requires a resolved
manifest at each source and compares the predecessor matrix, scenario, seed policy, planner roster,
kinematics, and legacy Social Navigation Quality Index (SNQI) asset declarations; new v2
weights, anchors, and family path/checksum declarations are the only permitted additions.
Unexpected additions inside the scientific manifest also fail the gate.

For the 0.0.8 candidate, pass `--require-robot-force-metrics` to the same gate. It requires
the four unconditional robot-force reductions on every row, finite and nonnegative; the
per-exposed-pedestrian impulse and mean-active values must be `null` exactly when no pedestrian
was exposed. The experimental pedestrian–pedestrian-equivalent variant may be absent when its
prerequisites are unavailable, but a present variant must contain all six fields with the same
denominator rule. The gate records counts of zero-exposure and absent-variant rows and up to 100
invalid examples. This implements the denominator-aware correction recorded on issue #9668;
`null` is a declared unavailable value, not a finite measurement.

Diagnostic-only check: the force-field gate passed on all 384 source
`71464357ca89e6578ee593a15167bd1eda612d99` rows retrieved from Slurm job `15754`
for issue #9666. It found 91 zero-exposure rows and four rows where the optional
pedestrian–pedestrian-equivalent variant was absent; the four rows had too few time samples
for that variant. The metric layer serializes exposed-pedestrian counts as whole-valued JSON
floats such as `0.0`, so the gate checks finiteness and integrality instead of Python's
integer type. This narrow source/row-shape check does not establish 0.0.8 equivalence or
the full-campaign result.

The gate's archive self-test on the actual 0.0.7 bundle paired all 20,160 rows with zero
mismatches and zero scientific-manifest differences. It found the historical
`min_separation_corrupted_m` field is NaN (not a number) on every row;
420 rows also use NaN in `min_predicted_separation_m`. These are predecessor sentinels, not new
0.0.8 results. Eleven focused comparator tests, the actual-archive manifest extension check,
and Ruff passed. A successor campaign has not yet
been submitted, so the actual 0.0.7-to-0.0.8 equivalence result is unavailable.

## Compute and release boundaries

The tracked 0.0.8 campaign draft is
`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_template.yaml`
(SHA-256 `cd749189831cc6cd940aed695687b52a476f02e0c11f4afeb792e84861ee2774`). Parsed YAML
comparison with the predecessor campaign template found every scientific field identical after
removing the new `snqi_v2_spec` block and resetting the publication-export flag to `true`.
The block names the weights, anchors, and family files; the anchors and their checksum remain
unfrozen until the 1,344-row development-seed calibration is accepted. The export flag is an
execution-stage change: the required force-validation report is produced after the predecessor
equivalence check. This template comparison establishes configuration scope only, not episode
equivalence.

The predecessor is `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml`,
as named by `payload/release/release_manifest.resolved.json` inside the checksum-verified 0.0.7
publication archive. The older
`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml` is not that
archive's canonical campaign config.
Against the same archived resolved manifest, the current template also matches the ordered
14 planner keys and every planner-config path and SHA-256, plus the scenario-matrix and seed-set
paths and SHA-256 checksums. This is a source-byte check before execution; the final campaign
manifest and episode outputs still require the release equivalence gate.

The 0.0.7 full campaign used Slurm job `15715`: 36 allocated CPUs and 03:00:36 elapsed. Its
terminal scheduler state was `FAILED (2:0)` because of a documented post-run custody promotion
error; the accepted 20,160 episode rows and independently preserved publication bundle remain
the benchmark evidence. The 0.0.8 evaluation therefore needs a similar resource budget plus
the separate 1,344-episode calibration split and doorway slice; these are estimates, not run
results.

## Publication sequencing still to close

The release runner normally exports after benchmark acceptance, before the separate
predecessor-metric equivalence and robot-force validation reports can be produced. A complete
Social Navigation Quality Index (SNQI) v2 bundle requires those reports. The 0.0.8 campaign
template therefore defers publication export. The separate
`scripts/tools/finalize_benchmark_data_v008.py` now copies the accepted producer, runs the
two evidence gates, rechecks full release acceptance, and exports a publication candidate.
Six focused finalizer tests passed on provisional source snapshot
`40c18118cdbbc92ca6ab4bc498d80b394adf27df`. The actual full-campaign and cold-bundle
path remains untested until the metric and SNQI-v2 branches merge and the author approves DOI
creation. A failed gate leaves the candidate invalid and the producer unchanged.

The source branch also stages `CITATION.cff` version `0.0.8` with
`configs/releases/release_0_0_8_preparation.yaml` explicitly awaiting maintainer approval.
The version-alignment guard and 41 focused tests passed in the untagged preparation state.
This marker does not authorize a tag, DOI, or publication.

Issue #9431 remains open and a native blocker for #9668. Its existing 0.0.7 Zenodo deposition
22814343 was published on 2026-09-24, and the public archive passed anonymous checksum readback
at the same SHA-256 above. The issue records an explicit release-owner exception to the prospective
doctor's tag-collision gate. A later #9431 readback at 2026-09-24 11:55 UTC found the digital
object identifier (DOI) resolver and DataCite record live (HTTP 200); the published 0.0.7 files
were unchanged.
This programme does not authorize changing the frozen 0.0.7 record. No 0.0.8 tag, Zenodo
reservation, DOI, or publication action follows from this audit.

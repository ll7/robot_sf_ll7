# 0.0.8 source audit before the campaign

This audit identifies changes since the frozen 0.0.7 benchmark source that could affect old
episode metrics. It is a preparation record, not acceptance evidence for a 0.0.8 campaign.

## Compared sources and evidence

- Frozen 0.0.7 benchmark source: `07f7e8d43084de748915e1b1eb8b2a1603357c6e`.
- First audit base: `origin/main@120c870d80daba4a06389f9df3c469a2f467da94`.
- Command: `git diff --name-status 07f7e8d43084de748915e1b1eb8b2a1603357c6e..120c870d80daba4a06389f9df3c469a2f467da94 -- robot_sf/benchmark robot_sf/sim robot_sf/ped_npc configs/benchmarks configs/scenarios scripts/tools/run_benchmark_release.py`.
- The 0.0.7 publication archive is the independently preserved issue #9431 bundle, SHA-256
  `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`.
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

## Equivalence gate prepared

`scripts/validation/check_release_metric_equivalence.py` compares the SHA-pinned 0.0.7 archive
with a candidate campaign by arm, scenario, and seed. It checks every predecessor field under
`metrics` and `metric_values`, plus outcome, status, and executed steps. New metric fields are
permitted; missing old fields, missing or extra identities, and changed finite values beyond
absolute tolerance `1e-12` fail the gate. Existing nonfinite unavailable sentinels must match
their class exactly; they are never counted as finite measurements. It also requires a resolved
manifest at each source and compares the predecessor matrix, scenario, seed policy, planner roster,
kinematics, and legacy SNQI asset declarations; new v2 declarations are permitted.

The gate's archive self-test on the actual 0.0.7 bundle paired all 20,160 rows with zero
mismatches and zero scientific-manifest differences. It found the historical
`min_separation_corrupted_m` field is NaN on every row;
420 rows also use NaN in `min_predicted_separation_m`. These are predecessor sentinels, not new
0.0.8 results. Seven focused comparator tests and Ruff passed. A successor campaign has not yet
been submitted, so the actual 0.0.7-to-0.0.8 equivalence result is unavailable.

## Compute and release boundaries

The 0.0.7 full campaign used Slurm job `15715`: 36 allocated CPUs and 03:00:36 elapsed. Its
terminal scheduler state was `FAILED (2:0)` because of a documented post-run custody promotion
error; the accepted 20,160 episode rows and independently preserved publication bundle remain
the benchmark evidence. The 0.0.8 evaluation therefore needs a similar resource budget plus
the separate 1,344-episode calibration split and doorway slice; these are estimates, not run
results.

Issue #9431 remains open and a native blocker for #9668. Its existing 0.0.7 Zenodo deposition
22814343 was published on 2026-09-24, and the public archive passed anonymous checksum readback
at the same SHA-256 above. The issue records an explicit release-owner exception to the prospective
doctor's tag-collision gate. A later #9431 readback at 2026-09-24 11:55 UTC found the DOI resolver
and DataCite record live (HTTP 200); the published 0.0.7 files were unchanged.
This programme does not authorize changing the frozen 0.0.7 record. No 0.0.8 tag, Zenodo
reservation, DOI, or publication action follows from this audit.

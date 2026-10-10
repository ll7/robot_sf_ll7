# Robot SF benchmark data 0.0.8 — main track only

The [author's 2026-10-09 amendment](main_only_scope_amendment_20261009.md)
approves publication of the main benchmark only. Publication execution is a
separate gated operation; this document makes no claim that it has occurred.

The main dataset contains 20,160 cells: 14 arms × 48 scenarios × 30 evaluation
seeds, source `373dbfde4f39667cf9e8732dabe7df5118bdeab1`, with authored scenario
budgets retained. It passed the public main release acceptance and strict bundle
reconciliation. SNQI-v2 reports, raw/component metrics and provenance are retained
unchanged. Job 21880's combined chain nevertheless ended `FAILED`, exit `2:0`,
because the separate doorway track refused; this is not a successful two-track
release claim.

## Excluded doorway track

The three-width doorway dataset is excluded from 0.0.8 and deferred to 0.1.0.
Its matrix bound the stale `predictive_mppi_camera_ready.yaml`: a 12-step planning
horizon against the registered predictor's eight-step forecast. Predictive_mppi
produced zero episode rows; risk_dwa was not started because `stop_on_failure`
was enabled. Twelve other arms completed 1,080 rows, retained as incomplete
diagnostic evidence. They are not an accepted doorway dataset and are never
pooled into the main denominator.

The successor must use a reviewed compatible binding and **fresh seeds**, with
new scientific admission and complete acceptance; see
[the 0.1.0 doorway work item](https://github.com/ll7/robot_sf_ll7/issues/10284).
No failed arm is counted as successful, omitted silently, or patched into the
0.0.8 producer results.

## Retained notes and correction lineage

The [original F3 release notes](https://github.com/ll7/robot_sf_ll7/blob/373dbfde4f39667cf9e8732dabe7df5118bdeab1/docs/release/0.0.8/release_notes.md)
remain immutable. Their lines 58 and 102 describe the planned doorway inclusion
and admission contract. The [issued amendment](main_only_scope_amendment_20261009.md)
supersedes that scope; the retained text is not a statement that doorway passed.

The bound description's stale “calibration failed” wording is also retained
under the author's route-A ruling. A separately reviewed metadata-only correction
successor under concept DOI `10.5281/zenodo.23150471` must correct that wording and
state the main-only exclusion lineage, while retaining the published predecessor
and every scientific/result byte unchanged. It is not a new experiment or a
retroactive claim that the two-track chain passed.

## Main integrity and public identity

The original main archive and sidecars are the only three assets in this
publication. The archive basename remains
`issue9850_release_snqi_v2_0_0_8_373dbfde4f_r39_20261008_publication_bundle.tar.gz`.

| Asset | SHA-256 |
| --- | --- |
| Main archive | `0b4585705c0655982ac64e94ff09bb3e65340c26494cb386c4ff0c0c21f1bbfc` |
| `checksums.sha256` | `15b5e5267a80f1716492282b95463811809c745a6da3d1a0a254ca79b688be42` |
| `publication_manifest.json` | `f4669cd2e2670e4a381512bc2d20f5fb53c22b6d560a18f36a26f2034bcff452` |

The exact source tag is
`paper-matrix-v2-h600-s30-2026-10-373dbfde4f39667cf9e8732dabe7df5118bdeab1`.
The reserved version DOI is `10.5281/zenodo.23150472`; the concept DOI remains
`10.5281/zenodo.23150471`. Before citing them as published results, verify public
asset readbacks and the final publication receipt.

[Issue #10261](https://github.com/ll7/robot_sf_ll7/issues/10261) affects the
separate policy-analysis outcome assembler at F3. The main benchmark runner uses
exclusive collision/success versus timeout assembly. Read-only inspection of all
20,160 stored main outcomes found zero such contradictions; that policy-analysis
fix changes no main reported number. This does not claim that every possible
simulator or evaluator defect has been excluded. No simulation or result rewrite
was performed for this check.

Cross-arm episode-ID reuse is intentional resume identity, not duplicate main
acceptance cells. Public acceptance keys each cell by arm, kinematics, scenario
and seed. All 20,160 such cells are unique. The private chain's global-ID check
is a separately tracked consumer defect and does not invalidate this dataset.

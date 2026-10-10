# 0.0.8 main-only scope amendment — 2026-10-09

**Author-approved scope; publication execution remains gated.** The author issued
this ruling in chat to the orchestrator on 2026-10-09 at approximately 13:50 UTC:

> Option 1 approved — publish robot_sf_ll7 0.0.8 with the main track only; the
> doorway track is disclosed as failed (stale predictive-MPPI binding) and moves
> to 0.1.0 with the fixed binding and fresh seeds. Reopen only on evidence that
> the main bundle is invalid.

The effective 0.0.8 dataset is the independently accepted main benchmark at
source `373dbfde4f39667cf9e8732dabe7df5118bdeab1`: 14 arms × 48 scenarios ×
30 evaluation seeds, 20,160 cells, with its authored scenario budgets unchanged.
No doorway rows enter its denominator. This records the issued author ruling;
it does not assert that the tag, GitHub release or DOI has already been published.

Job 21880 completed main but failed in the separate doorway track. That matrix
selected the older `predictive_mppi_camera_ready.yaml`, whose 12-step planning
horizon exceeds the registered checkpoint's eight-step forecast. Twelve doorway
arms completed 1,080 rows; predictive_mppi produced no episode rows and
`stop_on_failure` prevented risk_dwa from starting. The doorway output remains
incomplete diagnostic evidence. There is no successful two-track chain result.

## Correction and exclusion lineage

This ruling supersedes two statements of **planned scope** in the retained F3
notes:

- [Line 58, “The three-width doorway slice stays in 0.0.8”](https://github.com/ll7/robot_sf_ll7/blob/373dbfde4f39667cf9e8732dabe7df5118bdeab1/docs/release/0.0.8/release_notes.md#L58): the slice is now excluded and deferred to 0.1.0.
- [Line 102, doorway slice admission](https://github.com/ll7/robot_sf_ll7/blob/373dbfde4f39667cf9e8732dabe7df5118bdeab1/docs/release/0.0.8/release_notes.md#L102): the planned 14-arm/1,260-cell doorway contract was not satisfied. Main remains the separate accepted 20,160-cell dataset.

The source-bound notes, original producer bytes, identities, manifests,
checksums, scientific inputs, seeds and failed attempt record remain unchanged.
This is an additional scope/correction record, not an edit to those bound bytes.
The [publication disclosure](publication_disclosure_main_only.md) carries this
lineage into the GitHub release body. Do not describe partial doorway results
as an admitted dataset, add them to main, or synthesize a successful chain receipt.

The reserved version DOI remains `10.5281/zenodo.23150472`; its concept DOI
remains `10.5281/zenodo.23150471`. The author's 2026-10-05 route-A instruction
still requires publishing the bound description unchanged, then a separately
reviewed metadata-only `benchmark-release-erratum.v1` successor correcting the
stale “calibration failed” wording and disclosing this scope lineage. The old
F2-specific successor plan is historical; any actual successor must bind the
published F3 main predecessor and unchanged scientific bytes. No new reservation,
restamping, numerical correction or simulation is authorized by this document.

## Follow-up and gates

- [Predictor compatibility and a real planning-step preflight, #10283](https://github.com/ll7/robot_sf_ll7/issues/10283).
- [Doorway successor with fixed binding and fresh seeds, #10284](https://github.com/ll7/robot_sf_ll7/issues/10284).

Independent publication review must accept the exact three-asset main inventory,
source/tag/DOI identity, disclosure and cold readbacks before execution. Scope
approval does not waive those gates. The reopen clause is **evidence that the
main bundle is invalid**; elapsed time, the already-disclosed doorway refusal or
legitimate cross-arm resume IDs do not reopen it by themselves.

# 0.0.8 doorway companion claim scope

This supplemental publication disclosure records the orchestrator's reversible
2026-10-05 ruling. It accompanies the exact-source release notes; it does not
change the immutable F2 execution source or its disclosure-admission receipt.
See the [runbook](runbook.md#companion-claim-admission--ruling-of-2026-10-05)
for publication admission.

The fixed H400 2.2/2.8/3.6 m doorway companion is reported with **raw metrics
only** in 0.0.8: success, collision, time and the other unscored metrics.
**SNQI v2 applicability to this slice was not established.** Box 3 is resolved
by this explicit scope exclusion, **not by a PASS**. No review box is ticked by
this disclosure.

The frozen companion configuration, template and campaign code remain unchanged.
The sealed companion run continues to compute SNQI v2 diagnostics. Those values
remain in the raw companion data; the publication README and dataset data
dictionary must label all such columns:

> not validated for this slice (box 3 inconclusive, 2026-10-05)

SNQI v2 diagnostics for this slice are **not admitted as results, not ranked and
not compared**, either across doorway widths or against the main campaign.
Publication must preserve raw values while applying this claim boundary to
release prose, tables, plots and dataset documentation. It must not publish
SNQI-derived rankings or comparative claims for this companion.

## Evidence and reopening

[Slurm 21364 preserved evidence](https://wandb.ai/ll7/robot_sf/artifacts/campaign-preservation/campaign-issue9667_snqi_box3_doorway_f2_dev1004_1005_20261004/v0)
contains the predeclared criteria, all 112 development rows, per-width/per-seed
U/D tables, 28 control classifications and custody receipts. The recorded
outcome is **INCONCLUSIVE** under precedence 4(a). All feasible-width U/D counts
are below 7/14; these are facts without an applicability claim. The infeasible
2.0 m control has eight contact rows, from goal, ppo, prediction_planner and
sacadrl in both seeds; the orchestrator's 2026-10-05 ruling treats these as
deterministic planner behaviour rather than measurement inconsistency. This
ruling does not reinterpret the predeclared outcome as PASS.

Reopen only through a newly predeclared doorway applicability test in 0.0.9,
tracked by [issue #10140](https://github.com/ll7/robot_sf_ll7/issues/10140).
There is no expansion to dev seeds 1006–1030 under the old packet. An independent
review and a new scope ruling are required before admitting SNQI v2 results for
this slice. Other release, custody, DOI and sealed-execution gates remain separate.

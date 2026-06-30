# SNQI Governance Disposition Packet

Plain-language summary: current Social Navigation Quality Index (SNQI) governance diagnostics are visible and fail closed, but owner decisions remain before any canonical SNQI weight or normalization claim can be made.

- Schema: `snqi_governance_disposition_issue_3800.v1`
- Packet status: `blocked_on_owner_decisions`
- Source preflight status: `failed`
- Claim boundary: secondary_diagnostic_only: this preflight reports unresolved SNQI governance blockers from issues #3723 and #3699. It does not choose canonical weights, change normalization, change compute_snqi output, or make SNQI a primary safety ranking.

## Summary

- Open governance issues summarized: 2
- Resolved provenance checks separated here: 6
- Remaining owner decisions separated here: 4
- Claim-boundary notes separated here: 4

## Issue Dispositions

### Issue #3723: Conflicting canonical-labeled SNQI weight sets

- Current status: `decision-required`

Resolved provenance checks:
- Weight-source inventory enumerates code default and shipped JSON weight sets.
- Fail-closed preflight exposes blocking canonical-label conflicts.
- User guide documents the current provenance conflict without selecting a winner.

Remaining owner decisions:
- Choose the single canonical SNQI weight source, or explicitly retire the canonical label.
- Update shipped artifact metadata and fail-closed tests after the canonical decision.

Claim-boundary notes:
- Current checks are secondary diagnostic evidence only.
- The packet does not make SNQI a primary safety ranking or dissertation claim.

Source references:
- docs/snqi-weight-tools/weights_provenance.md
- robot_sf/benchmark/snqi/weights_inventory.py
- scripts/validation/check_snqi_governance.py

### Issue #3699: Mixed SNQI normalization basis

- Current status: `decision-required`

Resolved provenance checks:
- Normalization inventory mirrors the current compute_snqi term-scaling regimes.
- Combined governance preflight reports mixed raw and baseline-normalized penalty terms.
- Optional baseline coverage check reports missing normalized median/p95 inputs.

Remaining owner decisions:
- Choose whether to normalize raw terms or document the bounded raw-term asymmetry.
- Only after that decision, update scoring semantics, docs, and tests together.

Claim-boundary notes:
- The packet does not change emitted SNQI values.
- Mixed-scale diagnostics must stay caveats for benchmark or paper-facing SNQI claims.

Source references:
- docs/snqi-weight-tools/weights_provenance.md
- robot_sf/benchmark/snqi/normalization_inventory.py
- scripts/validation/check_snqi_governance.py

## Read-Only Boundary

- This packet does not run a benchmark campaign.
- This packet does not submit Slurm or GPU work.
- This packet does not edit dissertation, paper, or benchmark claims.
- This packet does not change SNQI scoring, weights, normalization, or artifacts.

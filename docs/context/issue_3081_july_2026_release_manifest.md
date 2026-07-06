# Issue #3081 July 2026 Release Manifest

Issue: <https://github.com/ll7/robot_sf_ll7/issues/3081>

## Summary

Issue #3081 tracks the July 2026 research package release and claim audit. It establishes a fail-closed preflight checklist to ensure that all release prerequisites (independent reproduction, tables/figures regeneration, claim card validation, and sprint issue closure ledger) are completed and audit-ready before publication.

This document serves as the release manifest gating the ready status of the `release_july_2026` package in the research package registry.

## Release Prerequisites & Checklist Configuration

The checklist is declaratively configured in `configs/benchmarks/releases/release_july_2026_preflight_issue_3081.yaml`. It maps acceptance criteria to the following required durable artifacts:

1. **Reproduction Record**: [reproduction_record.md](file:///home/luttkule/git/robot_sf_ll7/docs/context/evidence/release_july_2026/reproduction_record.md)
   Tracks the clean-worktree reproduction commands and verification details.
2. **Tables & Figures Manifest**: [tables_figures_manifest.json](file:///home/luttkule/git/robot_sf_ll7/docs/context/evidence/release_july_2026/tables_figures_manifest.json)
   Checksum manifest mapping generated publication assets to their canonical source digests.
3. **Claim Cards**: [claim_cards.yaml](file:///home/luttkule/git/robot_sf_ll7/docs/context/evidence/release_july_2026/claim_cards.yaml)
   Contains the verified research claims, auditing them to ensure no claims rely on degraded or fallback execution rows.
4. **Sprint Issue Ledger**: [sprint_issue_ledger.yaml](file:///home/luttkule/git/robot_sf_ll7/docs/context/evidence/release_july_2026/sprint_issue_ledger.yaml)
   Assures every sprint issue in the release scope is closed or has an approved terminal classification.

## Validation

The preflight checklist is validated using the release preflight checker tool:

```bash
uv run python scripts/tools/run_release_preflight.py \
  --checklist configs/benchmarks/releases/release_july_2026_preflight_issue_3081.yaml \
  --repo-root . \
  --out-json docs/context/evidence/release_july_2026/preflight_report.json \
  --out-md docs/context/evidence/release_july_2026/preflight_report.md
```

This verifies the status of all prerequisites. The overall preflight fails closed and remains `blocked` until all listed evidence files are fully populated.

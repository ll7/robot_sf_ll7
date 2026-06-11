# Issue #2523 Scenario-Prior Smoke Evidence

This bundle contains one compact `scenario_prior.v1` proxy artifact for the first
scenario-prior smoke path.

SDD was not staged locally when this issue ran: `python scripts/tools/manage_external_data.py list`
reported `sdd` as missing. The artifact therefore uses a small embedded proxy fixture that covers
the #2479 representation shape without claiming real-data provenance, realism, training readiness,
benchmark usefulness, or planner performance.

Files:

- `scenario_prior.v1.json`: proxy scenario-prior artifact with source/provenance status,
  representation fields, prior weights, and claim boundary.
- `summary.json`: compact review summary and validation commands.
- `SHA256SUMS`: checksums for the tracked evidence files.

Current classification: `proxy_schema_adequate_for_smoke`. The next valid step is to stage one
license-approved SDD annotations tree and regenerate this artifact shape from importer outputs.

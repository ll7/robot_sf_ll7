# Issue #1609 Seed Mechanism Synthesis Evidence

This bundle is a derived diagnostic synthesis over tracked Issue #1608 and Issue #1454 compact
evidence. It does not contain raw trajectories, videos, or benchmark output.

Files:

- `mechanism_synthesis_summary.json`: machine-readable mechanism-status table for the 25
  seed-sensitive scenarios from Issue #1608.
- `mechanism_synthesis_table.csv`: compact review table with hard-vs-easy aggregate metrics.

Boundary:

- Evidence class: diagnostic mechanism prioritization.
- Claim boundary: aggregate-supported, trace-limited hypotheses; not causal proof and not
  paper-facing significance evidence.
- Source bundle: `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/`.
- Source classification: `docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30/`.

Generation used the tracked Issue #1608 scenario table plus tracked Issue #1454
`seed_episode_rows.csv`, restricted to the top-four planners selected by Issue #1608.

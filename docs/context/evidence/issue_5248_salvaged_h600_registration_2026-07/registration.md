# Salvaged trace-capable h600 registration receipt

- Status: `blocked_campaign_registration`
- Job: `13334`
- Claim boundary: Registration readiness only: this receipt verifies campaign structure and trace-label coverage for the issue #4206 reanalysis. It does not establish benchmark, planner, paper, or dissertation claims.

| Check | Observed |
| --- | --- |
| Total episodes | 6480 / expected 6480 |
| Execution status | `completed` |
| Episode rows | 6480 |
| Trace-labeled rows | 2415 (0.373) |
| Minimum trace-labeled fraction | 0.5 |

## Mechanism-label sidecar (post-hoc derivation)

- Path: `docs/context/evidence/issue_4831_trace_verified_failure_mechanisms/mechanism_labels.csv`
- SHA-256: `6bab8e40d58bc9254e28c0a9c66898e9c42403f592acd6369e3c3b25560396b1`
- Sidecar rows: 2612 (matched 2612 campaign rows)
- Raw-row trace-labeled fraction (without sidecar): 0.000

## Blockers

- trace-verified labeled fraction must meet preregistration minimum 0.500; got 0.373

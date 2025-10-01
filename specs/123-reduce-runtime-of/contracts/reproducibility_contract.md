# Contract: Accelerated Reproducibility Test

**Purpose**: Define the behavioral assertions for the optimized reproducibility integration test.

## Preconditions
- Feature code & benchmark orchestration imported without side effects.
- Minimal scenario matrix (1 scenario) resolvable.
- Root seed defined (e.g., 12345) and stable.

## Contract Assertions
1. Two consecutive runs with identical config+seed produce identical ordered `episode_ids` lists.
2. `scenario_matrix_hash` string identical across both runs.
3. Episode count > 0 and equals planned seeds count.
4. No duplicate `episode_id` within a single run.
5. Wall clock runtime for the full two-run procedure < CI upper bound threshold (default 4s). Soft local target <2s.
6. Videos/plots not generated (or explicitly skipped) – ensures performance isolation.
7. Bootstrap samples either disabled or produce no CI keys (avoid overhead).

## Failure Modes & Expected Handling
| Failure | Expected Test Response |
|---------|-----------------------|
| Episode ordering mismatch | AssertionError with diff length or first differing index |
| Hash mismatch | AssertionError citing expected vs actual hash |
| Duplicate episode IDs | AssertionError listing duplicates |
| Zero episodes planned | AssertionError explaining misconfiguration |
| Runtime > threshold | Warning or failure (depending on STRICT_REPRO_TEST) |

## Non-Goals
- Validating metric numeric equality beyond structural sequencing.
- Performance benchmarking of worker scaling.

## Extension Hooks
- Future multi-worker determinism test may import this contract and parameterize worker counts.

Prepared: 2025-09-20

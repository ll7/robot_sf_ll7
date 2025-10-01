# Research: Reducing Runtime of Reproducibility Integration Test

**Purpose**: Analyze why `tests/benchmark_full/test_integration_reproducibility.py::test_reproducibility_same_seed` is slow and define safe acceleration strategies without weakening reproducibility guarantees.

## Current Behavior (Hypothesized)
- The test likely executes a mini benchmark twice (run A & run B) with identical configuration and seed.
- It probably uses a default scenario matrix or larger-than-needed episode count (e.g., multiple scenarios * multiple seeds), inflating runtime.
- Parallel worker invocation may add process startup overhead disproportionate to small workloads.
- Plot/video generation or aggregation extras may still be running if not explicitly disabled.
- Manifest/episode file writes might cause redundant disk I/O for large JSONL lines when only episode_id ordering is needed.

## Root Causes of Slowness
| Cause | Impact | Evidence Needed | Mitigation |
|-------|--------|-----------------|------------|
| Excess episodes (scenario * seeds) | O(n) environment resets/steps | Inspect test parameters | Reduce to minimal (e.g., 2 episodes total) |
| Unneeded artifact generation (plots/videos) | Extra CPU time | Check config flags | Force smoke mode, disable videos/plots |
| Parallel worker spawn overhead | Startup > workload | Observe if workers>1 used | Set workers=1 (minimal) |
| Full metric aggregation & bootstrap | Adds statistical passes | Inspect if bootstrap enabled | Disable bootstrap (set samples=0) |
| Disk I/O for large metrics | Delay on JSON writes | Count metrics fields | Use minimal metrics or early flush | 
| Re-reading scenario matrix each run | Duplicate YAML parse cost | Small but additive | Acceptable (microseconds) |

## Non-Negotiable Reproducibility Assertions
1. Ordered list of `episode_id` values identical across both runs.
2. `scenario_matrix_hash` identical across runs.
3. Episode count identical and > 0.
4. Deterministic planning (seed expansion) exercised (not stubbed out).

Optional (NOT required to keep runtime low): comparing detailed metric values, effect sizes, bootstrap intervals.

## Optimization Strategies
| Strategy | Risk | Mitigation |
|----------|------|------------|
| Reduce episodes to 2 (1 scenario * 2 seeds) | Lower coverage | Coverage still sufficient for ordering & hashing; document rationale |
| Min horizon/steps per episode (if configurable) | Might skip code path | Ensure horizon still > 1; keep at minimal functional value |
| Disable videos & heavy plots | Losing detection of ordering? | Not needed for reproducibility semantics |
| workers=1 | Less scaling coverage | Scaling is not goal of this test; separate perf test handles it |
| Skip bootstrap aggregation | Lose CI stability check | Repro test focuses on structural determinism, not CI precision |
| Soft performance assert (<2s local) | Flaky on slow CI | Use generous CI threshold (<4s) and soft warning or conditional assert |

## Selected Approach
- Minimal scenario matrix: single archetype-density combination.
- Two seeds (e.g., base_seed and base_seed+1) to ensure planning loop enumerates >1.
- Smoke mode flags to disable heavy artifact generation.
- workers=1 to avoid process pool overhead and potential nondeterministic ordering.
- bootstrap_samples=0 in any invoked aggregation; if not directly controlled, stub aggregation step by using only planning + job execution minimal path (if orchestrator supports).
- Add timing capture around the two runs; assert wall clock < target threshold with fallback note (soft failure turned into warning if environment variable `STRICT_REPRO_TEST=1` not set).

## Alternatives Considered
1. Caching first run output and only re-validating planning (Rejected: reduces end-to-end determinism validation of execution path).
2. Mocking environment step to skip physics (Rejected: undermines real integration semantics; risk of false positives if ordering logic depends on runtime events).
3. Parameterizing across multiple worker counts (Rejected: multiplies runtime; scaling determinism validated elsewhere if needed).

## Risks
- Over-minimization could mask a race condition that only appears with >1 worker. (Mitigation: add TODO reference to a separate multi-worker determinism test if needed.)
- Soft performance assertion may become flaky under extreme CI load. (Mitigation: keep generous upper bound; allow override.)

## Open Questions
- Is episode ordering guaranteed identical under parallel workers currently? (If not documented, keep workers=1.)
- Are there hidden side effects in manifest writing that depend on number of episodes? (Assumed no; stable append semantics.)

## Decisions
| Decision | Rationale | Alternatives |
|----------|-----------|--------------|
| 2 episodes total | Sufficient to test ordering hash determinism | 1 episode (too trivial), >2 (slower) |
| workers=1 | Eliminates parallel nondeterminism and overhead | 2+ workers adds spawn cost |
| Disable videos/plots | Not needed for reproducibility semantics | Keep enabled (slower) |
| Soft timing assert | Enforces performance goal | Omit (no feedback if regression) |

## Next Steps (Feeds Phase 1)
1. Define minimal data model entities relevant to test (EpisodeSequence, ScenarioHashSnapshot).
2. Specify contract conditions in contracts/ (expected manifest keys, ordering invariant).
3. Draft quickstart snippet showing how to manually reproduce the two-run determinism check.

---
Prepared: 2025-09-20

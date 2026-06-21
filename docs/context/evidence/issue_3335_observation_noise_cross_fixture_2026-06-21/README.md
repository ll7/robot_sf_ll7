# Issue #3335 Observation-Noise Cross-Fixture Live Grid

## Claim Boundary

Diagnostic-only cross-fixture synthesis of tracked native live step-diagnostics summaries. This checks whether observation-noise behavior sensitivity appears across more than one fixture surface, but it is not benchmark-strength, paper-grade, planner-superiority, robustness, or hardware-calibrated sensor-realism evidence.

## Reproducibility

- Command: `uv run python scripts/benchmark/summarize_observation_noise_live_grid.py --source-summary docs/context/evidence/issue_2777_live_observation_noise_replay/issue_3330_seed_amplitude_grid/summary.json --source-summary docs/context/evidence/issue_3335_observation_noise_cross_fixture_2026-06-21/issue_3320_seed_amplitude_grid/summary.json --output-json docs/context/evidence/issue_3335_observation_noise_cross_fixture_2026-06-21/summary.json --output-md docs/context/evidence/issue_3335_observation_noise_cross_fixture_2026-06-21/README.md`
- Source summaries: `2`
- Usable native live sources: `1`
- Failed-closed sources: `1`

## Classification

- Label: `fixture_candidate_failed_closed_after_sensitive_grid`
- Rationale: The committed #3330 near-field grid is behavior-sensitive, but the attempted second matrix failed closed under the near-field guardrail. Treat this as a useful negative external-validity check, not robustness evidence.

## Source Rows

| Source | Scenario | Seed | Near-field | Status | Command changed | Closest delta |
|---|---|---:|---|---|---|---:|
| `docs/context/evidence/issue_2777_live_observation_noise_replay/issue_3330_seed_amplitude_grid/summary.json` | `issue_2756_occluded_emergence` | `111` | `True` | `behavior_sensitive_grid` | `True` | `0.008230813627037481` |
| `docs/context/evidence/issue_3335_observation_noise_cross_fixture_2026-06-21/issue_3320_seed_amplitude_grid/summary.json` | `issue_2756_occluded_emergence` | `111` | `False` | `failed_closed` | `False` | `0.0` |

## Caveats

- This is a synthesis of tracked compact native live summaries, not a new full benchmark.
- Fallback, degraded, unavailable, or malformed sources fail closed.
- Fixture/profile-specific results remain diagnostic-only and do not support sensor-realism claims.

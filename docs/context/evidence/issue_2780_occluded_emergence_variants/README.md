# Issue 2780 Occluded Emergence Variants

This evidence note records five simulated occluded-emergence variant fixtures generated for
issue #2780.  The variants explore different emergence configurations beyond the single
hand-authored issue #2756 fixture.

## Claim Boundary

Smoke/diagnostic/stress only.  The fixtures prove that the repository can generate and validate
multiple occluded-emergence configurations varying emergence side, pedestrian speed, first-visible
distance, conflict timing, and robot approach speed.  They do not establish real-world occlusion
representativeness, paper-facing benchmark coverage, or live-replay safety sufficiency.  The
fixtures do not replace the #2777 live replay.

## Variant Summary

| Variant | Side | Ped Speed | First Visible | Robot Speed | Expected Failure Mode | Safety-Relevant |
|---------|------|-----------|---------------|-------------|-----------------------|-----------------|
| `left_close` | left | 1.2 m/s | step 4 | 1.0 m/s | late_detection | yes |
| `right_close` | right | 1.2 m/s | step 4 | 1.0 m/s | wrong_source_selection | yes |
| `late_visibility` | center | 1.0 m/s | step 8 | 1.0 m/s | insufficient_braking_distance | yes |
| `slow_pedestrian` | center | 0.4 m/s | step 5 | 0.8 m/s | unnecessary_stop | no |
| `fast_pedestrian` | center | 2.0 m/s | step 3 | 1.2 m/s | forecast_miss | yes |

## Variation Dimensions

- **Emergence side**: left, right, center (different occluder x-bounds)
- **Pedestrian speed**: 0.4 to 2.0 m/s
- **First-visible step**: 3 to 8 (reaction time pressure)
- **Robot approach speed**: 0.8 to 1.2 m/s
- **Conflict timing**: varies with speed and initial position

## Failure Modes

- `late_detection`: pedestrian detected too close to conflict for comfortable response
- `wrong_source_selection`: right-side emergence may confuse source attribution
- `insufficient_braking_distance`: very late visibility leaves no braking room
- `unnecessary_stop`: slow pedestrian does not actually require a stop
- `forecast_miss`: fast pedestrian exceeds forecast horizon coverage

## Safety Relevance Boundary

Four variants (`left_close`, `right_close`, `late_visibility`, `fast_pedestrian`) are
safety-relevant under live replay because they exercise detection latency, source selection,
or braking distance under stress.  The `slow_pedestrian` variant is non-safety-relevant
because the low speed gives ample reaction time.  No variant replaces the #2777 live replay
evidence.

## Durable Inputs

- Summary: `docs/context/evidence/issue_2780_occluded_emergence_variants/summary.json`
- Generator: `scripts/tools/generate_occluded_emergence_variants.py`
- Test file: `tests/benchmark/test_occluded_emergence_variants.py`
- Trace fixtures: `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_*_episode_0000.json`
- Source metadata: `tests/fixtures/analysis_workbench/simulation_trace_export_v1/sources/issue_2780_*.meta.json`

## Validation

```bash
uv run pytest tests/benchmark/test_occluded_emergence_variants.py -q
uv run pytest tests/benchmark/test_occluded_emergence_fixture.py tests/benchmark/test_occluded_emergence_variants.py -q
uv run ruff check scripts/tools/generate_occluded_emergence_variants.py tests/benchmark/test_occluded_emergence_variants.py
uv run ruff format --check scripts/tools/generate_occluded_emergence_variants.py tests/benchmark/test_occluded_emergence_variants.py
uv run python scripts/tools/generate_occluded_emergence_variants.py
```

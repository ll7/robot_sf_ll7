---
title: Migration Guide – Environment Factory Ergonomics (Feature 130)
updated: 2025-09-23
status: Final
---

## Purpose
Explain how to migrate from legacy environment factory usage (implicit kwargs) to the new explicit, option‑based API with deterministic seeding and strengthened performance guarantees.

## Before vs After Summary
| Concern | Legacy Pattern | New Pattern |
|---------|---------------|-------------|
| Recording enable | `record_video=True` | `RecordingOptions(record=True, video_path=...)` or convenience flag (same) |
| FPS cap | `fps=30` | `RenderOptions(max_fps_override=30)` |
| Video path | `video_path="run.mp4"` / `video_output_path` | `RecordingOptions(video_path="run.mp4")` |
| Ped recording opt-out | Overridden by boolean | Explicit `RecordingOptions(record=False)` preserved |
| Seeding | Manual / ad-hoc | `seed=123` parameter (Python, NumPy, Torch, hash) |
| Legacy unknown kw | Often silently ignored | Error unless `ROBOT_SF_FACTORY_LEGACY=1` (warn & map) |
| Perf guard | Informal | Test-enforced (+5% mean budget) |

## Key Changes
1. Explicit factory signatures eliminate ambiguous `**kwargs` for new features.
2. Option dataclasses group semantically related parameters; convenience flags remain for quick usage.
3. Legacy compatibility preserved during deprecation window using mapping + warnings.
4. Deterministic seeding unified via `_apply_global_seed` and `seed` param.
5. Pedestrian factory divergence: respects explicit opt-out for recording.

## Seeding Details
```
env = make_robot_env(seed=42)
print(env.applied_seed)  # 42
```
Sequence: Python `random.seed`, NumPy `np.random.seed`, optional Torch `manual_seed`, set `PYTHONHASHSEED`.

## Legacy Environment Variables
| Variable | Effect |
|----------|--------|
| `ROBOT_SF_FACTORY_LEGACY=1` | Permissive legacy kw acceptance (warnings) |
| `ROBOT_SF_FACTORY_STRICT=1` | Strict mode: unknown legacy kw raises |

## Precedence Rules (Robot/Image)
1. Map legacy → structured first.
2. Explicit options override convenience booleans.
3. `record_video=True` upgrades `RecordingOptions.record=False` (user convenience).
4. `video_fps` sets `RenderOptions.max_fps_override` if unset.

## Precedence Rules (Pedestrian Differences)
* Explicit `RecordingOptions(record=False)` is honored even with `record_video=True`.
* Debug may auto-enable when effective recording is active.

## Migration Steps
1. Replace legacy kwargs with option instances (optional: keep booleans short-term).
2. Add `seed=` argument where reproducibility needed.
3. Set `ROBOT_SF_FACTORY_STRICT=1` in CI to surface any stale legacy params early.
4. Monitor performance test to ensure no regression after refactors.

## Examples
```python
from robot_sf.gym_env.environment_factory import make_robot_env, RenderOptions, RecordingOptions

env = make_robot_env(
    seed=123,
    render_options=RenderOptions(max_fps_override=30),
    recording_options=RecordingOptions(record=True, video_path="demo.mp4"),
)
```

## Deprecation Timeline (Tentative)
| Phase | Window | Action |
|-------|--------|--------|
| 1 | Current release | Warnings for legacy params; mapping active |
| 2 | +2 releases | Introduce strict mode default (permissive opt-in) |
| 3 | +3 releases | Remove legacy mapping layer |

## Troubleshooting
| Symptom | Likely Cause | Resolution |
|---------|--------------|-----------|
| Legacy param error | Strict mode enabled | Set `ROBOT_SF_FACTORY_LEGACY=1` temporarily; update code |
| Recording disabled unexpectedly (pedestrian) | Explicit opt-out | Remove `record=False` or drop convenience flag |
| Perf test failure | Creation mean > +5% | Profile import path; defer heavy imports; rebaseline only after justified improvement |

## Related Docs
* `specs/130-improve-environment-factory/plan.md`
* `specs/130-improve-environment-factory/tasks.md`
* Development Guide (seeding + performance sections)

## Rejected Alternatives (FR-019)
| Alternative | Decision | Rationale |
|-------------|----------|-----------|
| Builder Pattern (`Factory().with_recording().with_seed().build()`) | Rejected | Adds ceremony, hides discoverability in IDE; no multi-step state needed. |
| Single Monolithic Options Object | Rejected | Reduces autocomplete clarity; groups unrelated concerns. |
| Dynamic Registry of Factories | Rejected | YAGNI; current set of factories stable and small. |
| Immediate Removal of Legacy Mapping | Rejected | Violates backward compatibility (Principle VII); need deprecation window. |

## Finalization
Deprecation timeline consolidated (three phases). All references updated. This document satisfies FR-010, FR-011, FR-019 and completes T035.

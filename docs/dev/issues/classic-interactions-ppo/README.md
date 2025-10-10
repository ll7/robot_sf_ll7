# Classic Interactions PPO Visualization (Feature 128)

Purpose: Provide a deterministic, constants-configured demonstration of classic interaction scenarios using a pre-trained PPO policy with optional recording and structured episode summaries.

## Documents
- Spec: `specs/128-classic-interactions-ppo/spec.md`
- Plan: `specs/128-classic-interactions-ppo/plan.md`
- Tasks: `specs/128-classic-interactions-ppo/tasks.md`

(Planned additions)
- Data model: `docs/dev/issues/classic-interactions-ppo/data-model.md` (T036)
- Quickstart: `docs/dev/issues/classic-interactions-ppo/quickstart.md` (T035)

## Key Requirements (MVP)
See tasks mapping (FR-001..FR-021). Multi-scenario chaining (FR-022) and frame sampling (FR-023) deferred.

## Run Preview (target API)
```python
from examples.classic_interactions_pygame import run_demo

episodes = run_demo(dry_run=True, enable_recording=False)
print(episodes[0])
```

## Determinism
- Seeds sorted deterministically before episode execution.
- Model inference uses `predict(..., deterministic=True)`.

## Recording
- Optional; gracefully skipped if video backend (e.g., moviepy) unavailable.

## Headless
- Uses SDL dummy video driver when environment variable set.

## Next Steps
Follow tasks in `tasks.md` Phase 2–3 for test authoring then core implementation.

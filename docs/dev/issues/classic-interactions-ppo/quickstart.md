"""Quickstart: Classic Interactions PPO Visualization (Feature 128)

How to run and customize the deterministic classic interactions PPO visualization.
"""

# Quickstart

## 1. Dry Run Validation

```sh
uv run python -c "from examples.classic_interactions_pygame import run_demo; run_demo(dry_run=True)"
```

Checks that the scenario matrix and (optionally) model path are accessible.

## 2. Run One Episode

```sh
uv run python examples/classic_interactions_pygame.py
```

Default constants: first scenario, MAX_EPISODES=1, recording disabled.

## 3. Enable Recording

```python
from examples.classic_interactions_pygame import run_demo
run_demo(enable_recording=True, max_episodes=2)
```

If `moviepy` or `ffmpeg` missing the code logs a skip message and continues.

## 4. Select Scenario

```python
run_demo(scenario_name="classic_crossing_low", max_episodes=3)
```

## 5. Programmatic Use

```python
episodes = run_demo(max_episodes=5)
print(episodes[0]["outcome"], episodes[0]["steps"])  # structured fields
```

## 6. Headless Mode

Set before invocation:

```sh
SDL_VIDEODRIVER=dummy uv run python examples/classic_interactions_pygame.py
```

## 7. Logging Control

Inside module: set `LOGGING_ENABLED=False` before calling `run_demo` to suppress informational prints.

## 8. Performance Tip

Model load dominates first call; subsequent runs reuse an in-memory cache.

## 9. Episode Summary Schema

See `data-model.md` in the same folder for field definitions and invariants.

# Quickstart: Phase A — Extract visualization & formatting helpers

Run locally to verify changes (smoke path):

1. Ensure submodules initialized:

```bash
git submodule update --init --recursive
```

2. Create and activate venv & install dev deps (project uses `uv`):

```bash
uv sync && source .venv/bin/activate
```

3. Run linter/format (Ruff) and tests focusing on new units:

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pytest tests/unit/benchmark -q
```

4. Smoke-run the example (dry-run) to confirm example-level behavior:

```bash
python examples/classic_interactions_pygame.py --dry-run
```

Notes:
- The Phase A helpers are intentionally small and testable. If moviepy or SimulationView are missing, tests will monkeypatch or run in a dry-run mode.

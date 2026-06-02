# Issue #2141 Active-Doc Checker Classification

Status: Current for PR #2141.

Issue #2141 classified the active-doc checker diagnostics introduced by
`scripts/validation/check_active_doc_examples.py` after examples were added to the default scan.

## Real Active-Doc Drift Fixed

- `docs/dev_guide.md` referenced removed `examples/demo_*.py` entry points. The active demo block
  now points at existing quickstart and advanced examples under `examples/quickstart/` and
  `examples/advanced/`.
- `docs/trajectory_visualization.md` referenced removed `examples/trajectory_demo.py`. The doc now
  points at `examples/advanced/14_trajectory_visualization.py`.
- `docs/benchmark_visuals.md`, `docs/dev/baselines/random.md`, and
  `docs/performance_notes.md` used active `results/` artifact paths. They now use the repository
  `output/` artifact root.
- `docs/snqi-weight-tools/README.md` used bare `python scripts/...` commands in recommended
  workflows. Those examples now use `uv run python`.
- `docs/ENVIRONMENT.md`, `docs/UV_MIGRATION.md`, and `docs/dev_guide.md` preferred `pip install`
  setup paths for uv. They now point at the official uv installer or package-manager route.

## Intentional Or Historical Mentions

- `CHANGELOG.md` keeps the historical `results/videos/` changelog entry with an explicit
  `active-docs-check: allow` marker.
- `scripts/README.md` intentionally describes a migration tool that consumes legacy artifact roots.
  The line now names `results/` as legacy so the checker can distinguish it from current guidance.

## Validation

- `uv run python scripts/validation/check_active_doc_examples.py`
- `git diff --check`

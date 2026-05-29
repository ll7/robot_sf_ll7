"""Fail-closed guard for the retired root-level Optuna entrypoint."""

from __future__ import annotations

import sys

_MIGRATION_MESSAGE = """\
scripts/hparam_opt.py is retired.

Use the maintained config-first Optuna workflow instead:
  uv run python scripts/training/launch_optuna_expert_ppo.py --config <config.yaml>

Inspect existing Optuna sqlite stores with:
  uv run python scripts/tools/inspect_optuna_db.py --db <path-to-study.db>
"""


def main(_argv: object = None) -> int:
    """Exit non-zero with supported Optuna workflow alternatives."""
    sys.stderr.write(_MIGRATION_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

"""Fail-closed guard for the removed legacy PPO training entrypoint."""

from __future__ import annotations

import sys

_MIGRATION_MESSAGE = """\
scripts/training_ppo.py is no longer supported.

Use scripts/training/train_ppo.py with --config for PPO training.
See docs/training/ppo_training_workflow.md for migration details.
"""


def main(_argv: object = None) -> int:
    """Exit non-zero with a migration command for legacy PPO invocations."""
    sys.stderr.write(_MIGRATION_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

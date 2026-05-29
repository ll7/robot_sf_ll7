"""Fail-closed guard for the retired root-level W&B PPO training entrypoint."""

from __future__ import annotations

import sys

_MIGRATION_MESSAGE = """\
scripts/wandb_ppo_training.py is retired.

Use config-first PPO training instead and enable W&B in the YAML config:
  uv run python scripts/training/train_ppo.py --config <config.yaml>

This legacy root script used import-time side effects and is intentionally blocked
to avoid accidental long training runs or W&B writes.
"""


def main(_argv: object = None) -> int:
    """Exit non-zero with the supported W&B-enabled training path."""
    sys.stderr.write(_MIGRATION_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

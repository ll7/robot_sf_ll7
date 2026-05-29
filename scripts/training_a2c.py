"""Fail-closed guard for the retired root-level A2C training entrypoint."""

from __future__ import annotations

import sys

_MIGRATION_MESSAGE = """\
scripts/training_a2c.py is retired.

Use config-first PPO training via:
  uv run python scripts/training/train_ppo.py --config <config.yaml>

If an A2C baseline is still required, add a reviewed config-first launcher under
scripts/training/ instead of reviving this root-level script.
"""


def main(_argv: object = None) -> int:
    """Exit non-zero with the supported training entrypoint."""
    sys.stderr.write(_MIGRATION_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

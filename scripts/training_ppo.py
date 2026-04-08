"""Fail-closed guard for the removed legacy PPO training entrypoint."""

from __future__ import annotations

import sys

_MIGRATION_MESSAGE = """\
scripts/training_ppo.py has been removed from the supported PPO training workflow.

Use the config-driven canonical entrypoint instead:
  uv run python scripts/training/train_ppo.py --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml

The canonical trainer requires evaluation.step_schedule for checkpoint cadence.
Legacy evaluation.frequency_episodes values are ignored when present.
"""


def main(_argv: object = None) -> int:
    """Exit non-zero with a migration command for legacy PPO invocations."""
    sys.stderr.write(_MIGRATION_MESSAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

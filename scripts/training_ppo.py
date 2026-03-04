"""Deprecated PPO training entrypoint.

This script intentionally hard-fails to prevent accidental usage of the legacy,
non-config-driven workflow. Use the structured training entrypoint instead:

    uv run python scripts/training/train_expert_ppo.py --config <config.yaml>
"""

from __future__ import annotations

import sys


def main() -> int:
    """Print migration guidance and return a non-zero exit code."""
    message = (
        "scripts/training_ppo.py is deprecated and disabled to prevent accidental "
        "legacy training runs.\n"
        "Use: uv run python scripts/training/train_expert_ppo.py "
        "--config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml"
    )
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

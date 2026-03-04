"""Compatibility shim for the renamed PPO training entrypoint.

Canonical entrypoint: ``scripts/training/train_ppo.py``.
This module remains for short-term compatibility with existing imports/scripts.
"""

from __future__ import annotations

from scripts.training import train_ppo as _train_ppo

main = _train_ppo.main

# Re-export all public and private names for short-term compatibility with tests/tools.
globals().update(vars(_train_ppo))

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

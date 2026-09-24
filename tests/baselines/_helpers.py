"""Shared helpers for baseline adapter tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def make_robot_observation() -> dict[str, object]:
    """Return a minimal robot observation accepted by external baseline wrappers."""
    return {
        "dt": 0.1,
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [1.0, 0.0],
            "radius": 0.3,
        },
        "agents": [],
        "obstacles": [],
    }


def write_fake_module_file(path: Path, text: str) -> None:
    """Write a fake external-package file while creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

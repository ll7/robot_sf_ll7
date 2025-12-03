"""Guidance heuristics for slow test optimization suggestions.

Converts a test duration and contextual hints into human-readable optimization
suggestions to help contributors shrink runtime while preserving semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

SUGGESTIONS_ORDER = [
    ("episode", "Reduce episode count / seeds"),
    ("horizon", "Lower horizon or early-stop condition"),
    ("matrix", "Use minimal scenario matrix helper"),
    ("bootstrap", "Disable bootstrap / sampling"),
    ("workers", "Reduce workers to 1 if contention"),
]


def default_guidance(duration_seconds: float, breach_type: str) -> list[str]:
    """Return a list of guidance strings for a breached test.

    The heuristics are intentionally lightweight and deterministic.
    """
    if breach_type == "none":
        return []
    rec: list[str] = []
    # If extremely long (> 2 * soft threshold heuristic ~40s), prioritize drastic steps first.
    prioritized = list(SUGGESTIONS_ORDER)
    if duration_seconds > 40:  # heuristic threshold
        # Move matrix + horizon suggestions to front for large overruns.
        prioritized.sort(key=lambda kv: 0 if kv[0] in {"matrix", "horizon"} else 1)
    for _key, msg in prioritized:
        rec.append(msg)
    # Provide a summarizing hint referencing the soft budget.
    if breach_type == "soft":
        rec.append("Aim to bring runtime < soft threshold (20s).")
    else:  # hard
        rec.append("Hard timeout risk: drastically reduce scenario complexity.")
    return rec


def format_guidance_lines(lines: Iterable[str]) -> str:
    """TODO docstring. Document this function.

    Args:
        lines: TODO docstring.

    Returns:
        TODO docstring.
    """
    return "\n".join(f"  - {line}" for line in lines)

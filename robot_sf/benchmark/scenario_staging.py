"""Fail-closed helpers for selecting scenarios during campaign staging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


class ScenarioStagingError(ValueError):
    """Raised when a campaign cannot identify one exact source scenario."""


def select_unique_scenario(
    scenarios: Sequence[Mapping[str, Any]],
    scenario_name: str,
    *,
    source: str | Path,
) -> Mapping[str, Any]:
    """Return the one exact name match or report the source and match count.

    Returns:
        The only scenario whose ``name`` equals ``scenario_name``.

    Raises:
        ScenarioStagingError: If the source has zero or multiple exact matches.
    """

    matches = [row for row in scenarios if str(row.get("name")) == scenario_name]
    if len(matches) != 1:
        raise ScenarioStagingError(
            f"Scenario staging expected exactly one match for {scenario_name!r} "
            f"in source {str(source)!r}; found {len(matches)} matches."
        )
    return matches[0]


__all__ = ["ScenarioStagingError", "select_unique_scenario"]

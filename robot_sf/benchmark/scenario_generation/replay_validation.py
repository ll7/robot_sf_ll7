"""Record explicit replay/load status for generated scenario hypotheses."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.scenario_generation.catalog_schema import validate_catalog_entry
from robot_sf.training.scenario_loader import build_robot_config_from_scenario

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


def assess_replay_status(
    entry: Mapping[str, Any],
    *,
    source_scenario: Mapping[str, Any],
    scenario_path: Path,
    config_builder: Callable[..., Any] = build_robot_config_from_scenario,
) -> dict[str, Any]:
    """Load-check the source scenario and preserve the exact-state replay gap.

    A successful source load is recorded as a warning detail, but the generated
    segment remains ``not_representable_yet`` until a standalone scenario can be
    constructed.  Source-template loading does not prove segment replay.

    Returns:
        A validated copy of the entry with an explicit replay status.
    """

    updated = deepcopy(dict(entry))
    try:
        config_builder(source_scenario, scenario_path=scenario_path)
    except (OSError, RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
        updated["replay"]["status"] = "not_representable_yet"
        updated["replay"]["warnings"] = [
            f"replay_gap: source scenario load failed: {type(exc).__name__}: {exc}"
        ]
    else:
        updated["replay"]["status"] = "not_representable_yet"
        updated["replay"]["warnings"] = [
            "replay_gap: source scenario load passed, but distilled mid-episode state is not "
            "representable as standalone scenario YAML"
        ]
    validate_catalog_entry(updated)
    return updated


__all__ = ["assess_replay_status"]

"""Versioned task-bundle loading for reusable scenario packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TASK_BUNDLE_SCHEMA_VERSION = "robot_sf.task_bundle.v1"
TASK_BUNDLE_REF_PREFIX = "bundle:"


@dataclass(frozen=True)
class TaskBundle:
    """Parsed task-bundle metadata and scenario file references."""

    name: str
    path: Path
    description: str
    scenario_files: tuple[Path, ...]
    select_scenarios: tuple[str, ...]


def repo_root() -> Path:
    """Return the repository root inferred from this module path."""
    return Path(__file__).resolve().parents[2]


def task_bundle_dir() -> Path:
    """Return the default tracked task-bundle registry directory."""
    return repo_root() / "configs" / "bundles"


def is_task_bundle_reference(reference: str | Path) -> bool:
    """Return whether a user reference points at a task bundle."""
    text = str(reference)
    if text.startswith(TASK_BUNDLE_REF_PREFIX):
        return True
    path = Path(reference)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return False
    if not path.exists() or not path.is_file():
        return False
    if not _has_task_bundle_header(path):
        return False
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(data, Mapping) and data.get("schema_version") == TASK_BUNDLE_SCHEMA_VERSION


def resolve_task_bundle_path(reference: str | Path) -> Path:
    """Resolve a bundle name, ``bundle:<name>``, or explicit bundle YAML path.

    Returns:
        Path: Resolved bundle definition path.
    """
    text = str(reference)
    if text.startswith(TASK_BUNDLE_REF_PREFIX):
        name = text.removeprefix(TASK_BUNDLE_REF_PREFIX).strip()
        _validate_bundle_name(name, source=reference)
        return task_bundle_dir() / f"{name}.yaml"

    path = Path(reference)
    if path.exists():
        return path.resolve()

    if path.parent == Path(".") and not path.suffix:
        _validate_bundle_name(text, source=reference)
        return task_bundle_dir() / f"{text}.yaml"

    return path.resolve()


def load_task_bundle(reference: str | Path) -> TaskBundle:
    """Load and validate a task-bundle definition.

    Returns:
        TaskBundle: Parsed bundle metadata and resolved scenario files.
    """
    path = resolve_task_bundle_path(reference)
    if not path.exists():
        raise FileNotFoundError(f"Task bundle '{reference}' not found at '{path}'.")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Task bundle '{path}' must contain a mapping.")

    schema_version = data.get("schema_version")
    if schema_version != TASK_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Task bundle '{path}' has unsupported schema_version '{schema_version}'. "
            f"Expected '{TASK_BUNDLE_SCHEMA_VERSION}'."
        )

    name = _require_string(data, "name", source=path)
    _validate_bundle_name(name, source=path)
    description = _optional_string(data, "description", source=path)
    scenario_files = _resolve_scenario_files(data.get("scenario_files"), source=path)
    select_scenarios = _resolve_select_scenarios(data.get("select_scenarios"), source=path)

    return TaskBundle(
        name=name,
        path=path,
        description=description,
        scenario_files=tuple(scenario_files),
        select_scenarios=tuple(select_scenarios),
    )


def load_task_bundle_scenarios(
    reference: str | Path,
    *,
    _bundle_stack: tuple[Path, ...] = (),
) -> list[Mapping[str, Any]]:
    """Expand a task bundle into scenario entries using the scenario loader.

    Returns:
        list[Mapping[str, Any]]: Expanded scenario entries in deterministic bundle order.
    """
    bundle = load_task_bundle(reference)
    bundle_path = bundle.path.resolve()
    if bundle_path in _bundle_stack:
        cycle = (*_bundle_stack, bundle_path)
        raise ValueError(
            "Task bundle include cycle detected: " + " -> ".join(str(path) for path in cycle)
        )

    scenarios: list[Mapping[str, Any]] = []
    bundle_stack = (*_bundle_stack, bundle_path)
    for scenario_file in bundle.scenario_files:
        scenarios.extend(_load_bundle_scenario_file(scenario_file, bundle_stack=bundle_stack))
    if bundle.select_scenarios:
        return _apply_bundle_selection(
            scenarios,
            selected_names=bundle.select_scenarios,
            source=bundle.path,
        )
    return scenarios


def describe_task_bundle_source(reference: str | Path) -> dict[str, object]:
    """Describe a task-bundle source for CLI validation and preview reports.

    Returns:
        dict[str, object]: Source metadata compatible with scenario-matrix reports.
    """
    bundle = load_task_bundle(reference)
    return {
        "path": str(bundle.path),
        "format": "task_bundle",
        "bundle_name": bundle.name,
        "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
        "scenario_files": [str(path) for path in bundle.scenario_files],
        "select_scenarios": list(bundle.select_scenarios),
    }


def _require_string(data: Mapping[str, Any], key: str, *, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Task bundle '{source}' must define non-empty string '{key}'.")
    return value.strip()


def _optional_string(data: Mapping[str, Any], key: str, *, source: Path) -> str:
    value = data.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Task bundle '{source}' field '{key}' must be a string.")
    return value.strip()


def _has_task_bundle_header(path: Path) -> bool:
    """Return whether the first YAML lines look like a task-bundle document."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = "".join(handle.readline() for _ in range(20))
    except OSError:
        return False
    return "schema_version:" in header and TASK_BUNDLE_SCHEMA_VERSION in header


def _validate_bundle_name(name: str, *, source: object) -> None:
    if not name:
        raise ValueError(f"Task bundle reference '{source}' must name a bundle.")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    if any(char not in allowed for char in name):
        raise ValueError(
            f"Task bundle name '{name}' in '{source}' may only contain lowercase "
            "letters, digits, hyphen, and underscore."
        )


def _resolve_scenario_files(raw: object, *, source: Path) -> list[Path]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Task bundle '{source}' must define non-empty list 'scenario_files'.")

    scenario_files: list[Path] = []
    seen: set[Path] = set()
    for entry in raw:
        if not isinstance(entry, (str, Path)) or not str(entry).strip():
            raise ValueError(
                f"Task bundle '{source}' scenario_files entries must be non-empty strings."
            )
        candidate = _resolve_bundle_relative_path(entry, source=source)
        _reject_local_output_path(candidate, source=source)
        if candidate in seen:
            raise ValueError(f"Task bundle '{source}' has duplicate scenario file '{candidate}'.")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(
                f"Task bundle '{source}' references missing scenario file '{candidate}'."
            )
        seen.add(candidate)
        scenario_files.append(candidate)
    return scenario_files


def _resolve_bundle_relative_path(entry: str | Path, *, source: Path) -> Path:
    """Resolve a bundle-owned file path relative to the bundle source file.

    Returns:
        Path: Absolute path for the referenced bundle-owned file.
    """
    candidate = Path(str(entry).strip())
    if not candidate.is_absolute():
        candidate = source.parent / candidate
    return candidate.resolve()


def _resolve_select_scenarios(raw: object, *, source: Path) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"Task bundle '{source}' field 'select_scenarios' must be a list.")
    selected: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                f"Task bundle '{source}' select_scenarios entries must be non-empty strings."
            )
        name = entry.strip()
        key = name.lower()
        if key in seen:
            raise ValueError(f"Task bundle '{source}' repeats select_scenarios entry '{name}'.")
        seen.add(key)
        selected.append(name)
    return selected


def _reject_local_output_path(candidate: Path, *, source: Path) -> None:
    output_root = (repo_root() / "output").resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError:
        return
    raise ValueError(
        f"Task bundle '{source}' references local output path '{candidate}'. "
        "Task bundles must expand from durable config files."
    )


def _load_bundle_scenario_file(
    scenario_file: Path,
    *,
    bundle_stack: tuple[Path, ...],
) -> list[Mapping[str, Any]]:
    """Load a scenario file or nested task bundle with bundle cycle tracking.

    Returns:
        list[Mapping[str, Any]]: Expanded scenario entries.
    """
    if is_task_bundle_reference(scenario_file):
        return load_task_bundle_scenarios(scenario_file, _bundle_stack=bundle_stack)

    from robot_sf.training.scenario_loader import load_scenarios  # noqa: PLC0415

    return load_scenarios(scenario_file, base_dir=scenario_file)


def _scenario_name(scenario: Mapping[str, Any], *, source: Path, index: int) -> str:
    value = scenario.get("name") or scenario.get("scenario_id")
    if value is None:
        raise ValueError(
            f"Task bundle '{source}' scenario entry {index} is missing a name or scenario_id."
        )
    if not isinstance(value, str):
        raise ValueError(f"Task bundle '{source}' scenario entry {index} name must be a string.")
    if not value.strip():
        raise ValueError(f"Task bundle '{source}' scenario entry {index} name must be non-empty.")
    return value.strip()


def _apply_bundle_selection(
    scenarios: list[Mapping[str, Any]],
    *,
    selected_names: tuple[str, ...],
    source: Path,
) -> list[Mapping[str, Any]]:
    scenario_map: dict[str, Mapping[str, Any]] = {}
    for index, scenario in enumerate(scenarios):
        name = _scenario_name(scenario, source=source, index=index)
        key = name.lower()
        if key in scenario_map:
            raise ValueError(
                f"Duplicate scenario name '{name}' in task bundle '{source}' prevents selection."
            )
        scenario_map[key] = scenario

    selected: list[Mapping[str, Any]] = []
    for name in selected_names:
        key = name.lower()
        if key not in scenario_map:
            raise ValueError(f"Unknown select_scenarios entry '{name}' in task bundle '{source}'.")
        selected.append(scenario_map[key])
    return selected

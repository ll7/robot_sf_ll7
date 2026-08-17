"""Fail-closed, analysis-only contracts for the Flint chart foundation.

The foundation normalizes a candidate surface against a canonical report and
builds a context-separated atlas manifest.  It deliberately does not render,
download, promote, or admit a dissertation figure.  Release and replay inputs
are kept as separate records so a downstream promotion decision cannot silently
mix their provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SURFACE_INPUT_SCHEMA_VERSION = "flint-chart-surface-input.v1"
SURFACE_SCHEMA_VERSION = "flint-chart-surface.v1"
ATLAS_SCHEMA_VERSION = "flint-chart-atlas-manifest.v1"
CLAIM_BOUNDARY = (
    "analysis-only candidate surface; not a promoted figure, benchmark result, "
    "evidence admission, or dissertation claim"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PARITY_FIELDS: tuple[str, ...] = (
    "value",
    "denominator",
    "exposure_definition",
    "exclusions",
    "uncertainty",
    "capability_track",
    "evidence_status",
)


class FlintChartContractError(ValueError):
    """Raised when an analysis-only Flint input violates its contract."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Return a mapping or raise a stable contract error."""
    if not isinstance(value, Mapping):
        raise FlintChartContractError(f"{name} must be an object")
    return value


def _string(value: Any, *, name: str) -> str:
    """Return a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise FlintChartContractError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, *, name: str) -> int | float:
    """Return a finite JSON number, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FlintChartContractError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise FlintChartContractError(f"{name} must be a finite number")
    return value


def _sha256(value: Any, *, name: str) -> str:
    """Return a lowercase SHA-256 digest."""
    digest = _string(value, name=name).lower()
    if not _SHA256_RE.fullmatch(digest):
        raise FlintChartContractError(f"{name} must be a 64-character SHA-256 digest")
    return digest


def sha256_file(path: Path) -> str:
    """Hash a regular file in bounded chunks.

    Returns:
        Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FlintChartContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    """Load a JSON object from a path.

    Returns:
        Parsed JSON mapping.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlintChartContractError(f"cannot read JSON {path}: {exc}") from exc
    return _mapping(value, name=str(path))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic, sorted JSON for compact tracked evidence."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise FlintChartContractError(f"cannot write JSON {path}: {exc}") from exc


def _validate_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable source and artifact-catalog metadata.

    Returns:
        Normalized source metadata.
    """
    context = _string(source.get("context"), name="source.context")
    if context not in {"release", "replay"}:
        raise FlintChartContractError("source.context must be 'release' or 'replay'")
    commit = _string(source.get("source_commit"), name="source.source_commit").lower()
    if not _GIT_SHA_RE.fullmatch(commit):
        raise FlintChartContractError("source.source_commit must be a full 40-character commit")
    release_id = _string(source.get("release_id"), name="source.release_id")
    catalog = _mapping(source.get("artifact_catalog"), name="source.artifact_catalog")
    catalog_path = _string(catalog.get("path"), name="source.artifact_catalog.path")
    catalog_sha = _sha256(catalog.get("sha256"), name="source.artifact_catalog.sha256")
    raw_hashes = _mapping(source.get("input_hashes"), name="source.input_hashes")
    if not raw_hashes:
        raise FlintChartContractError("source.input_hashes must not be empty")
    input_hashes = {
        _string(path, name="source.input_hashes path"): _sha256(
            digest, name=f"source.input_hashes[{path}]"
        )
        for path, digest in raw_hashes.items()
    }
    durability = _string(source.get("durability"), name="source.durability")
    if durability != "durable_pinned":
        raise FlintChartContractError("source.durability must be 'durable_pinned'")
    return {
        "context": context,
        "source_commit": commit,
        "release_id": release_id,
        "artifact_catalog": {"path": catalog_path, "sha256": catalog_sha},
        "input_hashes": dict(sorted(input_hashes.items())),
        "durability": durability,
    }


def _validate_uncertainty(value: Any, *, name: str) -> dict[str, Any]:
    """Validate available or explicitly unavailable uncertainty metadata.

    Returns:
        Normalized uncertainty metadata.
    """
    uncertainty = _mapping(value, name=name)
    status = _string(uncertainty.get("status"), name=f"{name}.status")
    if status == "available":
        lower = _finite_number(uncertainty.get("lower"), name=f"{name}.lower")
        upper = _finite_number(uncertainty.get("upper"), name=f"{name}.upper")
        if float(lower) > float(upper):
            raise FlintChartContractError(f"{name}.lower must not exceed {name}.upper")
        method = _string(uncertainty.get("method"), name=f"{name}.method")
        return {"status": status, "lower": lower, "upper": upper, "method": method}
    if status == "unavailable":
        reason = _string(uncertainty.get("reason"), name=f"{name}.reason")
        return {"status": status, "reason": reason}
    raise FlintChartContractError(f"{name}.status must be 'available' or 'unavailable'")


def _validate_cell(value: Any, *, name: str) -> dict[str, Any]:
    """Validate one planner-by-scenario-family result cell.

    Returns:
        Normalized result cell.
    """
    cell = _mapping(value, name=name)
    allowed_fields = {
        "planner_key",
        "scenario_family",
        "value",
        "denominator",
        "exposure_definition",
        "exclusions",
        "uncertainty",
        "capability_track",
        "evidence_status",
    }
    unexpected_fields = set(cell) - allowed_fields
    if unexpected_fields:
        raise FlintChartContractError(
            f"{name} contains unsupported fields: {sorted(unexpected_fields)!r}"
        )
    planner_key = _string(cell.get("planner_key"), name=f"{name}.planner_key")
    scenario_family = _string(cell.get("scenario_family"), name=f"{name}.scenario_family")
    exclusions = cell.get("exclusions")
    if not isinstance(exclusions, list) or any(not isinstance(item, str) for item in exclusions):
        raise FlintChartContractError(f"{name}.exclusions must be a list of strings")
    denominator = cell.get("denominator")
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator < 0:
        raise FlintChartContractError(f"{name}.denominator must be a non-negative integer")
    return {
        "planner_key": planner_key,
        "scenario_family": scenario_family,
        "value": _finite_number(cell.get("value"), name=f"{name}.value"),
        "denominator": denominator,
        "exposure_definition": _string(
            cell.get("exposure_definition"), name=f"{name}.exposure_definition"
        ),
        "exclusions": list(exclusions),
        "uncertainty": _validate_uncertainty(cell.get("uncertainty"), name=f"{name}.uncertainty"),
        "capability_track": _string(cell.get("capability_track"), name=f"{name}.capability_track"),
        "evidence_status": _string(cell.get("evidence_status"), name=f"{name}.evidence_status"),
    }


def _cell_key(cell: Mapping[str, Any]) -> tuple[str, str]:
    """Return a stable planner/scenario-family cell key."""
    return (str(cell["planner_key"]), str(cell["scenario_family"]))


def _index_cells(value: Any, *, name: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate and index a non-empty cell list, rejecting duplicates.

    Returns:
        Cell mapping keyed by planner and scenario family.
    """
    if not isinstance(value, list) or not value:
        raise FlintChartContractError(f"{name} must be a non-empty list")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_cell in enumerate(value):
        cell = _validate_cell(raw_cell, name=f"{name}[{index}]")
        key = _cell_key(cell)
        if key in indexed:
            raise FlintChartContractError(f"{name} contains duplicate cell {key!r}")
        indexed[key] = cell
    return indexed


def _population_keys(value: Any) -> list[tuple[str, str]]:
    """Validate the explicit display population and return sorted keys.

    Returns:
        Sorted planner and scenario-family keys.
    """
    if not isinstance(value, list) or not value:
        raise FlintChartContractError("display_population must be a non-empty list")
    keys: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        population_item = _mapping(item, name=f"display_population[{index}]")
        key = (
            _string(
                population_item.get("planner_key"), name=f"display_population[{index}].planner_key"
            ),
            _string(
                population_item.get("scenario_family"),
                name=f"display_population[{index}].scenario_family",
            ),
        )
        if key in keys:
            raise FlintChartContractError(f"display_population contains duplicate cell {key!r}")
        keys.add(key)
    return sorted(keys)


def _canonical_json(value: Any) -> str:
    """Return deterministic JSON for exact structural parity comparisons."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _compare_cells(
    canonical: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate: Mapping[tuple[str, str], Mapping[str, Any]],
    population: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Compare all canonical and candidate fields before rendering.

    Returns:
        A passed parity report.
    """
    canonical_keys = set(canonical)
    candidate_keys = set(candidate)
    population_keys = set(population)
    if canonical_keys != population_keys:
        raise FlintChartContractError(
            "canonical cells do not match display population: "
            f"missing={sorted(population_keys - canonical_keys)!r}, "
            f"extra={sorted(canonical_keys - population_keys)!r}"
        )
    if candidate_keys != population_keys:
        raise FlintChartContractError(
            "candidate cells do not match display population: "
            f"missing={sorted(population_keys - candidate_keys)!r}, "
            f"extra={sorted(candidate_keys - population_keys)!r}"
        )
    for key in population:
        for field in _PARITY_FIELDS:
            if _canonical_json(canonical[key][field]) != _canonical_json(candidate[key][field]):
                raise FlintChartContractError(f"canonical parity drift for {field!r} at {key!r}")
    return {
        "status": "passed",
        "compared_cells": len(population),
        "compared_fields": list(_PARITY_FIELDS),
        "missing_cells": [],
        "extra_cells": [],
    }


def _validate_figure_policy(
    policy: Mapping[str, Any],
    *,
    figure_id: str,
    cells: Mapping[tuple[str, str], Any],
) -> None:
    """Validate figure-specific uncertainty, labels, and tie policies."""
    if figure_id == "figure_7_1":
        if policy.get("requires_uncertainty") is not True:
            raise FlintChartContractError("Figure 7.1 requires uncertainty metadata")
        if policy.get("requires_direct_labels") is not True:
            raise FlintChartContractError("Figure 7.1 requires direct-label metadata")
        if any(cell["uncertainty"]["status"] != "available" for cell in cells.values()):
            raise FlintChartContractError("Figure 7.1 cells must include available uncertainty")
    if figure_id == "figure_7_6":
        if policy.get("requires_tie_preservation") is not True:
            raise FlintChartContractError("Figure 7.6 requires exact-tie preservation")
        if any("rank" in cell or "catalog_rank" in cell for cell in cells.values()):
            raise FlintChartContractError("Figure 7.6 must not carry catalog-order ranks")


def _validate_renderer_policy(
    value: Any,
    *,
    figure_id: str,
    cells: Mapping[tuple[str, str], Any],
) -> dict[str, Any]:
    """Validate the non-promotional renderer and tie policies.

    Returns:
        Normalized renderer policy.
    """
    policy = _mapping(value, name="renderer_policy")
    canonical_renderer = _string(
        policy.get("canonical_renderer"), name="renderer_policy.canonical_renderer"
    )
    if canonical_renderer != "matplotlib/pgf/tikz":
        raise FlintChartContractError(
            "renderer_policy.canonical_renderer must remain 'matplotlib/pgf/tikz'"
        )
    tie_policy = _string(policy.get("tie_policy"), name="renderer_policy.tie_policy")
    if tie_policy != "exact_ties_no_catalog_rank":
        raise FlintChartContractError(
            "renderer_policy.tie_policy must be 'exact_ties_no_catalog_rank'"
        )
    if policy.get("source_context_separation") is not True:
        raise FlintChartContractError("renderer_policy.source_context_separation must be true")
    _validate_figure_policy(policy, figure_id=figure_id, cells=cells)
    return dict(policy)


def _validate_metric(value: Any, *, name: str) -> dict[str, str]:
    """Validate and normalize one metric descriptor.

    Returns:
        Normalized metric descriptor.
    """
    metric = _mapping(value, name=name)
    return {
        "id": _string(metric.get("id"), name=f"{name}.id"),
        "unit": _string(metric.get("unit"), name=f"{name}.unit"),
    }


def _add_tie_metadata(
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return sorted cells with explicit exact-tie groups and no generated ranks.

    Returns:
        Mapping with normalized ``cells`` and ``tie_groups`` lists.
    """
    by_family: dict[str, dict[float, list[tuple[str, str]]]] = {}
    for key, cell in cells.items():
        by_family.setdefault(key[1], {}).setdefault(float(cell["value"]), []).append(key)
    tie_group_by_key: dict[tuple[str, str], str] = {}
    tie_groups: list[dict[str, Any]] = []
    for family, value_groups in sorted(by_family.items()):
        group_index = 0
        for value, keys in sorted(value_groups.items(), key=lambda item: item[0]):
            if len(keys) < 2:
                continue
            group_index += 1
            group_id = f"{family}:tie:{group_index}"
            sorted_keys = sorted(keys)
            for key in sorted_keys:
                tie_group_by_key[key] = group_id
            tie_groups.append(
                {
                    "group_id": group_id,
                    "scenario_family": family,
                    "value": value,
                    "members": [list(key) for key in sorted_keys],
                }
            )
    normalized_cells: list[dict[str, Any]] = []
    for key in sorted(cells):
        normalized = dict(cells[key])
        normalized["rank"] = None
        normalized["tie_group"] = tie_group_by_key.get(key)
        normalized_cells.append(normalized)
    return {"cells": normalized_cells, "tie_groups": tie_groups}


def build_surface(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build one deterministic, analysis-only Flint surface candidate.

    Returns:
        Normalized surface candidate document.
    """
    if payload.get("schema_version") != SURFACE_INPUT_SCHEMA_VERSION:
        raise FlintChartContractError(f"schema_version must be {SURFACE_INPUT_SCHEMA_VERSION!r}")
    surface_id = _string(payload.get("surface_id"), name="surface_id")
    figure_id = _string(payload.get("figure_id"), name="figure_id")
    metric = _validate_metric(payload.get("metric"), name="metric")
    source = _validate_source(_mapping(payload.get("source"), name="source"))
    population = _population_keys(payload.get("display_population"))
    canonical = _index_cells(payload.get("canonical_cells"), name="canonical_cells")
    candidate = _index_cells(payload.get("candidate_cells"), name="candidate_cells")
    parity = _compare_cells(canonical, candidate, population)
    renderer_policy = _validate_renderer_policy(
        payload.get("renderer_policy"), figure_id=figure_id, cells=candidate
    )
    tie_result = _add_tie_metadata(candidate)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "surface_id": surface_id,
        "figure_id": figure_id,
        "metric": metric,
        "source_context": source["context"],
        "source": source,
        "display_population": [
            {"planner_key": planner_key, "scenario_family": scenario_family}
            for planner_key, scenario_family in population
        ],
        "coverage": {
            "status": "complete",
            "expected_cells": len(population),
            "actual_cells": len(candidate),
            "missing_cells": [],
            "duplicate_cells": [],
            "dropped_cells": [],
        },
        "cells": tie_result["cells"],
        "tie_groups": tie_result["tie_groups"],
        "parity": parity,
        "renderer_policy": renderer_policy,
        "render_status": "not_run",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _index_output_cells(value: Any, *, name: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate output cells while preserving only the normalized cell fields.

    Returns:
        Cell mapping keyed by planner and scenario family.
    """
    if not isinstance(value, list) or not value:
        raise FlintChartContractError(f"{name} must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, raw_cell in enumerate(value):
        cell = dict(_mapping(raw_cell, name=f"{name}[{index}]"))
        if cell.get("rank") is not None:
            raise FlintChartContractError(f"{name}[{index}].rank must remain null")
        tie_group = cell.get("tie_group")
        if tie_group is not None:
            _string(tie_group, name=f"{name}[{index}].tie_group")
        cell.pop("rank", None)
        cell.pop("tie_group", None)
        normalized.append(cell)
    return _index_cells(normalized, name=name)


def _output_tie_references(
    raw_cells: Any,
    *,
    name: str,
) -> dict[tuple[str, str], str | None]:
    """Normalize output-cell tie references.

    Returns:
        Tie-group reference keyed by planner and scenario family.
    """
    if not isinstance(raw_cells, list) or not raw_cells:
        raise FlintChartContractError(f"{name} requires a non-empty cells list")
    observed_ties: dict[tuple[str, str], str | None] = {}
    for index, raw_cell in enumerate(raw_cells):
        cell = _mapping(raw_cell, name=f"{name}.cells[{index}]")
        key = (
            _string(cell.get("planner_key"), name=f"{name}.cells[{index}].planner_key"),
            _string(
                cell.get("scenario_family"),
                name=f"{name}.cells[{index}].scenario_family",
            ),
        )
        if key in observed_ties:
            raise FlintChartContractError(f"{name}.cells contains duplicate cell {key!r}")
        tie_group = cell.get("tie_group")
        observed_ties[key] = (
            None
            if tie_group is None
            else _string(tie_group, name=f"{name}.cells[{index}].tie_group")
        )
    return observed_ties


def _normalize_tie_members(value: Any, *, name: str) -> list[list[str]]:
    """Normalize the members of one exact-value tie group.

    Returns:
        Normalized planner and scenario-family member pairs.
    """
    if not isinstance(value, list) or len(value) < 2:
        raise FlintChartContractError(f"{name} must contain at least two cells")
    members: list[list[str]] = []
    member_keys: set[tuple[str, str]] = set()
    for member_index, raw_member in enumerate(value):
        if not isinstance(raw_member, list) or len(raw_member) != 2:
            raise FlintChartContractError(f"{name}[{member_index}] must be a two-item list")
        member = (
            _string(raw_member[0], name=f"{name}[{member_index}][0]"),
            _string(raw_member[1], name=f"{name}[{member_index}][1]"),
        )
        if member in member_keys:
            raise FlintChartContractError(f"{name} contains duplicate cell {member!r}")
        member_keys.add(member)
        members.append(list(member))
    return members


def _normalize_tie_groups(
    value: Any,
    *,
    name: str,
) -> dict[str, dict[str, Any]]:
    """Normalize exact-value tie groups keyed by group id.

    Returns:
        Normalized tie groups keyed by group id.
    """
    if not isinstance(value, list):
        raise FlintChartContractError(f"{name} must be a list")
    observed_groups: dict[str, dict[str, Any]] = {}
    for index, raw_group in enumerate(value):
        group = _mapping(raw_group, name=f"{name}[{index}]")
        allowed_fields = {"group_id", "scenario_family", "value", "members"}
        unexpected_fields = set(group) - allowed_fields
        if unexpected_fields:
            raise FlintChartContractError(
                f"{name}[{index}] contains unsupported fields: {sorted(unexpected_fields)!r}"
            )
        group_id = _string(group.get("group_id"), name=f"{name}[{index}].group_id")
        if group_id in observed_groups:
            raise FlintChartContractError(f"{name} contains duplicate group {group_id!r}")
        observed_groups[group_id] = {
            "group_id": group_id,
            "scenario_family": _string(
                group.get("scenario_family"), name=f"{name}[{index}].scenario_family"
            ),
            "value": _finite_number(group.get("value"), name=f"{name}[{index}].value"),
            "members": _normalize_tie_members(
                group.get("members"), name=f"{name}[{index}].members"
            ),
        }
    return observed_groups


def _validate_tie_metadata(
    raw_cells: Any,
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
    value: Any,
    *,
    name: str,
) -> None:
    """Validate output-cell tie references and exact tie-group membership."""
    observed_ties = _output_tie_references(raw_cells, name=name)

    expected = _add_tie_metadata(cells)
    expected_ties = {
        (cell["planner_key"], cell["scenario_family"]): cell["tie_group"]
        for cell in expected["cells"]
    }
    if observed_ties != expected_ties:
        raise FlintChartContractError(f"{name} cell tie references do not match exact values")

    observed_groups = _normalize_tie_groups(value, name=name)
    expected_groups = {group["group_id"]: group for group in expected["tie_groups"]}
    if observed_groups != expected_groups:
        raise FlintChartContractError(f"{name} do not match exact-value tie groups")


def _validate_surface_coverage(
    value: Any,
    *,
    name: str,
    expected_cells: int,
    actual_cells: int,
) -> dict[str, Any]:
    """Validate and normalize complete surface coverage metadata.

    Returns:
        Normalized complete-coverage metadata.
    """
    coverage = _mapping(value, name=name)
    if coverage.get("status") != "complete":
        raise FlintChartContractError(f"{name}: status must be complete")
    if (
        coverage.get("expected_cells") != expected_cells
        or coverage.get("actual_cells") != actual_cells
    ):
        raise FlintChartContractError(f"{name}: counts do not match cells")
    for field in ("missing_cells", "duplicate_cells", "dropped_cells"):
        if coverage.get(field) != []:
            raise FlintChartContractError(f"{name}.{field} must be empty")
    return {
        "status": "complete",
        "expected_cells": expected_cells,
        "actual_cells": actual_cells,
        "missing_cells": [],
        "duplicate_cells": [],
        "dropped_cells": [],
    }


def _validate_surface_parity(value: Any, *, name: str, expected_cells: int) -> dict[str, Any]:
    """Validate and normalize complete canonical parity metadata.

    Returns:
        Normalized passed-parity metadata.
    """
    parity = _mapping(value, name=name)
    if parity.get("status") != "passed":
        raise FlintChartContractError(f"{name}: status must be passed")
    if parity.get("compared_cells") != expected_cells:
        raise FlintChartContractError(f"{name}: compared cell count does not match cells")
    if parity.get("compared_fields") != list(_PARITY_FIELDS):
        raise FlintChartContractError(f"{name}: fields are incomplete or reordered")
    if parity.get("missing_cells") != [] or parity.get("extra_cells") != []:
        raise FlintChartContractError(f"{name}: contains missing or extra cells")
    return {
        "status": "passed",
        "compared_cells": expected_cells,
        "compared_fields": list(_PARITY_FIELDS),
        "missing_cells": [],
        "extra_cells": [],
    }


def _validate_surface_document(surface: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    """Validate and normalize the surface contract consumed by the atlas builder.

    Returns:
        Normalized fields needed to build one atlas entry.
    """
    if surface.get("schema_version") != SURFACE_SCHEMA_VERSION:
        raise FlintChartContractError(f"{path}: unsupported surface schema")
    surface_id = _string(surface.get("surface_id"), name=f"{path}.surface_id")
    figure_id = _string(surface.get("figure_id"), name=f"{path}.figure_id")
    context = _string(surface.get("source_context"), name=f"{path}.source_context")
    if context not in {"release", "replay"}:
        raise FlintChartContractError(f"{path}: invalid source_context")
    source = _validate_source(_mapping(surface.get("source"), name=f"{path}.source"))
    if source["context"] != context:
        raise FlintChartContractError(f"{path}: source context disagrees with source_context")
    metric = _validate_metric(surface.get("metric"), name=f"{path}.metric")
    if surface.get("render_status") != "not_run":
        raise FlintChartContractError(
            f"{path}: render status must remain not_run in the foundation"
        )
    cells = _index_output_cells(surface.get("cells"), name=f"{path}.cells")
    population = _population_keys(surface.get("display_population"))
    if set(cells) != set(population):
        raise FlintChartContractError(f"{path}: cells do not match display population")
    coverage = _validate_surface_coverage(
        surface.get("coverage"),
        name=f"{path}.coverage",
        expected_cells=len(population),
        actual_cells=len(cells),
    )
    parity = _validate_surface_parity(
        surface.get("parity"), name=f"{path}.parity", expected_cells=len(population)
    )
    _validate_renderer_policy(surface.get("renderer_policy"), figure_id=figure_id, cells=cells)
    _validate_tie_metadata(
        surface.get("cells"),
        cells,
        surface.get("tie_groups"),
        name=f"{path}.tie_groups",
    )
    if _string(surface.get("claim_boundary"), name=f"{path}.claim_boundary") != CLAIM_BOUNDARY:
        raise FlintChartContractError(f"{path}: claim boundary is not analysis-only")
    return {
        "surface_id": surface_id,
        "figure_id": figure_id,
        "source_context": context,
        "source": source,
        "metric": metric,
        "coverage": coverage,
        "parity": parity,
    }


def build_atlas_manifest(surface_paths: Sequence[Path], *, atlas_id: str) -> dict[str, Any]:
    """Build a context-separated atlas manifest from surface candidate files.

    Returns:
        Normalized atlas manifest document.
    """
    if not surface_paths:
        raise FlintChartContractError("at least one surface is required")
    atlas_name = _string(atlas_id, name="atlas_id")
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    context_ids: dict[str, list[str]] = {"release": [], "replay": []}
    for path in surface_paths:
        surface = load_json(path)
        validated_surface = _validate_surface_document(surface, path=path)
        key = (validated_surface["surface_id"], validated_surface["source_context"])
        if key in seen:
            raise FlintChartContractError(f"duplicate surface context {key!r}")
        seen.add(key)
        context = key[1]
        context_ids[context].append(key[0])
        entries.append(
            {
                "surface_id": key[0],
                "figure_id": validated_surface["figure_id"],
                "source_context": context,
                "source": validated_surface["source"],
                "metric": validated_surface["metric"],
                "coverage": validated_surface["coverage"],
                "parity": validated_surface["parity"],
                "surface_sha256": sha256_file(path),
            }
        )
    entries.sort(key=lambda item: (item["surface_id"], item["source_context"]))
    for values in context_ids.values():
        values.sort()
    return {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "atlas_id": atlas_name,
        "claim_boundary": CLAIM_BOUNDARY,
        "renderer_policy": {
            "canonical_renderer": "matplotlib/pgf/tikz",
            "promotion_status": "not_admitted",
            "source_context_separation": "release_and_replay_are_separate_entries",
        },
        "surfaces": entries,
        "contexts": context_ids,
        "coverage": {
            "status": "complete",
            "surface_count": len(entries),
            "release_surface_count": len(context_ids["release"]),
            "replay_surface_count": len(context_ids["replay"]),
            "duplicate_surface_contexts": [],
            "dropped_surfaces": [],
        },
    }

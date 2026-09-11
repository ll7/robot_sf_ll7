"""Source-bound component-contrast analysis for issue #8566.

This module is a small, deterministic adapter for the author-authorized
fixture slice.  It binds tracked source bytes, keeps release lineages separate,
computes paired scenario-block uncertainty for synthetic rows, and reports
unavailable source-complete inputs without substitution.  It does not run a
campaign, hydrate external artifacts, change metric definitions, or promote a
scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.hierarchical_paired_release_analysis import holm_multiplicity
from robot_sf.benchmark.identity.hash_utils import sha256_file, stable_hash
from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "issue_8566_source_bound_component_contrasts.v1"
FIXTURE_SCHEMA_VERSION = "issue_8566_source_bound_component_rows.v1"
ISSUE = 8566
GENERATOR_PATH = "robot_sf/benchmark/source_bound_component_contrasts.py"
UNCERTAINTY_METHOD = "paired_scenario_block_percentile_bootstrap"
MULTIPLICITY_METHOD = "holm_step_down"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_LOCAL_ONLY_PARTS = frozenset({".git", ".venv", "output", "results"})


class SourceBoundComponentContrastError(RobotSfError, ValueError):
    """Raised when the source-bound fixture contract cannot be trusted."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    """Require a mapping at a named contract field.

    Returns:
        The validated mapping.
    """

    if not isinstance(value, Mapping):
        raise SourceBoundComponentContrastError(f"{field} must be a mapping")
    return value


def _text(value: Any, field: str) -> str:
    """Require a non-empty text field.

    Returns:
        The stripped text value.
    """

    if not isinstance(value, str) or not value.strip():
        raise SourceBoundComponentContrastError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    """Require a positive integer, rejecting booleans.

    Returns:
        The validated integer.
    """

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourceBoundComponentContrastError(f"{field} must be a positive integer")
    return value


def _finite_number(value: Any, field: str) -> float:
    """Require a finite numeric scalar.

    Returns:
        The value normalized to a float.
    """

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SourceBoundComponentContrastError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SourceBoundComponentContrastError(f"{field} must be a finite number")
    return number


def _sha256(value: Any, field: str) -> str:
    """Require a lowercase SHA-256 digest.

    Returns:
        The validated digest.
    """

    digest = _text(value, field)
    if _SHA256_RE.fullmatch(digest) is None:
        raise SourceBoundComponentContrastError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _repo_file(repo_root: Path, value: Any, field: str) -> tuple[str, Path]:
    """Resolve one safe repository-relative file and reject local-only paths.

    Returns:
        The normalized relative path and its resolved file path.
    """

    relative = _text(value, field)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise SourceBoundComponentContrastError(f"{field} must be repository-relative")
    if path.parts and path.parts[0] in _LOCAL_ONLY_PARTS:
        raise SourceBoundComponentContrastError(f"{field} points to local-only data: {relative}")
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise SourceBoundComponentContrastError(
            f"{field} resolves outside the repository: {relative}"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise SourceBoundComponentContrastError(f"{field} is not a regular file: {relative}")
    return relative, resolved


def _git_file_sha256(repo_root: Path, commit: str, relative_path: str) -> str:
    """Hash one path as stored in a git commit.

    Returns:
        The SHA-256 digest of the committed bytes.
    """

    if _COMMIT_RE.fullmatch(commit) is None:
        raise SourceBoundComponentContrastError(
            f"tracked commit for {relative_path} must be a 40-character SHA-1"
        )
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SourceBoundComponentContrastError(
            f"tracked commit {commit} does not contain {relative_path}"
        )
    return hashlib.sha256(result.stdout).hexdigest()


def load_source_bound_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML issue #8566 configuration without executing any analysis.

    Returns:
        The parsed configuration mapping.
    """

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SourceBoundComponentContrastError(
            f"could not read source-bound config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SourceBoundComponentContrastError("source-bound config must be a mapping")
    return dict(payload)


def load_fixture_rows(path: str | Path) -> dict[str, Any]:
    """Load the versioned synthetic paired-row fixture.

    Returns:
        The parsed fixture mapping.
    """

    fixture_path = Path(path)
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceBoundComponentContrastError(
            f"could not read source-bound fixture {fixture_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SourceBoundComponentContrastError("source-bound fixture must be a mapping")
    return dict(payload)


def _validate_config_identity(config: Mapping[str, Any]) -> None:
    """Validate the immutable issue and diagnostic status fields."""

    if config.get("schema_version") != f"{SCHEMA_VERSION}.config":
        raise SourceBoundComponentContrastError(f"schema_version must be {SCHEMA_VERSION!r}.config")
    if config.get("issue") != ISSUE:
        raise SourceBoundComponentContrastError(f"issue must be {ISSUE}")
    if config.get("status") != "diagnostic_only":
        raise SourceBoundComponentContrastError("status must remain diagnostic_only")
    if (
        config.get("base_commit") is None
        or _COMMIT_RE.fullmatch(str(config["base_commit"])) is None
    ):
        raise SourceBoundComponentContrastError("base_commit must be a 40-character SHA-1")


def _validate_execution(config: Mapping[str, Any]) -> None:
    """Reject campaign, compute, paper, and fallback execution authorization."""

    execution = _mapping(config.get("execution"), "execution")
    for field in (
        "campaign_execution",
        "compute_submit_authorized",
        "paper_facing_claim",
        "fallback_or_degraded_success_allowed",
    ):
        if execution.get(field) is not False:
            raise SourceBoundComponentContrastError(f"execution.{field} must be false")


def _validate_analysis(config: Mapping[str, Any]) -> None:
    """Validate the frozen pairing, clustering, resampling, and multiplicity rules."""

    analysis = _mapping(config.get("analysis"), "analysis")
    if analysis.get("pairing_key") != [
        "corpus_id",
        "release_id",
        "contrast_id",
        "scenario_id",
        "seed",
    ]:
        raise SourceBoundComponentContrastError("analysis.pairing_key is not the frozen key")
    if analysis.get("clustering_key") != "scenario_id":
        raise SourceBoundComponentContrastError("analysis.clustering_key must be scenario_id")
    uncertainty = _mapping(analysis.get("uncertainty"), "analysis.uncertainty")
    if uncertainty.get("method") != UNCERTAINTY_METHOD:
        raise SourceBoundComponentContrastError(
            f"analysis.uncertainty.method must be {UNCERTAINTY_METHOD!r}"
        )
    _positive_int(uncertainty.get("replicates"), "analysis.uncertainty.replicates")
    confidence = _finite_number(uncertainty.get("confidence"), "analysis.uncertainty.confidence")
    if not 0.0 < confidence < 1.0:
        raise SourceBoundComponentContrastError("analysis.uncertainty.confidence must be in (0, 1)")
    if isinstance(uncertainty.get("seed"), bool) or not isinstance(uncertainty.get("seed"), int):
        raise SourceBoundComponentContrastError("analysis.uncertainty.seed must be an integer")
    multiplicity = _mapping(analysis.get("multiplicity"), "analysis.multiplicity")
    if multiplicity.get("method") != MULTIPLICITY_METHOD:
        raise SourceBoundComponentContrastError(
            f"analysis.multiplicity.method must be {MULTIPLICITY_METHOD!r}"
        )
    alpha = _finite_number(multiplicity.get("alpha"), "analysis.multiplicity.alpha")
    if not 0.0 < alpha < 1.0:
        raise SourceBoundComponentContrastError("analysis.multiplicity.alpha must be in (0, 1)")
    _text(multiplicity.get("family"), "analysis.multiplicity.family")


def _validate_components(config: Mapping[str, Any]) -> list[str]:
    """Validate component identities and return their ordered ids.

    Returns:
        Component identifiers in the order used for multiplicity adjustment.
    """

    components = config.get("components")
    if not isinstance(components, list) or not components:
        raise SourceBoundComponentContrastError("components must be a non-empty list")
    component_ids: list[str] = []
    for index, component in enumerate(components):
        item = _mapping(component, f"components[{index}]")
        component_id = _text(item.get("id"), f"components[{index}].id")
        if component_id in component_ids:
            raise SourceBoundComponentContrastError(f"duplicate component id: {component_id}")
        component_ids.append(component_id)
        _text(item.get("family"), f"components[{index}].family")
        _text(item.get("unit"), f"components[{index}].unit")
        _text(item.get("metric_owner"), f"components[{index}].metric_owner")
    return component_ids


def _validate_generator(config: Mapping[str, Any], repo_root: Path) -> None:
    """Validate that the configured generator is this canonical module."""

    generator = _mapping(config.get("generator"), "generator")
    generator_path, _ = _repo_file(repo_root, generator.get("path"), "generator.path")
    if generator_path != GENERATOR_PATH:
        raise SourceBoundComponentContrastError(
            f"generator.path must be {GENERATOR_PATH!r}, got {generator_path!r}"
        )


def _validate_one_source_binding(item: Mapping[str, Any], *, index: int, repo_root: Path) -> str:
    """Validate one source binding and return its source identifier.

    Returns:
        The validated source identifier.
    """

    field_prefix = f"source_bindings[{index}]"
    source_id = _text(item.get("source_id"), f"{field_prefix}.source_id")
    relative, resolved = _repo_file(repo_root, item.get("path"), f"{field_prefix}.path")
    declared_sha = _sha256(item.get("sha256"), f"{field_prefix}.sha256")
    actual_sha = sha256_file(resolved)
    if actual_sha != declared_sha:
        raise SourceBoundComponentContrastError(
            f"source binding {source_id!r} digest mismatch: declared {declared_sha}, actual {actual_sha}"
        )
    kind = _text(item.get("kind"), f"{field_prefix}.kind")
    if kind == "release_metadata":
        source_commit = _text(item.get("source_commit"), f"{field_prefix}.source_commit")
        if _COMMIT_RE.fullmatch(source_commit) is None:
            raise SourceBoundComponentContrastError(
                f"{field_prefix}.source_commit must be a 40-character SHA-1"
            )
        tracked_commit = _text(item.get("tracked_commit"), f"{field_prefix}.tracked_commit")
        if _git_file_sha256(repo_root, tracked_commit, relative) != declared_sha:
            raise SourceBoundComponentContrastError(
                f"source binding {source_id!r} tracked bytes do not match {relative}"
            )
        _text(item.get("release_id"), f"{field_prefix}.release_id")
    elif kind in {"synthetic_fixture", "analysis_config"}:
        if item.get("source_commit") is not None or item.get("tracked_commit") is not None:
            raise SourceBoundComponentContrastError(
                f"synthetic source binding {source_id!r} cannot claim a release commit"
            )
    else:
        raise SourceBoundComponentContrastError(f"{field_prefix}.kind is unsupported: {kind!r}")
    return source_id


def _validate_source_bindings(config: Mapping[str, Any], repo_root: Path) -> set[str]:
    """Validate source files, digests, and release-byte provenance.

    Returns:
        The validated source identifiers.
    """

    bindings = config.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise SourceBoundComponentContrastError("source_bindings must be a non-empty list")
    source_ids: set[str] = set()
    for index, binding in enumerate(bindings):
        item = _mapping(binding, f"source_bindings[{index}]")
        source_id = _validate_one_source_binding(item, index=index, repo_root=repo_root)
        if source_id in source_ids:
            raise SourceBoundComponentContrastError(f"duplicate source id: {source_id}")
        source_ids.add(source_id)
    return source_ids


def _validate_fixture_contract(
    config: Mapping[str, Any], source_ids: set[str], repo_root: Path
) -> None:
    """Validate fixture identity, scenarios, and declared seeds."""

    fixture = _mapping(config.get("fixture"), "fixture")
    fixture_source_id = _text(fixture.get("source_id"), "fixture.source_id")
    if fixture_source_id not in source_ids:
        raise SourceBoundComponentContrastError("fixture.source_id is not a source binding")
    fixture_relative, fixture_file = _repo_file(repo_root, fixture.get("path"), "fixture.path")
    fixture_binding = next(
        item for item in config["source_bindings"] if item["source_id"] == fixture_source_id
    )
    if fixture_binding["path"] != fixture_relative:
        raise SourceBoundComponentContrastError(
            "fixture.path must match the fixture source binding path"
        )
    if fixture_binding["sha256"] != sha256_file(fixture_file):
        raise SourceBoundComponentContrastError(
            "fixture source binding digest must match fixture.path"
        )
    for field in ("corpus_id", "release_id", "contrast_id", "reference_arm", "comparison_arm"):
        _text(fixture.get(field), f"fixture.{field}")
    scenarios = fixture.get("scenarios")
    seeds = fixture.get("seeds")
    if not isinstance(scenarios, list) or not scenarios or len(set(scenarios)) != len(scenarios):
        raise SourceBoundComponentContrastError("fixture.scenarios must be unique and non-empty")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise SourceBoundComponentContrastError("fixture.seeds must be unique and non-empty")
    for scenario in scenarios:
        _text(scenario, "fixture.scenarios[]")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SourceBoundComponentContrastError("fixture.seeds[] must contain integers")


def _validate_external_targets(config: Mapping[str, Any], component_ids: Sequence[str]) -> None:
    """Require source-complete targets to remain explicitly unavailable."""

    external_targets = config.get("external_targets")
    if not isinstance(external_targets, list) or not external_targets:
        raise SourceBoundComponentContrastError("external_targets must be a non-empty list")
    for index, target in enumerate(external_targets):
        item = _mapping(target, f"external_targets[{index}]")
        _text(item.get("target_id"), f"external_targets[{index}].target_id")
        _positive_int(item.get("owner_issue"), f"external_targets[{index}].owner_issue")
        if item.get("status") != "unavailable_in_fixture_slice":
            raise SourceBoundComponentContrastError(
                f"external_targets[{index}].status must be unavailable_in_fixture_slice"
            )
        if item.get("component_ids") != list(component_ids):
            raise SourceBoundComponentContrastError(
                f"external_targets[{index}].component_ids must match declared components"
            )
        _text(item.get("reason"), f"external_targets[{index}].reason")


def validate_source_bound_config(
    config: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """Validate static identity, uncertainty, and fail-closed config rules.

    Returns:
        A shallow copy of the validated configuration mapping.
    """

    normalized = dict(config)
    _validate_config_identity(normalized)
    _validate_execution(normalized)
    _validate_analysis(normalized)
    component_ids = _validate_components(normalized)
    root = Path(repo_root).resolve()
    _validate_generator(normalized, root)
    source_ids = _validate_source_bindings(normalized, root)
    _validate_fixture_contract(normalized, source_ids, root)
    _validate_external_targets(normalized, component_ids)
    return normalized


def _validate_fixture_component_values(
    components: Mapping[str, Any], *, component_ids: Sequence[str], row_index: int
) -> None:
    """Validate available values and require reasons for unavailable cells."""

    if set(components) != set(component_ids):
        raise SourceBoundComponentContrastError(
            f"rows[{row_index}].components must match the declared component set"
        )
    for component_id in component_ids:
        component = _mapping(
            components[component_id], f"rows[{row_index}].components.{component_id}"
        )
        status = component.get("status")
        if status not in {"available", "unavailable"}:
            raise SourceBoundComponentContrastError(
                f"rows[{row_index}].components.{component_id}.status is invalid"
            )
        reference = component.get("reference")
        comparison = component.get("comparison")
        if status == "available":
            _finite_number(reference, f"rows[{row_index}].components.{component_id}.reference")
            _finite_number(comparison, f"rows[{row_index}].components.{component_id}.comparison")
            if component.get("reason") is not None:
                raise SourceBoundComponentContrastError(
                    f"available component {component_id!r} cannot carry an unavailable reason"
                )
        elif reference is not None or comparison is not None:
            raise SourceBoundComponentContrastError(
                f"unavailable component {component_id!r} must not carry numeric values"
            )
        else:
            _text(component.get("reason"), f"rows[{row_index}].components.{component_id}.reason")


def _validate_fixture_row(
    raw_row: Any,
    *,
    row_index: int,
    config: Mapping[str, Any],
    component_ids: Sequence[str],
) -> tuple[tuple[str, int], dict[str, Any]]:
    """Validate one fixture row and return its scenario/seed key and copy.

    Returns:
        The unique pair key and the validated row mapping.
    """

    row = _mapping(raw_row, f"rows[{row_index}]")
    for field in (
        "row_id",
        "source_id",
        "corpus_id",
        "release_id",
        "contrast_id",
        "scenario_id",
        "seed",
        "reference_arm",
        "comparison_arm",
        "components",
    ):
        if field not in row:
            raise SourceBoundComponentContrastError(f"rows[{row_index}] missing {field}")
    fixture = _mapping(config["fixture"], "fixture")
    _text(row["row_id"], f"rows[{row_index}].row_id")
    if row["source_id"] != fixture["source_id"]:
        raise SourceBoundComponentContrastError(f"rows[{row_index}] source_id is not fixture-bound")
    if row["corpus_id"] != fixture["corpus_id"] or row["release_id"] != fixture["release_id"]:
        raise SourceBoundComponentContrastError(
            f"rows[{row_index}] crosses the declared fixture release identity"
        )
    if row["contrast_id"] != fixture["contrast_id"]:
        raise SourceBoundComponentContrastError(
            f"rows[{row_index}] contrast_id is not fixture-bound"
        )
    if (
        row["reference_arm"] != fixture["reference_arm"]
        or row["comparison_arm"] != fixture["comparison_arm"]
    ):
        raise SourceBoundComponentContrastError(
            f"rows[{row_index}] arm identities do not match the fixture contract"
        )
    scenario = _text(row["scenario_id"], f"rows[{row_index}].scenario_id")
    declared_scenarios = {str(value) for value in fixture["scenarios"]}
    if scenario not in declared_scenarios:
        raise SourceBoundComponentContrastError(f"rows[{row_index}] has an undeclared scenario")
    seed = row["seed"]
    declared_seeds = {int(value) for value in fixture["seeds"]}
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in declared_seeds:
        raise SourceBoundComponentContrastError(f"rows[{row_index}].seed is not declared")
    _validate_fixture_component_values(
        _mapping(row["components"], f"rows[{row_index}].components"),
        component_ids=component_ids,
        row_index=row_index,
    )
    return (scenario, seed), dict(row)


def _validate_fixture_rows(
    config: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[tuple[str, int]]]:
    """Validate row identity and return rows plus explicitly missing pair keys.

    Returns:
        A pair of validated rows keyed by scenario/seed and missing expected keys.
    """

    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise SourceBoundComponentContrastError(
            f"fixture schema_version must be {FIXTURE_SCHEMA_VERSION!r}"
        )
    fixture = _mapping(config["fixture"], "fixture")
    if payload.get("source_id") != fixture["source_id"]:
        raise SourceBoundComponentContrastError("fixture source_id does not match config")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise SourceBoundComponentContrastError("fixture rows must be a list")
    component_ids = [str(item["id"]) for item in config["components"]]
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row_index, raw_row in enumerate(raw_rows):
        key, row = _validate_fixture_row(
            raw_row, row_index=row_index, config=config, component_ids=component_ids
        )
        if key in rows_by_key:
            raise SourceBoundComponentContrastError(f"duplicate fixture pair key: {key!r}")
        rows_by_key[key] = row
    expected_keys = sorted(
        (str(scenario), int(seed)) for scenario in fixture["scenarios"] for seed in fixture["seeds"]
    )
    missing = [key for key in expected_keys if key not in rows_by_key]
    return rows_by_key, missing


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    if not values:
        raise SourceBoundComponentContrastError("cannot compute a percentile from no samples")
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="linear"))


def _component_contrast(
    component: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    rows_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    missing_keys: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    """Compute one fixture component or emit a blocked component row.

    Returns:
        A diagnostic component contrast with explicit support and uncertainty.
    """

    component_id = str(component["id"])
    fixture = _mapping(config["fixture"], "fixture")
    analysis = _mapping(config["analysis"], "analysis")
    uncertainty = _mapping(analysis["uncertainty"], "analysis.uncertainty")
    method = str(uncertainty["method"])
    confidence = float(uncertainty["confidence"])
    replicates = int(uncertainty["replicates"])
    seed = int(uncertainty["seed"])
    source_refs = [
        {
            "source_id": str(binding["source_id"]),
            "path": str(binding["path"]),
            "sha256": str(binding["sha256"]),
        }
        for binding in config["source_bindings"]
        if binding["source_id"] == fixture["source_id"]
    ]
    if len(source_refs) != 1:
        raise SourceBoundComponentContrastError(
            f"fixture source binding is not unique: {fixture['source_id']!r}"
        )
    expected_keys = sorted(
        (str(scenario), int(pair_seed))
        for scenario in fixture["scenarios"]
        for pair_seed in fixture["seeds"]
    )
    excluded: list[dict[str, Any]] = []
    by_scenario: dict[str, list[float]] = {str(scenario): [] for scenario in fixture["scenarios"]}
    for key in expected_keys:
        row = rows_by_key.get(key)
        if row is None:
            excluded.append({"scenario_id": key[0], "seed": key[1], "reason": "missing_pair_row"})
            continue
        value = _mapping(row["components"], "row.components")[component_id]
        if value["status"] != "available":
            excluded.append(
                {
                    "scenario_id": key[0],
                    "seed": key[1],
                    "reason": str(value["reason"]),
                }
            )
            continue
        delta = _finite_number(value["comparison"], "component.comparison") - _finite_number(
            value["reference"], "component.reference"
        )
        by_scenario[key[0]].append(delta)

    if excluded:
        return {
            "component_id": component_id,
            "component_family": str(component["family"]),
            "unit": str(component["unit"]),
            "source_ids": [str(fixture["source_id"])],
            "source_refs": source_refs,
            "status": "blocked",
            "availability_status": "partial" if any(by_scenario.values()) else "unavailable",
            "reference_arm": str(fixture["reference_arm"]),
            "comparison_arm": str(fixture["comparison_arm"]),
            "estimand": "comparison_minus_reference_mean",
            "support": sum(len(values) for values in by_scenario.values()),
            "denominator": len(expected_keys),
            "excluded_cells": excluded,
            "effect": None,
            "uncertainty": {
                "declared": True,
                "method": method,
                "confidence": confidence,
                "replicates": replicates,
                "seed": seed,
                "ci_low": None,
                "ci_high": None,
                "p_value_raw": None,
                "p_value_adjusted": None,
            },
            "multiplicity": {
                "declared": False,
                "method": str(
                    _mapping(analysis["multiplicity"], "analysis.multiplicity")["method"]
                ),
                "n_comparisons": None,
                "status": "not_applied_blocked",
            },
            "interpretation_boundary": "No contrast is computed when a declared pair is unavailable.",
        }

    scenario_ids = [str(scenario) for scenario in fixture["scenarios"]]
    if any(not by_scenario[scenario] for scenario in scenario_ids):
        raise SourceBoundComponentContrastError(
            f"component {component_id!r} has no paired values in one or more scenarios"
        )
    seed_counts = {len(by_scenario[scenario]) for scenario in scenario_ids}
    if len(seed_counts) != 1:
        raise SourceBoundComponentContrastError(
            f"component {component_id!r} has unequal seed counts across scenarios"
        )
    deltas = [delta for scenario in scenario_ids for delta in by_scenario[scenario]]
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(replicates):
        sampled_indices = rng.integers(0, len(scenario_ids), size=len(scenario_ids))
        sampled = [
            delta
            for scenario_index in sampled_indices
            for delta in by_scenario[scenario_ids[int(scenario_index)]]
        ]
        samples.append(float(np.mean(sampled)))
    samples.sort()
    tail_low = (sum(value <= 0.0 for value in samples) + 1) / (len(samples) + 1)
    tail_high = (sum(value >= 0.0 for value in samples) + 1) / (len(samples) + 1)
    return {
        "component_id": component_id,
        "component_family": str(component["family"]),
        "unit": str(component["unit"]),
        "source_ids": [str(fixture["source_id"])],
        "source_refs": source_refs,
        "status": "diagnostic_only",
        "availability_status": "available",
        "reference_arm": str(fixture["reference_arm"]),
        "comparison_arm": str(fixture["comparison_arm"]),
        "estimand": "comparison_minus_reference_mean",
        "support": len(deltas),
        "denominator": len(expected_keys),
        "excluded_cells": [],
        "effect": float(np.mean(deltas)),
        "uncertainty": {
            "declared": True,
            "method": method,
            "confidence": confidence,
            "replicates": replicates,
            "seed": seed,
            "ci_low": _percentile(samples, (1.0 - confidence) / 2.0),
            "ci_high": _percentile(samples, 1.0 - (1.0 - confidence) / 2.0),
            "p_value_raw": min(1.0, 2.0 * min(tail_low, tail_high)),
            "p_value_adjusted": None,
        },
        "multiplicity": {
            "declared": True,
            "method": str(_mapping(analysis["multiplicity"], "analysis.multiplicity")["method"]),
            "n_comparisons": len(config["components"]),
            "status": "pending",
        },
        "interpretation_boundary": "Synthetic fixture calculation only; not release or benchmark evidence.",
    }


def _source_coverage(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Emit explicit unavailable coverage for source-complete targets.

    Returns:
        One unavailable coverage record for each external target.
    """

    analysis = _mapping(config["analysis"], "analysis")
    method = str(_mapping(analysis["uncertainty"], "analysis.uncertainty")["method"])
    components = [str(item["id"]) for item in config["components"]]
    coverage: list[dict[str, Any]] = []
    for target in config["external_targets"]:
        item = dict(target)
        expected_contrasts = item.get("expected_contrast_count")
        coverage.append(
            {
                "target_id": str(item["target_id"]),
                "owner_issue": int(item["owner_issue"]),
                "status": str(item["status"]),
                "release_id": item.get("release_id"),
                "source_commit": item.get("source_commit"),
                "source_uri": item.get("source_uri"),
                "expected_contrast_count": expected_contrasts,
                "expected_source_rows": item.get("expected_source_rows"),
                "component_coverage": [
                    {
                        "component_id": component_id,
                        "status": "unavailable",
                        "support": 0,
                        "denominator": expected_contrasts,
                        "denominator_status": (
                            "declared" if expected_contrasts is not None else "not_available"
                        ),
                        "uncertainty": {"status": "not_available", "method": method},
                        "reason": str(item["reason"]),
                    }
                    for component_id in components
                ],
            }
        )
    return coverage


def _artifact_digest(report: Mapping[str, Any]) -> str:
    """Hash a report while excluding only its self-describing digest value.

    Returns:
        The canonical report digest.
    """

    digest_input = json.loads(json.dumps(report))
    artifact_identity = dict(digest_input["artifact_identity"])
    artifact_identity.pop("sha256", None)
    digest_input["artifact_identity"] = artifact_identity
    return stable_hash(digest_input)


def validate_source_bound_report(report: Mapping[str, Any]) -> None:
    """Validate the report schema identity and canonical self-digest."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise SourceBoundComponentContrastError(f"report schema_version must be {SCHEMA_VERSION!r}")
    if report.get("issue") != ISSUE:
        raise SourceBoundComponentContrastError(f"report issue must be {ISSUE}")
    artifact_identity = _mapping(report.get("artifact_identity"), "artifact_identity")
    declared = _sha256(artifact_identity.get("sha256"), "artifact_identity.sha256")
    if _artifact_digest(report) != declared:
        raise SourceBoundComponentContrastError("report artifact identity digest does not match")


def build_source_bound_report(config_path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    """Build the deterministic fixture report from tracked inputs only.

    Returns:
        The validated diagnostic report.
    """

    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    try:
        config_file.relative_to(root)
    except ValueError as exc:
        raise SourceBoundComponentContrastError(
            "config_path must be inside the repository"
        ) from exc
    config = validate_source_bound_config(load_source_bound_config(config_file), repo_root=root)
    fixture = _mapping(config["fixture"], "fixture")
    fixture_relative, fixture_file = _repo_file(root, fixture["path"], "fixture.path")
    fixture_payload = load_fixture_rows(fixture_file)
    rows_by_key, missing_keys = _validate_fixture_rows(config, fixture_payload)
    components = [dict(item) for item in config["components"]]
    component_contrasts = [
        _component_contrast(
            component,
            config=config,
            rows_by_key=rows_by_key,
            missing_keys=missing_keys,
        )
        for component in components
    ]
    blockers = []
    if missing_keys:
        blockers.append(f"missing fixture pair cells: {missing_keys!r}")
    blockers.extend(
        f"component {row['component_id']} blocked: {row['excluded_cells']}"
        for row in component_contrasts
        if row["status"] == "blocked"
    )
    available = [row for row in component_contrasts if row["status"] == "diagnostic_only"]
    p_values = [float(row["uncertainty"]["p_value_raw"]) for row in available]
    multiplicity_config = _mapping(
        _mapping(config["analysis"], "analysis")["multiplicity"], "analysis.multiplicity"
    )
    multiplicity_decisions = holm_multiplicity(p_values, alpha=float(multiplicity_config["alpha"]))
    for row, decision in zip(available, multiplicity_decisions, strict=True):
        row["uncertainty"]["p_value_adjusted"] = decision.adjusted_p_value
        row["multiplicity"]["status"] = "applied"
        row["multiplicity"]["rejected"] = decision.rejected

    config_relative = config_file.relative_to(root).as_posix()
    generator_relative, generator_file = _repo_file(
        root, config["generator"]["path"], "generator.path"
    )
    source_bindings = []
    for binding in config["source_bindings"]:
        item = dict(binding)
        item["sha256"] = _sha256(item["sha256"], f"source_bindings.{item['source_id']}.sha256")
        source_bindings.append(item)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "status": "blocked" if blockers else "diagnostic_only",
        "evidence_status": "not_benchmark_evidence",
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "claim_boundary": {
            "status": "diagnostic_only" if not blockers else "blocked",
            "allowed": [
                "Tracked source-binding, pairing, uncertainty, determinism, and fail-closed behavior of this fixture packet."
            ],
            "forbidden": [
                "Any empirical, benchmark, causal, safety, planner-ranking, release, paper, or dissertation claim.",
                "Pooling August b1d5 and September 59577 release lineages.",
                "Treating fallback, degraded, inactive, or unavailable input as success or zero.",
            ],
        },
        "provenance": {
            "base_commit": str(config["base_commit"]),
            "config": {"path": config_relative, "sha256": sha256_file(config_file)},
            "fixture": {
                "path": fixture_relative,
                "source_id": str(fixture["source_id"]),
                "sha256": sha256_file(fixture_file),
            },
            "generator": {"path": generator_relative, "sha256": sha256_file(generator_file)},
            "source_bindings": source_bindings,
        },
        "analysis": {
            "estimand": "comparison_minus_reference_mean",
            "pairing_key": list(_mapping(config["analysis"], "analysis")["pairing_key"]),
            "clustering_key": "scenario_id",
            "uncertainty": dict(_mapping(config["analysis"], "analysis")["uncertainty"]),
            "multiplicity": {
                "method": str(multiplicity_config["method"]),
                "family": str(multiplicity_config["family"]),
                "alpha": float(multiplicity_config["alpha"]),
                "n_comparisons": len(available),
                "applied_to": [str(row["component_id"]) for row in available],
            },
        },
        "fixture": {
            "corpus_id": str(fixture["corpus_id"]),
            "release_id": str(fixture["release_id"]),
            "contrast_id": str(fixture["contrast_id"]),
            "reference_arm": str(fixture["reference_arm"]),
            "comparison_arm": str(fixture["comparison_arm"]),
            "expected_pair_cells": len(fixture["scenarios"]) * len(fixture["seeds"]),
            "observed_pair_cells": len(rows_by_key),
            "missing_pair_cells": [
                {"scenario_id": scenario, "seed": seed} for scenario, seed in missing_keys
            ],
        },
        "fixture_coverage": [
            {
                "scenario_id": scenario,
                "seed": seed,
                "component_id": component_id,
                "status": (
                    "unavailable"
                    if (scenario, seed) not in rows_by_key
                    else str(
                        _mapping(rows_by_key[(scenario, seed)]["components"], "row.components")[
                            component_id
                        ]["status"]
                    )
                ),
                "reason": (
                    "missing_pair_row"
                    if (scenario, seed) not in rows_by_key
                    else _mapping(rows_by_key[(scenario, seed)]["components"], "row.components")[
                        component_id
                    ].get("reason")
                ),
            }
            for scenario in sorted(str(value) for value in fixture["scenarios"])
            for seed in sorted(int(value) for value in fixture["seeds"])
            for component_id in [str(item["id"]) for item in components]
        ],
        "component_contrasts": component_contrasts,
        "source_coverage": _source_coverage(config),
        "lineage_guard": dict(config["lineage_guard"]),
        "blockers": blockers,
        "semantics": {
            "campaign_executed": False,
            "external_artifact_hydrated": False,
            "new_episode_created": False,
            "fallback_rows": 0,
            "degraded_rows": 0,
            "benchmark_evidence": False,
            "claim_promotion": "none",
        },
        "artifact": {
            "report_path": str(config["artifact"]["report_path"]),
            "receipt_path": str(config["artifact"]["receipt_path"]),
        },
        "artifact_identity": {
            "hash_scope": "canonical report JSON with artifact_identity.sha256 omitted",
            "sha256": "0" * 64,
        },
    }
    report["artifact_identity"]["sha256"] = _artifact_digest(report)
    validate_source_bound_report(report)
    return report


def write_source_bound_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Write one deterministic, human-readable JSON report."""

    validate_source_bound_report(report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_source_bound_receipt(
    report: Mapping[str, Any],
    *,
    config_path: str | Path,
    repo_root: str | Path,
    report_path: str | Path,
    receipt_path: str | Path,
) -> None:
    """Write a root-relative SHA-256 receipt for all durable packet inputs."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    generator_file = root / str(report["provenance"]["generator"]["path"])
    report_file = Path(report_path).resolve()
    files = [config_file]
    files.extend(root / str(binding["path"]) for binding in report["provenance"]["source_bindings"])
    files.extend((generator_file, report_file))
    unique_files: list[Path] = []
    for path in files:
        if path not in unique_files:
            unique_files.append(path)
    lines = [
        "# AI-GENERATED NEEDS-REVIEW",
        "# SHA-256 receipt for issue #8566 source-bound fixture packet",
        "# Paths are repository-relative; run sha256sum -c from the repository root.",
    ]
    for path in unique_files:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise SourceBoundComponentContrastError(
                f"receipt path is outside the repository: {path}"
            ) from exc
        lines.append(f"{sha256_file(path)}  {relative}")
    output = Path(receipt_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Build the fixture report and optional SHA-256 receipt.

    Returns:
        Zero on success and two when the source-bound contract is rejected.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        report = build_source_bound_report(args.config, repo_root=repo_root)
        write_source_bound_report(report, args.output)
        write_source_bound_receipt(
            report,
            config_path=args.config,
            repo_root=repo_root,
            report_path=args.output,
            receipt_path=args.receipt,
        )
    except (OSError, SourceBoundComponentContrastError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.output),
                "receipt": str(args.receipt),
                "artifact_sha256": report["artifact_identity"]["sha256"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


__all__ = [
    "FIXTURE_SCHEMA_VERSION",
    "GENERATOR_PATH",
    "ISSUE",
    "SCHEMA_VERSION",
    "SourceBoundComponentContrastError",
    "build_source_bound_report",
    "load_fixture_rows",
    "load_source_bound_config",
    "main",
    "validate_source_bound_config",
    "validate_source_bound_report",
    "write_source_bound_receipt",
    "write_source_bound_report",
]

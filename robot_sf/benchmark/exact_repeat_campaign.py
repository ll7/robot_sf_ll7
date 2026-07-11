"""Fail-closed exact-repeat campaign packets for issue #5263.

The retained #4978 fixture identifies seven knife-edge scenario/planner cells,
but it deliberately omits runnable scenario and planner definitions.  This
module turns that bounded, durable evidence into a campaign manifest and checks
the result files produced by a future CPU-only run.  It does not execute a
campaign, change metric semantics, or infer a determinism verdict from missing
trajectory evidence.

Every requested ``(scenario, planner, seed)`` target needs exactly three
repeats.  A target is bitwise-identical only when its binary outcome and
SHA-256 trajectory digest agree for all repeats.  A differing target must name
its first divergence.  Two verified host reports can then be compared only when
their pinned NumPy and Numba versions agree.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

BASELINE_SCHEMA_VERSION = "scenario_flakiness.v1"
MANIFEST_SCHEMA_VERSION = "scenario_exact_repeat_campaign.v1"
HOST_REPORT_SCHEMA_VERSION = "scenario_exact_repeat_host_result.v1"
VERIFIED_HOST_REPORT_SCHEMA_VERSION = "scenario_exact_repeat_verified_host_result.v1"
CROSS_HOST_SCHEMA_VERSION = "scenario_exact_repeat_cross_host.v1"
DEFAULT_REPEATS = 3


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_text(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
        raise ValueError(f"{label} must be a lowercase-or-uppercase SHA-256 digest")
    return digest.lower()


def _planner(record: Mapping[str, Any]) -> str:
    return _require_text(record.get("algo", record.get("planner")), "episode planner")


def _target_key(target: Mapping[str, Any]) -> tuple[str, str, int]:
    seed = target.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("target seed must be an integer")
    return (
        _require_text(target.get("scenario_id"), "target scenario_id"),
        _require_text(target.get("planner"), "target planner"),
        seed,
    )


def build_manifest(  # noqa: C901 - validation branches make the fail-closed contract explicit.
    baseline_report: Mapping[str, Any],
    source_episodes: Sequence[Mapping[str, Any]],
    *,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Build an exact-repeat request from a flakiness report and source rows.

    The request pins the selected cells, their original seed/horizon/commit and
    config hashes.  It records that runnable definitions are still required;
    this is intentional because the committed #4978 fixture retains only audit
    fields, not the original scenario/planner configurations.

    Returns:
        An immutable, schema-versioned 420-run request with source provenance.
    """
    if repeats != DEFAULT_REPEATS:
        raise ValueError(f"issue #5263 requires exactly {DEFAULT_REPEATS} repeats, got {repeats}")
    if baseline_report.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError("baseline report is not scenario_flakiness.v1")
    cells = baseline_report.get("cells")
    if not isinstance(cells, list):
        raise ValueError("baseline report cells must be a list")
    knife_edges = [
        cell for cell in cells if isinstance(cell, Mapping) and cell.get("knife_edge") is True
    ]
    if not knife_edges:
        raise ValueError("baseline report contains no knife-edge cells")

    selected_cells = {
        (
            _require_text(cell.get("scenario_id"), "knife-edge scenario_id"),
            _require_text(cell.get("planner"), "knife-edge planner"),
        )
        for cell in knife_edges
    }
    selected_rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in source_episodes:
        row = _require_mapping(row, "source episode")
        key = (_require_text(row.get("scenario_id"), "episode scenario_id"), _planner(row))
        if key not in selected_cells:
            continue
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"source episode {key} has no integer seed")
        target = (*key, seed)
        if target in selected_rows:
            raise ValueError(f"duplicate source episode for target {target}")
        selected_rows[target] = row

    expected_targets = sum(int(cell.get("n_seeds", 0)) for cell in knife_edges)
    if len(selected_rows) != expected_targets:
        raise ValueError(
            "source episodes do not cover every seed in the knife-edge cells: "
            f"expected {expected_targets}, found {len(selected_rows)}"
        )

    targets = []
    for key, row in sorted(selected_rows.items()):
        horizon = row.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"source episode {key} has no positive integer horizon")
        targets.append(
            {
                "scenario_id": key[0],
                "planner": key[1],
                "seed": key[2],
                "horizon": horizon,
                "source_git_hash": _require_text(row.get("git_hash"), "source git_hash"),
                "source_config_hash": _require_text(row.get("config_hash"), "source config_hash"),
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "claim_boundary": "diagnostic-only; no campaign result is represented by this manifest",
        "execution_contract": {
            "cpu_only": True,
            "workers": 1,
            "repeats_per_target": repeats,
            "trajectory_hash": "sha256",
            "required_runtime_versions": ["numpy_version", "numba_version"],
        },
        "source": {
            "baseline_report_sha256": canonical_sha256(baseline_report),
            "source_episodes_sha256": canonical_sha256(list(source_episodes)),
            "runnable_definitions_required": ["scenario_params", "planner_config"],
        },
        "cells": [
            {"scenario_id": scenario_id, "planner": planner}
            for scenario_id, planner in sorted(selected_cells)
        ],
        "targets": targets,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _check_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, int], Mapping[str, Any]], int]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest has an unsupported schema_version")
    contract = _require_mapping(manifest.get("execution_contract"), "manifest execution_contract")
    if contract.get("cpu_only") is not True or contract.get("workers") != 1:
        raise ValueError("manifest must require CPU-only single-worker execution")
    repeats = contract.get("repeats_per_target")
    if repeats != DEFAULT_REPEATS:
        raise ValueError(f"manifest must require exactly {DEFAULT_REPEATS} repeats")
    expected_hash = manifest.get("manifest_sha256")
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if expected_hash != canonical_sha256(without_hash):
        raise ValueError("manifest_sha256 does not match the manifest content")
    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("manifest targets must be a non-empty list")
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for target in targets:
        target = _require_mapping(target, "manifest target")
        key = _target_key(target)
        if key in indexed:
            raise ValueError(f"manifest contains duplicate target {key}")
        _require_text(target.get("source_git_hash"), "target source_git_hash")
        _require_text(target.get("source_config_hash"), "target source_config_hash")
        horizon = target.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("target horizon must be a positive integer")
        indexed[key] = target
    return indexed, repeats


def _repeat_fingerprint(repeat: Mapping[str, Any]) -> tuple[int, str, Any]:
    outcome = repeat.get("outcome")
    if isinstance(outcome, bool):
        outcome = int(outcome)
    if outcome not in (0, 1):
        raise ValueError("repeat outcome must be binary")
    trajectory = _require_sha256(repeat.get("trajectory_sha256"), "repeat trajectory_sha256")
    return int(outcome), trajectory, repeat.get("near_misses")


def _first_divergence(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    reference = _repeat_fingerprint(repeats[0])
    for index, repeat in enumerate(repeats[1:], start=1):
        current = _repeat_fingerprint(repeat)
        for field, expected, observed in zip(
            ("outcome", "trajectory_sha256", "near_misses"), reference, current, strict=True
        ):
            if expected != observed:
                return {
                    "repeat_index": index,
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                }
    return None


def verify_host_report(  # noqa: C901, PLR0912 - each rejected report state needs a specific error.
    manifest: Mapping[str, Any], host_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify one host's exact-repeat results against the immutable manifest.

    Missing, surplus, malformed, or divergent results are errors or explicit
    non-identical verdicts; none can become an implicit successful cell.

    Returns:
        A verified report with per-target and per-cell determinism verdicts.
    """
    targets, repeats_per_target = _check_manifest(manifest)
    if host_report.get("schema_version") != HOST_REPORT_SCHEMA_VERSION:
        raise ValueError("host report has an unsupported schema_version")
    if host_report.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("host report was not produced for this manifest")
    environment = _require_mapping(host_report.get("environment"), "host report environment")
    for field in ("machine_id", "numpy_version", "numba_version", "python_version", "git_commit"):
        _require_text(environment.get(field), f"host environment {field}")
    if environment.get("cpu_only") is not True or environment.get("workers") != 1:
        raise ValueError("host report must record CPU-only single-worker execution")
    expected_commits = {target["source_git_hash"] for target in targets.values()}
    if environment["git_commit"] not in expected_commits:
        raise ValueError("host report git_commit does not match the manifest source revision")

    results = host_report.get("results")
    if not isinstance(results, list):
        raise ValueError("host report results must be a list")
    indexed_results: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for result in results:
        result = _require_mapping(result, "host result")
        key = _target_key(result)
        if key in indexed_results:
            raise ValueError(f"host report contains duplicate result for {key}")
        if key not in targets:
            raise ValueError(f"host report contains unexpected target {key}")
        expected_target = targets[key]
        for field in ("horizon", "source_config_hash"):
            if result.get(field) != expected_target[field]:
                raise ValueError(f"host report target {key} has a mismatched {field}")
        indexed_results[key] = result
    missing = sorted(set(targets) - set(indexed_results))
    if missing:
        raise ValueError(f"host report is missing manifest targets: {missing[:3]}")

    verified_targets = []
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(targets):
        result = indexed_results[key]
        repeats = result.get("repeats")
        if not isinstance(repeats, list) or len(repeats) != repeats_per_target:
            raise ValueError(f"target {key} must contain exactly {repeats_per_target} repeats")
        repeat_maps = [_require_mapping(item, "repeat") for item in repeats]
        divergence = _first_divergence(repeat_maps)
        reported = result.get("first_divergence")
        if divergence is None and reported is not None:
            raise ValueError(f"target {key} reports a divergence but repeats are identical")
        if divergence is not None and reported != divergence:
            raise ValueError(f"target {key} must record the computed first divergence")
        verified = {
            "scenario_id": key[0],
            "planner": key[1],
            "seed": key[2],
            "bitwise_identical": divergence is None,
            "first_divergence": divergence,
            "repeat_fingerprints": [_repeat_fingerprint(item) for item in repeat_maps],
        }
        verified_targets.append(verified)
        by_cell[key[:2]].append(verified)

    cells = []
    for (scenario_id, planner), target_results in sorted(by_cell.items()):
        first = next(
            (item["first_divergence"] for item in target_results if item["first_divergence"]), None
        )
        cells.append(
            {
                "scenario_id": scenario_id,
                "planner": planner,
                "exact_repeat_determinism": first is None,
                "first_divergence": first,
                "n_targets": len(target_results),
            }
        )
    verified = {
        "schema_version": VERIFIED_HOST_REPORT_SCHEMA_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "environment": dict(environment),
        "targets": verified_targets,
        "cells": cells,
        "summary": {
            "n_targets": len(verified_targets),
            "n_cells": len(cells),
            "all_cells_bitwise_identical": all(cell["exact_repeat_determinism"] for cell in cells),
        },
    }
    return verified


def compare_verified_hosts(
    manifest: Mapping[str, Any], first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the fail-closed two-host exact-repeat comparison matrix.

    Returns:
        A matrix whose cells are identical only with matching runtime versions
        and matching repeat fingerprints.
    """
    targets, _ = _check_manifest(manifest)
    for report in (first, second):
        if report.get("schema_version") != VERIFIED_HOST_REPORT_SCHEMA_VERSION:
            raise ValueError("cross-host input is not a verified host report")
        if report.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("cross-host input belongs to a different manifest")
    first_env = _require_mapping(first.get("environment"), "first host environment")
    second_env = _require_mapping(second.get("environment"), "second host environment")
    first_machine = _require_text(first_env.get("machine_id"), "first machine_id")
    second_machine = _require_text(second_env.get("machine_id"), "second machine_id")
    if first_machine == second_machine:
        raise ValueError("cross-host comparison requires two distinct machine_id values")
    version_match = all(
        first_env.get(key) == second_env.get(key) for key in ("numpy_version", "numba_version")
    )

    def index(report: Mapping[str, Any]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
        rows = report.get("targets")
        if not isinstance(rows, list):
            raise ValueError("verified host report targets must be a list")
        return {_target_key(_require_mapping(row, "verified target")): row for row in rows}

    first_targets, second_targets = index(first), index(second)
    if set(first_targets) != set(targets) or set(second_targets) != set(targets):
        raise ValueError("verified host reports do not cover the manifest target set")
    cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for key in sorted(targets):
        identical = first_targets[key].get("repeat_fingerprints") == second_targets[key].get(
            "repeat_fingerprints"
        )
        cells[key[:2]].append(bool(identical))
    matrix = [
        {
            "scenario_id": scenario_id,
            "planner": planner,
            "bitwise_identical": version_match and all(target_matches),
            "comparison_status": "identical"
            if version_match and all(target_matches)
            else "divergent",
        }
        for (scenario_id, planner), target_matches in sorted(cells.items())
    ]
    return {
        "schema_version": CROSS_HOST_SCHEMA_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "hosts": [first_machine, second_machine],
        "pinned_runtime_versions_match": version_match,
        "matrix": matrix,
        "summary": {
            "n_cells": len(matrix),
            "all_cells_bitwise_identical": version_match
            and all(row["bitwise_identical"] for row in matrix),
        },
    }

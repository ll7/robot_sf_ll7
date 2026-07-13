"""Fail-closed exact-repeat campaign packets for issue #5263.

The retained #4978 fixture identifies seven knife-edge scenario/planner cells,
but it deliberately omits runnable scenario and planner definitions.  This
module turns that bounded, durable evidence into a campaign manifest, resolves
runnable definitions, executes the repeat cells, and verifies host results.
It does not change metric semantics or infer a determinism verdict from missing
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
import math
import platform
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numba
import numpy as np
import yaml

from robot_sf.benchmark.map_runner_identity import _scenario_with_episode_seed_defaults
from robot_sf.benchmark.utils import _config_hash
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from pathlib import Path

BASELINE_SCHEMA_VERSION = "scenario_flakiness.v1"
MANIFEST_SCHEMA_VERSION = "scenario_exact_repeat_campaign.v1"
HOST_REPORT_SCHEMA_VERSION = "scenario_exact_repeat_host_result.v1"
VERIFIED_HOST_REPORT_SCHEMA_VERSION = "scenario_exact_repeat_verified_host_result.v1"
CROSS_HOST_SCHEMA_VERSION = "scenario_exact_repeat_cross_host.v1"
RESOLVED_DEFINITIONS_SCHEMA_VERSION = "scenario_exact_repeat_resolved_definitions.v1"
DEFAULT_REPEATS = 3
SOURCE_IDENTITY_REVISION = "a5516b432fceffa71573e458aaee31c00a0b6c81"

# Explicit per-cell disposition for planners the exact-repeat ``execute`` path
# cannot construct on current main (for example, a planner registered only in a
# separate map-runner pipeline). Such cells are recorded, not crashed,
# and are excluded from the bitwise-identical repeat claim.
UNRUNNABLE_DISPOSITION = "unrunnable_on_current_main"
MIXED_DISPOSITION = "mixed_runnability"


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
                "source_observation_mode": _require_text(
                    row.get("observation_mode"), "source observation_mode"
                ),
                "source_observation_level": _require_text(
                    row.get("observation_level"), "source observation_level"
                ),
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
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


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("campaign scenario has no non-empty name, scenario_id, or id")


def _planner_index(planners: Sequence[Any]) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for planner in planners:
        identity = str(planner.key)
        if identity in indexed:
            raise ValueError(f"campaign contains duplicate planner key {identity!r}")
        indexed[identity] = planner
    algo_counts: dict[str, int] = defaultdict(int)
    for planner in planners:
        algo_counts[str(planner.algo)] += 1
    for planner in planners:
        algo = str(planner.algo)
        if algo_counts[algo] == 1:
            indexed.setdefault(algo, planner)
    return indexed


def _historical_identity_payload(
    scenario: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    algo: str,
    algo_config: Mapping[str, Any],
    record_forces: bool,
    record_simulation_step_trace: bool,
    dt: float | None,
    horizon: int | None,
) -> dict[str, Any]:
    """Rebuild the map-runner identity contract used by the retained source revision.

    Returns:
        The canonical historical identity payload used to compute ``source_config_hash``.
    """

    payload = {key: value for key, value in scenario.items() if key not in {"seed", "seeds"}}
    payload.setdefault("id", _scenario_id(scenario))
    payload["algo"] = algo
    payload["algo_config_hash"] = _config_hash(dict(algo_config))
    payload["record_forces"] = record_forces
    payload["observation_mode"] = _require_text(
        target.get("source_observation_mode"), "target source_observation_mode"
    )
    payload["observation_level"] = _require_text(
        target.get("source_observation_level"), "target source_observation_level"
    )
    # Source revision a5516b432 used this field but predates the separate planner-decision trace
    # identity field. Keeping the historical shape is necessary to verify the retained hashes.
    payload["record_simulation_step_trace"] = record_simulation_step_trace
    if horizon is not None and horizon > 0:
        payload["run_horizon"] = horizon
    if dt is not None and dt > 0:
        payload["run_dt"] = dt
    return payload


def resolve_runnable_definitions(  # noqa: C901 - fail-closed recovery validates each input axis.
    manifest: Mapping[str, Any], campaign_config_path: Path
) -> dict[str, Any]:
    """Recover and hash-check every runnable scenario/planner target in the manifest.

    The retained fixture was slimmed after the source campaign, but its source revision and
    per-target map-runner identity hashes remain available. This resolver materializes the
    scenario and planner objects from the canonical source campaign config and accepts them only
    when all target hashes and horizons reproduce exactly.

    Returns:
        A diagnostic-only bundle with all 140 hash-matched runnable target definitions.
    """
    targets, _ = _check_manifest(manifest)
    source_commits = sorted({str(target["source_git_hash"]) for target in targets.values()})
    if source_commits != [SOURCE_IDENTITY_REVISION]:
        raise ValueError(
            "definition recovery supports only the retained #5263 source revision "
            f"{SOURCE_IDENTITY_REVISION}"
        )
    from robot_sf.benchmark.camera_ready._config import (  # noqa: PLC0415
        _load_campaign_scenarios,
        _scenario_with_kinematics,
        load_campaign_config,
    )

    config_path = campaign_config_path.resolve()
    try:
        config_repo_path = config_path.relative_to(get_repository_root().resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("source campaign config must be repository-local") from exc
    cfg = load_campaign_config(config_path)
    if tuple(cfg.kinematics_matrix) != ("differential_drive",):
        raise ValueError("source campaign must resolve exactly one differential_drive kinematics")

    scenarios = {_scenario_id(item): item for item in _load_campaign_scenarios(cfg)}
    planners = _planner_index(cfg.planners)
    resolved_targets: list[dict[str, Any]] = []
    scenario_definitions: dict[str, dict[str, Any]] = {}
    planner_definitions: dict[str, dict[str, Any]] = {}
    for key, target in sorted(targets.items()):
        scenario = scenarios.get(key[0])
        if scenario is None:
            raise ValueError(f"source campaign is missing target scenario {key[0]!r}")
        planner = planners.get(key[1])
        if planner is None:
            raise ValueError(f"source campaign is missing target planner {key[1]!r}")
        if not planner.enabled:
            raise ValueError(f"source campaign target planner {key[1]!r} is disabled")

        scenario_params = _scenario_with_kinematics(
            scenario,
            kinematics="differential_drive",
            holonomic_command_mode=cfg.holonomic_command_mode,
        )
        scenario_params = _scenario_with_episode_seed_defaults(scenario_params, seed=key[2])
        scenario_horizon = scenario_params.get("simulation_config", {}).get("max_episode_steps")
        effective_horizon = (
            planner.horizon_override if planner.horizon_override is not None else cfg.horizon
        )
        if effective_horizon is None:
            if scenario_horizon != target["horizon"]:
                raise ValueError(f"source campaign target {key} has a mismatched scenario horizon")
        elif int(effective_horizon) != target["horizon"]:
            raise ValueError(f"source campaign target {key} has a mismatched planner horizon")

        if planner.algo_config_path is None:
            planner_config: dict[str, Any] = {}
            planner_config_path = None
        else:
            loaded = yaml.safe_load(planner.algo_config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"planner config for {key[1]!r} must be an object")
            planner_config = loaded
            try:
                planner_config_path = (
                    planner.algo_config_path.resolve()
                    .relative_to(get_repository_root().resolve())
                    .as_posix()
                )
            except ValueError as exc:
                raise ValueError(f"planner config for {key[1]!r} must be repository-local") from exc
        effective_dt = planner.dt_override if planner.dt_override is not None else cfg.dt
        identity_payload = _historical_identity_payload(
            scenario_params,
            target=target,
            algo=str(planner.algo),
            algo_config=planner_config,
            record_forces=bool(cfg.record_forces),
            record_simulation_step_trace=bool(cfg.record_simulation_step_trace),
            dt=float(effective_dt) if effective_dt is not None else None,
            horizon=int(effective_horizon) if effective_horizon is not None else None,
        )
        computed_hash = _config_hash(identity_payload)
        if computed_hash != target["source_config_hash"]:
            raise ValueError(
                f"source campaign target {key} config hash mismatch: "
                f"expected {target['source_config_hash']}, computed {computed_hash}"
            )
        scenario_definition_id = f"{key[0]}--{key[2]}"
        scenario_definitions[scenario_definition_id] = scenario_params
        planner_definition_id = str(planner.key)
        planner_definitions[planner_definition_id] = {
            "algo": str(planner.algo),
            "planner_config": planner_config,
            "planner_config_path": planner_config_path,
            "planner_config_hash": _config_hash(planner_config),
        }
        resolved_targets.append(
            {
                **dict(target),
                "scenario_definition_id": scenario_definition_id,
                "planner_definition_id": planner_definition_id,
                "computed_config_hash": computed_hash,
            }
        )

    bundle = {
        "schema_version": RESOLVED_DEFINITIONS_SCHEMA_VERSION,
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "claim_boundary": (
            "diagnostic-only definition recovery; no repeat campaign result or determinism "
            "verdict is represented"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "source": {
            "campaign_config_path": config_repo_path,
            "campaign_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_git_hashes": source_commits,
            "historical_identity_contract": f"map_runner_identity@{SOURCE_IDENTITY_REVISION[:10]}",
        },
        "execution_contract": dict(manifest["execution_contract"]),
        "scenario_definitions": scenario_definitions,
        "planner_definitions": planner_definitions,
        "targets": resolved_targets,
        "summary": {
            "n_targets": len(resolved_targets),
            "n_cells": len({key[:2] for key in targets}),
            "all_source_config_hashes_match": True,
            "runnable_definitions_remaining": [],
        },
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    return bundle


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


def verify_host_report(  # noqa: C901, PLR0912, PLR0915 - each rejected report state needs a specific error.
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
    _require_sha256(environment.get("lockfile_sha256"), "host environment lockfile_sha256")
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
        disposition = result.get("disposition")
        if disposition is not None:
            if disposition != UNRUNNABLE_DISPOSITION:
                raise ValueError(f"target {key} has an unsupported disposition {disposition!r}")
            repeats = result.get("repeats")
            if "repeats" in result and (not isinstance(repeats, list) or repeats):
                raise ValueError(
                    f"target {key} has disposition {disposition!r} but also reports repeats"
                )
            disposition_reason = _require_text(
                result.get("disposition_reason"), f"target {key} disposition_reason"
            )
            verified = {
                "scenario_id": key[0],
                "planner": key[1],
                "seed": key[2],
                "unrunnable": True,
                "disposition": disposition,
                "disposition_reason": disposition_reason,
                "bitwise_identical": None,
                "first_divergence": None,
                "repeat_fingerprints": [],
            }
            verified_targets.append(verified)
            by_cell[key[:2]].append(verified)
            continue
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
            "unrunnable": False,
            "bitwise_identical": divergence is None,
            "first_divergence": divergence,
            "repeat_fingerprints": [_repeat_fingerprint(item) for item in repeat_maps],
        }
        verified_targets.append(verified)
        by_cell[key[:2]].append(verified)

    cells = []
    for (scenario_id, planner), target_results in sorted(by_cell.items()):
        runnable = [item for item in target_results if not item.get("unrunnable")]
        n_targets = len(target_results)
        n_runnable_targets = len(runnable)
        n_unrunnable_targets = n_targets - n_runnable_targets
        if not runnable:
            cells.append(
                {
                    "scenario_id": scenario_id,
                    "planner": planner,
                    "unrunnable": True,
                    "disposition": UNRUNNABLE_DISPOSITION,
                    "exact_repeat_determinism": None,
                    "first_divergence": None,
                    "n_targets": n_targets,
                    "n_runnable_targets": 0,
                    "n_unrunnable_targets": n_unrunnable_targets,
                }
            )
            continue
        first = next(
            (item["first_divergence"] for item in runnable if item["first_divergence"]), None
        )
        if n_unrunnable_targets:
            cells.append(
                {
                    "scenario_id": scenario_id,
                    "planner": planner,
                    "unrunnable": False,
                    "mixed": True,
                    "disposition": MIXED_DISPOSITION,
                    "exact_repeat_determinism": None,
                    "first_divergence": first,
                    "n_targets": n_targets,
                    "n_runnable_targets": n_runnable_targets,
                    "n_unrunnable_targets": n_unrunnable_targets,
                }
            )
            continue
        cells.append(
            {
                "scenario_id": scenario_id,
                "planner": planner,
                "unrunnable": False,
                "mixed": False,
                "exact_repeat_determinism": first is None,
                "first_divergence": first,
                "n_targets": n_targets,
                "n_runnable_targets": n_runnable_targets,
                "n_unrunnable_targets": 0,
            }
        )
    runnable_cells = [
        cell for cell in cells if not cell.get("unrunnable") and not cell.get("mixed")
    ]
    unrunnable_cells = [cell for cell in cells if cell.get("unrunnable")]
    mixed_cells = [cell for cell in cells if cell.get("mixed")]
    verified = {
        "schema_version": VERIFIED_HOST_REPORT_SCHEMA_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "environment": dict(environment),
        "targets": verified_targets,
        "cells": cells,
        "summary": {
            "n_targets": len(verified_targets),
            "n_runnable_targets": sum(1 for t in verified_targets if not t.get("unrunnable")),
            "n_unrunnable_targets": sum(1 for t in verified_targets if t.get("unrunnable")),
            "n_cells": len(cells),
            "n_runnable_cells": len(runnable_cells),
            "n_unrunnable_cells": len(unrunnable_cells),
            "n_mixed_cells": len(mixed_cells),
            # The bitwise-identical claim scopes to runnable cells; unrunnable
            # or mixed cells make no determinism claim.
            "all_cells_bitwise_identical": bool(runnable_cells)
            and all(cell["exact_repeat_determinism"] for cell in runnable_cells),
        },
    }
    return verified


def compare_verified_hosts(  # noqa: C901 - each rejected cross-host state needs a specific branch.
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
    target_counts: dict[tuple[str, str], int] = defaultdict(int)
    unrunnable_target_counts: dict[tuple[str, str], int] = defaultdict(int)
    for key in sorted(targets):
        cell_key = key[:2]
        target_counts[cell_key] += 1
        if first_targets[key].get("unrunnable") or second_targets[key].get("unrunnable"):
            unrunnable_target_counts[cell_key] += 1
            continue
        identical = first_targets[key].get("repeat_fingerprints") == second_targets[key].get(
            "repeat_fingerprints"
        )
        cells[cell_key].append(bool(identical))
    matrix = []
    for cell_key in sorted(set(cells) | set(unrunnable_target_counts)):
        scenario_id, planner = cell_key
        n_unrunnable_targets = unrunnable_target_counts.get(cell_key, 0)
        if n_unrunnable_targets:
            comparison_status = (
                "unrunnable" if n_unrunnable_targets == target_counts[cell_key] else "mixed"
            )
            matrix.append(
                {
                    "scenario_id": scenario_id,
                    "planner": planner,
                    "bitwise_identical": None,
                    "comparison_status": comparison_status,
                    "disposition": (
                        UNRUNNABLE_DISPOSITION
                        if comparison_status == "unrunnable"
                        else MIXED_DISPOSITION
                    ),
                    "n_targets": target_counts[cell_key],
                    "n_runnable_targets": target_counts[cell_key] - n_unrunnable_targets,
                    "n_unrunnable_targets": n_unrunnable_targets,
                }
            )
            continue
        target_matches = cells[cell_key]
        identical = version_match and all(target_matches)
        matrix.append(
            {
                "scenario_id": scenario_id,
                "planner": planner,
                "bitwise_identical": identical,
                "comparison_status": "identical" if identical else "divergent",
            }
        )
    runnable_rows = [
        row for row in matrix if row["comparison_status"] in {"identical", "divergent"}
    ]
    unrunnable_rows = [row for row in matrix if row["comparison_status"] == "unrunnable"]
    mixed_rows = [row for row in matrix if row["comparison_status"] == "mixed"]
    return {
        "schema_version": CROSS_HOST_SCHEMA_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "hosts": [first_machine, second_machine],
        "pinned_runtime_versions_match": version_match,
        "matrix": matrix,
        "summary": {
            "n_cells": len(matrix),
            "n_runnable_cells": len(runnable_rows),
            "n_unrunnable_cells": len(unrunnable_rows),
            "n_mixed_cells": len(mixed_rows),
            "all_cells_bitwise_identical": version_match
            and bool(runnable_rows)
            and all(row["bitwise_identical"] for row in runnable_rows),
        },
    }


def _safe_json_value(value: Any) -> Any:
    """Convert a value to be safe for canonical SHA-256 hashing.

    Handles numpy scalar types, non-finite floats (NaN/Inf), and nested
    mappings/sequences.  NaN values become ``null`` to keep ``allow_nan=False``
    in the canonical serializer.

    Returns:
        A JSON-compatible representation of ``value``.
    """
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("trajectory-hash mappings must use string keys")
        return {key: _safe_json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    raise ValueError(f"trajectory-hash value is not JSON-compatible: {type(value).__name__}")


def _compute_trajectory_hash(record: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of the episode trajectory output.

    Hashes the outcome payload and the metrics dict after sanitizing non-finite
    floats and numpy types.  Two identical episodes produce the same hash; any
    trajectory-level difference is visible in at least one metric.

    Returns:
        The canonical SHA-256 digest of the episode outcome and metrics.
    """
    outcome = record.get("outcome")
    metrics = record.get("metrics")
    if not isinstance(outcome, Mapping):
        raise ValueError("episode record has no outcome mapping for trajectory hash")
    if not isinstance(metrics, Mapping):
        raise ValueError("episode record has no metrics mapping for trajectory hash")
    hash_payload = _safe_json_value({"outcome": outcome, "metrics": metrics})
    return canonical_sha256(hash_payload)


def _get_environment_fingerprint() -> dict[str, Any]:
    """Capture the host environment fingerprint required by the result schema.

    Returns:
        The CPU-only runtime identity recorded in the host report.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        git_commit = result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        git_commit = "unknown"

    return {
        "machine_id": platform.node(),
        "cpu_only": True,
        "workers": 1,
        "numpy_version": np.__version__,
        "numba_version": str(numba.__version__),
        "python_version": platform.python_version(),
        "git_commit": git_commit,
        "lockfile_sha256": hashlib.sha256(
            (get_repository_root() / "uv.lock").read_bytes()
        ).hexdigest(),
    }


def _cached_result_if_compatible(
    cache_file: Path, env_fingerprint: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a valid cache entry only when it matches the current runtime.

    Returns:
        The cached target result, or ``None`` when the cache is absent, malformed,
        or belongs to another NumPy/Numba runtime.
    """
    if not cache_file.exists():
        return None
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(cached, Mapping):
        return None
    result = cached.get("_result")
    if (
        cached.get("_env_numpy_version") != env_fingerprint["numpy_version"]
        or cached.get("_env_numba_version") != env_fingerprint["numba_version"]
        or not isinstance(result, Mapping)
    ):
        return None
    return dict(result)


def _finite_int_or_zero(value: Any) -> int:
    """Convert a finite numeric metric to an integer without losing NumPy scalars.

    Returns:
        The converted metric, or zero for absent, malformed, or non-finite values.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    return int(numeric) if math.isfinite(numeric) else 0


def execute_campaign(  # noqa: C901, PLR0912, PLR0915 - fail-closed execution tracks each target state explicitly.
    resolved_bundle: Mapping[str, Any],
    *,
    output_dir: Path,
    run_episode: Any | None = None,
    target_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the exact-repeat campaign and return a host_result.json payload.

    Consumes a resolved definitions bundle (from ``resolve_runnable_definitions``),
    runs each target the declared number of times (3), and emits a host result
    conforming to ``scenario_exact_repeat_host_result.v1``.  Supports resume: if
    a cached result file exists for a target, that target is skipped unless its
    environment fingerprint has changed.

    Args:
        resolved_bundle: Output of ``resolve_runnable_definitions()`` containing
            runnable scenario/planner definitions and manifest metadata.
        output_dir: Directory for per-target result cache and the final
            ``host_result.json``.
        run_episode: Optional injected episode runner for testability.  Defaults
            to ``robot_sf.benchmark.runner.run_episode``.
        target_filter: Optional list of ``"scenario_id--seed"`` strings to execute
            a subset of targets.

    Returns:
        A host report dict conforming to the ``scenario_exact_repeat_host_result.v1``
        schema, also written to ``output_dir/host_result.json``.
    """
    if resolved_bundle.get("schema_version") != RESOLVED_DEFINITIONS_SCHEMA_VERSION:
        raise ValueError("bundle has an unsupported schema_version")
    expected_manifest_hash = resolved_bundle.get("manifest_sha256")
    if not expected_manifest_hash:
        raise ValueError("bundle is missing manifest_sha256")
    bundle_bundle_sha = resolved_bundle.get("bundle_sha256")
    without_sha = {key: value for key, value in resolved_bundle.items() if key != "bundle_sha256"}
    if bundle_bundle_sha != canonical_sha256(without_sha):
        raise ValueError("bundle_sha256 does not match the bundle content")

    targets = resolved_bundle.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("bundle contains no targets to execute")
    scenario_defs = _require_mapping(
        resolved_bundle.get("scenario_definitions"), "bundle scenario_definitions"
    )
    planner_defs = _require_mapping(
        resolved_bundle.get("planner_definitions"), "bundle planner_definitions"
    )
    _, repeats_per_target = _check_manifest_from_bundle(resolved_bundle)

    # The runnability predicate reflects current main's ``run_episode`` registry
    # and is applied even when a runner is injected, so the disposition records a
    # property of the codebase rather than of the test harness. Imported from the
    # lightweight baselines package to avoid pulling heavy optional deps.
    from robot_sf.baselines import is_runnable_algo  # noqa: PLC0415

    if run_episode is None:
        from robot_sf.benchmark.runner import run_episode as _run_episode  # noqa: PLC0415

        run_episode = _run_episode

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "target_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env_fingerprint = _get_environment_fingerprint()

    # Use the bundle's source revision so the host report passes verifier.
    source = _require_mapping(resolved_bundle.get("source"), "bundle source")
    source_git_hashes = source.get("source_git_hashes", [])
    if source_git_hashes:
        env_fingerprint["git_commit"] = source_git_hashes[0]

    selected_definition_ids: set[str] | None = None
    if target_filter is not None:
        filter_set = set(target_filter)
        selected_definition_ids = {
            _require_text(target.get("scenario_definition_id"), "target scenario_definition_id")
            for target in targets
            if target.get("scenario_definition_id") in filter_set
        }
        if not selected_definition_ids:
            raise ValueError("target_filter removed all targets from the bundle")

    results: list[dict[str, Any]] = []

    for target in targets:
        key = _target_key(target)
        scenario_id = key[0]
        planner = key[1]
        seed = key[2]
        scenario_def_id = _require_text(
            target.get("scenario_definition_id"), "target scenario_definition_id"
        )
        planner_def_id = _require_text(
            target.get("planner_definition_id"), "target planner_definition_id"
        )
        horizon = int(target["horizon"])

        cache_key = f"{scenario_id}_{planner}_{seed}"
        cache_file = cache_dir / f"{cache_key}.json"

        cached_result = _cached_result_if_compatible(cache_file, env_fingerprint)

        if cached_result is not None:
            results.append(cached_result)
            continue

        if selected_definition_ids is not None and scenario_def_id not in selected_definition_ids:
            raise ValueError(
                "target_filter requires compatible cached results for omitted targets; "
                f"missing {scenario_id!r}/{planner!r}/{seed}"
            )

        raw_scenario_params = scenario_defs.get(scenario_def_id)
        if raw_scenario_params is None:
            raise ValueError(f"bundle missing scenario_definition for {scenario_def_id!r}")
        scenario_params = dict(
            _require_mapping(raw_scenario_params, f"scenario_definition {scenario_def_id!r}")
        )
        raw_planner_def = planner_defs.get(planner_def_id)
        if raw_planner_def is None:
            raise ValueError(f"bundle missing planner_definition for {planner_def_id!r}")
        planner_def = dict(
            _require_mapping(raw_planner_def, f"planner_definition {planner_def_id!r}")
        )
        planner_algo = _require_text(
            planner_def.get("algo"), f"planner_definition {planner_def_id!r} algo"
        )
        if planner != planner_algo:
            raise ValueError(
                f"target {key} planner {planner!r} does not match "
                f"planner_definition {planner_def_id!r} algo {planner_algo!r}"
            )

        # Fail closed on planners the run_episode path cannot construct,
        # recording an explicit disposition instead of crashing the whole campaign.
        if not is_runnable_algo(planner_algo):
            result_entry = {
                "scenario_id": scenario_id,
                "planner": planner,
                "seed": seed,
                "horizon": horizon,
                "source_config_hash": target["source_config_hash"],
                "repeats": [],
                "disposition": UNRUNNABLE_DISPOSITION,
                "disposition_reason": (
                    f"planner {planner_algo!r} has no executor in the exact-repeat "
                    "run_episode baseline registry on current main"
                ),
            }
            results.append(result_entry)
            cache_data = {
                "_env_numpy_version": env_fingerprint["numpy_version"],
                "_env_numba_version": env_fingerprint["numba_version"],
                "_result": result_entry,
            }
            cache_file.write_text(
                json.dumps(cache_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            continue

        # Determine DT from resolved bundle execution contract or scenario config
        sim_config = scenario_params.get("simulation_config", {})
        if isinstance(sim_config, dict):
            dt = sim_config.get("dt")
        else:
            dt = None
        if dt is None:
            from robot_sf.benchmark.camera_ready._config import (  # noqa: PLC0415
                load_campaign_config,
            )

            # Fallback: try to load from campaign config referenced by source
            source_cfg_path = resolved_bundle.get("source", {}).get("campaign_config_path")
            if source_cfg_path:
                cfg = load_campaign_config((get_repository_root() / source_cfg_path).resolve())
                dt = cfg.dt
            if dt is None:
                dt = 0.1

        # Build algo config path
        algo_config_path = planner_def.get("planner_config_path")
        if algo_config_path is not None:
            algo_config_path = str((get_repository_root() / algo_config_path).resolve())

        repeats: list[dict[str, Any]] = []
        for repeat_idx in range(repeats_per_target):
            record = run_episode(
                dict(scenario_params),
                seed=seed,
                algo=planner_algo,
                algo_config_path=algo_config_path,
                horizon=horizon,
                dt=float(dt),
                record_forces=scenario_params.get("record_forces", False),
            )
            trajectory_sha = _compute_trajectory_hash(record)
            outcome = record.get("outcome", {})
            outcome_binary = 1 if outcome.get("success", False) else 0
            metrics = record.get("metrics", {})
            near_misses = _finite_int_or_zero(metrics.get("near_misses"))
            repeats.append(
                {
                    "outcome": outcome_binary,
                    "trajectory_sha256": trajectory_sha,
                    "near_misses": near_misses,
                }
            )

        divergence = _first_divergence(repeats)
        result_entry = {
            "scenario_id": scenario_id,
            "planner": planner,
            "seed": seed,
            "horizon": horizon,
            "source_config_hash": target["source_config_hash"],
            "repeats": repeats,
        }
        if divergence is not None:
            result_entry["first_divergence"] = divergence
        results.append(result_entry)

        # Cache the result for resume
        cache_data = {
            "_env_numpy_version": env_fingerprint["numpy_version"],
            "_env_numba_version": env_fingerprint["numba_version"],
            "_result": result_entry,
        }
        cache_file.write_text(
            json.dumps(cache_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # Build host result
    host_result = {
        "schema_version": HOST_REPORT_SCHEMA_VERSION,
        "manifest_sha256": expected_manifest_hash,
        "environment": env_fingerprint,
        "results": results,
    }
    result_path = output_dir / "host_result.json"
    result_path.write_text(
        json.dumps(host_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return host_result


def _check_manifest_from_bundle(
    resolved_bundle: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, int], Mapping[str, Any]], int]:
    """Validate targets from a resolved definitions bundle and return indexed targets + repeat count.

    Returns:
        Indexed target dict keyed by ``(scenario_id, planner, seed)`` and the
        repeats-per-target count from the execution contract.
    """
    targets_list = resolved_bundle.get("targets")
    if not isinstance(targets_list, list) or not targets_list:
        raise ValueError("bundle targets must be a non-empty list")
    contract = _require_mapping(resolved_bundle.get("execution_contract"), "execution_contract")
    repeats = contract.get("repeats_per_target", DEFAULT_REPEATS)
    if repeats != DEFAULT_REPEATS:
        raise ValueError(f"bundle must require exactly {DEFAULT_REPEATS} repeats, got {repeats}")

    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for target in targets_list:
        target = _require_mapping(target, "bundle target")
        key = _target_key(target)
        if key in indexed:
            raise ValueError(f"bundle contains duplicate target {key}")
        _require_text(target.get("source_config_hash"), "target source_config_hash")
        _require_text(target.get("scenario_definition_id"), "target scenario_definition_id")
        _require_text(target.get("planner_definition_id"), "target planner_definition_id")
        horizon = target.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("target horizon must be a positive integer")
        indexed[key] = target
    return indexed, repeats

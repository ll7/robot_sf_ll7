"""Helpers for benchmark freeze manifests and runtime contract validation.

The freeze manifest captures reproducibility-critical benchmark settings used for
publication-grade runs. This module supports:
1. Loading a freeze manifest from YAML/JSON.
2. Building a runtime freeze contract from benchmark config/runtime metadata.
3. Comparing freeze vs runtime contracts and returning structured mismatches.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_METRIC_SUBSET = (
    "success_rate",
    "collision_rate",
    "snqi",
    "time_to_goal",
    "path_efficiency",
)


def _canonical_str_list(value: Any) -> list[str]:
    """Normalize string lists with dedupe + deterministic ordering.

    Returns:
        Sorted unique string list.
    """

    return sorted(set(_as_list_of_str(value)))


def _as_list_of_str(value: Any) -> list[str]:
    """Normalize values into a list of non-empty strings.

    Returns:
        List of normalized string values.
    """

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list | tuple | set):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _safe_int(value: Any) -> int | None:
    """Best-effort integer conversion.

    Returns:
        Integer value when conversion succeeds, else None.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """Best-effort float conversion.

    Returns:
        Float value when conversion succeeds, else None.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonical_planner_configs(value: Any) -> list[dict[str, Any]]:
    """Normalize and deterministically sort planner config entries.

    Returns:
        Canonicalized planner configuration list.
    """

    if isinstance(value, dict):
        items: list[dict[str, Any]] = [dict(value)]
    elif isinstance(value, list):
        items = [dict(entry) for entry in value if isinstance(entry, dict)]
    else:
        items = []
    if not items:
        return [{"planner_backend": "default", "planner_classic_config": None}]
    seen: set[str] = set()
    canonical: list[dict[str, Any]] = []
    for entry in items:
        payload = {
            "planner_backend": entry.get("planner_backend", "default"),
            "planner_classic_config": entry.get("planner_classic_config"),
        }
        key = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        if key in seen:
            continue
        seen.add(key)
        canonical.append(payload)
    canonical.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return canonical


def derive_planner_configs(raw_scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract planner settings from scenario definitions.

    Returns:
        Canonicalized planner configuration list derived from scenarios.
    """

    planner_entries: list[dict[str, Any]] = []
    for scenario in raw_scenarios:
        sim_cfg = scenario.get("simulation_config")
        if not isinstance(sim_cfg, dict):
            sim_cfg = {}
        planner_backend = sim_cfg.get("planner_backend", scenario.get("planner_backend"))
        planner_classic = sim_cfg.get(
            "planner_classic_config",
            scenario.get("planner_classic_config"),
        )
        if planner_backend is None and planner_classic is None:
            continue
        planner_entries.append(
            {
                "planner_backend": planner_backend if planner_backend is not None else "default",
                "planner_classic_config": planner_classic,
            }
        )
    return _canonical_planner_configs(planner_entries)


def build_runtime_freeze_contract(
    cfg,
    *,
    scenario_matrix_hash: str,
    git_commit: str,
    raw_scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build runtime freeze contract payload from benchmark config and runtime state.

    Returns:
        Canonical runtime freeze contract dictionary.
    """

    matrix_raw = getattr(cfg, "scenario_matrix_path", None)
    matrix_path = "unknown"
    if matrix_raw is not None:
        try:
            matrix_path = str(Path(str(matrix_raw)).resolve())
        except (OSError, RuntimeError, ValueError, TypeError):
            matrix_path = str(matrix_raw)

    algorithms = _canonical_str_list(getattr(cfg, "baseline_set", None))
    if not algorithms:
        algorithms = _canonical_str_list(getattr(cfg, "algo", None))
    if not algorithms:
        algorithms = ["unknown"]

    metrics_subset = _canonical_str_list(
        getattr(cfg, "metrics_subset", None) or getattr(cfg, "metric_subset", None)
    )
    if not metrics_subset:
        metrics_subset = sorted(DEFAULT_METRIC_SUBSET)

    bootstrap_seed = getattr(cfg, "bootstrap_seed", None)
    if bootstrap_seed is None:
        bootstrap_seed = getattr(cfg, "master_seed", None)

    base_seed = getattr(cfg, "base_seed", None)
    if base_seed is None:
        base_seed = getattr(cfg, "master_seed", None)

    repeats = getattr(cfg, "repeats", None)
    if repeats is None:
        repeats = getattr(cfg, "initial_episodes", None)

    planner_configs_override = getattr(cfg, "freeze_planner_configs", None)
    planner_configs = _canonical_planner_configs(planner_configs_override)
    if planner_configs_override is None:
        planner_configs = derive_planner_configs(raw_scenarios)

    return {
        "scenario": {
            "matrix_path": matrix_path,
            "matrix_hash": str(scenario_matrix_hash),
        },
        "baselines": {
            "algorithms": algorithms,
            "planner_configs": planner_configs,
        },
        "seed_plan": {
            "base_seed": _safe_int(base_seed),
            "repeats": _safe_int(repeats),
        },
        "metrics": {
            "subset": metrics_subset,
        },
        "bootstrap": {
            "samples": _safe_int(getattr(cfg, "bootstrap_samples", 1000)),
            "confidence": _safe_float(getattr(cfg, "bootstrap_confidence", 0.95)),
            "seed": _safe_int(bootstrap_seed),
        },
        "software": {
            "identifiers": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.platform(),
                "robot_sf_commit": str(git_commit),
            },
        },
    }


def normalize_freeze_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize arbitrary freeze payloads to canonical contract shape.

    Returns:
        Canonicalized freeze contract dictionary.
    """

    scenario = payload.get("scenario")
    if not isinstance(scenario, dict):
        scenario = {}
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict):
        baselines = {}
    seed_plan = payload.get("seed_plan")
    if not isinstance(seed_plan, dict):
        seed_plan = {}
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, dict):
        bootstrap = {}
    software = payload.get("software")
    if not isinstance(software, dict):
        software = {}
    software_identifiers = software.get("identifiers")
    if not isinstance(software_identifiers, dict):
        software_identifiers = software

    matrix_path = scenario.get("matrix_path", payload.get("matrix_path", "unknown"))
    matrix_hash = scenario.get(
        "matrix_hash",
        payload.get("matrix_hash", payload.get("scenario_matrix_hash", "unknown")),
    )
    algorithms = baselines.get("algorithms", payload.get("baseline_set"))
    planner_configs = baselines.get("planner_configs", baselines.get("planner"))
    if planner_configs is None:
        planner_configs = payload.get("planner_configs")

    metric_subset = metrics.get("subset", payload.get("metrics_subset"))
    if metric_subset is None:
        metric_subset = payload.get("metric_subset")

    return {
        "scenario": {
            "matrix_path": str(matrix_path),
            "matrix_hash": str(matrix_hash),
        },
        "baselines": {
            "algorithms": _canonical_str_list(algorithms),
            "planner_configs": _canonical_planner_configs(planner_configs),
        },
        "seed_plan": {
            "base_seed": _safe_int(seed_plan.get("base_seed", payload.get("base_seed"))),
            "repeats": _safe_int(seed_plan.get("repeats", payload.get("repeats"))),
        },
        "metrics": {
            "subset": _canonical_str_list(metric_subset) or sorted(DEFAULT_METRIC_SUBSET),
        },
        "bootstrap": {
            "samples": _safe_int(bootstrap.get("samples", payload.get("bootstrap_samples", 1000))),
            "confidence": _safe_float(
                bootstrap.get("confidence", payload.get("bootstrap_confidence", 0.95))
            ),
            "seed": _safe_int(bootstrap.get("seed", payload.get("bootstrap_seed"))),
        },
        "software": {
            "identifiers": {
                str(k): str(v)
                for k, v in dict(software_identifiers).items()
                if str(k).strip() and v is not None
            },
        },
    }


def load_freeze_manifest(path: str | Path) -> dict[str, Any]:
    """Load and normalize freeze manifest from YAML or JSON.

    Returns:
        Canonicalized freeze manifest dictionary.
    """

    freeze_path = Path(path)
    if not freeze_path.exists():
        raise FileNotFoundError(f"Freeze manifest not found: {freeze_path}")
    text = freeze_path.read_text(encoding="utf-8")
    suffix = freeze_path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("Freeze manifest must deserialize to a mapping")
    contract = normalize_freeze_contract(dict(raw))
    matrix_path = str(contract["scenario"]["matrix_path"])
    if matrix_path and matrix_path != "unknown":
        resolved = Path(matrix_path)
        if not resolved.is_absolute():
            resolved = (freeze_path.parent / resolved).resolve()
        else:
            resolved = resolved.resolve()
        contract["scenario"]["matrix_path"] = str(resolved)
    return contract


def compare_freeze_contracts(
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare freeze and runtime contracts and return structured mismatches.

    Returns:
        List of mismatch records.
    """

    mismatches: list[dict[str, Any]] = []

    paths = (
        ("scenario", "matrix_path"),
        ("scenario", "matrix_hash"),
        ("baselines", "algorithms"),
        ("baselines", "planner_configs"),
        ("seed_plan", "base_seed"),
        ("seed_plan", "repeats"),
        ("metrics", "subset"),
        ("bootstrap", "samples"),
        ("bootstrap", "confidence"),
        ("bootstrap", "seed"),
    )
    for left, right in paths:
        expected_val = expected[left][right]
        observed_val = observed[left][right]
        if expected_val != observed_val:
            mismatches.append(
                {
                    "path": f"{left}.{right}",
                    "expected": expected_val,
                    "observed": observed_val,
                }
            )

    expected_software = expected["software"]["identifiers"]
    observed_software = observed["software"]["identifiers"]
    for key, expected_val in sorted(expected_software.items()):
        observed_val = observed_software.get(key)
        if observed_val != expected_val:
            mismatches.append(
                {
                    "path": f"software.identifiers.{key}",
                    "expected": expected_val,
                    "observed": observed_val,
                }
            )
    return mismatches


def evaluate_freeze_manifest(
    freeze_manifest_path: str | Path,
    cfg,
    *,
    scenario_matrix_hash: str,
    git_commit: str,
    raw_scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate runtime benchmark contract against a configured freeze manifest.

    Returns:
        Structured validation report with status and mismatch details.
    """

    runtime_contract = build_runtime_freeze_contract(
        cfg,
        scenario_matrix_hash=scenario_matrix_hash,
        git_commit=git_commit,
        raw_scenarios=raw_scenarios,
    )
    try:
        freeze_contract = load_freeze_manifest(freeze_manifest_path)
    except Exception as exc:
        return {
            "path": str(Path(freeze_manifest_path)),
            "status": "error",
            "error": f"{exc.__class__.__name__}: {exc}",
            "mismatches": [],
            "runtime_contract": runtime_contract,
        }

    mismatches = compare_freeze_contracts(freeze_contract, runtime_contract)
    return {
        "path": str(Path(freeze_manifest_path).resolve()),
        "status": "match" if not mismatches else "mismatch",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "runtime_contract": runtime_contract,
    }


__all__ = [
    "DEFAULT_METRIC_SUBSET",
    "build_runtime_freeze_contract",
    "compare_freeze_contracts",
    "derive_planner_configs",
    "evaluate_freeze_manifest",
    "load_freeze_manifest",
    "normalize_freeze_contract",
]

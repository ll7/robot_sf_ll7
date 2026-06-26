"""Build scenario denominator manifests from canonical benchmark configs.

The manifest is an audit artifact: it expands tracked campaign YAML, resolves the
configured seed policy, and reports the intended scenario-family and planner
episode denominators before any benchmark execution or runtime exclusions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.training.scenario_loader import load_scenarios

SCENARIO_DENOMINATOR_SCHEMA_VERSION = "scenario_denominator_manifest.v1"

_DEFAULT_SEED_SETS_PATH = Path("configs/benchmarks/seed_sets_v1.yaml")
_UNKNOWN = "unknown"
_TABLE_COLUMNS = (
    "config_name",
    "config_path",
    "family",
    "planner",
    "planner_algo",
    "scenario_count",
    "cell_count",
    "seed_count",
    "denominator_episodes",
    "kinematics_count",
    "denominator_episodes_with_kinematics",
    "densities",
)
_TABLE_INT_COLUMNS = {
    "scenario_count",
    "cell_count",
    "seed_count",
    "denominator_episodes",
    "kinematics_count",
    "denominator_episodes_with_kinematics",
}


class DenominatorManifestError(ValueError):
    """Raised when configs or consumer tables cannot produce a closed denominator audit."""


@dataclass(frozen=True)
class _Planner:
    """Normalized planner row from a benchmark campaign config."""

    key: str
    algo: str
    enabled: bool


def _repo_root() -> Path:
    """Return the repository root for repo-relative path normalization.

    Returns:
        Absolute repository root path.
    """

    return Path(__file__).resolve().parents[2]


def _read_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    """Read a YAML file that must contain a top-level mapping.

    Returns:
        Parsed YAML mapping.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise DenominatorManifestError(f"{label} must contain a YAML mapping: {path}")
    return payload


def _resolve_path(raw: Any, *, config_path: Path, repo_root: Path, field: str) -> Path:
    """Resolve config paths using local-relative first, then repository-relative fallback.

    Returns:
        Resolved filesystem path.
    """

    if raw is None:
        raise DenominatorManifestError(
            f"Campaign config missing required field '{field}': {config_path}"
        )
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    local_candidate = (config_path.parent / candidate).resolve()
    if local_candidate.exists():
        return local_candidate

    repo_candidate = (repo_root / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    raise FileNotFoundError(
        f"Could not resolve '{field}' path '{raw}' relative to {config_path.parent} or {repo_root}"
    )


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    """Return a stable repository-relative path when possible.

    Returns:
        Repository-relative path when possible, otherwise absolute path.
    """

    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256_file(path: Path) -> str:
    """Return SHA256 hash for a source config or matrix file.

    Returns:
        Hex-encoded SHA256 digest.
    """

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_string_list(raw: Any, *, field: str) -> list[str]:
    """Normalize scalar or list config values into non-empty stripped strings.

    Returns:
        Normalized string values.
    """

    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        values = [str(raw)]
    elif isinstance(raw, list):
        values = [str(value) for value in raw if isinstance(value, (str, int, float))]
        if len(values) != len(raw):
            raise DenominatorManifestError(f"{field} entries must be scalar strings or numbers")
    else:
        raise DenominatorManifestError(f"{field} must be a scalar or list")

    normalized = [value.strip() for value in values if value.strip()]
    if len(normalized) != len(values):
        raise DenominatorManifestError(f"{field} must not contain empty values")
    return normalized


def _normalize_seed_list(raw: Any, *, field: str) -> list[int]:
    """Normalize a seed list and fail closed when it is absent or empty.

    Returns:
        Integer seed list.
    """

    if not isinstance(raw, list) or not raw:
        raise DenominatorManifestError(f"{field} must be a non-empty list of integer seeds")
    seeds: list[int] = []
    for value in raw:
        try:
            seeds.append(int(value))
        except (TypeError, ValueError) as exc:
            raise DenominatorManifestError(
                f"{field} contains a non-integer seed: {value!r}"
            ) from exc
    return seeds


def _load_seed_sets(path: Path) -> dict[str, list[int]]:
    """Load named seed sets from the canonical seed-set YAML file.

    Returns:
        Mapping from seed-set name to integer seeds.
    """

    payload = _read_yaml_mapping(path, label="seed sets")
    seed_sets: dict[str, list[int]] = {}
    for name, raw_seeds in payload.items():
        seed_sets[str(name)] = _normalize_seed_list(raw_seeds, field=f"seed set {name!r}")
    return seed_sets


def _resolve_seed_override(
    payload: dict[str, Any], *, config_path: Path, repo_root: Path
) -> tuple[list[int] | None, dict[str, Any]]:
    """Resolve the campaign seed policy and return both override and provenance payload.

    Returns:
        Tuple of optional seed override and manifest seed-policy payload.
    """

    raw_policy = payload.get("seed_policy") or {}
    if not isinstance(raw_policy, dict):
        raise DenominatorManifestError(f"seed_policy must be a mapping: {config_path}")

    mode = str(raw_policy.get("mode", "scenario-default")).strip().lower()
    policy_payload: dict[str, Any] = {
        "mode": mode,
        "seed_set": raw_policy.get("seed_set"),
        "seeds": [],
        "resolved_seeds": None,
        "seed_sets_path": None,
    }

    if mode == "scenario-default":
        return None, policy_payload

    if mode == "fixed-list":
        seeds = _normalize_seed_list(raw_policy.get("seeds"), field="seed_policy.seeds")
        policy_payload["seeds"] = seeds
        policy_payload["resolved_seeds"] = seeds
        return seeds, policy_payload

    if mode == "seed-set":
        seed_set = str(raw_policy.get("seed_set") or "").strip()
        if not seed_set:
            raise DenominatorManifestError(f"seed_policy.seed_set is required: {config_path}")
        seed_sets_path = _resolve_path(
            raw_policy.get("seed_sets_path", _DEFAULT_SEED_SETS_PATH),
            config_path=config_path,
            repo_root=repo_root,
            field="seed_policy.seed_sets_path",
        )
        seed_sets = _load_seed_sets(seed_sets_path)
        if seed_set not in seed_sets:
            known = ", ".join(sorted(seed_sets))
            raise DenominatorManifestError(f"Unknown seed set {seed_set!r}. Available: {known}")
        seeds = list(seed_sets[seed_set])
        policy_payload["seed_set"] = seed_set
        policy_payload["resolved_seeds"] = seeds
        policy_payload["seed_sets_path"] = _repo_relative(seed_sets_path, repo_root=repo_root)
        return seeds, policy_payload

    raise DenominatorManifestError(f"Unsupported seed_policy.mode {mode!r}: {config_path}")


def _normalize_kinematics(raw: Any) -> list[str]:
    """Return normalized campaign kinematics values with camera-ready default.

    Returns:
        Normalized kinematics labels.
    """

    values = _normalize_string_list(raw, field="kinematics_matrix")
    return [value.lower() for value in values] or ["differential_drive"]


def _normalize_planners(raw: Any, *, config_path: Path) -> tuple[list[_Planner], list[_Planner]]:
    """Return enabled and disabled planner rows from campaign config.

    Returns:
        Tuple of enabled planner rows and disabled planner rows.
    """

    if not isinstance(raw, list) or not raw:
        raise DenominatorManifestError(
            f"Campaign config requires non-empty planners list: {config_path}"
        )

    enabled: list[_Planner] = []
    disabled: list[_Planner] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise DenominatorManifestError(
                f"Planner entry #{index} must be a mapping: {config_path}"
            )
        key = str(entry.get("key") or entry.get("algo") or "").strip()
        algo = str(entry.get("algo") or key).strip()
        if not key:
            raise DenominatorManifestError(
                f"Planner entry #{index} missing key/algo: {config_path}"
            )
        if key in seen:
            raise DenominatorManifestError(f"Duplicate planner key {key!r}: {config_path}")
        seen.add(key)
        planner = _Planner(key=key, algo=algo, enabled=bool(entry.get("enabled", True)))
        if planner.enabled:
            enabled.append(planner)
        else:
            disabled.append(planner)

    if not enabled:
        raise DenominatorManifestError(f"Campaign config has no enabled planners: {config_path}")
    return enabled, disabled


def _scenario_id(scenario: dict[str, Any], *, index: int, matrix_path: Path) -> str:
    """Return the stable scenario cell identifier used by benchmark configs.

    Returns:
        Scenario identifier.
    """

    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise DenominatorManifestError(f"Scenario #{index} missing name/scenario_id/id: {matrix_path}")


def _scenario_family(scenario: dict[str, Any]) -> str:
    """Return scenario family from metadata, falling back to explicit top-level fields.

    Returns:
        Scenario family label, or ``unknown``.
    """

    metadata = scenario.get("metadata")
    if isinstance(metadata, dict):
        for key in ("archetype", "family", "scenario_family"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("archetype", "family", "scenario_family"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _UNKNOWN


def _scenario_density(scenario: dict[str, Any]) -> str:
    """Return scenario density from metadata or top-level density-like fields.

    Returns:
        Scenario density label, or ``unknown``.
    """

    metadata = scenario.get("metadata")
    if isinstance(metadata, dict):
        for key in ("density", "pedestrian_density", "density_label"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    for key in ("density", "pedestrian_density", "ped_density", "density_label"):
        value = scenario.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return _UNKNOWN


def _scenario_seeds(
    scenario: dict[str, Any],
    *,
    seed_override: list[int] | None,
    scenario_id: str,
) -> list[int]:
    """Resolve per-scenario seeds after applying campaign seed overrides.

    Returns:
        Integer seeds for the scenario cell.
    """

    if seed_override is not None:
        return list(seed_override)
    raw_seeds = scenario.get("seeds")
    if isinstance(raw_seeds, list) and raw_seeds:
        return _normalize_seed_list(raw_seeds, field=f"scenario {scenario_id!r} seeds")
    raw_seed = scenario.get("seed")
    if raw_seed is not None:
        try:
            return [int(raw_seed)]
        except (TypeError, ValueError) as exc:
            raise DenominatorManifestError(
                f"scenario {scenario_id!r} contains a non-integer seed: {raw_seed!r}"
            ) from exc
    raise DenominatorManifestError(
        f"scenario {scenario_id!r} has no seeds and campaign seed_policy is scenario-default"
    )


def _filter_scenario_candidates(
    scenarios: list[dict[str, Any]],
    *,
    candidate_names: list[str],
    matrix_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Apply campaign scenario candidate selection with missing-name validation.

    Returns:
        Filtered scenario list, or the original list when no candidates are configured.
    """

    if not candidate_names:
        return scenarios
    counts = dict.fromkeys(candidate_names, 0)
    filtered: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        sid = _scenario_id(scenario, index=index, matrix_path=matrix_path)
        if sid in counts:
            counts[sid] += 1
            filtered.append(scenario)
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise DenominatorManifestError(
            "scenario_candidates did not resolve in "
            f"{_repo_relative(matrix_path, repo_root=repo_root)}: {', '.join(missing)}"
        )
    return filtered


def _family_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize scenario cells by family.

    Returns:
        Family denominator rows.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(str(cell["family"]), []).append(cell)

    rows: list[dict[str, Any]] = []
    for family, family_cells in sorted(grouped.items()):
        densities = sorted({str(cell["density"]) for cell in family_cells})
        density_counts = {
            density: sum(1 for cell in family_cells if cell["density"] == density)
            for density in densities
        }
        seeds = sorted({int(seed) for cell in family_cells for seed in cell["seeds"]})
        rows.append(
            {
                "family": family,
                "scenario_count": len(family_cells),
                "cell_count": len(family_cells),
                "density_counts": density_counts,
                "densities": densities,
                "scenario_ids": sorted(str(cell["scenario_id"]) for cell in family_cells),
                "seeds": seeds,
                "seed_count": len(seeds),
                "episode_denominator": sum(int(cell["seed_count"]) for cell in family_cells),
            }
        )
    return rows


def _per_family_planner_denominators(
    family_rows: list[dict[str, Any]],
    *,
    planners: list[_Planner],
    kinematics: list[str],
) -> list[dict[str, Any]]:
    """Cross family episode denominators with enabled planners.

    Returns:
        Per-family x planner denominator rows.
    """

    rows: list[dict[str, Any]] = []
    for family in family_rows:
        for planner in planners:
            denominator = int(family["episode_denominator"])
            rows.append(
                {
                    "family": family["family"],
                    "planner": planner.key,
                    "planner_algo": planner.algo,
                    "scenario_count": int(family["scenario_count"]),
                    "cell_count": int(family["cell_count"]),
                    "seed_count": int(family["seed_count"]),
                    "denominator_episodes": denominator,
                    "kinematics": list(kinematics),
                    "kinematics_count": len(kinematics),
                    "denominator_episodes_with_kinematics": denominator * len(kinematics),
                    "densities": list(family["densities"]),
                }
            )
    return rows


def _build_config_manifest(config_path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Build one config entry for the scenario denominator manifest.

    Returns:
        Manifest entry for one benchmark config.
    """

    config_path = config_path.resolve()
    payload = _read_yaml_mapping(config_path, label="benchmark config")
    name = str(payload.get("name") or config_path.stem).strip()
    scenario_matrix_path = _resolve_path(
        payload.get("scenario_matrix"),
        config_path=config_path,
        repo_root=repo_root,
        field="scenario_matrix",
    )
    candidate_names = _normalize_string_list(
        payload.get("scenario_candidates"), field="scenario_candidates"
    )
    seed_override, seed_policy = _resolve_seed_override(
        payload, config_path=config_path, repo_root=repo_root
    )
    enabled_planners, disabled_planners = _normalize_planners(
        payload.get("planners"), config_path=config_path
    )
    kinematics = _normalize_kinematics(payload.get("kinematics_matrix"))

    loaded_scenarios = load_scenarios(scenario_matrix_path)
    scenarios = [dict(scenario) for scenario in loaded_scenarios]
    scenarios = _filter_scenario_candidates(
        scenarios,
        candidate_names=candidate_names,
        matrix_path=scenario_matrix_path,
        repo_root=repo_root,
    )
    if not scenarios:
        raise DenominatorManifestError(
            f"Scenario matrix resolved no scenarios: {scenario_matrix_path}"
        )

    cells: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        sid = _scenario_id(scenario, index=index, matrix_path=scenario_matrix_path)
        seeds = _scenario_seeds(scenario, seed_override=seed_override, scenario_id=sid)
        family = _scenario_family(scenario)
        density = _scenario_density(scenario)
        cells.append(
            {
                "scenario_id": sid,
                "cell_id": sid,
                "family": family,
                "density": density,
                "seeds": seeds,
                "seed_count": len(seeds),
                "episode_denominator": len(seeds),
            }
        )

    families = _family_rows(cells)
    per_family_planner = _per_family_planner_denominators(
        families,
        planners=enabled_planners,
        kinematics=kinematics,
    )
    all_seeds = sorted({int(seed) for cell in cells for seed in cell["seeds"]})
    densities = sorted({str(cell["density"]) for cell in cells})
    episode_denominator = sum(int(cell["seed_count"]) for cell in cells)

    return {
        "name": name,
        "config_path": _repo_relative(config_path, repo_root=repo_root),
        "config_sha256": _sha256_file(config_path),
        "scenario_matrix": _repo_relative(scenario_matrix_path, repo_root=repo_root),
        "scenario_matrix_sha256": _sha256_file(scenario_matrix_path),
        "scenario_candidates": candidate_names,
        "seed_policy": seed_policy,
        "kinematics_matrix": kinematics,
        "planners": [
            {"key": planner.key, "algo": planner.algo, "enabled": True}
            for planner in enabled_planners
        ],
        "disabled_planners": [
            {"key": planner.key, "algo": planner.algo, "enabled": False}
            for planner in disabled_planners
        ],
        "summary": {
            "scenario_count": len(cells),
            "cell_count": len(cells),
            "family_count": len(families),
            "density_count": len(densities),
            "densities": densities,
            "seed_count": len(all_seeds),
            "seeds": all_seeds,
            "planner_count": len(enabled_planners),
            "kinematics_count": len(kinematics),
            "episode_denominator_without_planner": episode_denominator,
            "planner_episode_denominator": episode_denominator * len(enabled_planners),
            "planner_kinematics_episode_denominator": (
                episode_denominator * len(enabled_planners) * len(kinematics)
            ),
        },
        "families": families,
        "cells": sorted(cells, key=lambda cell: str(cell["scenario_id"])),
        "per_family_planner_denominators": per_family_planner,
    }


def build_scenario_denominator_manifest(
    config_paths: list[Path], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Build a denominator manifest from one or more canonical benchmark configs.

    Returns:
        Complete scenario denominator manifest payload.
    """

    if not config_paths:
        raise DenominatorManifestError("At least one benchmark config path is required")
    root = (repo_root or _repo_root()).resolve()
    configs = [_build_config_manifest(Path(path), repo_root=root) for path in config_paths]
    total_planner_denominator = sum(
        int(config["summary"]["planner_episode_denominator"]) for config in configs
    )
    total_planner_kinematics_denominator = sum(
        int(config["summary"]["planner_kinematics_episode_denominator"]) for config in configs
    )
    return {
        "schema_version": SCENARIO_DENOMINATOR_SCHEMA_VERSION,
        "summary": {
            "config_count": len(configs),
            "planner_episode_denominator": total_planner_denominator,
            "planner_kinematics_episode_denominator": total_planner_kinematics_denominator,
        },
        "configs": configs,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Write a deterministic JSON denominator manifest.

    Returns:
        Path written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML denominator manifest from disk.

    Returns:
        Loaded manifest mapping.
    """

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise DenominatorManifestError(f"Manifest must contain a mapping: {path}")
    return payload


def check_manifest(expected: dict[str, Any], observed_path: Path) -> None:
    """Fail closed when an observed manifest differs from expected generated content."""

    observed = load_manifest(observed_path)
    if observed != expected:
        raise DenominatorManifestError(f"Manifest mismatch: {observed_path}")


def denominator_table_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Return stable table rows for per-family x planner denominator consumers.

    Returns:
        Sorted table rows with string values.
    """

    rows: list[dict[str, str]] = []
    for config in manifest.get("configs", []):
        for row in config.get("per_family_planner_denominators", []):
            rows.append(
                {
                    "config_name": str(config["name"]),
                    "config_path": str(config["config_path"]),
                    "family": str(row["family"]),
                    "planner": str(row["planner"]),
                    "planner_algo": str(row["planner_algo"]),
                    "scenario_count": str(int(row["scenario_count"])),
                    "cell_count": str(int(row["cell_count"])),
                    "seed_count": str(int(row["seed_count"])),
                    "denominator_episodes": str(int(row["denominator_episodes"])),
                    "kinematics_count": str(int(row["kinematics_count"])),
                    "denominator_episodes_with_kinematics": str(
                        int(row["denominator_episodes_with_kinematics"])
                    ),
                    "densities": ";".join(str(value) for value in row["densities"]),
                }
            )
    return sorted(rows, key=lambda item: (item["config_name"], item["family"], item["planner"]))


def write_denominator_table(manifest: dict[str, Any], path: Path) -> Path:
    """Write per-family x planner denominator rows as CSV.

    Returns:
        Path written.
    """

    rows = denominator_table_rows(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _normalize_table_row(row: dict[str, Any]) -> dict[str, str]:
    """Normalize CSV/Markdown/JSON consumer rows for closed comparison.

    Returns:
        Normalized table row with string values.
    """

    missing = [column for column in _TABLE_COLUMNS if column not in row]
    if missing:
        raise DenominatorManifestError(f"Denominator table row missing columns: {missing}")
    normalized: dict[str, str] = {}
    for column in _TABLE_COLUMNS:
        value = row[column]
        if column in _TABLE_INT_COLUMNS:
            try:
                normalized[column] = str(int(value))
            except (TypeError, ValueError) as exc:
                raise DenominatorManifestError(
                    f"Denominator table column {column!r} must be integer, got {value!r}"
                ) from exc
        elif column == "densities":
            parts = [
                part.strip()
                for chunk in str(value).split(";")
                for part in chunk.split(",")
                if part.strip()
            ]
            normalized[column] = ";".join(sorted(parts))
        else:
            normalized[column] = str(value).strip()
    return normalized


def _parse_markdown_table(path: Path) -> list[dict[str, str]]:
    """Parse a GitHub-style Markdown table containing the required denominator columns.

    Returns:
        Parsed table rows.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    required = set(_TABLE_COLUMNS)
    for index, line in enumerate(lines[:-1]):
        if "|" not in line:
            continue
        headers = [cell.strip() for cell in line.strip().strip("|").split("|")]
        normalized_headers = [header.lower().replace(" ", "_") for header in headers]
        if not required.issubset(normalized_headers):
            continue
        separator = lines[index + 1]
        if not set(separator.strip()).issubset({"|", "-", ":", " "}):
            continue
        rows: list[dict[str, str]] = []
        header_map = dict(zip(normalized_headers, range(len(headers)), strict=False))
        for row_line in lines[index + 2 :]:
            if "|" not in row_line:
                break
            cells = [cell.strip() for cell in row_line.strip().strip("|").split("|")]
            if len(cells) < len(headers):
                break
            rows.append({column: cells[header_map[column]] for column in _TABLE_COLUMNS})
        return rows
    raise DenominatorManifestError(f"No denominator table with required columns found: {path}")


def load_denominator_table(path: Path) -> list[dict[str, str]]:
    """Load denominator consumer table from CSV, JSON/YAML, or Markdown.

    Returns:
        Normalized denominator table rows.
    """

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix in {".json", ".yaml", ".yml"}:
        payload = load_manifest(path)
        raw_rows = payload.get("rows", payload.get("per_family_planner_denominators"))
        if not isinstance(raw_rows, list):
            raise DenominatorManifestError(
                f"JSON/YAML denominator table must contain rows list: {path}"
            )
        rows = raw_rows
    elif suffix == ".md":
        rows = _parse_markdown_table(path)
    else:
        raise DenominatorManifestError(
            f"Unsupported denominator table format {suffix!r}; use CSV, JSON, YAML, or Markdown"
        )
    return [_normalize_table_row(dict(row)) for row in rows]


def check_denominator_table(expected_manifest: dict[str, Any], observed_path: Path) -> None:
    """Fail closed when a consumer table does not match manifest-derived denominators."""

    expected = [_normalize_table_row(row) for row in denominator_table_rows(expected_manifest)]
    observed = load_denominator_table(observed_path)
    expected_keys = {tuple(row[column] for column in _TABLE_COLUMNS) for row in expected}
    observed_keys = {tuple(row[column] for column in _TABLE_COLUMNS) for row in observed}
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing or extra:
        parts = [f"Denominator table mismatch: {observed_path}"]
        if missing:
            parts.append(f"missing rows: {len(missing)}")
        if extra:
            parts.append(f"extra rows: {len(extra)}")
        raise DenominatorManifestError("; ".join(parts))

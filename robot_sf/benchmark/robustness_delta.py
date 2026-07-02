"""Build nominal-vs-perturbed observation robustness delta reports."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = "observation_robustness_delta.v1"
ISSUE = 3952
PAIR_KEYS = (
    "planner_key",
    "algo",
    "scenario_id",
    "seed",
    "kinematics",
    "observation_mode",
    "observation_level",
)
CLAIM_BOUNDARY = (
    "Same-scenario same-seed diagnostic robustness delta under non-calibrated "
    "observation perturbations. Not hardware sensor model and not paper-facing "
    "benchmark evidence."
)


def load_episode_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load benchmark episode JSONL rows.

    Returns:
        Parsed JSON object rows.
    """
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} contains a non-object JSONL row")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no episode rows")
    return rows


def build_robustness_delta_report(
    *,
    nominal_jsonl: Path,
    perturbed_jsonl: Path,
    nominal_rows: Sequence[Mapping[str, Any]] | None = None,
    perturbed_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a per-planner robustness delta report from paired episode rows.

    Returns:
        Report payload with pairing diagnostics and per-planner delta rows.
    """
    clean_rows = (
        list(nominal_rows) if nominal_rows is not None else load_episode_jsonl(nominal_jsonl)
    )
    noisy_rows = (
        list(perturbed_rows) if perturbed_rows is not None else load_episode_jsonl(perturbed_jsonl)
    )
    active_pair_keys = _active_pair_keys(clean_rows, noisy_rows)
    nominal_by_key = _index_rows(clean_rows, active_pair_keys, "nominal")
    perturbed_by_key = _index_rows(noisy_rows, active_pair_keys, "perturbed")

    nominal_keys = set(nominal_by_key)
    perturbed_keys = set(perturbed_by_key)
    paired_keys = sorted(nominal_keys & perturbed_keys, key=_sort_key)
    unmatched_nominal = sorted(nominal_keys - perturbed_keys, key=_sort_key)
    unmatched_perturbed = sorted(perturbed_keys - nominal_keys, key=_sort_key)

    grouped: dict[str, dict[str, Any]] = {}
    for key in paired_keys:
        nominal = nominal_by_key[key]
        perturbed = perturbed_by_key[key]
        planner_key = _planner_identity(nominal)
        planner = grouped.setdefault(
            planner_key,
            {
                "_algo": _algo(nominal),
                "_success_nominal": [],
                "_success_perturbed": [],
                "_collision_nominal": [],
                "_collision_perturbed": [],
                "_profiles": set(),
                "_hashes": set(),
                "_noise_stats": defaultdict(int),
            },
        )
        planner["_success_nominal"].append(_success(nominal))
        planner["_success_perturbed"].append(_success(perturbed))
        planner["_collision_nominal"].append(_collision(nominal))
        planner["_collision_perturbed"].append(_collision(perturbed))
        profile = _perturbation_profile(perturbed)
        if profile:
            planner["_profiles"].add(profile)
        noise_hash = perturbed.get("observation_noise_hash")
        if noise_hash:
            planner["_hashes"].add(str(noise_hash))
        for stat_key, value in _numeric_stats(perturbed.get("observation_noise_stats")).items():
            planner["_noise_stats"][stat_key] += value

    planner_rows = [
        _finalize_planner_row(planner_key, planner)
        for planner_key, planner in sorted(
            grouped.items(), key=lambda item: (item[0], item[1]["_algo"])
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "nominal_jsonl": nominal_jsonl.as_posix(),
            "perturbed_jsonl": perturbed_jsonl.as_posix(),
        },
        "pairing": {
            "paired_rows": len(paired_keys),
            "unmatched_nominal_rows": len(unmatched_nominal),
            "unmatched_perturbed_rows": len(unmatched_perturbed),
            "pair_keys": active_pair_keys,
            "unmatched_nominal_keys": [_key_as_dict(key) for key in unmatched_nominal],
            "unmatched_perturbed_keys": [_key_as_dict(key) for key in unmatched_perturbed],
        },
        "planner_rows": planner_rows,
    }


def write_report_json(report: Mapping[str, Any], path: Path) -> None:
    """Write report JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report_csv(report: Mapping[str, Any], path: Path) -> None:
    """Write planner delta rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "planner_key",
        "algo",
        "paired_episodes",
        "nominal_success_incidence",
        "perturbed_success_incidence",
        "success_delta",
        "nominal_collision_incidence",
        "perturbed_collision_incidence",
        "collision_delta",
        "perturbation_profiles",
        "perturbation_hashes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in report.get("planner_rows", []):
            if not isinstance(row, Mapping):
                continue
            writer.writerow(
                {
                    field: ";".join(row[field])
                    if field in {"perturbation_profiles", "perturbation_hashes"}
                    else row.get(field)
                    for field in fieldnames
                }
            )


def format_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown robustness report.

    Returns:
        Markdown report text.
    """
    inputs = _mapping(report.get("inputs"))
    pairing = _mapping(report.get("pairing"))
    rows = [row for row in report.get("planner_rows", []) if isinstance(row, Mapping)]
    lines = [
        "# Issue #3952 Observation Robustness Smoke",
        "",
        "## Claim Boundary",
        str(report.get("claim_boundary", CLAIM_BOUNDARY)),
        "",
        "## Inputs",
        f"- Nominal JSONL: `{inputs.get('nominal_jsonl', '')}`",
        f"- Perturbed JSONL: `{inputs.get('perturbed_jsonl', '')}`",
        f"- Paired rows: `{pairing.get('paired_rows', 0)}`",
        f"- Unmatched nominal rows: `{pairing.get('unmatched_nominal_rows', 0)}`",
        f"- Unmatched perturbed rows: `{pairing.get('unmatched_perturbed_rows', 0)}`",
        "",
        "## Planner Robustness Delta",
        "",
        "| planner | algo | paired episodes | nominal success | perturbed success | success delta | nominal collision | perturbed collision | collision delta | perturbation profile |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {planner} | {algo} | {paired} | {nom_s:.3f} | {pert_s:.3f} | {delta_s:.3f} | "
            "{nom_c:.3f} | {pert_c:.3f} | {delta_c:.3f} | {profile} |".format(
                planner=row.get("planner_key", ""),
                algo=row.get("algo", ""),
                paired=row.get("paired_episodes", 0),
                nom_s=float(row.get("nominal_success_incidence", 0.0)),
                pert_s=float(row.get("perturbed_success_incidence", 0.0)),
                delta_s=float(row.get("success_delta", 0.0)),
                nom_c=float(row.get("nominal_collision_incidence", 0.0)),
                pert_c=float(row.get("perturbed_collision_incidence", 0.0)),
                delta_c=float(row.get("collision_delta", 0.0)),
                profile=", ".join(row.get("perturbation_profiles", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "- CPU smoke only.",
            "- Same-scenario same-seed diagnostic comparison only.",
            "- Observation perturbations are non-calibrated benchmark perturbations, not hardware sensor evidence.",
            "- Nominal metric definitions unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_markdown(report: Mapping[str, Any], path: Path) -> None:
    """Write Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report_markdown(report), encoding="utf-8")


def _active_pair_keys(
    nominal_rows: Sequence[Mapping[str, Any]],
    perturbed_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    rows = [*nominal_rows, *perturbed_rows]
    active = ["planner_identity"]
    for key in PAIR_KEYS:
        if key == "planner_key":
            continue
        if any(_get_nested(row, key) is not None for row in rows):
            active.append(key)
    return active


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    pair_keys: Sequence[str],
    label: str,
) -> dict[tuple[tuple[str, Any], ...], Mapping[str, Any]]:
    indexed: dict[tuple[tuple[str, Any], ...], Mapping[str, Any]] = {}
    for row in rows:
        key = _pair_key(row, pair_keys)
        if key in indexed:
            raise ValueError(f"duplicate {label} row for pairing key {_key_as_dict(key)!r}")
        indexed[key] = row
    return indexed


def _pair_key(row: Mapping[str, Any], pair_keys: Sequence[str]) -> tuple[tuple[str, Any], ...]:
    values: list[tuple[str, Any]] = []
    for key in pair_keys:
        value = _planner_identity(row) if key == "planner_identity" else _get_nested(row, key)
        values.append((key, _jsonable_scalar(value)))
    return tuple(values)


def _planner_identity(row: Mapping[str, Any]) -> str:
    for value in (
        row.get("planner_key"),
        row.get("planner"),
        row.get("algo"),
        _get_nested(row, "scenario_params.algo"),
    ):
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _algo(row: Mapping[str, Any]) -> str:
    value = row.get("algo") or _get_nested(row, "scenario_params.algo") or _planner_identity(row)
    return str(value)


def _success(row: Mapping[str, Any]) -> bool:
    outcome = _mapping(row.get("outcome"))
    if "route_complete" in outcome:
        return bool(outcome["route_complete"])
    metrics = _mapping(row.get("metrics"))
    return _number(metrics.get("success")) > 0.0 or _number(metrics.get("success_rate")) > 0.0


def _collision(row: Mapping[str, Any]) -> bool:
    outcome = _mapping(row.get("outcome"))
    if "collision" in outcome:
        return bool(outcome["collision"])
    if "collision_event" in outcome:
        return bool(outcome["collision_event"])
    metrics = _mapping(row.get("metrics"))
    return _number(metrics.get("collisions")) > 0.0 or _number(metrics.get("collision_rate")) > 0.0


def _finalize_planner_row(planner_key: str, planner: Mapping[str, Any]) -> dict[str, Any]:
    nominal_success = planner["_success_nominal"]
    perturbed_success = planner["_success_perturbed"]
    nominal_collision = planner["_collision_nominal"]
    perturbed_collision = planner["_collision_perturbed"]
    nominal_success_incidence = _incidence(nominal_success)
    perturbed_success_incidence = _incidence(perturbed_success)
    nominal_collision_incidence = _incidence(nominal_collision)
    perturbed_collision_incidence = _incidence(perturbed_collision)
    return {
        "planner_key": planner_key,
        "algo": planner["_algo"],
        "paired_episodes": len(nominal_success),
        "nominal_success_incidence": nominal_success_incidence,
        "perturbed_success_incidence": perturbed_success_incidence,
        "success_delta": perturbed_success_incidence - nominal_success_incidence,
        "nominal_collision_incidence": nominal_collision_incidence,
        "perturbed_collision_incidence": perturbed_collision_incidence,
        "collision_delta": perturbed_collision_incidence - nominal_collision_incidence,
        "perturbation_profiles": sorted(planner["_profiles"]),
        "perturbation_hashes": sorted(planner["_hashes"]),
        "noise_stats_sum": dict(sorted(planner["_noise_stats"].items())),
    }


def _incidence(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _perturbation_profile(row: Mapping[str, Any]) -> str | None:
    spec = row.get("observation_noise")
    if isinstance(spec, Mapping):
        profile = spec.get("profile")
        if profile:
            return str(profile)
    profile = row.get("observation_noise_profile")
    return str(profile) if profile else None


def _numeric_stats(value: Any) -> dict[str, int]:
    stats = _mapping(value)
    return {
        str(key): int(number)
        for key, raw in stats.items()
        if isinstance((number := _number(raw)), int | float)
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _get_nested(row: Mapping[str, Any], key: str) -> Any:
    current: Any = row
    for part in key.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _key_as_dict(key: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return dict(key)


def _sort_key(key: tuple[tuple[str, Any], ...]) -> tuple[str, ...]:
    return tuple(f"{name}={value}" for name, value in key)

"""Build the bounded Chapter 7 release-cell evidence package v2 projection.

The v2 package is a deterministic, digest-bound projection of the immutable
issue #6792 package.  It does not rerun a campaign, copy traces, reinterpret
collision metrics, or create an admission decision.  The source package and
its review sidecars are verified before the safe release-cell projection is
written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, ValidationError

from scripts.analysis import build_ch7_evidence_package as v1
from scripts.analysis import verify_ch7_evidence_admission as admission

PACKAGE_SCHEMA = "ch7-evidence-package.v2"
REDUCED_PUBLICATION_ATLAS_SCHEMA = "ch7-reduced-publication-atlas.v3"
DEFAULT_CONFIG = Path("configs/analysis/ch7_evidence_package.v2.yaml")
PORTFOLIO_CONFIG_PATH = Path("configs/analysis/ch7_worked_example_portfolio.v2.yaml")
PORTFOLIO_CONFIG_SHA256 = "ebf2e943b6cea7e647f71171c08e904edf19b818cd2e1853ee5409a80d74f010"
SOURCE_PACKAGE_SHA256SUMS = "6807fdc9275133365812c8f51f51e057da6054f8dcaf77cb5fa8a32b08c4a87f"
SOURCE_AUDIT_MEMBER = "audit/campaign_atlas.csv"
SOURCE_AUDIT_SHA256 = "18768f5cf1d9f360487e9203fa1c538de136aaff240070ae3684b5243c44bc10"
SOURCE_REDUCED_ATLAS_MEMBER = "publication/reduced_atlas.json"
SOURCE_REDUCED_ATLAS_SHA256 = "e5bed5855b7048291530615c86c88257f02622a1abbbecef4a7a6d7fb37c29b6"
TOPOLOGY_SCENARIOS = (
    "classic_realworld_double_bottleneck_high",
    "francis2023_blind_corner",
)
MECHANISM_SCENARIOS = (
    "francis2023_pedestrian_obstruction",
    "francis2023_join_group",
)
HYBRID_ARMS = (
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
)
TOPOLOGY_PLANNERS = ("ppo", *HYBRID_ARMS)
MECHANISM_PLANNERS = ("orca", "socnav_sampling")
DOORWAY_SCENARIO = "francis2023_narrow_doorway"
DOORWAY_ARMS = (
    "goal",
    "guarded_ppo",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "orca",
    "ppo",
    "prediction_planner",
    "predictive_mppi",
    "risk_dwa",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "social_force",
    "socnav_sampling",
)
SAFE_METRICS = (
    "success_fraction",
    "near_misses_mean",
    "time_to_goal_norm_mean",
    "path_efficiency_mean",
)
EXCLUDED_METRICS = (
    "collision_fraction",
    "ped_collision_fraction",
    "obstacle_collision_fraction",
    "total_collision_fraction",
    "collision_count_mean",
    "ped_collision_count_mean",
    "obstacle_collision_count_mean",
    "total_collision_count_mean",
    "snqi_mean",
    "collision_derived_composites",
)
CLAIM_BOUNDARY = (
    "Release-cell descriptive projections only. Collision-related metrics remain omitted pending "
    "#7042 metric naming resolution. No trace-level, causal, universal-ranking, or admission claim "
    "is produced by this builder."
)


class Ch7EvidencePackageV2Error(ValueError):
    """Raised when a v2 source or projection contract is not satisfied."""


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Ch7EvidencePackageV2Error(f"invalid JSON input: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidencePackageV2Error(f"JSON input must be an object: {path}")
    return dict(payload)


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Ch7EvidencePackageV2Error(f"invalid package config: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidencePackageV2Error("package config must be a mapping")
    return dict(payload)


def _validate_config(config: Mapping[str, Any]) -> None:  # noqa: C901
    if not config:
        return
    if config.get("schema_version") != PACKAGE_SCHEMA:
        raise Ch7EvidencePackageV2Error("unsupported Chapter 7 v2 package config schema")
    if config.get("issue") != 7087:
        raise Ch7EvidencePackageV2Error("Chapter 7 v2 package config issue mismatch")
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise Ch7EvidencePackageV2Error("v2 package config lacks source binding")
    if source.get("sha256sums_sha256") != SOURCE_PACKAGE_SHA256SUMS:
        raise Ch7EvidencePackageV2Error("v2 config source package digest does not match v1 input")
    if source.get("audit_member") != SOURCE_AUDIT_MEMBER:
        raise Ch7EvidencePackageV2Error("v2 config audit member changed")
    if source.get("audit_member_sha256") != SOURCE_AUDIT_SHA256:
        raise Ch7EvidencePackageV2Error("v2 config audit member digest does not match v1 input")
    if source.get("reduced_atlas_member") != SOURCE_REDUCED_ATLAS_MEMBER:
        raise Ch7EvidencePackageV2Error("v2 config reduced atlas member changed")
    if source.get("reduced_atlas_member_sha256") != SOURCE_REDUCED_ATLAS_SHA256:
        raise Ch7EvidencePackageV2Error("v2 config reduced atlas digest does not match v1 input")
    if config.get("safe_metrics") != list(SAFE_METRICS):
        raise Ch7EvidencePackageV2Error("v2 safe metric selection changed")
    if config.get("excluded_metrics") != list(EXCLUDED_METRICS):
        raise Ch7EvidencePackageV2Error("v2 excluded metric selection changed")
    admission_config = config.get("admission")
    if (
        not isinstance(admission_config, Mapping)
        or admission_config.get("status") != "not_admitted"
        or admission_config.get("receipt_schema") != "ch7-evidence-admission.v2"
    ):
        raise Ch7EvidencePackageV2Error("v2 config must retain the not-admitted boundary")


def _load_portfolio_contract(config: Mapping[str, Any]) -> dict[str, Any]:  # noqa: C901
    """Verify the separate v2 selection contract without touching the frozen v1 file."""

    metadata = config.get(
        "portfolio_config",
        {"path": PORTFOLIO_CONFIG_PATH.as_posix(), "sha256": PORTFOLIO_CONFIG_SHA256},
    )
    if not isinstance(metadata, Mapping):
        raise Ch7EvidencePackageV2Error("v2 package config lacks portfolio binding")
    if metadata.get("path") != PORTFOLIO_CONFIG_PATH.as_posix():
        raise Ch7EvidencePackageV2Error("v2 portfolio path changed")
    portfolio_path = Path(__file__).parents[2] / PORTFOLIO_CONFIG_PATH
    if metadata.get("sha256") != PORTFOLIO_CONFIG_SHA256:
        raise Ch7EvidencePackageV2Error("v2 portfolio digest changed")
    if _sha256_file(portfolio_path) != PORTFOLIO_CONFIG_SHA256:
        raise Ch7EvidencePackageV2Error("v2 portfolio file digest mismatch")
    try:
        payload = yaml.safe_load(portfolio_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Ch7EvidencePackageV2Error("v2 portfolio config is unreadable") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "ch7_case_portfolio.v3":
        raise Ch7EvidencePackageV2Error("v2 portfolio schema is unsupported")
    source = payload.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("package_sha256sums_sha256") != SOURCE_PACKAGE_SHA256SUMS
    ):
        raise Ch7EvidencePackageV2Error("v2 portfolio source binding changed")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise Ch7EvidencePackageV2Error("v2 portfolio lacks selection binding")
    expected = {
        "cross_topology": {
            "scenarios": list(TOPOLOGY_SCENARIOS),
            "planners": list(TOPOLOGY_PLANNERS),
        },
        "cross_mechanism": {
            "scenarios": list(MECHANISM_SCENARIOS),
            "planners": list(MECHANISM_PLANNERS),
        },
        "narrow_doorway_terminal": {
            "scenarios": [DOORWAY_SCENARIO],
            "planners": list(DOORWAY_ARMS),
        },
    }
    if selection != expected:
        raise Ch7EvidencePackageV2Error("v2 portfolio selection changed")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("included") != list(SAFE_METRICS):
        raise Ch7EvidencePackageV2Error("v2 portfolio safe metrics changed")
    if metrics.get("excluded") != list(EXCLUDED_METRICS) or metrics.get("blocking_issue") != 7042:
        raise Ch7EvidencePackageV2Error("v2 portfolio metric boundary changed")
    return dict(payload)


def verify_v1_source_package(source_package: Path) -> dict[str, Any]:  # noqa: C901
    """Verify the immutable v1 package and return its bound member metadata."""

    try:
        sums_sha, listed = admission._verify_members(
            source_package, label="Chapter 7 v1 source package"
        )
    except admission.Ch7EvidenceAdmissionError as exc:
        raise Ch7EvidencePackageV2Error(f"v1 source package verification failed: {exc}") from exc
    if sums_sha != SOURCE_PACKAGE_SHA256SUMS:
        raise Ch7EvidencePackageV2Error("v1 source package SHA256SUMS digest is not approved")
    required = {"manifest.json", SOURCE_AUDIT_MEMBER, SOURCE_REDUCED_ATLAS_MEMBER}
    if not required.issubset(listed):
        raise Ch7EvidencePackageV2Error("v1 source package lacks the bound audit projection")
    source_package = source_package.resolve()
    manifest = _read_json(source_package / "manifest.json")
    schema_path = (
        Path(__file__).parents[2] / "robot_sf/benchmark/schemas/ch7-evidence-package.v1.json"
    )
    schema = _read_json(schema_path)
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=str)
    if errors:
        raise Ch7EvidencePackageV2Error(f"v1 source manifest schema error: {errors[0].message}")
    if manifest.get("schema_version") != "ch7-evidence-package.v1" or manifest.get("issue") != 6792:
        raise Ch7EvidencePackageV2Error("v1 source manifest is not the issue #6792 package")
    if manifest.get("status") != "blocked_pending_domain_approval":
        raise Ch7EvidencePackageV2Error("v1 source manifest admission boundary changed")
    if manifest.get("admission_status") != "not_admitted":
        raise Ch7EvidencePackageV2Error("v1 source manifest must remain not admitted")
    source_terminal_mapping = manifest.get("terminal_label_normalization")
    if (
        source_terminal_mapping is not None
        and source_terminal_mapping != v1.terminal_label_normalization()
    ):
        raise Ch7EvidencePackageV2Error(
            "v1 source terminal-label mapping is not the approved contract"
        )
    atlas = manifest.get("atlas")
    if (
        not isinstance(atlas, Mapping)
        or atlas.get("audit_cells") != 672
        or atlas.get("planner_arms") != 14
    ):
        raise Ch7EvidencePackageV2Error(
            "v1 source audit dimensions do not match the frozen package"
        )
    audit_path = source_package / SOURCE_AUDIT_MEMBER
    audit_sha = _sha256_file(audit_path)
    if audit_sha != SOURCE_AUDIT_SHA256:
        raise Ch7EvidencePackageV2Error("v1 source audit member digest is not approved")
    reduced_atlas_path = source_package / SOURCE_REDUCED_ATLAS_MEMBER
    reduced_atlas_sha = _sha256_file(reduced_atlas_path)
    if reduced_atlas_sha != SOURCE_REDUCED_ATLAS_SHA256:
        raise Ch7EvidencePackageV2Error("v1 source reduced atlas digest is not approved")
    return {
        "package_sha256sums_sha256": sums_sha,
        "manifest_sha256": _sha256_file(source_package / "manifest.json"),
        "audit_member": SOURCE_AUDIT_MEMBER,
        "audit_member_sha256": audit_sha,
        "reduced_atlas_member": SOURCE_REDUCED_ATLAS_MEMBER,
        "reduced_atlas_member_sha256": reduced_atlas_sha,
        "manifest": manifest,
    }


def _read_audit_rows(source_package: Path) -> list[dict[str, str]]:
    path = source_package / SOURCE_AUDIT_MEMBER
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise Ch7EvidencePackageV2Error("v1 source audit member is unreadable") from exc
    if len(rows) != 672:
        raise Ch7EvidencePackageV2Error(f"v1 source audit row count changed: {len(rows)}")
    if not rows:
        raise Ch7EvidencePackageV2Error("v1 source audit member is empty")
    return rows


def _read_terminal_counts(  # noqa: C901
    source_package: Path,
) -> dict[tuple[str, str], dict[str, int]]:
    """Read only the existing normalized terminal panel from the immutable v1 atlas."""

    payload = _read_json(source_package / SOURCE_REDUCED_ATLAS_MEMBER)
    if payload.get("schema_version") != "ch7-reduced-publication-atlas.v1":
        raise Ch7EvidencePackageV2Error("v1 reduced atlas schema changed")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise Ch7EvidencePackageV2Error("v1 reduced atlas cells are missing")
    expected = {(DOORWAY_SCENARIO, planner) for planner in DOORWAY_ARMS}
    result: dict[tuple[str, str], dict[str, int]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise Ch7EvidencePackageV2Error("v1 reduced atlas contains a malformed cell")
        key = (str(cell.get("scenario_id", "")), str(cell.get("planner_key", "")))
        if key not in expected:
            continue
        if key in result or cell.get("episodes") != 30:
            raise Ch7EvidencePackageV2Error("v1 terminal panel has duplicate or non-30 cell")
        counts = cell.get("terminal_counts")
        if not isinstance(counts, Mapping) or not counts:
            raise Ch7EvidencePackageV2Error("v1 terminal panel lacks normalized terminal counts")
        normalized: dict[str, int] = {}
        for label, count in counts.items():
            if label not in v1.TERMINAL_LABEL_PRECEDENCE or not isinstance(count, int) or count < 0:
                raise Ch7EvidencePackageV2Error("v1 terminal panel contains an invalid label count")
            normalized[str(label)] = count
        if sum(normalized.values()) != 30:
            raise Ch7EvidencePackageV2Error("v1 terminal panel does not sum to 30 episodes")
        result[key] = normalized
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise Ch7EvidencePackageV2Error(f"v1 terminal panel is missing cells: {missing}")
    return result


def _finite_float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise Ch7EvidencePackageV2Error(f"audit row has malformed {field}: {row}") from exc
    if not math.isfinite(value):
        raise Ch7EvidencePackageV2Error(f"audit row has non-finite {field}: {row}")
    return value


def select_v2_cells(  # noqa: C901
    rows: Sequence[Mapping[str, str]],
    *,
    source: Mapping[str, Any],
    terminal_counts: Mapping[tuple[str, str], Mapping[str, int]],
) -> list[dict[str, Any]]:
    """Select the three v2 projections while retaining per-cell source binding."""

    index: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        scenario = row.get("scenario_id", "")
        planner = row.get("planner_key", "")
        key = (scenario, planner)
        if key in index:
            raise Ch7EvidencePackageV2Error(f"duplicate v1 audit cell: {scenario}/{planner}")
        index[key] = row
    requests = (
        ("cross_topology", TOPOLOGY_SCENARIOS, TOPOLOGY_PLANNERS),
        ("cross_mechanism", MECHANISM_SCENARIOS, MECHANISM_PLANNERS),
        ("narrow_doorway_terminal", (DOORWAY_SCENARIO,), DOORWAY_ARMS),
    )
    selected: list[dict[str, Any]] = []
    for panel, scenarios, planners in requests:
        for scenario in scenarios:
            for planner in planners:
                row = index.get((scenario, planner))
                if row is None:
                    raise Ch7EvidencePackageV2Error(
                        f"v1 audit is missing the v2 cell: {scenario}/{planner}"
                    )
                try:
                    episodes = int(row["episodes"])
                    source_row_sha256 = row["source_row_sha256"]
                    if len(source_row_sha256) != 64 or any(
                        char not in "0123456789abcdef" for char in source_row_sha256
                    ):
                        raise ValueError("source_row_sha256 is not lowercase SHA-256")
                    if episodes != 30:
                        raise ValueError("episodes is not 30")
                    cell: dict[str, Any] = {
                        "panel": panel,
                        "scenario_id": scenario,
                        "scenario_family": row["scenario_family"],
                        "planner_key": planner,
                        "arm_id": row["arm_id"],
                        "configuration_identity": row["configuration_identity"],
                        "kinematics": row["kinematics"],
                        "episodes": episodes,
                    }
                    for metric in SAFE_METRICS:
                        cell[metric] = _finite_float(row, metric)
                except (KeyError, TypeError, ValueError) as exc:
                    raise Ch7EvidencePackageV2Error(
                        f"v1 audit cell is malformed: {scenario}/{planner}"
                    ) from exc
                if not cell["scenario_family"] or not cell["arm_id"] or not cell["kinematics"]:
                    raise Ch7EvidencePackageV2Error(
                        f"v1 audit cell lacks identity metadata: {scenario}/{planner}"
                    )
                if panel == "narrow_doorway_terminal":
                    cell["terminal_counts"] = dict(terminal_counts[(scenario, planner)])
                    cell["terminal_counts_status"] = "available"
                    cell["terminal_counts_provenance"] = {
                        "status": "available",
                        "member": source["reduced_atlas_member"],
                        "member_sha256": source["reduced_atlas_member_sha256"],
                    }
                else:
                    cell["terminal_counts"] = {}
                    cell["terminal_counts_status"] = "unavailable"
                    cell["terminal_counts_provenance"] = {
                        "status": "unavailable",
                        "reason": "v1 audit/campaign_atlas.csv does not contain normalized terminal counts",
                    }
                cell["source_provenance"] = {
                    "package_sha256sums_sha256": source["package_sha256sums_sha256"],
                    "member": source["audit_member"],
                    "member_sha256": source["audit_member_sha256"],
                    "source_row_sha256": source_row_sha256,
                }
                selected.append(cell)
    return selected


def _projection_metadata() -> dict[str, Any]:
    return {
        "cross_topology": {
            "scenarios": list(TOPOLOGY_SCENARIOS),
            "planners": list(TOPOLOGY_PLANNERS),
            "cell_count": len(TOPOLOGY_SCENARIOS) * len(TOPOLOGY_PLANNERS),
        },
        "cross_mechanism": {
            "scenarios": list(MECHANISM_SCENARIOS),
            "planners": list(MECHANISM_PLANNERS),
            "cell_count": len(MECHANISM_SCENARIOS) * len(MECHANISM_PLANNERS),
        },
        "narrow_doorway_terminal": {
            "scenarios": [DOORWAY_SCENARIO],
            "planners": list(DOORWAY_ARMS),
            "cell_count": len(DOORWAY_ARMS),
        },
    }


def _excluded_metric_records() -> list[dict[str, Any]]:
    return [
        {
            "metric": metric,
            "issue": 7042,
            "status": "excluded",
            "reason": "collision-related metric naming remains blocked; v2 does not quote this field",
        }
        for metric in EXCLUDED_METRICS
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "panel",
        "scenario_id",
        "scenario_family",
        "planner_key",
        "arm_id",
        "configuration_identity",
        "kinematics",
        "episodes",
        *SAFE_METRICS,
        "terminal_counts_json",
        "terminal_counts_status",
        "terminal_counts_member",
        "terminal_counts_member_sha256",
        "terminal_counts_reason",
        "source_package_sha256sums_sha256",
        "source_member",
        "source_member_sha256",
        "source_row_sha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for cell in rows:
            provenance = cell["source_provenance"]
            flat = dict(cell)
            flat.update(
                {
                    "terminal_counts_json": json.dumps(
                        cell["terminal_counts"], sort_keys=True, separators=(",", ":")
                    ),
                    "terminal_counts_status": cell["terminal_counts_status"],
                    "source_package_sha256sums_sha256": provenance["package_sha256sums_sha256"],
                    "source_member": provenance["member"],
                    "source_member_sha256": provenance["member_sha256"],
                    "source_row_sha256": provenance["source_row_sha256"],
                }
            )
            terminal_provenance = cell["terminal_counts_provenance"]
            flat["terminal_counts_member"] = terminal_provenance.get("member", "")
            flat["terminal_counts_member_sha256"] = terminal_provenance.get("member_sha256", "")
            flat["terminal_counts_reason"] = terminal_provenance.get("reason", "")
            flat.pop("source_provenance", None)
            writer.writerow(flat)


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rows.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="ascii")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.relative_to(root).as_posix(),
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update((root / "SHA256SUMS").read_bytes())
    return digest.hexdigest()


def _validate_output(output: Path) -> dict[str, Any]:
    manifest = _read_json(output / "manifest.json")
    manifest_schema = _read_json(
        Path(__file__).parents[2] / "robot_sf/benchmark/schemas/ch7-evidence-package.v2.json"
    )
    Draft202012Validator(manifest_schema).validate(manifest)
    atlas = _read_json(output / "publication/reduced_atlas.json")
    atlas_schema = _read_json(
        Path(__file__).parents[2]
        / "robot_sf/benchmark/schemas/ch7-reduced-publication-atlas.v3.json"
    )
    Draft202012Validator(atlas_schema).validate(atlas)
    try:
        admission._verify_members(
            output, label="generated Chapter 7 v2 package", require_review_sidecars=False
        )
    except admission.Ch7EvidenceAdmissionError as exc:
        raise Ch7EvidencePackageV2Error(
            f"generated package checksum verification failed: {exc}"
        ) from exc
    return manifest


def _build_once(*, source_package: Path, output: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists():
        raise Ch7EvidencePackageV2Error(f"refusing to overwrite package output: {output}")
    source_package = source_package.resolve()
    output = output.resolve()
    if output == source_package or source_package in output.parents:
        raise Ch7EvidencePackageV2Error("v2 output must not be the v1 source package or its child")
    _load_portfolio_contract(config)
    source = verify_v1_source_package(source_package)
    rows = _read_audit_rows(source_package)
    terminal_counts = _read_terminal_counts(source_package)
    selected = select_v2_cells(rows, source=source, terminal_counts=terminal_counts)
    metadata = _projection_metadata()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise Ch7EvidencePackageV2Error(f"refusing to reuse staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        binding = {
            "schema_version": "ch7-v2-source-projection-binding.v1",
            "source_package": {
                "sha256sums_sha256": source["package_sha256sums_sha256"],
                "manifest_sha256": source["manifest_sha256"],
                "member": source["audit_member"],
                "member_sha256": source["audit_member_sha256"],
                "terminal_member": source["reduced_atlas_member"],
                "terminal_member_sha256": source["reduced_atlas_member_sha256"],
                "audit_cell_count": len(rows),
            },
            "selection": metadata,
            "safe_metrics": list(SAFE_METRICS),
            "excluded_metrics": _excluded_metric_records(),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        _write_json(staging / "source/projection_binding.json", binding)
        _write_csv(staging / "publication/reduced_atlas.csv", selected)
        atlas = {
            "schema_version": REDUCED_PUBLICATION_ATLAS_SCHEMA,
            "metric_contract": "release_cell_safe_metrics_without_collision_fields.v1",
            "terminal_label_normalization": v1.terminal_label_normalization(),
            "cells": selected,
            "projections": metadata,
            "excluded_metrics": _excluded_metric_records(),
            "roles": [
                "cross_topology_inversion",
                "cross_mechanism_inversion",
                "feasibility_criticism",
            ],
            "claim_boundary": CLAIM_BOUNDARY,
        }
        _write_json(staging / "publication/reduced_atlas.json", atlas)
        for panel, label in (
            ("cross_topology", "cross_topology_inversion"),
            ("cross_mechanism", "cross_mechanism_inversion"),
            ("narrow_doorway_terminal", "narrow_doorway_terminal_signature"),
        ):
            panel_cells = [cell for cell in selected if cell["panel"] == panel]
            _write_json(
                staging / f"publication/{label}.sidecar.json",
                {
                    "schema_version": "ch7-publication-sidecar.v2",
                    "figure_id": label,
                    "status": "preview_pending_domain_approval",
                    "evidence_grain": "release_cell",
                    "release_cell_count": len(panel_cells),
                    "scenarios": metadata[panel]["scenarios"],
                    "planners": metadata[panel]["planners"],
                    "terminal_label_normalization": v1.terminal_label_normalization(),
                    "source_package_sha256sums_sha256": source["package_sha256sums_sha256"],
                    "observed_result": (
                        "The v2 projection binds the declared scenario/planner cross-product; "
                        "cell values remain release-cell descriptive and pending v2 domain approval."
                    ),
                    "limitations": [
                        "release-cell grain only",
                        "collision-related metrics omitted pending #7042",
                        "no trace or mechanism-level causal evidence",
                        "no universal ranking claim",
                        "external admission receipt remains required",
                    ],
                    "planner_identity_note": (
                        "socnav_sampling is the in-repository sampling adapter; this projection does not "
                        "assert upstream SocNavBench equivalence."
                        if panel == "cross_mechanism"
                        else ""
                    ),
                    "causal_language_allowed": False,
                },
            )
        _write_json(
            staging / "review/source_verification.json",
            {
                "schema_version": "ch7-v2-source-verification.v1",
                "status": "verified_but_domain_approval_pending",
                "source_package": source,
                "raw_traces_included": False,
                "release_archive_included": False,
                "admission_receipt": {
                    "required": True,
                    "status": "not_created",
                    "schema": "ch7-evidence-admission.v2",
                    "reason": "v2 domain approval is outside this builder",
                },
            },
        )
        manifest = {
            "schema_version": PACKAGE_SCHEMA,
            "issue": 7087,
            "status": "blocked_pending_domain_approval",
            "admission_status": "not_admitted",
            "source_integrity_gate": "blocked_pending_domain_approval",
            "source": {
                "v1_package_sha256sums": source["package_sha256sums_sha256"],
                "v1_manifest_sha256": source["manifest_sha256"],
                "v1_audit_member": source["audit_member"],
                "v1_audit_member_sha256": source["audit_member_sha256"],
                "v1_reduced_atlas_member": source["reduced_atlas_member"],
                "v1_reduced_atlas_member_sha256": source["reduced_atlas_member_sha256"],
            },
            "inputs": {
                "portfolio_config": {
                    "path": PORTFOLIO_CONFIG_PATH.as_posix(),
                    "sha256": PORTFOLIO_CONFIG_SHA256,
                }
            },
            "atlas": {
                "source_audit_cells": len(rows),
                "publication_cells": len(selected),
                "planner_arms": 14,
            },
            "projection": metadata,
            "metrics": {
                "included": list(SAFE_METRICS),
                "excluded": _excluded_metric_records(),
            },
            "terminal_label_normalization": v1.terminal_label_normalization(),
            "roles": {
                "cross_topology_inversion": {"status": "available", "grain": "release_cell"},
                "cross_mechanism_inversion": {"status": "available", "grain": "release_cell"},
                "feasibility_criticism": {
                    "status": "available",
                    "grain": "release_cell_geometry",
                },
            },
            "admission": {
                "status": "not_admitted",
                "receipt_required": True,
                "receipt_schema": "ch7-evidence-admission.v2",
                "reason": "v2 domain approval and the external admission receipt remain pending",
            },
            "claim_boundary": CLAIM_BOUNDARY,
            "raw_traces_included": False,
            "release_archive_included": False,
            "deterministic_serialization": "strict-json-sort-keys-utf8-newline.v1",
        }
        _write_json(staging / "manifest.json", manifest)
        _write_checksums(staging)
        _validate_output(staging)
        staging.rename(output)
        return _read_json(output / "manifest.json")
    except (Ch7EvidencePackageV2Error, OSError, TypeError, ValueError, ValidationError):
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_ch7_evidence_package_v2(
    *,
    source_package: Path,
    output: Path,
    config_path: Path | None = None,
    check_determinism: bool = False,
) -> dict[str, Any]:
    """Build the v2 projection, optionally proving byte-identical rebuilds."""

    config = _load_config(config_path)
    _validate_config(config)
    if check_determinism:
        with (
            tempfile.TemporaryDirectory(prefix="ch7-v2-build-a-") as first_root,
            tempfile.TemporaryDirectory(prefix="ch7-v2-build-b-") as second_root,
        ):
            first = Path(first_root) / "package"
            second = Path(second_root) / "package"
            first_manifest = _build_once(source_package=source_package, output=first, config=config)
            second_manifest = _build_once(
                source_package=source_package, output=second, config=config
            )
            if first_manifest != second_manifest or _tree_hash(first) != _tree_hash(second):
                raise Ch7EvidencePackageV2Error("Chapter 7 v2 package is not byte deterministic")
    return _build_once(source_package=source_package, output=output, config=config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-determinism", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v2 projection builder and return a typed CLI status code."""

    args = _parser().parse_args(argv)
    try:
        manifest = build_ch7_evidence_package_v2(
            source_package=args.source_package,
            output=args.output,
            config_path=args.config,
            check_determinism=args.check_determinism,
        )
    except (Ch7EvidencePackageV2Error, OSError, ValidationError) as exc:
        print(f"ch7 v2 evidence package unavailable: {exc}")
        return 2
    print(f"ch7 v2 evidence package status: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

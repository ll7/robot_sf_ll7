#!/usr/bin/env python3
"""Run the issue #2011 AMV actuation-envelope sensitivity sweep.

The sweep is intentionally config-first: a manifest declares each variant, source status, and
pilot bounds; this tool materializes ordinary camera-ready benchmark configs, optionally preflights
or runs them, then aggregates outcome deltas against each field group's nominal variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.camera_ready_campaign import (
    load_campaign_config,
    prepare_campaign_preflight,
    run_campaign,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file
from robot_sf.benchmark.synthetic_actuation import (
    SYNTHETIC_ACTUATION_CLAIM_BOUNDARY,
    SyntheticActuationProfile,
    known_latency_modes,
    known_update_modes,
    sample_synthetic_actuation_profile,
    summarize_synthetic_actuation_samples,
    validate_synthetic_actuation_profile,
    validate_synthetic_actuation_variability_distribution,
)

_SCHEMA_VERSION = "robot-sf-amv-actuation-sensitivity-results.v1"
_SAMPLING_MODES = ("fixed-variants", "variability-sweep", "all")
_METRICS = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "command_clip_fraction",
    "yaw_rate_saturation_fraction",
    "signed_braking_peak_m_s2",
)
_LATENCY_MODE_TO_STEPS = {
    "zero-step-delay": 0,
    "one-step-delay": 1,
    "two-step-delay": 2,
}
_UPDATE_MODE_TO_PERIOD = {
    "10hz-matched": ("every-step", 1),
    "5hz-hold": ("hold-last", 2),
    "2.5hz-hold": ("hold-last", 4),
}


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/benchmarks/issue_2011_amv_actuation_sensitivity_sweep_v0.yaml"),
        help="Sweep manifest YAML path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for generated configs, indexes, summaries, and figures.",
    )
    parser.add_argument(
        "--mode",
        choices=("materialize", "preflight", "pilot", "aggregate"),
        default="materialize",
        help="Materialize configs, run preflight, run the pilot matrix, or aggregate existing runs.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=_SAMPLING_MODES,
        default=None,
        help=(
            "Actuation profile materialization mode. Defaults to the manifest "
            "variability_sampling.default_mode, which remains fixed-variants."
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Override the manifest seed for deterministic variability-sweep sampling.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        action="append",
        default=[],
        help="Existing campaign root to aggregate; may be passed multiple times.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"),
        help="Log level forwarded to camera-ready campaigns through invoked-command metadata.",
    )
    return parser


def _repo_root() -> Path:
    """Return the repository root."""
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def _repo_relative(path: Path, *, root: Path | None = None) -> str:
    """Return a repository-relative path when possible."""
    root = _repo_root() if root is None else root
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _repo_path(path: Path | str, *, root: Path | None = None) -> Path:
    """Resolve an absolute or repository-relative path."""
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    root = _repo_root() if root is None else root
    return root / parsed


def _write_json(path: Path, payload: Any) -> None:
    """Write stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _sampling_settings(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return optional variability sampling settings from the manifest."""
    settings = manifest.get("variability_sampling")
    if settings is None:
        return {}
    if not isinstance(settings, Mapping):
        raise TypeError("variability_sampling must be a mapping when provided")
    return settings


def _resolve_sampling_mode(
    manifest: Mapping[str, Any],
    sampling_mode: str | None,
) -> str:
    """Resolve the requested actuation materialization mode."""
    settings = _sampling_settings(manifest)
    mode = str(sampling_mode or settings.get("default_mode", "fixed-variants")).strip()
    if mode not in _SAMPLING_MODES:
        raise ValueError(
            f"Unsupported sampling mode '{mode}'. Expected one of: {', '.join(_SAMPLING_MODES)}"
        )
    return mode


def _resolve_sampling_seed(
    manifest: Mapping[str, Any],
    sampling_seed: int | None,
) -> int:
    """Resolve the deterministic variability sampling seed."""
    if sampling_seed is not None:
        return int(sampling_seed)
    settings = _sampling_settings(manifest)
    return int(settings.get("seed", 3284))


def _resolve_sample_count(manifest: Mapping[str, Any]) -> int:
    """Resolve the number of variability samples to materialize."""
    settings = _sampling_settings(manifest)
    sample_count = int(settings.get("sample_count", 3))
    if sample_count <= 0:
        raise ValueError("variability_sampling.sample_count must be > 0")
    return sample_count


def _variability_distribution(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return validated variability distribution metadata when the manifest declares it."""
    distribution = manifest.get("variability_distribution")
    if distribution is None:
        return None
    if not isinstance(distribution, Mapping):
        raise TypeError("variability_distribution must be a mapping when provided")
    validate_synthetic_actuation_variability_distribution(distribution)
    return distribution


def _base_synthetic_profile(manifest: Mapping[str, Any]) -> SyntheticActuationProfile:
    """Return the manifest baseline profile as a validated typed profile."""
    baseline = manifest.get("baseline_profile")
    if not isinstance(baseline, Mapping):
        raise ValueError("Sweep manifest baseline_profile must be a mapping")
    profile = SyntheticActuationProfile(
        name=f"{manifest['name']}_baseline",
        profile_version="v0",
        claim_scope="synthetic-only",
        claim_boundary=SYNTHETIC_ACTUATION_CLAIM_BOUNDARY,
        max_linear_accel_m_s2=float(baseline["max_linear_accel_m_s2"]),
        max_linear_decel_m_s2=float(baseline["max_linear_decel_m_s2"]),
        max_yaw_rate_rad_s=float(baseline["max_yaw_rate_rad_s"]),
        max_angular_accel_rad_s2=float(baseline["max_angular_accel_rad_s2"]),
        latency_mode=str(baseline["latency_mode"]),
        update_mode=str(baseline["update_mode"]),
    )
    validate_synthetic_actuation_profile(profile)
    return profile


def _validate_manifest(manifest: Mapping[str, Any]) -> None:  # noqa: C901
    """Validate the sweep manifest before materializing configs."""
    if manifest.get("schema_version") != "robot-sf-amv-actuation-sensitivity-sweep.v1":
        raise ValueError("Unsupported sweep manifest schema_version")
    if str(manifest.get("claim_boundary", "")).strip() != "diagnostic-only":
        raise ValueError("Sweep claim_boundary must stay diagnostic-only")
    _base_synthetic_profile(manifest)
    _resolve_sampling_mode(manifest, None)
    _resolve_sample_count(manifest)
    _resolve_sampling_seed(manifest, None)
    _variability_distribution(manifest)
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Sweep manifest must define non-empty variants")

    seen_names: set[str] = set()
    groups: defaultdict[str, set[str]] = defaultdict(set)
    for variant in variants:
        if not isinstance(variant, dict):
            raise TypeError("Sweep variants must be mappings")
        name = str(variant.get("name", "")).strip()
        field_group = str(variant.get("field_group", "")).strip()
        level = str(variant.get("level", "")).strip()
        source_status = str(variant.get("source_status", "")).strip()
        if not name or name in seen_names:
            raise ValueError(f"Sweep variant names must be non-empty and unique: {name!r}")
        if not field_group or not level or not source_status:
            raise ValueError(f"Sweep variant missing field_group/level/source_status: {name}")
        if "calibrated" in source_status:
            raise ValueError(f"Sweep variant must not claim calibrated evidence: {name}")
        if not isinstance(variant.get("profile"), dict):
            raise TypeError(f"Sweep variant profile must be a mapping: {name}")
        seen_names.add(name)
        groups[field_group].add(level)

    for field_group, levels in groups.items():
        missing = {"low", "nominal", "high"} - levels
        if missing:
            raise ValueError(
                f"Sweep field_group '{field_group}' is missing levels: {', '.join(sorted(missing))}"
            )


def _pilot_overrides(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return pilot overrides from the manifest."""
    pilot = manifest.get("pilot")
    if not isinstance(pilot, dict):
        raise ValueError("Sweep manifest must define pilot overrides")
    return pilot


def _materialize_variant_config(  # noqa: C901
    *,
    manifest: Mapping[str, Any],
    base_payload: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a concrete camera-ready config for one sweep variant."""
    payload: dict[str, Any] = deepcopy(dict(base_payload))
    baseline = deepcopy(manifest.get("baseline_profile"))
    if not isinstance(baseline, dict):
        raise ValueError("Sweep manifest baseline_profile must be a mapping")
    profile_patch = variant.get("profile")
    if not isinstance(profile_patch, dict):
        raise TypeError("Sweep variant profile must be a mapping")
    baseline.update(profile_patch)
    distribution = _variability_distribution(manifest)
    if distribution is not None and "variability_distribution" not in baseline:
        baseline["variability_distribution"] = deepcopy(dict(distribution))

    update_mode = str(baseline.get("update_mode", "")).strip()
    latency_mode = str(baseline.get("latency_mode", "")).strip()
    if update_mode not in known_update_modes():
        raise ValueError(
            f"Unsupported update_mode for sweep variant '{variant['name']}': {update_mode}"
        )
    if latency_mode not in known_latency_modes():
        raise ValueError(
            f"Unsupported latency_mode for sweep variant '{variant['name']}': {latency_mode}"
        )

    payload["name"] = f"{manifest['name']}_{variant['name']}"
    payload["paper_facing"] = False
    payload["export_publication_bundle"] = False
    payload["include_videos_in_publication"] = False
    payload["overwrite_publication_bundle"] = False
    payload["paper_interpretation_profile"] = "issue-2011-amv-actuation-sensitivity-diagnostic"
    payload["synthetic_actuation_profile"] = {
        "name": str(variant["name"]),
        "profile_version": "v0",
        "claim_scope": "synthetic-only",
        "claim_boundary": "diagnostic-only",
        "max_linear_accel_m_s2": float(baseline["max_linear_accel_m_s2"]),
        "max_linear_decel_m_s2": float(baseline["max_linear_decel_m_s2"]),
        "max_yaw_rate_rad_s": float(baseline["max_yaw_rate_rad_s"]),
        "max_angular_accel_rad_s2": float(baseline["max_angular_accel_rad_s2"]),
        "latency_mode": latency_mode,
        "update_mode": update_mode,
    }
    for metadata_key in ("variability_distribution", "variability_sample"):
        if isinstance(baseline.get(metadata_key), Mapping):
            payload["synthetic_actuation_profile"][metadata_key] = deepcopy(
                dict(baseline[metadata_key])
            )

    latency_steps = _LATENCY_MODE_TO_STEPS[latency_mode]
    update_mode_label, update_period = _UPDATE_MODE_TO_PERIOD[update_mode]
    latency_profile = deepcopy(payload.get("latency_stress_profile") or {})
    latency_profile.update(
        {
            "name": "issue-2011-actuation-sensitivity-latency-provenance-v0",
            "profile_version": "v0",
            "claim_scope": "synthetic-only",
            "observation_delay_steps": 0,
            "action_delay_steps": latency_steps,
            "planner_update_mode": update_mode_label,
            "planner_update_period_steps": update_period,
            "inference_timeout_ms": 200.0,
            "non_success_statuses": [
                "fallback",
                "degraded",
                "timeout",
                "not_available",
                "failed",
            ],
        }
    )
    payload["latency_stress_profile"] = latency_profile

    pilot = _pilot_overrides(manifest)
    for key in ("scenario_candidates", "horizon", "workers", "bootstrap_samples"):
        if key in pilot:
            payload[key] = deepcopy(pilot[key])
    if "scenario_candidates" in payload and isinstance(payload.get("scenario_amv_overrides"), dict):
        selected_scenarios = {str(name) for name in payload["scenario_candidates"]}
        payload["scenario_amv_overrides"] = {
            scenario_name: overrides
            for scenario_name, overrides in payload["scenario_amv_overrides"].items()
            if str(scenario_name) in selected_scenarios
        }
    if isinstance(pilot.get("seed_policy"), dict):
        payload["seed_policy"] = deepcopy(pilot["seed_policy"])
    planner_keys = pilot.get("planners")
    if isinstance(planner_keys, list):
        selected = {str(key) for key in planner_keys}
        payload["planners"] = [
            planner
            for planner in payload.get("planners", [])
            if str(planner.get("key")) in selected
        ]
        missing = selected - {str(planner.get("key")) for planner in payload["planners"]}
        if missing:
            raise ValueError(
                f"Pilot planner override references missing planners: {sorted(missing)}"
            )
    payload["issue_2011_sweep_variant"] = {
        "name": str(variant["name"]),
        "field_group": str(variant["field_group"]),
        "level": str(variant["level"]),
        "source_status": str(variant["source_status"]),
        "supported_fields": list(variant.get("supported_fields") or []),
        "source_context": str(variant.get("source_context", "")),
        "caveat": str(variant.get("caveat", "")),
        "sampling_mode": str(variant.get("sampling_mode", "fixed-variants")),
    }
    for metadata_key in ("sample_id", "sample_index", "sampling_seed", "sampled_parameters"):
        if metadata_key in variant:
            payload["issue_2011_sweep_variant"][metadata_key] = deepcopy(variant[metadata_key])
    return payload


def _entry_from_variant(
    *,
    variant: Mapping[str, Any],
    config_path: Path,
    root: Path,
) -> dict[str, Any]:
    """Return one resolved manifest entry for a materialized config."""
    entry = {
        "variant_name": str(variant["name"]),
        "field_group": str(variant["field_group"]),
        "level": str(variant["level"]),
        "source_status": str(variant["source_status"]),
        "config_path": _repo_relative(config_path, root=root),
        "config_sha256": _sha256_file(config_path),
        "supported_fields": list(variant.get("supported_fields") or []),
        "source_context": str(variant.get("source_context", "")),
        "caveat": str(variant.get("caveat", "")),
        "sampling_mode": str(variant.get("sampling_mode", "fixed-variants")),
    }
    for key in ("sample_id", "sample_index", "sampling_seed", "sampled_parameters"):
        if key in variant:
            entry[key] = deepcopy(variant[key])
    return entry


def _sample_variants(
    *,
    manifest: Mapping[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[SyntheticActuationProfile]]:
    """Return deterministic sampled-variability pseudo-variants and profiles."""
    distribution = _variability_distribution(manifest)
    if distribution is None:
        raise ValueError("variability-sweep mode requires manifest variability_distribution")
    sample_count = _resolve_sample_count(manifest)
    base_profile = _base_synthetic_profile(manifest)
    settings = _sampling_settings(manifest)
    source_context = str(
        distribution.get("source_context")
        or settings.get("source_context")
        or "configs/benchmarks/issue_2011_amv_actuation_sensitivity_sweep_v0.yaml"
    )
    caveat = str(
        settings.get("caveat")
        or "Synthetic/provisional variability samples are diagnostic-only and not "
        "hardware-calibrated AMV evidence."
    )
    parameters = distribution.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("variability_distribution.parameters must be a mapping")

    variants: list[dict[str, Any]] = []
    profiles: list[SyntheticActuationProfile] = []
    for sample_index in range(sample_count):
        variant_name = f"variability_sample_{sample_index:03d}"
        profile = sample_synthetic_actuation_profile(
            base_profile,
            distribution,
            seed=seed,
            sample_index=sample_index,
            name=variant_name,
        )
        metadata = profile.to_metadata()
        sample = metadata.get("variability_sample")
        sampled_parameters = (
            dict(sample.get("sampled_parameters", {})) if isinstance(sample, Mapping) else {}
        )
        variants.append(
            {
                "name": variant_name,
                "field_group": "variability_sample",
                "level": f"sample-{sample_index:03d}",
                "source_status": "synthetic_provisional_distribution",
                "supported_fields": list(parameters),
                "source_context": source_context,
                "caveat": caveat,
                "profile": metadata,
                "sampling_mode": "variability-sweep",
                "sample_id": f"sample-{sample_index:03d}",
                "sample_index": sample_index,
                "sampling_seed": seed,
                "sampled_parameters": sampled_parameters,
            }
        )
        profiles.append(profile)
    return variants, profiles


def _sample_summary_from_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a compact sampled-parameter summary from resolved manifest entries."""
    rows = []
    for entry in entries:
        if str(entry.get("sampling_mode", "")) != "variability-sweep":
            continue
        rows.append(
            {
                "profile_name": str(entry.get("variant_name", "")),
                "sample_id": str(entry.get("sample_id", "")),
                "sample_index": int(entry.get("sample_index", -1)),
                "sampling_seed": int(entry.get("sampling_seed", 0)),
                "sampled_parameters": dict(entry.get("sampled_parameters", {})),
                "claim_boundary": SYNTHETIC_ACTUATION_CLAIM_BOUNDARY,
            }
        )
    return {
        "schema_version": "synthetic-actuation-sampled-parameter-summary.v1",
        "claim_boundary": SYNTHETIC_ACTUATION_CLAIM_BOUNDARY,
        "row_count": len(rows),
        "rows": rows,
    }


def _write_sampled_parameter_summary(
    output_dir: Path,
    summary: Mapping[str, Any],
) -> None:
    """Write JSON, CSV, and Markdown summaries for sampled variability parameters."""
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        return
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(reports_dir / "sampled_parameter_summary.json", summary)

    field_names = sorted(
        {
            str(field_name)
            for row in rows
            if isinstance(row, Mapping)
            for field_name in dict(row.get("sampled_parameters", {}))
        }
    )
    csv_path = reports_dir / "sampled_parameter_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "profile_name",
                "sample_id",
                "sample_index",
                "sampling_seed",
                "claim_boundary",
                *field_names,
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            sampled = dict(row.get("sampled_parameters", {}))
            writer.writerow(
                {
                    "profile_name": str(row.get("profile_name", "")),
                    "sample_id": str(row.get("sample_id", "")),
                    "sample_index": int(row.get("sample_index", -1)),
                    "sampling_seed": int(row.get("sampling_seed", 0)),
                    "claim_boundary": str(row.get("claim_boundary", "")),
                    **{field_name: sampled.get(field_name, "") for field_name in field_names},
                }
            )

    lines = [
        "# Issue #3284 Sampled Actuation Parameters",
        "",
        "diagnostic-only synthetic/provisional variability samples. These rows are not "
        "hardware-calibrated AMV evidence.",
        "",
        "| Profile | Sample | Seed | Claim boundary | Sampled parameters |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        sampled = json.dumps(dict(row.get("sampled_parameters", {})), sort_keys=True)
        lines.append(
            "| "
            f"{row.get('profile_name')} | {row.get('sample_id')} | "
            f"{row.get('sampling_seed')} | {row.get('claim_boundary')} | "
            f"`{sampled}` |"
        )
    (reports_dir / "sampled_parameter_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def materialize_configs(
    *,
    manifest_path: Path,
    output_dir: Path,
    sampling_mode: str | None = None,
    sampling_seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write generated camera-ready configs and return the resolved manifest/index."""
    root = _repo_root()
    manifest = _load_yaml(manifest_path)
    _validate_manifest(manifest)
    resolved_sampling_mode = _resolve_sampling_mode(manifest, sampling_mode)
    resolved_sampling_seed = _resolve_sampling_seed(manifest, sampling_seed)
    base_config_path = root / str(manifest["base_config"])
    base_payload = _load_yaml(base_config_path)
    configs_dir = output_dir / "generated_configs"
    entries: list[dict[str, Any]] = []

    materialized_variants: list[dict[str, Any]] = []
    if resolved_sampling_mode in {"fixed-variants", "all"}:
        materialized_variants.extend(dict(variant) for variant in manifest["variants"])
    sampled_profiles: list[SyntheticActuationProfile] = []
    if resolved_sampling_mode in {"variability-sweep", "all"}:
        sample_variants, sampled_profiles = _sample_variants(
            manifest=manifest,
            seed=resolved_sampling_seed,
        )
        materialized_variants.extend(sample_variants)

    for variant in materialized_variants:
        config_payload = _materialize_variant_config(
            manifest=manifest,
            base_payload=base_payload,
            variant=variant,
        )
        config_path = configs_dir / f"{variant['name']}.yaml"
        _write_yaml(config_path, config_payload)
        entries.append(_entry_from_variant(variant=variant, config_path=config_path, root=root))

    sampled_parameter_summary = summarize_synthetic_actuation_samples(sampled_profiles)
    if not sampled_parameter_summary["rows"]:
        sampled_parameter_summary = _sample_summary_from_entries(entries)
    _write_sampled_parameter_summary(output_dir, sampled_parameter_summary)

    resolved = {
        "schema_version": _SCHEMA_VERSION,
        "name": str(manifest["name"]),
        "manifest_path": _repo_relative(manifest_path, root=root),
        "manifest_sha256": _sha256_file(manifest_path),
        "base_config": _repo_relative(base_config_path, root=root),
        "base_config_sha256": _sha256_file(base_config_path),
        "claim_boundary": "diagnostic-only",
        "paper_facing": False,
        "pilot": manifest["pilot"],
        "variability_sampling": {
            "mode": resolved_sampling_mode,
            "seed": resolved_sampling_seed,
            "sample_count": _resolve_sample_count(manifest),
        },
        "variability_distribution": (
            deepcopy(dict(_variability_distribution(manifest)))
            if _variability_distribution(manifest) is not None
            else None
        ),
        "sampled_parameter_summary": sampled_parameter_summary,
        "variants": entries,
    }
    _write_json(output_dir / "resolved_sweep_manifest.json", resolved)
    return resolved, entries


def _invoked_command(config_path: Path, raw_argv: Sequence[str]) -> str:
    """Build an invoked command string for campaign metadata."""
    return shlex.join(
        [sys.executable, "scripts/tools/run_amv_actuation_sensitivity_sweep.py", *raw_argv]
    )


def run_preflights(
    *,
    entries: Sequence[Mapping[str, Any]],
    output_dir: Path,
    manifest: Mapping[str, Any],
    raw_argv: Sequence[str],
) -> list[dict[str, Any]]:
    """Run camera-ready preflight for every generated variant config."""
    root = _repo_root()
    pilot = _pilot_overrides(manifest)
    campaign_output_root = Path(str(pilot.get("output_root", output_dir / "campaigns")))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        config_path = _repo_path(str(entry["config_path"]), root=root)
        cfg = load_campaign_config(config_path)
        prepared = prepare_campaign_preflight(
            cfg,
            output_root=campaign_output_root,
            label=str(entry["variant_name"]),
            invoked_command=_invoked_command(config_path, raw_argv),
        )
        rows.append(
            {
                **{
                    key: entry[key]
                    for key in ("variant_name", "field_group", "level", "source_status")
                },
                "campaign_id": str(prepared["campaign_id"]),
                "campaign_root": _repo_relative(Path(prepared["campaign_root"])),
                "validate_config_path": _repo_relative(Path(prepared["validate_config_path"])),
                "preview_scenarios_path": _repo_relative(Path(prepared["preview_scenarios_path"])),
                "matrix_summary_json": _repo_relative(Path(prepared["matrix_summary_json_path"])),
                "matrix_summary_csv": _repo_relative(Path(prepared["matrix_summary_csv_path"])),
            }
        )
    _write_json(
        output_dir / "preflight_index.json", {"schema_version": _SCHEMA_VERSION, "rows": rows}
    )
    return rows


def run_pilot(
    *,
    entries: Sequence[Mapping[str, Any]],
    output_dir: Path,
    manifest: Mapping[str, Any],
    raw_argv: Sequence[str],
) -> list[dict[str, Any]]:
    """Run the pilot matrix for every generated variant config."""
    root = _repo_root()
    pilot = _pilot_overrides(manifest)
    campaign_output_root = Path(str(pilot.get("output_root", output_dir / "campaigns")))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        config_path = _repo_path(str(entry["config_path"]), root=root)
        cfg = load_campaign_config(config_path)
        result = run_campaign(
            cfg,
            output_root=campaign_output_root,
            label=str(entry["variant_name"]),
            skip_publication_bundle=True,
            invoked_command=_invoked_command(config_path, raw_argv),
        )
        rows.append(
            {
                **{
                    key: entry[key]
                    for key in ("variant_name", "field_group", "level", "source_status")
                },
                "campaign_id": str(result.get("campaign_id", "")),
                "campaign_root": _repo_relative(Path(str(result.get("campaign_root", "")))),
                "status": str(result.get("status", "")),
                "benchmark_success": bool(result.get("benchmark_success", False)),
                "exit_code": int(result.get("exit_code", 2)),
            }
        )
    _write_json(output_dir / "pilot_index.json", {"schema_version": _SCHEMA_VERSION, "rows": rows})
    aggregate_campaigns(
        output_dir=output_dir,
        entries=entries,
        campaign_roots=[Path(row["campaign_root"]) for row in rows],
    )
    return rows


def _safe_float(value: Any) -> float | None:
    """Parse a finite float or return None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _metric_from_episode(record: Mapping[str, Any], metric: str) -> float | None:
    """Extract one metric from an episode record."""
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping) and metric in metrics:
        return _safe_float(metrics[metric])
    return _safe_float(record.get(metric))


def _scenario_family(record: Mapping[str, Any]) -> str:
    """Extract a scenario family label from one episode record."""
    params = record.get("scenario_params")
    if isinstance(params, Mapping):
        for key in ("scenario_family", "family"):
            value = params.get(key)
            if value:
                return str(value)
        metadata = params.get("metadata")
        if isinstance(metadata, Mapping):
            archetype = metadata.get("archetype")
            if archetype:
                return str(archetype)
        map_id = params.get("map_id")
        if map_id:
            return str(map_id)
    return str(record.get("scenario_family", "all")) or "all"


def _iter_campaign_episode_records(campaign_root: Path) -> Iterable[dict[str, Any]]:
    """Yield annotated episode records from one campaign summary."""
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for run_entry in summary.get("runs", []):
        if not isinstance(run_entry, dict):
            continue
        planner = run_entry.get("planner") if isinstance(run_entry.get("planner"), dict) else {}
        episodes_path_raw = run_entry.get("episodes_path")
        if not isinstance(episodes_path_raw, str):
            continue
        raw_path = Path(episodes_path_raw)
        episodes_path = raw_path if raw_path.is_absolute() else _repo_root() / raw_path
        if not episodes_path.exists():
            episodes_path = campaign_root / raw_path
        if not episodes_path.exists():
            logger.warning("Skipping missing episode JSONL: {}", episodes_path_raw)
            continue
        for record in read_jsonl(str(episodes_path)):
            if isinstance(record, dict):
                annotated = dict(record)
                annotated["planner_key"] = str(planner.get("key", "unknown"))
                annotated["algo"] = str(planner.get("algo", "unknown"))
                yield annotated


def _mean(values: Sequence[float]) -> float | None:
    """Return arithmetic mean or None."""
    if not values:
        return None
    return float(sum(values) / len(values))


def _format_optional_float(value: float | None) -> str:
    """Format optional floats for CSV and Markdown."""
    return "nan" if value is None else f"{value:.6f}"


def aggregate_campaigns(  # noqa: C901, PLR0912, PLR0915
    *,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
    campaign_roots: Sequence[Path],
) -> list[dict[str, Any]]:
    """Aggregate field-group effect sizes from pilot campaign roots."""
    if not campaign_roots:
        raise ValueError("At least one campaign root is required for aggregation")
    by_name = {str(entry["variant_name"]): dict(entry) for entry in entries}
    variant_roots: dict[str, tuple[Path, dict[str, Any]]] = {}
    for root in campaign_roots:
        summary_path = root / "reports" / "campaign_summary.json"
        if not summary_path.exists():
            logger.warning("Skipping campaign root without summary: {}", root)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        campaign = summary.get("campaign")
        if not isinstance(campaign, dict):
            campaign = {}
        profile = campaign.get("synthetic_actuation_profile")
        if isinstance(profile, dict):
            variant_name = str(profile.get("name", ""))
            if variant_name in by_name:
                variant_roots[variant_name] = (root, campaign)
    if not variant_roots:
        raise ValueError("No valid campaign summaries matched the sweep variants")

    buckets: defaultdict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"episodes": 0, **{metric: [] for metric in _METRICS}}
    )
    campaign_status_by_variant: dict[str, dict[str, Any]] = {}
    for variant_name, (root, campaign) in variant_roots.items():
        entry = by_name[variant_name]
        campaign_status_by_variant[variant_name] = {
            "campaign_status": str(campaign.get("status", "unknown")),
            "campaign_benchmark_success": bool(campaign.get("benchmark_success", False)),
            "campaign_exit_code": int(campaign.get("exit_code", 2)),
            "campaign_status_reason": str(campaign.get("status_reason", "")),
        }
        for record in _iter_campaign_episode_records(root):
            key = (
                str(entry["field_group"]),
                variant_name,
                str(record.get("planner_key", "unknown")),
                _scenario_family(record),
            )
            bucket = buckets[key]
            bucket["episodes"] += 1
            for metric in _METRICS:
                value = _metric_from_episode(record, metric)
                if value is not None:
                    bucket[metric].append(value)

    nominal_by_group_planner_family: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for (field_group, variant_name, planner_key, scenario_family), bucket in buckets.items():
        entry = by_name[variant_name]
        if str(entry.get("level")) != "nominal":
            continue
        nominal_by_group_planner_family[(field_group, planner_key, scenario_family)] = {
            metric: _mean(bucket[metric]) for metric in _METRICS
        }

    rows: list[dict[str, Any]] = []
    for (field_group, variant_name, planner_key, scenario_family), bucket in sorted(
        buckets.items()
    ):
        entry = by_name[variant_name]
        baseline = nominal_by_group_planner_family.get(
            (field_group, planner_key, scenario_family), {}
        )
        row: dict[str, Any] = {
            "field_group": field_group,
            "variant_name": variant_name,
            "level": str(entry["level"]),
            "source_status": str(entry["source_status"]),
            "sampling_mode": str(entry.get("sampling_mode", "fixed-variants")),
            "sample_id": str(entry.get("sample_id", "")),
            **campaign_status_by_variant.get(
                variant_name,
                {
                    "campaign_status": "unknown",
                    "campaign_benchmark_success": False,
                    "campaign_exit_code": 2,
                    "campaign_status_reason": "",
                },
            ),
            "planner_key": planner_key,
            "scenario_family": scenario_family,
            "episodes": int(bucket["episodes"]),
        }
        sampled_parameters = dict(entry.get("sampled_parameters", {}))
        if sampled_parameters:
            row["sampled_parameters"] = json.dumps(sampled_parameters, sort_keys=True)
        for metric in _METRICS:
            mean = _mean(bucket[metric])
            base_mean = baseline.get(metric)
            row[f"{metric}_mean"] = _format_optional_float(mean)
            row[f"{metric}_delta_vs_nominal"] = _format_optional_float(
                None if mean is None or base_mean is None else mean - base_mean
            )
        rows.append(row)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "effect_size_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text(
            "field_group,variant_name,level,planner_key,scenario_family,episodes\n",
            encoding="utf-8",
        )
    _write_effect_markdown(reports_dir / "effect_size_summary.md", rows)
    _write_sensitivity_svg(output_dir / "figures" / "outcome_sensitivity.svg", rows)
    _write_json(
        reports_dir / "effect_size_summary.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "claim_boundary": "diagnostic-only",
            "sampled_parameter_summary": _sample_summary_from_entries(entries),
            "rows": rows,
        },
    )
    _write_checksums(output_dir)
    return rows


def _write_effect_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a compact Markdown effect-size table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Issue #2011 AMV Actuation Sensitivity Summary",
        "",
        "diagnostic-only pilot summary. Longitudinal rows use platform-class proxy values; "
        "yaw, latency, and update-rate rows remain synthetic stress factors.",
        "Variability-sweep rows, when present, are synthetic/provisional samples and not "
        "hardware-calibrated AMV evidence.",
        "",
        "| Field group | Level | Campaign status | Benchmark success | Planner | Scenario family | Episodes | Success delta | Collision delta | Near-miss delta |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('field_group')} | {row.get('level')} | "
            f"{row.get('campaign_status')} | {row.get('campaign_benchmark_success')} | "
            f"{row.get('planner_key')} | "
            f"{row.get('scenario_family')} | {row.get('episodes')} | "
            f"{row.get('success_delta_vs_nominal')} | "
            f"{row.get('collisions_delta_vs_nominal')} | "
            f"{row.get('near_misses_delta_vs_nominal')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sensitivity_svg(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a simple SVG bar chart of absolute success sensitivity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        delta = _safe_float(row.get("success_delta_vs_nominal"))
        if delta is None:
            continue
        key = f"{row.get('field_group')} / {row.get('scenario_family')}"
        grouped[key] = max(grouped[key], abs(delta))
    items = sorted(grouped.items())
    width = 960
    row_height = 28
    height = max(120, 70 + row_height * max(1, len(items)))
    max_value = max((value for _, value in items), default=1.0) or 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        "<title>Issue #2011 outcome sensitivity by actuation field group</title>",
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        '<text x="24" y="32" font-family="sans-serif" font-size="18" font-weight="700">Outcome sensitivity by field group</text>',
    ]
    if not items:
        lines.append(
            '<text x="24" y="72" font-family="sans-serif" font-size="14">No pilot rows available.</text>'
        )
    for idx, (label, value) in enumerate(items):
        y = 66 + idx * row_height
        bar_width = int((value / max_value) * 420)
        lines.append(
            f'<text x="24" y="{y + 15}" font-family="sans-serif" font-size="12">{label}</text>'
        )
        lines.append(f'<rect x="430" y="{y}" width="{bar_width}" height="18" fill="#2f7d8c"/>')
        lines.append(
            f'<text x="{440 + bar_width}" y="{y + 14}" font-family="sans-serif" font-size="12">{value:.3f}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checksums(output_dir: Path) -> None:
    """Write checksums for compact output artifacts."""
    targets = [
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    ]
    lines = []
    for path in sorted(targets):
        rel = path.relative_to(output_dir).as_posix()
        lines.append(f"{_sha256_file(path)}  {rel}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = _build_parser().parse_args(raw_argv)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, entries = materialize_configs(
        manifest_path=args.manifest,
        output_dir=args.output,
        sampling_mode=args.sampling_mode,
        sampling_seed=args.sampling_seed,
    )
    source_manifest = _load_yaml(args.manifest)
    if args.mode == "preflight":
        run_preflights(
            entries=entries, output_dir=args.output, manifest=source_manifest, raw_argv=raw_argv
        )
    elif args.mode == "pilot":
        run_pilot(
            entries=entries, output_dir=args.output, manifest=source_manifest, raw_argv=raw_argv
        )
    elif args.mode == "aggregate":
        aggregate_campaigns(
            output_dir=args.output, entries=entries, campaign_roots=args.campaign_root
        )
    else:
        _write_checksums(args.output)
    _write_json(
        args.output / "run_summary.json",
        {"schema_version": _SCHEMA_VERSION, "mode": args.mode, "manifest": manifest},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build issue #4014 matched PPO/LSTM/Mamba smoke comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = (
    "diagnostic-only matched smoke comparison; not benchmark-strength, "
    "paper-grade, or dissertation claim evidence"
)
SCHEMA_VERSION = "issue_4014.ppo_sequence_smoke_comparison.v1"
REQUIRED_MODELS = ("ppo", "recurrent_ppo_lstm", "ppo_mamba")
THROUGHPUT_FIELDS = ("total_wall_clock_sec", "train_env_steps_per_sec_mean")
PARAMETER_FIELDS = (
    "policy_parameter_count",
    "policy_trainable_parameter_count",
    "model_parameter_count",
    "model_trainable_parameter_count",
)
REQUIRED_PARAMETER_FIELDS = (
    "policy_parameter_count",
    "policy_trainable_parameter_count",
)


def _label_path(value: str) -> tuple[str, Path]:
    """Parse ``label=path`` CLI values."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected label=path")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("label must not be empty")
    return label, Path(raw_path)


def _sha256(path: Path) -> str:
    """Return SHA-256 hex digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"summary file missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"summary file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"summary file must contain a JSON object: {path}")
    return payload


def _require_number(payload: dict[str, Any], field: str, *, path: Path) -> float:
    """Read a required numeric summary field."""
    value = payload.get(field)
    # ``bool`` is a subclass of ``int``; reject it so a JSON ``true``/``false`` cannot masquerade
    # as a throughput number.
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{path}: required numeric field missing: {field}")
    return float(value)


def _load_perf_row(label: str, summary_path: Path, config_path: Path) -> dict[str, Any]:
    """Load and validate one performance summary row."""
    payload = _load_json(summary_path)
    parameter_summary = payload.get("parameter_summary")
    if not isinstance(parameter_summary, dict):
        raise ValueError(f"{summary_path}: parameter_summary missing")
    if parameter_summary.get("available") is not True:
        raise ValueError(f"{summary_path}: parameter_summary.available must be true")

    parameters: dict[str, int | None] = {}
    for field in PARAMETER_FIELDS:
        value = parameter_summary.get(field)
        # ``bool`` is a subclass of ``int``; a JSON ``true``/``false`` must not count as a
        # parameter count, so reject bool explicitly on both the required and optional branches.
        if field in REQUIRED_PARAMETER_FIELDS and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"{summary_path}: parameter_summary.{field} must be an integer")
        if (
            field not in REQUIRED_PARAMETER_FIELDS
            and value is not None
            and (not isinstance(value, int) or isinstance(value, bool))
        ):
            raise ValueError(
                f"{summary_path}: parameter_summary.{field} must be an integer or null"
            )
        parameters[field] = value

    throughput = {
        field: _require_number(payload, field, path=summary_path) for field in THROUGHPUT_FIELDS
    }
    if throughput["total_wall_clock_sec"] <= 0.0:
        raise ValueError(f"{summary_path}: total_wall_clock_sec must be positive")
    if throughput["train_env_steps_per_sec_mean"] <= 0.0:
        raise ValueError(f"{summary_path}: train_env_steps_per_sec_mean must be positive")

    config_sha256 = _sha256(config_path) if config_path.is_file() else None
    if config_sha256 is None:
        raise ValueError(f"config file missing for {label}: {config_path}")

    return {
        "model_key": label,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "perf_summary_path": str(summary_path),
        "perf_summary_sha256": _sha256(summary_path),
        "run_id": payload.get("run_id"),
        "throughput": throughput,
        "parameter_summary": parameters,
        "fallback_degraded_status": "not_reported",
    }


def build_summary(
    *,
    perf_summaries: dict[str, Path],
    config_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Build the strict comparison summary payload."""
    missing_summaries = sorted(set(REQUIRED_MODELS) - set(perf_summaries))
    missing_configs = sorted(set(REQUIRED_MODELS) - set(config_paths))
    if missing_summaries or missing_configs:
        raise ValueError(
            "missing required model inputs: "
            f"summaries={missing_summaries or 'none'} configs={missing_configs or 'none'}"
        )

    rows = {
        label: _load_perf_row(label, perf_summaries[label], config_paths[label])
        for label in REQUIRED_MODELS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": "ll7/robot_sf_ll7#4014",
        "follow_up_issue": "ll7/robot_sf_ll7#4585",
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_tier": "diagnostic-only",
        "smoke_not_campaign_evidence": True,
        "required_models": list(REQUIRED_MODELS),
        "rows": rows,
        "artifact_dir": str(output_dir),
        "closure_eligible": True,
        "closure_note": (
            "All three smoke rows supplied real parameter and throughput summaries. "
            "This supports #4014 closure at diagnostic smoke tier only."
        ),
    }


def _write_outputs(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    """Write summary, table, README, and checksum artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    throughput_path = output_dir / "throughput.csv"
    readme_path = output_dir / "README.md"
    checksums_path = output_dir / "SHA256SUMS"

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with throughput_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_key",
                "total_wall_clock_sec",
                "train_env_steps_per_sec_mean",
                "policy_parameter_count",
                "policy_trainable_parameter_count",
                "model_parameter_count",
                "model_trainable_parameter_count",
            ],
        )
        writer.writeheader()
        for row in summary["rows"].values():
            writer.writerow(
                {
                    "model_key": row["model_key"],
                    **row["throughput"],
                    **row["parameter_summary"],
                }
            )

    lines = [
        "# Issue #4014 PPO sequence encoder smoke comparison",
        "",
        f"Claim boundary: {summary['claim_boundary']}.",
        "",
        "This artifact compares proximal policy optimization (PPO), true RecurrentPPO "
        "long short-term memory (LSTM), and PPO-Mamba smoke summaries only when every row "
        "has populated throughput and parameter-count metadata.",
        "",
        "It is diagnostic-only evidence and does not promote benchmark, paper, or "
        "dissertation claims.",
        "",
        "| Model | Wall-clock seconds | Steps/sec | Trainable policy parameters |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["rows"].values():
        lines.append(
            "| {model_key} | {wall:.6g} | {fps:.6g} | {params} |".format(
                model_key=row["model_key"],
                wall=row["throughput"]["total_wall_clock_sec"],
                fps=row["throughput"]["train_env_steps_per_sec_mean"],
                params=row["parameter_summary"]["policy_trainable_parameter_count"],
            )
        )
    lines.extend(
        [
            "",
            f"Closure note: {summary['closure_note']}",
            "",
        ]
    )
    readme_path.write_text("\n".join(lines), encoding="utf-8")

    checksum_lines = []
    checksum_paths = [summary_path, throughput_path, readme_path]
    for row in summary["rows"].values():
        source_path = Path(row["perf_summary_path"])
        if source_path.exists():
            checksum_paths.append(source_path)
    for path in checksum_paths:
        # Preserve the path relative to the packet root (forward slashes) so nested entries such
        # as ``source_perf/ppo.json`` verify correctly with ``sha256sum -c`` run from the packet
        # directory; bare ``path.name`` collapsed them to root-level names that fail verification.
        try:
            rel_path = path.relative_to(output_dir).as_posix()
        except ValueError:
            rel_path = path.name
        checksum_lines.append(f"{_sha256(path)}  {rel_path}")
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return [summary_path, throughput_path, readme_path, checksums_path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-summary", action="append", type=_label_path, required=True)
    parser.add_argument("--config", action="append", type=_label_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    perf_summaries = dict(args.perf_summary)
    config_paths = dict(args.config)
    summary = build_summary(
        perf_summaries=perf_summaries,
        config_paths=config_paths,
        output_dir=args.output_dir,
    )
    for path in _write_outputs(summary, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

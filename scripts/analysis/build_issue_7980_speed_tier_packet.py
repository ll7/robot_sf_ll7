#!/usr/bin/env python3
"""Build the issue #7980 source-complete speed-tier interpretation packet.

This projects the 24 canonical issue #5578 synthesis decisions into the existing
``result_interpretation_packet.v1`` contract.  It does not rerun the campaign or
admit a benchmark claim.  The immutable synthesis member is bound through the
reviewed issue #6102 recovery manifest, and every packet metric carries one
versioned source-binding payload containing its canonical decision row in full.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.result_interpretation_packet import (  # noqa: E402
    compute_packet_digest,
    load_result_interpretation_packet,
    render_caption,
    validate_packet,
    write_deterministic_json,
)
from robot_sf.evidence.writers import (  # noqa: E402
    REVIEW_SIDECAR_SCHEMA_VERSION,
    review_marker,
    review_marker_comment,
    review_marker_json,
    write_review_sidecar,
    write_text,
)

RECOVERY_DIR = Path("docs/context/evidence/issue_6102_robot_speed_tier_recovery")
DEFAULT_RECOVERY_MANIFEST = RECOVERY_DIR / "recovery_manifest.json"
DEFAULT_PREVIOUS_PACKET = RECOVERY_DIR / "result_interpretation_packet.v1.json"
DEFAULT_PREREGISTRATION = Path(
    "configs/benchmarks/issue_5578_robot_speed_tier_preregistration.yaml"
)
DEFAULT_OUTPUT = RECOVERY_DIR / "result_interpretation_packet.issue_7980.v1.json"
DEFAULT_CAPTION = RECOVERY_DIR / "result_interpretation_caption.issue_7980.txt"
DEFAULT_CHECKSUM = RECOVERY_DIR / "SHA256SUMS.issue_7980"

BINDING_PREFIX = "issue_7980_source_binding.v1="
EXPECTED_CLASSIFICATIONS = {
    "no_material_shift",
    "inconclusive",
    "intervention_not_activated",
}
EXPECTED_SYNTHESIS_SCHEMA = "robot_sf.issue_5578_speed_tier_synthesis_adapter.v1"
EXPECTED_EVIDENCE_STATUS = "native_grid_synthesis_complete_provenance_unverified"


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object or fail closed with path context."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> Path:
    """Resolve a repository-relative path without permitting an external target."""

    resolved = path if path.is_absolute() else (_REPO_ROOT / path)
    resolved = resolved.resolve()
    try:
        resolved.relative_to(_REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside the repository: {path}") from exc
    return resolved


def _git(*args: str) -> str:
    """Run a read-only git query and return stripped stdout."""

    completed = subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _producer_base_commit() -> str:
    """Return the current branch's stable merge base with ``origin/main``."""

    return _git("merge-base", "HEAD", "origin/main")


def _read_preregistration(path: Path) -> dict[str, Any]:
    """Load the frozen preregistration mapping."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def _expected_design(preregistration: Mapping[str, Any]) -> tuple[set[str], int, dict[str, float]]:
    """Resolve exact contrast IDs, paired support, and harm margins from preregistration."""

    planners = [item.get("planner_id") for item in preregistration["planner_roster"]["arms"]]
    tiers = [
        item.get("tier_id")
        for item in preregistration["robot_speed_axis"]["tiers"]
        if item.get("role") != "nominal_reference"
    ]
    metrics = list(preregistration["inference_contract"]["primary_metrics"])
    seeds = list(preregistration["seed_policy"]["seeds"])
    scenarios = list(preregistration["scenario_contract"]["selected_scenarios"])
    if not all(isinstance(item, str) and item for item in [*planners, *tiers, *metrics]):
        raise ValueError("preregistration planner, tier, and metric IDs must be non-empty strings")
    expected_ids = {
        f"{planner}__{tier}__{metric}"
        for planner in planners
        for tier in tiers
        for metric in metrics
    }
    paired_denominator = len(seeds) * len(scenarios)
    rules = preregistration["inference_contract"]["decision_rule"]
    thresholds = {
        "success_rate": float(rules["success_rate_harm_threshold"]),
        "collision_rate": float(rules["collision_rate_harm_threshold"]),
        "near_miss_rate": float(rules["near_miss_rate_harm_threshold"]),
    }
    if len(expected_ids) != 24 or paired_denominator != 180:
        raise ValueError(
            "issue #7980 requires the frozen 24-contrast, 180-pair preregistration design"
        )
    return expected_ids, paired_denominator, thresholds


def _require_finite(row: Mapping[str, Any], fields: Sequence[str], test_id: str) -> None:
    """Require finite numeric fields on one canonical decision row."""

    for field in fields:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{test_id}: {field} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{test_id}: {field} must be finite")


def _validate_synthesis_row(
    raw_row: object,
    *,
    numeric_fields: Sequence[str],
) -> dict[str, Any]:
    """Validate one canonical decision row and its activation-classification contract."""

    if not isinstance(raw_row, dict):
        raise ValueError("every synthesis decision row must be an object")
    row = dict(raw_row)
    test_id = str(row["test_id"])
    _require_finite(row, numeric_fields, test_id)
    if int(row["n_scenarios"]) != 6:
        raise ValueError(f"{test_id}: n_scenarios must equal the frozen six-scenario suite")
    classification = row.get("classification")
    if classification not in EXPECTED_CLASSIFICATIONS:
        raise ValueError(f"{test_id}: unsupported classification {classification!r}")
    activated = row.get("intervention_activated")
    if not isinstance(activated, bool):
        raise ValueError(f"{test_id}: intervention_activated must be boolean")
    if (classification == "intervention_not_activated") != (not activated):
        raise ValueError(f"{test_id}: activation state and classification disagree")
    diagnostics = row.get("activation_diagnostics_summary")
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("intervention_activated") is not activated
    ):
        raise ValueError(f"{test_id}: activation diagnostics disagree with decision row")
    return row


def _validate_synthesis(
    synthesis: Mapping[str, Any],
    *,
    synthesis_sha256: str,
    recovery_manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int, dict[str, float]]:
    """Validate immutable custody, the frozen grid, and every source decision row."""

    recorded_sha = recovery_manifest["local_artifact_sha256"]["synthesis.json"]
    if synthesis_sha256 != recorded_sha:
        raise ValueError(
            "synthesis digest does not match the reviewed recovery manifest "
            f"(observed {synthesis_sha256}, expected {recorded_sha})"
        )
    required_header = {
        "schema_version": EXPECTED_SYNTHESIS_SCHEMA,
        "per_cell_count": 2160,
        "native_cell_count": 2160,
        "excluded_cell_count": 0,
        "all_native": True,
        "grid_complete": True,
        "evidence_status": EXPECTED_EVIDENCE_STATUS,
    }
    for field, expected in required_header.items():
        if synthesis.get(field) != expected:
            raise ValueError(
                f"synthesis {field} mismatch: observed {synthesis.get(field)!r}, "
                f"expected {expected!r}"
            )

    expected_ids, paired_denominator, thresholds = _expected_design(preregistration)
    rows = synthesis.get("decision_table")
    if not isinstance(rows, list):
        raise ValueError("synthesis decision_table must be a list")
    test_ids = [row.get("test_id") for row in rows if isinstance(row, Mapping)]
    if len(rows) != 24 or len(test_ids) != 24:
        raise ValueError("synthesis must contain exactly 24 decision rows")
    duplicates = sorted(item for item, count in Counter(test_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"synthesis contains duplicate test IDs: {duplicates}")
    observed_ids = set(test_ids)
    if observed_ids != expected_ids:
        raise ValueError(
            "synthesis contrast roster mismatch; "
            f"missing={sorted(expected_ids - observed_ids)}, "
            f"unexpected={sorted(observed_ids - expected_ids)}"
        )

    expected_counts = recovery_manifest["descriptive_synthesis"]["classification_counts"]
    observed_counts: Counter[str] = Counter()
    numeric_fields = (
        "n_scenarios",
        "pooled_delta_mean",
        "pooled_delta_se",
        "harm_bound_unadjusted",
        "noninferiority_bound_unadjusted",
        "harm_bound",
        "noninferiority_bound",
        "harm_adjusted_confidence_level",
        "noninferiority_adjusted_confidence_level",
        "p_value_harm_raw",
        "p_value_harm_holm",
        "p_value_noninferiority_raw",
        "p_value_noninferiority_holm",
        "familywise_alpha",
        "directional_family_alpha",
    )
    validated_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        row = _validate_synthesis_row(raw_row, numeric_fields=numeric_fields)
        classification = str(row["classification"])
        observed_counts[str(classification)] += 1
        validated_rows.append(row)
    if dict(observed_counts) != expected_counts:
        raise ValueError(
            f"classification accounting mismatch: {dict(observed_counts)} != {expected_counts}"
        )
    return (
        sorted(validated_rows, key=lambda item: str(item["test_id"])),
        paired_denominator,
        thresholds,
    )


def _composite_bounds(row: Mapping[str, Any], test_id: str) -> tuple[float, float]:
    """Return the exact lower and upper adjusted one-sided bounds without merging tests."""

    bounds: dict[str, float] = {}
    for prefix in ("harm", "noninferiority"):
        bound_type = row.get(f"{prefix}_bound_type")
        value = row.get(f"{prefix}_bound")
        if bound_type not in {"lower", "upper"} or not isinstance(value, (int, float)):
            raise ValueError(f"{test_id}: invalid {prefix} bound")
        if bound_type in bounds:
            raise ValueError(f"{test_id}: directional bounds do not provide lower and upper sides")
        bounds[str(bound_type)] = float(value)
    if set(bounds) != {"lower", "upper"}:
        raise ValueError(
            f"{test_id}: directional bounds must contain one lower and one upper bound"
        )
    return bounds["lower"], bounds["upper"]


def _source_binding(
    row: Mapping[str, Any],
    *,
    synthesis_sha256: str,
    paired_denominator: int,
    harm_threshold: float,
) -> str:
    """Encode one complete canonical row in the packet's versioned sensitivity binding."""

    binding = {
        "canonical_decision_row": row,
        "harm_threshold": harm_threshold,
        "paired_denominator": paired_denominator,
        "preregistration": {
            "path": "configs/benchmarks/issue_5578_robot_speed_tier_preregistration.yaml",
            "schema_version": "robot_sf.issue_5578_robot_speed_tier_preregistration.v1",
        },
        "source_artifact": {
            "artifact": "ll7/robot_sf/campaign-issue5578-native-speed-tier-job-13828:v0",
            "artifact_location": (
                "wandb-artifact://ll7/robot_sf/"
                "campaign-issue5578-native-speed-tier-job-13828:v0/synthesis.json"
            ),
            "artifact_path": "synthesis.json",
            "member": "synthesis.json",
            "sha256": synthesis_sha256,
        },
    }
    return BINDING_PREFIX + json.dumps(
        binding, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def decode_source_binding(value: str) -> dict[str, Any]:
    """Decode one issue #7980 source-binding sensitivity value."""

    if not value.startswith(BINDING_PREFIX):
        raise ValueError("missing issue #7980 source-binding prefix")
    payload = json.loads(value.removeprefix(BINDING_PREFIX))
    if not isinstance(payload, dict):
        raise ValueError("issue #7980 source binding must be a JSON object")
    return payload


def _metric_and_decision(
    row: Mapping[str, Any],
    *,
    synthesis_sha256: str,
    paired_denominator: int,
    harm_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project one canonical synthesis row into one metric and one fail-closed decision."""

    test_id = str(row["test_id"])
    metric_name = str(row["metric"])
    lower, upper = _composite_bounds(row, test_id)
    sensitivity = [
        _source_binding(
            row,
            synthesis_sha256=synthesis_sha256,
            paired_denominator=paired_denominator,
            harm_threshold=harm_threshold,
        )
    ]
    uncertainty = {
        "declared": True,
        "method": "paired_seed_block_two_directional_one_sided_holm_bounds",
        "ci_low": lower,
        "ci_high": upper,
        "p_value_raw": None,
        "p_value_adjusted": None,
    }
    multiplicity = {
        "declared": True,
        "method": "holm_bonferroni_per_planner_directional_family",
        "n_comparisons": 6,
    }
    desirability = {
        "success_rate": "higher_is_better",
        "collision_rate": "lower_is_better",
        "near_miss_rate": "lower_is_better",
    }[metric_name]
    metric = {
        "metric_id": test_id,
        "source_ids": ["recovery_manifest"],
        "unit": "paired_rate_delta",
        "desirability": desirability,
        "support": paired_denominator,
        "denominator": paired_denominator,
        "support_threshold": paired_denominator,
        "missingness": "complete",
        "unavailable_handling": (
            "fail_closed"
            if row["classification"] == "intervention_not_activated"
            else "diagnostic_only"
        ),
        "effect": float(row["pooled_delta_mean"]),
        "uncertainty": uncertainty,
        "null_value": harm_threshold,
        "multiplicity": multiplicity,
        "sensitivity": sensitivity,
    }
    comparator = {
        "reference": "cap_2_0_nominal",
        "comparison": str(row["speed_tier_id"]),
        "direction": "comparison_minus_reference",
    }
    contrast = {
        "comparator": comparator,
        "effect": metric["effect"],
        "support": paired_denominator,
        "denominator": paired_denominator,
        "support_threshold": paired_denominator,
        "null_value": harm_threshold,
        "uncertainty": uncertainty,
        "multiplicity": multiplicity,
    }
    is_invalid = row["classification"] == "intervention_not_activated"
    decision = {
        "decision_id": f"d_{test_id}",
        "metric_id": test_id,
        "outcome": "invalid" if is_invalid else "inconclusive",
        "rationale": (
            f"Canonical classification {row['classification']!r} is preserved for {test_id}; "
            "this source-complete packet remains diagnostic_only and grants no admission."
        ),
        "comparator": comparator,
        "contrast_result": contrast,
        "effect": metric["effect"],
        "refusal_reason": (
            "Speed intervention did not activate; the contrast is invalid for interpretation."
            if is_invalid
            else "A separate domain-aware admission decision is required."
        ),
    }
    return metric, decision


def build_packet(
    *,
    synthesis_path: Path,
    recovery_manifest_path: Path,
    previous_packet_path: Path,
    preregistration_path: Path,
    producer_commit: str,
) -> dict[str, Any]:
    """Build and semantically validate the source-complete diagnostic packet."""

    synthesis_path = synthesis_path.resolve()
    recovery_manifest_path = _repo_path(recovery_manifest_path)
    previous_packet_path = _repo_path(previous_packet_path)
    preregistration_path = _repo_path(preregistration_path)
    synthesis = _load_json(synthesis_path)
    recovery_manifest = _load_json(recovery_manifest_path)
    previous_packet = _load_json(previous_packet_path)
    preregistration = _read_preregistration(preregistration_path)
    synthesis_sha256 = _sha256(synthesis_path)
    rows, paired_denominator, thresholds = _validate_synthesis(
        synthesis,
        synthesis_sha256=synthesis_sha256,
        recovery_manifest=recovery_manifest,
        preregistration=preregistration,
    )

    recovery_source = next(
        source
        for source in previous_packet["sources"]
        if source["source_id"] == "recovery_manifest"
    )
    metrics: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for row in rows:
        metric, decision = _metric_and_decision(
            row,
            synthesis_sha256=synthesis_sha256,
            paired_denominator=paired_denominator,
            harm_threshold=thresholds[str(row["metric"])],
        )
        metrics.append(metric)
        decisions.append(decision)

    counts = Counter(str(row["classification"]) for row in rows)
    forbidden = [
        "Planner ranking or planner superiority claim.",
        "General safety, realism, causal, or population claim.",
        "Claim that prediction_planner is insensitive to the speed cap.",
        "Dissertation, release, or paper-facing admission claim.",
    ]
    packet = {
        "schema_version": "result_interpretation_packet.v1",
        "packet_id": "issue_7980_robot_speed_tier_contrast_binding_diagnostic",
        "question": {
            "question_id": "q_7980_robot_speed_tier_contrast_binding",
            "text": (
                "What do the exact 24 source-bound speed-tier contrasts establish before any "
                "separate benchmark-admission decision?"
            ),
            "issue_refs": [5578, 6102, 7980],
        },
        "evidence": {
            "evidence_id": "issue_7980_robot_speed_tier_contrast_binding",
            "tier": "smoke_diagnostic",
            "admission_state": "diagnostic_only",
            "rationale": (
                "All registered statistics are source-complete and custody-bound, but this "
                "packet intentionally preserves the existing non-admitted claim boundary."
            ),
        },
        "sources": [dict(recovery_source)],
        "population": previous_packet["population"],
        "execution_mode": previous_packet["execution_mode"],
        "estimand": {
            "estimand_id": "registered_speed_tier_contrast_source_binding",
            "analysis_unit": "planner_speed_tier_metric_contrast",
            "resampling_unit": "paired_seed_block",
            "description": (
                "Each non-nominal tier minus cap_2_0_nominal contrast is conditioned on the "
                "six fixed declared scenarios and 30 paired seeds."
            ),
            "pairing_key": "planner_id,scenario_id,seed",
            "clustering_key": "scenario_id",
            "contrast_direction": "non_nominal_tier_minus_cap_2_0_nominal",
        },
        "metrics": metrics,
        "decisions": decisions,
        "figure_links": [],
        "caption_assertions": [],
        "claim_boundary": {
            "allowed": [
                "The immutable synthesis contains exactly 24 registered contrast rows.",
                (
                    "The source-complete classification accounting is "
                    f"{counts['no_material_shift']} no_material_shift, "
                    f"{counts['inconclusive']} inconclusive, and "
                    f"{counts['intervention_not_activated']} intervention_not_activated."
                ),
                (
                    "All six intervention_not_activated rows are retained as invalid for a "
                    "speed-effect interpretation."
                ),
            ],
            "forbidden": forbidden,
        },
        "producer": {
            "actor_id": "codex_issue_7980_packet_builder",
            "commit": producer_commit,
            "command": (
                "uv run python scripts/analysis/build_issue_7980_speed_tier_packet.py "
                "--synthesis <verified-wandb-v0-synthesis.json>"
            ),
            "status": "draft",
        },
        "findings": [
            "All 24 registered planner-by-tier-by-metric contrasts are present exactly once.",
            "Every contrast binds the pooled effect, paired denominator, both directional tests and bounds, multiplicity, activation state, and immutable synthesis digest.",
            "The canonical 10/8/6 classification accounting reconciles exactly.",
            "The six non-activated prediction-planner contrasts remain invalid exclusions.",
        ],
        "limitations": [
            "No activated contrast is promoted above inconclusive by this diagnostic packet.",
            "The six prediction-planner contrasts cannot answer a speed-effect question because the intervention did not activate.",
            "The fixed six-scenario suite does not support unbounded scenario-population, causal, safety, ranking, dissertation, or paper-facing claims.",
            "A separate domain-aware decision is required before any bounded simulator-defined outcome is admitted.",
        ],
        "fail_closed_changes": [
            "Missing, duplicate, non-native, digest-mismatched, or activation-inconsistent source rows stop packet generation.",
            "All activated source classifications remain non-admitted inconclusive decisions.",
            "All non-activated source classifications remain invalid decisions.",
            "Fallback and degraded execution remain forbidden and absent.",
        ],
        "forbidden_claims": forbidden,
    }
    errors = validate_packet(packet)
    if errors:
        raise ValueError("generated packet failed validation:\n- " + "\n- ".join(errors))
    return packet


def _checksum_manifest_text(paths: Sequence[Path], destination: Path) -> str:
    """Return a marked checksum manifest with paths relative to its directory."""

    lines = []
    for path in sorted(paths, key=lambda item: item.name):
        if path.parent.resolve() != destination.parent.resolve():
            raise ValueError("issue #7980 checksum outputs must share one directory")
        lines.append(f"{_sha256(path)}  {path.name}")
    return review_marker_comment() + "\n" + "\n".join(lines) + "\n"


def _review_sidecar_payload(artifact: Path) -> dict[str, Any]:
    """Return the exact shared-writer sidecar payload expected for one artifact."""

    return {
        "artifact_path": artifact.resolve().relative_to(_REPO_ROOT.resolve()).as_posix(),
        "artifact_sha256": _sha256(artifact),
        "preserved_exact_bytes": True,
        "review_marker": review_marker_json(),
        "schema_version": REVIEW_SIDECAR_SCHEMA_VERSION,
    }


def _review_sidecar_path(artifact: Path) -> Path:
    """Return the canonical shared-writer review-sidecar path."""

    return artifact.with_name(artifact.name + ".review.json")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, default=DEFAULT_RECOVERY_MANIFEST)
    parser.add_argument("--previous-packet", type=Path, default=DEFAULT_PREVIOUS_PACKET)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--caption-output", type=Path, default=DEFAULT_CAPTION)
    parser.add_argument("--checksum-output", type=Path, default=DEFAULT_CHECKSUM)
    parser.add_argument("--producer-commit", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare regenerated bytes with the existing output instead of writing files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build or deterministically check the issue #7980 packet."""

    args = _parse_args(argv)
    output = _repo_path(args.output)
    caption_output = _repo_path(args.caption_output)
    checksum_output = _repo_path(args.checksum_output)
    producer_commit = args.producer_commit
    if producer_commit is None and args.check and output.is_file():
        producer_commit = _load_json(output)["producer"]["commit"]
    producer_commit = producer_commit or _producer_base_commit()
    packet = build_packet(
        synthesis_path=args.synthesis,
        recovery_manifest_path=args.recovery_manifest,
        previous_packet_path=args.previous_packet,
        preregistration_path=args.preregistration,
        producer_commit=producer_commit,
    )
    expected_packet = json.dumps(packet, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if args.check:
        actual_packet = output.read_text(encoding="utf-8")
        if actual_packet != expected_packet:
            print(f"error: regenerated packet differs from {output}", file=sys.stderr)
            return 1
        loaded = load_result_interpretation_packet(output)
        expected_caption = review_marker("robot_sf#7980") + "\n" + render_caption(loaded)
        if caption_output.read_text(encoding="utf-8") != expected_caption:
            print(f"error: regenerated caption differs from {caption_output}", file=sys.stderr)
            return 1
        expected_checksums = _checksum_manifest_text((caption_output, output), checksum_output)
        if checksum_output.read_text(encoding="utf-8") != expected_checksums:
            print(f"error: checksum manifest differs from {checksum_output}", file=sys.stderr)
            return 1
        for artifact in (output, caption_output, checksum_output):
            sidecar = _review_sidecar_path(artifact)
            if _load_json(sidecar) != _review_sidecar_payload(artifact):
                print(f"error: review sidecar differs from {sidecar}", file=sys.stderr)
                return 1
        print(f"packet check passed: {compute_packet_digest(loaded)}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_json(packet, output)
    loaded = load_result_interpretation_packet(output)
    write_text(caption_output, render_caption(loaded), issue_ref="robot_sf#7980")
    write_text(
        checksum_output,
        _checksum_manifest_text((output, caption_output), checksum_output),
    )
    for artifact in (output, caption_output, checksum_output):
        write_review_sidecar(artifact, repo_root=_REPO_ROOT)
    print(f"written {output.relative_to(_REPO_ROOT)}")
    print(f"packet_digest: {compute_packet_digest(loaded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

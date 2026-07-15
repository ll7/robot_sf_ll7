"""Matched calibration comparison for online collision-risk estimators (issue #5445).

This CLI runs the calibration harness in
:mod:`robot_sf.research.collision_risk.calibration` over a **frozen, preregistered**
comparison packet. Every estimator scores identical scenario histories, candidate
actions, footprints, and horizons; each is graded against the same realized
collision outcome; and each receives a ``use online`` / ``offline analysis only``
/ ``revise`` / ``stop`` verdict.

The realized labels are generated from a declared constant-velocity forecast
model (optionally *misspecified* per scenario family), so this is **API + fixture
evidence**: it validates the calibration machinery, demonstrates self-consistency
of the constant-velocity Monte Carlo estimator against its own model, and proves
the machinery detects miscalibration. It is **not** calibrated benchmark risk for
the simulator distribution and never a real-world risk claim. A real-distribution
run requires eligible simulator traces and is out of scope until an approved
compute packet exists (issue #5445 stop rule).

The report is written to the git-ignored ``output/`` tree by default; it changes
no benchmark metric semantics and emits no ``safe`` verdict.

Example:
    uv run python scripts/analysis/collision_risk_calibration_report.py \
        --config configs/analysis/issue_5445_matched_calibration.yaml --print-summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.nav.predictive_types import PedestrianState
from robot_sf.research.collision_risk import RiskEstimatorConfig, action_from_constant_velocity
from robot_sf.research.collision_risk.calibration import (
    AVAILABLE_ESTIMATORS,
    CALIBRATION_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    UNAVAILABLE_ESTIMATORS,
    CalibrationInputError,
    FamilySpec,
    MatchedActionPair,
    MatchedDatasetProvenance,
    VerdictThresholds,
    action_sensitivity,
    evaluate_estimator,
    evaluate_matched_action_sensitivity,
    generate_matched_dataset,
)

DEFAULT_CONFIG = Path("configs/analysis/issue_5445_matched_calibration.yaml")
DEFAULT_OUTPUT = Path("output/collision_risk/issue_5445_calibration_report.json")


def _load_config(path: Path) -> dict[str, Any]:
    """Load the YAML comparison packet, failing closed on missing blocks."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    required = {"estimator", "provenance", "families", "evaluation"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise CalibrationInputError(f"config {path} must define blocks: {sorted(required)}")
    return data


def _build_estimator_config(estimator: dict[str, Any]) -> RiskEstimatorConfig:
    """Build a :class:`RiskEstimatorConfig` from the config block."""
    fields = {f.name for f in RiskEstimatorConfig.__dataclass_fields__.values()}
    return RiskEstimatorConfig(**{key: value for key, value in estimator.items() if key in fields})


def _build_families(entries: list[dict[str, Any]]) -> list[FamilySpec]:
    """Build the scenario-family specs from the config block."""
    families = []
    for entry in entries:
        families.append(
            FamilySpec(
                name=str(entry["name"]),
                n_scenarios=int(entry["n_scenarios"]),
                gt_velocity_std_m_s=float(entry["gt_velocity_std_m_s"]),
                gt_velocity_bias_m_s=tuple(entry.get("gt_velocity_bias_m_s", (0.0, 0.0))),
                n_pedestrians=int(entry.get("n_pedestrians", 4)),
                robot_speed_range=tuple(entry.get("robot_speed_range", (0.6, 1.4))),
                ped_spawn_radius_m=float(entry.get("ped_spawn_radius_m", 3.0)),
                strata=dict(entry.get("strata", {})),
            )
        )
    return families


def _build_provenance(block: dict[str, Any], seed: int, n_samples: int) -> MatchedDatasetProvenance:
    """Build the preregistered provenance record, failing closed on gaps."""
    required = (
        "target_distribution",
        "collision_predicate",
        "horizon_steps",
        "dt_s",
        "planner_rows",
        "discovery_calibration_split",
        "prevalence_note",
        "compute_cap",
        "stop_rule",
    )
    missing = [key for key in required if not str(block.get(key, "")).strip()]
    if missing:
        raise CalibrationInputError(
            f"provenance is incomplete; preregister these fields before scoring: {missing}"
        )
    return MatchedDatasetProvenance(
        target_distribution=str(block["target_distribution"]),
        collision_predicate=str(block["collision_predicate"]),
        horizon_steps=int(block["horizon_steps"]),
        dt_s=float(block["dt_s"]),
        planner_rows=tuple(str(row) for row in block["planner_rows"]),
        discovery_calibration_split=str(block["discovery_calibration_split"]),
        prevalence_note=str(block["prevalence_note"]),
        n_samples=n_samples,
        compute_cap=str(block["compute_cap"]),
        stop_rule=str(block["stop_rule"]),
        seed=seed,
    )


def _vector2(value: object, field_name: str) -> tuple[float, float]:
    """Parse one finite two-dimensional vector from the YAML packet."""
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f"{field_name} must be a finite 2D vector") from exc
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise CalibrationInputError(f"{field_name} must be a finite 2D vector")
    return float(array[0]), float(array[1])


def _build_action_sensitivity_pairs(
    entries: object, estimator_config: RiskEstimatorConfig
) -> list[MatchedActionPair]:
    """Build typed same-state action pairs from the preregistered packet."""
    if not isinstance(entries, list):
        raise CalibrationInputError("evaluation.action_sensitivity.pairs must be a list")

    pairs: list[MatchedActionPair] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CalibrationInputError(f"action-sensitivity pair {index} must be a mapping")
        pair_id = str(entry.get("pair_id", "")).strip()
        if not pair_id:
            raise CalibrationInputError(f"action-sensitivity pair {index} has no pair_id")
        actor_entries = entry.get("pedestrians")
        if not isinstance(actor_entries, list) or not actor_entries:
            raise CalibrationInputError(f"action-sensitivity pair {pair_id!r} needs pedestrians")

        pedestrians: list[PedestrianState] = []
        actor_ids: set[int] = set()
        for actor_index, actor in enumerate(actor_entries):
            if not isinstance(actor, dict):
                raise CalibrationInputError(
                    f"action-sensitivity pair {pair_id!r} actor {actor_index} must be a mapping"
                )
            try:
                actor_id = int(actor["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CalibrationInputError(
                    f"action-sensitivity pair {pair_id!r} actor {actor_index} needs an integer id"
                ) from exc
            if actor_id in actor_ids:
                raise CalibrationInputError(
                    f"action-sensitivity pair {pair_id!r} repeats actor id {actor_id}"
                )
            actor_ids.add(actor_id)
            pedestrians.append(
                PedestrianState(
                    id=actor_id,
                    position=np.asarray(
                        _vector2(actor.get("position"), f"{pair_id}.pedestrians.position"),
                        dtype=float,
                    ),
                    velocity=np.asarray(
                        _vector2(actor.get("velocity"), f"{pair_id}.pedestrians.velocity"),
                        dtype=float,
                    ),
                )
            )

        start_position = _vector2(entry.get("start_position"), f"{pair_id}.start_position")
        higher_velocity = _vector2(
            entry.get("higher_risk_velocity"), f"{pair_id}.higher_risk_velocity"
        )
        lower_velocity = _vector2(
            entry.get("lower_risk_velocity"), f"{pair_id}.lower_risk_velocity"
        )
        pairs.append(
            MatchedActionPair(
                pair_id=pair_id,
                pedestrians=tuple(pedestrians),
                higher_risk_action=action_from_constant_velocity(
                    f"{pair_id}:higher_risk",
                    start_position,
                    higher_velocity,
                    horizon_steps=estimator_config.horizon_steps,
                    dt_s=estimator_config.dt_s,
                ),
                lower_risk_action=action_from_constant_velocity(
                    f"{pair_id}:lower_risk",
                    start_position,
                    lower_velocity,
                    horizon_steps=estimator_config.horizon_steps,
                    dt_s=estimator_config.dt_s,
                ),
            )
        )
    return pairs


def build_report(config_path: Path) -> dict[str, Any]:
    """Run the frozen comparison packet and return a JSON-safe report dict."""
    data = _load_config(config_path)
    estimator_config = _build_estimator_config(data["estimator"])
    families = _build_families(data["families"])
    evaluation = data["evaluation"]

    n_samples = sum(family.n_scenarios for family in families)
    min_samples = int(evaluation.get("min_eligible_samples", 1))
    if n_samples < min_samples:
        # Fail closed: the #5445 stop rule forbids scoring on insufficient traces.
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "status": "insufficient_traces",
            "config": str(config_path),
            "n_samples": n_samples,
            "min_eligible_samples": min_samples,
            "message": (
                "eligible matched-sample count is below the preregistered minimum; "
                "stopping before scoring per the issue #5445 stop rule"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    seed = int(data["estimator"].get("seed", 0))
    provenance = _build_provenance(data["provenance"], seed=seed, n_samples=n_samples)
    dataset = generate_matched_dataset(families, estimator_config, provenance)

    thresholds_block = evaluation.get("verdict_thresholds", {})
    thresholds = VerdictThresholds(
        ece_use_online_max=float(thresholds_block.get("ece_use_online_max", 0.05)),
        ece_revise_max=float(thresholds_block.get("ece_revise_max", 0.15)),
        min_ap_over_prevalence=float(thresholds_block.get("min_ap_over_prevalence", 1.5)),
        deadline_ms=estimator_config.deadline_ms,
    )

    estimator_results = [
        evaluate_estimator(
            estimator_id,
            dataset,
            estimator_config,
            n_reliability_bins=int(evaluation.get("n_reliability_bins", 10)),
            fnr_thresholds=tuple(evaluation.get("fnr_thresholds", (0.1, 0.25, 0.5))),
            n_bootstrap=int(evaluation.get("n_bootstrap", 200)),
            thresholds=thresholds,
        )
        for estimator_id in AVAILABLE_ESTIMATORS
    ]

    unavailable = [
        {
            "estimator_id": estimator_id,
            "available": False,
            "reason": reason,
            "verdict": "unavailable",
        }
        for estimator_id, reason in sorted(UNAVAILABLE_ESTIMATORS.items())
    ]

    action_sensitivity_block = evaluation.get("action_sensitivity")
    if action_sensitivity_block is None:
        # Preserve the v1 packet reader for older local packets while requiring
        # the current issue packet to declare typed same-state pairs below.
        action_pairs = [tuple(pair) for pair in evaluation.get("action_pairs", [])]
        sensitivity = (
            action_sensitivity(dataset, estimator_config, action_pairs) if action_pairs else None
        )
    else:
        if not isinstance(action_sensitivity_block, dict):
            raise CalibrationInputError("evaluation.action_sensitivity must be a mapping")
        min_pairs = int(action_sensitivity_block.get("min_pairs", 1))
        if min_pairs <= 0:
            raise CalibrationInputError("evaluation.action_sensitivity.min_pairs must be positive")
        action_pairs = _build_action_sensitivity_pairs(
            action_sensitivity_block.get("pairs"), estimator_config
        )
        if len(action_pairs) < min_pairs:
            raise CalibrationInputError(
                "action-sensitivity fixture has fewer pairs than its preregistered minimum: "
                f"{len(action_pairs)} < {min_pairs}"
            )
        sensitivity = evaluate_matched_action_sensitivity(action_pairs, estimator_config)
        sensitivity["min_pairs"] = min_pairs

    # Per-family stratified prevalence where denominators permit.
    strata_prevalence: dict[str, dict[str, float]] = {}
    for family in families:
        subset = [s for s in dataset.samples if s.family == family.name]
        positives = sum(1 for s in subset if s.realized_contact)
        strata_prevalence[family.name] = {
            "n": len(subset),
            "prevalence": (positives / len(subset)) if subset else float("nan"),
        }

    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "scored",
        "config": str(config_path),
        "estimator_config_hash": estimator_config.config_hash(),
        "provenance": provenance.to_dict(),
        "overall_prevalence": dataset.prevalence(),
        "stratified_prevalence": strata_prevalence,
        "estimators": estimator_results,
        "unavailable_estimators": unavailable,
        "action_sensitivity": sensitivity,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _print_summary(report: dict[str, Any]) -> None:
    """Print a compact human-readable summary of the calibration report."""
    if report.get("status") != "scored":
        print(f"status: {report.get('status')} -- {report.get('message', '')}")
        return
    print(f"overall prevalence: {report['overall_prevalence']:.3f}")
    for result in report["estimators"]:
        line = (
            f"  {result['estimator_id']} ({result['kind']}): verdict={result['verdict']} "
            f"AP={result['average_precision']:.3f} "
            f"latency={result['latency']['classification']}"
        )
        if result["kind"] == "probabilistic":
            line += (
                f" ECE={result['expected_calibration_error']:.3f} "
                f"Brier={result['brier_score']:.3f} (baseline {result['baseline_brier']:.3f})"
            )
        print(line)
    for entry in report["unavailable_estimators"]:
        print(f"  {entry['estimator_id']}: unavailable -- {entry['reason']}")
    if report["action_sensitivity"] is not None:
        sens = report["action_sensitivity"]
        print(
            f"action sensitivity: {sens['fraction_correct']:.2f} of {sens['n_pairs']} pairs "
            "in the expected direction"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build and write the matched calibration report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a compact human-readable summary to stdout.",
    )
    args = parser.parse_args(argv)

    report = build_report(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"wrote {args.output}")
    _print_summary(report)
    # Fail closed with a non-zero code when scoring was refused.
    return 0 if report.get("status") == "scored" else 2


if __name__ == "__main__":
    raise SystemExit(main())

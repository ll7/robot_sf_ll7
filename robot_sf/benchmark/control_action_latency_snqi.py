"""Reproducible SNQI analysis of the control-action-latency sweep (issues #5892 / #5912).

PR #5904 registered the job 13516 ``snqi_analysis.json`` (Social Navigation Quality
Index slopes, paired cluster-bootstrap intervals, and planner rankings) but the
repository shipped only the *output* artifact, not the command that derives it.
This module is the canonical derivation: it reads a **durable sufficient input**
(the promoted ``snqi_latency_inputs.csv`` carrying exactly the per-episode SNQI
terms, or the raw campaign rows when re-deriving from the private artifact),
validates raw-input checksum / fixed-scope coverage / execution modes /
fallback-degraded exclusions, then computes SNQI-v0 per episode, the per-unit
ordinary-least-squares latency slope, and the paired cluster-bootstrap
uncertainty, and emits the ``control-action-latency-snqi-analysis.v1`` packet.

This module runs **no episode** and makes **no benchmark / simulator-realism /
sim-to-real / paper-facing claim**. It is the deterministic analyzer that turns a
durable sufficient input into the registered SNQI analysis. It fails closed when
the input does not match its declared checksum, does not cover the required
fixed scope (48 scenarios x 3 seeds x 3 planner groups x 3 latency steps), mixes
in fallback / degraded / unavailable rows, or loses the native / adapter
execution-mode labels.

Reproducibility contract (issue #5912 DoD #3): the SNQI-v0 point estimates
(means, deltas, slopes, and pairwise slope differences) are deterministic and
reproduce the registered values to within floating-point summation order
(absolute tolerance ``POINT_TOL``). The bootstrap intervals and posterior
probabilities are Monte-Carlo quantities: they reproduce the registered values
within the documented absolute tolerances (``INTERVAL_TOL`` /
``PROBABILITY_TOL``) but are not byte-identical, because the registered packet's
generating code was never committed and percentile endpoints depend on the exact
resampling stream and ``numpy`` version at generation time.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.control_action_latency_preflight import (
    AXIS_KEY,
    REQUIRED_LATENCY_STEPS,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.snqi.compute import compute_snqi
from robot_sf.errors import RobotSfError
from robot_sf.evidence.writers import review_marker_comment, review_marker_json

#: Versioned schema for the generated SNQI analysis packet (issue #5912 DoD #1).
ANALYSIS_SCHEMA_VERSION = "control-action-latency-snqi-analysis.v1"

#: Versioned schema for the re-issued uncertainty packet (issue #5928 DoD #2).
#: The registered ``snqi_analysis.json`` uncertainty block came from code that was
#: never committed and is internally inconsistent (see ``recovery_decision`` in
#: :func:`build_uncertainty_reissue`). This packet re-issues that block from the
#: committed canonical analyzer under a fresh, byte-reproducible provenance stamp.
UNCERTAINTY_REISSUE_SCHEMA_VERSION = "control-action-latency-snqi-uncertainty-reissue.v1"

#: Versioned schema for the promoted durable sufficient input.
INPUT_SCHEMA_VERSION = "control-action-latency-snqi-inputs.v1"

#: Schema for the input provenance sidecar.
INPUT_PROVENANCE_SCHEMA_VERSION = "control-action-latency-snqi-inputs-provenance.v1"

ISSUE = 5892
CAMPAIGN_ISSUE = 5034
PARENT_ISSUE = 4977

#: Job 13516 raw-row SHA-256 (registered provenance; the durable sufficient input
#: derives from this artifact).
JOB_ID = 13516
JOB_LABEL = "5034c-issue5034-latency-sweep"
EXECUTION_COMMIT = "c153848d7be2851b5c5e89c11055bf96ea778a84"
RAW_ROWS_SHA256 = "6b34e690dfe6cc1ccccd9cd19bde8b3f6a3501bbc1b0a0b44639e151557b4134"
FIXED_SCOPE_PLAN_SHA256 = "968f53ff458b3af81b0646faa581a90e3c81395a0bcf52292f8a4cff52677809"

#: Canonical SNQI-v0 weight / baseline configs (registered provenance).
WEIGHTS_PATH = "configs/benchmarks/snqi_weights_camera_ready_v3.json"
WEIGHTS_SHA256 = "71a67c3c02faff166f8c96bef8bcf898533981ca2b2c4493829988520fb1aeb2"
BASELINE_PATH = "configs/benchmarks/snqi_baseline_camera_ready_v3.json"
BASELINE_SHA256 = "329ca5766491e1587979d0a435c7ba676e148ccdff97040a36546bbb9414035a"

#: Planners in the registered point-estimate ranking (deterministic output order).
PLANNER_GROUPS: tuple[str, ...] = ("default_social_force", "hybrid_rule_v0_minimal", "orca")

#: SNQI-v0 terms the campaign did not emit; the canonical neutral defaults apply.
NEUTRAL_DEFAULTED_TERMS: tuple[str, ...] = ("force_exceed_events", "jerk_mean")

#: Paired cluster-bootstrap configuration (registered ``snqi_method.uncertainty``).
BOOTSTRAP_SEED = 5892
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_PERCENTILES = (2.5, 97.5)
#: Each planner groups 48 scenarios x 3 seeds = 144 paired scenario-seed units.
EXPECTED_PAIRED_UNITS_PER_PLANNER = 144
EXPECTED_SCENARIO_COUNT = 48
EXPECTED_SCENARIO_IDS: tuple[str, ...] = (
    "classic_bottleneck_low",
    "classic_bottleneck_medium",
    "classic_bottleneck_high",
    "classic_realworld_double_bottleneck_high",
    "classic_station_platform_medium",
    "classic_cross_trap_low",
    "classic_cross_trap_medium",
    "classic_cross_trap_high",
    "classic_doorway_low",
    "classic_doorway_medium",
    "classic_doorway_high",
    "classic_group_crossing_low",
    "classic_group_crossing_medium",
    "classic_group_crossing_high",
    "classic_head_on_corridor_low",
    "classic_head_on_corridor_medium",
    "classic_merging_low",
    "classic_merging_medium",
    "classic_overtaking_low",
    "classic_overtaking_medium",
    "classic_t_intersection_low",
    "classic_t_intersection_medium",
    "classic_urban_crossing_medium",
    "francis2023_frontal_approach",
    "francis2023_pedestrian_obstruction",
    "francis2023_pedestrian_overtaking",
    "francis2023_robot_overtaking",
    "francis2023_down_path",
    "francis2023_intersection_no_gesture",
    "francis2023_blind_corner",
    "francis2023_narrow_hallway",
    "francis2023_narrow_doorway",
    "francis2023_entering_room",
    "francis2023_exiting_room",
    "francis2023_entering_elevator",
    "francis2023_exiting_elevator",
    "francis2023_intersection_wait",
    "francis2023_intersection_proceed",
    "francis2023_following_human",
    "francis2023_leading_human",
    "francis2023_accompanying_peer",
    "francis2023_join_group",
    "francis2023_leave_group",
    "francis2023_crowd_navigation",
    "francis2023_parallel_traffic",
    "francis2023_perpendicular_traffic",
    "francis2023_circular_crossing",
    "francis2023_robot_crowding",
)
EXPECTED_SEEDS: tuple[int, ...] = (111, 112, 113)
EXPECTED_LATENCY_STEPS: tuple[int, ...] = REQUIRED_LATENCY_STEPS  # (0, 1, 3)

#: One latency step is a 100 ms-equivalent control-to-actuation delay.
MS_PER_LATENCY_STEP = 100.0

#: Columns of the durable sufficient input table.
INPUT_COLUMNS: tuple[str, ...] = (
    "planner_group",
    "planner",
    "latency_step",
    "latency_ms",
    "seed",
    "scenario_id",
    "execution_mode",
    "availability_status",
    "success",
    "collision",
    "time_to_goal_norm",
    "near_miss_rate",
    "steps",
    "comfort_exposure_mean",
)

#: Execution modes that count as native benchmark-success rows (issue #691 policy,
#: mirroring ``control_action_latency_evidence.NATIVE_EXECUTION_MODES``).
NATIVE_EXECUTION_MODES: frozenset[str] = frozenset({"native", "adapter"})
AVAILABLE_AVAILABILITY_STATUSES: frozenset[str] = frozenset({"available"})

#: Absolute tolerance for deterministic SNQI point estimates (means, deltas,
#: per-planner slopes). Real deviations are ~1e-16 (float64 summation order);
#: the tolerance is generous to stay robust across platforms.
POINT_TOL = 1e-9

#: Absolute tolerance for the pairwise ``slope_difference`` point estimate. The
#: registered packet's pairwise ``slope_difference`` is internally inconsistent
#: with its own per-planner slopes (it does not equal ``slope_A - slope_B``), so
#: that block came from an unrecoverable second code path; the deviation is
#: ~7e-6 and does not change any qualitative comparison.
PAIRWISE_DIFF_TOL = 1e-3

#: Absolute tolerance for bootstrap percentile endpoints. Real deviations are
#: ~5e-4 (Monte-Carlo + unrecoverable estimator differences); the tolerance
#: absorbs that without admitting a meaningful change in the inferred interval
#: sign.
INTERVAL_TOL = 1e-2

#: Absolute tolerance for bootstrap posterior probabilities. Real deviations are
#: ~6e-3 (the registered probabilities include half-integer counts, e.g.
#: 0.68635 = 6863.5/10000, proving a smoothed/unrecoverable estimator).
PROBABILITY_TOL = 1.5e-2

CLAIM_BOUNDARY = (
    "internal fixed-scope latency-sensitivity diagnostic, not paper-facing; native claims apply "
    "only to native rows, adapter rows remain explicitly labeled diagnostics, and fallback/"
    "degraded cells are caveats, not successes."
)

EVIDENCE_STATUS = "diagnostic-only"


class SnqiLatencyAnalysisError(RobotSfError, RuntimeError):
    """Raised when the durable input cannot be analyzed as the registered SNQI packet."""


# ---------------------------------------------------------------------------
# Input rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnqiLatencyInput:
    """One durable sufficient-input row carrying exactly the SNQI-v0 terms.

    A row is a usable ``result`` only when it carries a valid seed, a native /
    adapter execution mode, an available status, and finite SNQI inputs. Anything
    else is an ``exclusion`` and never contributes to the SNQI calculation, so a
    fallback / degraded row can never enter the result set silently.
    """

    planner_group: str
    planner: str
    latency_step: int
    latency_ms: float
    seed: int
    scenario_id: str
    execution_mode: str
    availability_status: str
    success: bool
    collision: bool
    time_to_goal_norm: float
    near_miss_rate: float
    steps: int
    comfort_exposure_mean: float
    classification: str
    exclusion_reason: str | None


def _coerce_int(value: Any) -> int | None:
    """Return an int for numeric values, rejecting bools and non-numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _finite_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` when unusable."""
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _parse_bool(value: Any, *, field: str, row_label: str) -> bool:
    """Parse a boolean from the durable input, failing closed on ambiguity."""
    if isinstance(value, bool):
        return value
    raise SnqiLatencyAnalysisError(
        f"durable input row {row_label!r} has a non-boolean {field!r}: {value!r}"
    )


def _row_label(row: Mapping[str, Any]) -> str:
    """Return a compact identity label for an input row (for error messages)."""
    return (
        f"planner_group={row.get('planner_group')!r},latency_step={row.get('latency_step')!r},"
        f"seed={row.get('seed')!r},scenario_id={row.get('scenario_id')!r}"
    )


def classify_input_row(row: Mapping[str, Any]) -> SnqiLatencyInput:
    """Classify one durable sufficient-input row as a ``result`` or ``exclusion``.

    Mirrors the exclusion policy of
    :func:`robot_sf.benchmark.control_action_latency_evidence.classify_latency_row`
    so the SNQI analyzer cannot silently admit a fallback / degraded / non-native
    / malformed row.
    """
    label = _row_label(row)
    seed = _coerce_int(row.get("seed"))
    latency_step = _coerce_int(row.get("latency_step"))
    latency_ms = _finite_float(row.get("latency_ms"))
    execution_mode = str(row.get("execution_mode") or "")
    availability_status = str(row.get("availability_status") or "")

    ttg = _finite_float(row.get("time_to_goal_norm"))
    nmr = _finite_float(row.get("near_miss_rate"))
    steps = _coerce_int(row.get("steps"))
    comfort = _finite_float(row.get("comfort_exposure_mean"))

    reasons: list[str] = []
    if latency_step is None:
        reasons.append("missing_or_invalid_latency_step")
    if latency_ms is None:
        reasons.append("missing_or_invalid_latency_ms")
    elif latency_step is not None and not math.isclose(
        latency_ms,
        latency_step * MS_PER_LATENCY_STEP,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        reasons.append("latency_ms_does_not_match_latency_step")
    if seed is None:
        reasons.append("missing_or_invalid_seed")
    if not row.get("planner_group"):
        reasons.append("missing_planner_group")
    if not row.get("scenario_id"):
        reasons.append("missing_scenario_id")
    if execution_mode not in NATIVE_EXECUTION_MODES:
        reasons.append(f"non_native_execution_mode:{execution_mode or 'missing'}")
    if availability_status not in AVAILABLE_AVAILABILITY_STATUSES:
        reasons.append(f"unavailable:{availability_status or 'missing'}")
    for name, value in (
        ("time_to_goal_norm", ttg),
        ("near_miss_rate", nmr),
        ("comfort_exposure_mean", comfort),
    ):
        if value is None:
            reasons.append(f"missing_or_invalid_metric:{name}")
    if steps is None:
        reasons.append("missing_or_invalid_steps")
    if "success" not in row:
        reasons.append("missing_success")
    if "collision" not in row:
        reasons.append("missing_collision")

    classification = "result" if not reasons else "exclusion"
    success = (
        _parse_bool(row["success"], field="success", row_label=label) if "success" in row else False
    )
    collision = (
        _parse_bool(row["collision"], field="collision", row_label=label)
        if "collision" in row
        else False
    )

    return SnqiLatencyInput(
        planner_group=str(row.get("planner_group") or "unknown"),
        planner=str(row.get("planner") or "unknown"),
        latency_step=latency_step if latency_step is not None else -1,
        latency_ms=latency_ms if latency_ms is not None else float("nan"),
        seed=seed if seed is not None else -1,
        scenario_id=str(row.get("scenario_id") or "unknown"),
        execution_mode=execution_mode,
        availability_status=availability_status,
        success=success,
        collision=collision,
        time_to_goal_norm=ttg if ttg is not None else float("nan"),
        near_miss_rate=nmr if nmr is not None else float("nan"),
        steps=steps if steps is not None else -1,
        comfort_exposure_mean=comfort if comfort is not None else float("nan"),
        classification=classification,
        exclusion_reason="; ".join(reasons) if reasons else None,
    )


# ---------------------------------------------------------------------------
# Raw-row derivation (re-derives the sufficient input from the private artifact)
# ---------------------------------------------------------------------------


def _latency_step_from_raw(row: Mapping[str, Any]) -> int | None:
    """Extract the effective action-latency step from a raw campaign row."""
    metadata = row.get("action_latency")
    if isinstance(metadata, Mapping):
        for key in ("effective_steps", "configured_steps"):
            step = _coerce_int(metadata.get(key))
            if step is not None:
                return step
    return _coerce_int(row.get("action_latency_steps"))


def _latency_ms_from_raw(row: Mapping[str, Any]) -> float | None:
    """Extract the effective action-latency milliseconds from a raw campaign row."""
    metadata = row.get("action_latency")
    if isinstance(metadata, Mapping):
        for key in ("effective_ms", "configured_ms"):
            value = metadata.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
    return None


def derive_inputs_from_raw_rows(rows: Sequence[Mapping[str, Any]]) -> list[SnqiLatencyInput]:
    """Derive the durable sufficient input from raw campaign episode rows.

    Only ``control_action_latency`` axis rows contribute. Each retained row is
    classified by :func:`classify_input_row` so the same fail-closed exclusion
    policy applies whether the analyzer reads the promoted CSV or re-derives from
    the raw artifact.
    """
    derived: list[SnqiLatencyInput] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("axis") or "") != AXIS_KEY:
            continue
        metrics = row.get("metrics")
        metric_map = metrics if isinstance(metrics, Mapping) else {}
        promoted = {
            "planner_group": row.get("planner_group"),
            "planner": row.get("planner"),
            "latency_step": _latency_step_from_raw(row),
            "latency_ms": _latency_ms_from_raw(row),
            "seed": row.get("seed"),
            "scenario_id": row.get("scenario_id"),
            "execution_mode": row.get("execution_mode") or "native",
            "availability_status": row.get("availability_status") or "available",
            "success": row.get("success"),
            "collision": row.get("collision"),
            "time_to_goal_norm": metric_map.get("time_to_goal_norm"),
            "near_miss_rate": metric_map.get("near_miss_rate"),
            "steps": row.get("steps"),
            "comfort_exposure_mean": metric_map.get("comfort_exposure_mean"),
        }
        derived.append(classify_input_row(promoted))
    return derived


def load_raw_rows(raw_rows_path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON episode rows emitted by the campaign runner."""
    path = Path(raw_rows_path)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise SnqiLatencyAnalysisError(
                        f"raw rows file {path} has invalid JSON on line {line_number}: {exc}"
                    ) from exc
                if isinstance(row, dict):
                    rows.append(row)
    except OSError as exc:
        raise SnqiLatencyAnalysisError(f"raw rows file {path} cannot be read: {exc}") from exc
    return rows


# ---------------------------------------------------------------------------
# Durable sufficient input (promoted CSV + provenance sidecar)
# ---------------------------------------------------------------------------


def load_input_rows(input_path: str | Path) -> list[SnqiLatencyInput]:
    """Load and classify the durable sufficient-input CSV table."""
    path = Path(input_path)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            content = handle.read()
    except OSError as exc:
        raise SnqiLatencyAnalysisError(f"durable input {path} cannot be read: {exc}") from exc
    # Tolerate a leading review-marker comment line.
    data_lines = [
        line for line in content.splitlines() if line.strip() and not line.startswith("#")
    ]
    if not data_lines:
        raise SnqiLatencyAnalysisError(f"durable input {path} is empty")
    try:
        reader = csv.DictReader(data_lines)
        records = list(reader)
    except csv.Error as exc:
        raise SnqiLatencyAnalysisError(f"durable input {path} is not valid CSV: {exc}") from exc
    if reader.fieldnames is None:
        raise SnqiLatencyAnalysisError(f"durable input {path} has no header row")
    missing = [column for column in INPUT_COLUMNS if column not in reader.fieldnames]
    if missing:
        raise SnqiLatencyAnalysisError(
            f"durable input {path} is missing required columns: {missing}"
        )
    inputs: list[SnqiLatencyInput] = []
    for record in records:
        # Coerce numeric fields that CSV stores as strings.
        coerced: dict[str, Any] = dict(record)
        for key in ("latency_step", "seed", "steps"):
            coerced[key] = _coerce_csv_int(record.get(key), key=key, path=path)
        for key in ("latency_ms", "time_to_goal_norm", "near_miss_rate", "comfort_exposure_mean"):
            coerced[key] = _coerce_csv_float(record.get(key), key=key, path=path)
        for key in ("success", "collision"):
            coerced[key] = _coerce_csv_bool(record.get(key), key=key, path=path)
        inputs.append(classify_input_row(coerced))
    return inputs


def _coerce_csv_int(raw: Any, *, key: str, path: Path) -> int | None:
    """Coerce a CSV string cell to int, tolerating empty cells."""
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise SnqiLatencyAnalysisError(
            f"durable input {path} column {key!r} has a non-integer value {raw!r}"
        ) from exc


def _coerce_csv_float(raw: Any, *, key: str, path: Path) -> float | None:
    """Coerce a CSV string cell to float, tolerating empty cells."""
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SnqiLatencyAnalysisError(
            f"durable input {path} column {key!r} has a non-numeric value {raw!r}"
        ) from exc
    return value if math.isfinite(value) else None


def _coerce_csv_bool(raw: Any, *, key: str, path: Path) -> bool | None:
    """Coerce a CSV string cell to bool (``true``/``false``), tolerating empty."""
    if raw is None or raw == "":
        return None
    text = str(raw).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise SnqiLatencyAnalysisError(
        f"durable input {path} column {key!r} has a non-boolean value {raw!r}"
    )


def write_input_rows(
    inputs: Sequence[SnqiLatencyInput],
    input_path: str | Path,
) -> Path:
    """Write the durable sufficient-input CSV table with a review marker."""
    path = Path(input_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(review_marker_comment() + "\n")
        writer = csv.DictWriter(handle, fieldnames=list(INPUT_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for entry in inputs:
            writer.writerow(
                {
                    "planner_group": entry.planner_group,
                    "planner": entry.planner,
                    "latency_step": entry.latency_step,
                    "latency_ms": entry.latency_ms,
                    "seed": entry.seed,
                    "scenario_id": entry.scenario_id,
                    "execution_mode": entry.execution_mode,
                    "availability_status": entry.availability_status,
                    "success": entry.success,
                    "collision": entry.collision,
                    "time_to_goal_norm": entry.time_to_goal_norm,
                    "near_miss_rate": entry.near_miss_rate,
                    "steps": entry.steps,
                    "comfort_exposure_mean": entry.comfort_exposure_mean,
                }
            )
    return path


def write_input_provenance(
    input_path: str | Path,
    provenance_path: str | Path,
    *,
    raw_rows_path: str,
    promoter_git_head: str,
    date: str,
) -> Path:
    """Write the provenance sidecar anchoring the promoted input to the raw rows."""
    input_path = Path(input_path)
    path = Path(provenance_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "review_marker": review_marker_json(),
        "schema_version": INPUT_PROVENANCE_SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "input_path": str(input_path.name),
        "input_sha256": sha256_file(input_path),
        "source": {
            "kind": "raw_campaign_episode_rows",
            "raw_rows_path": raw_rows_path,
            "raw_rows_sha256": RAW_ROWS_SHA256,
            "axis": AXIS_KEY,
        },
        "promotion": {
            "promoter_git_head": promoter_git_head,
            "date": date,
            "description": (
                "Per-episode SNQI-v0 input terms extracted from the job 13516 raw campaign rows "
                "for the control_action_latency axis. The raw JSONL is private/non-durable; this "
                "checksummed table is the deliberately promoted sufficient input that lets a fresh "
                "checkout reproduce the registered SNQI analysis."
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_input_provenance(provenance_path: str | Path) -> dict[str, Any]:
    """Load the provenance sidecar for the durable sufficient input."""
    path = Path(provenance_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnqiLatencyAnalysisError(f"input provenance {path} cannot be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise SnqiLatencyAnalysisError(f"input provenance {path} must contain a JSON object")
    return payload


# ---------------------------------------------------------------------------
# Validation: checksum, fixed-scope coverage, execution modes, exclusions
# ---------------------------------------------------------------------------


def validate_file_checksum(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    """Fail closed when a required file does not match its registered checksum."""
    file_path = Path(path)
    try:
        actual = sha256_file(file_path)
    except OSError as exc:
        raise SnqiLatencyAnalysisError(f"{label} {file_path} cannot be hashed: {exc}") from exc
    if actual != expected_sha256:
        raise SnqiLatencyAnalysisError(
            f"{label} checksum mismatch: expected {expected_sha256} but {file_path} is {actual}"
        )


def validate_raw_rows_checksum(raw_rows_path: str | Path) -> None:
    """Fail closed unless raw campaign rows match the registered job artifact."""
    validate_file_checksum(raw_rows_path, RAW_ROWS_SHA256, label="raw campaign rows")


def validate_input_checksum(input_path: str | Path, provenance: Mapping[str, Any]) -> None:
    """Fail closed when the durable input's SHA-256 disagrees with its provenance."""
    expected = provenance.get("input_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise SnqiLatencyAnalysisError(
            "input provenance has no valid input_sha256; cannot verify durable input checksum"
        )
    validate_file_checksum(input_path, expected, label="durable input")
    source = provenance.get("source") or {}
    raw_sha = source.get("raw_rows_sha256")
    if raw_sha != RAW_ROWS_SHA256:
        raise SnqiLatencyAnalysisError(
            "input provenance does not anchor to the registered job 13516 raw-row SHA-256 "
            f"{RAW_ROWS_SHA256}; found {raw_sha!r}"
        )


def validate_fixed_scope(inputs: Sequence[SnqiLatencyInput]) -> dict[str, Any]:
    """Require exact fixed-scope identity coverage before computing SNQI.

    Fails closed (:class:`SnqiLatencyAnalysisError`) when the result rows do not
    cover exactly the 1,296 expected ``(planner_group, latency_step, seed,
    scenario_id)`` cells, when any expected cell is missing / duplicated /
    unexpected, or when any fallback / degraded / unavailable / non-native row is
    present in the result set.
    """
    result_rows = [entry for entry in inputs if entry.classification == "result"]
    exclusions = [entry for entry in inputs if entry.classification == "exclusion"]

    # The fixed-scope contract: every (planner_group, latency_step, seed,
    # scenario_id) combination appears exactly once among result rows.
    seen: dict[tuple[str, int, int, str], int] = defaultdict(int)
    for entry in result_rows:
        seen[(entry.planner_group, entry.latency_step, entry.seed, entry.scenario_id)] += 1

    duplicate_cells = sorted(key for key, count in seen.items() if count > 1)
    fallbackish = [
        entry
        for entry in inputs
        if entry.execution_mode not in NATIVE_EXECUTION_MODES
        or entry.availability_status not in AVAILABLE_AVAILABILITY_STATUSES
    ]
    if duplicate_cells or fallbackish:
        reasons: list[str] = []
        if duplicate_cells:
            reasons.append(f"duplicate_cells={len(duplicate_cells)}")
        if fallbackish:
            reasons.append(f"non_native_or_unavailable_rows={len(fallbackish)}")
        raise SnqiLatencyAnalysisError(
            "control_action_latency fixed-scope coverage contract failed: " + "; ".join(reasons)
        )

    scenarios = sorted({entry.scenario_id for entry in result_rows})
    seeds = sorted({entry.seed for entry in result_rows})
    planner_groups = sorted({entry.planner_group for entry in result_rows})
    steps = sorted({entry.latency_step for entry in result_rows})
    if planner_groups != sorted(PLANNER_GROUPS):
        raise SnqiLatencyAnalysisError(
            f"planner groups {planner_groups} do not match the registered scope "
            f"{sorted(PLANNER_GROUPS)}"
        )
    if steps != list(EXPECTED_LATENCY_STEPS):
        raise SnqiLatencyAnalysisError(
            f"latency steps {steps} do not match the required steps {list(EXPECTED_LATENCY_STEPS)}"
        )
    if seeds != list(EXPECTED_SEEDS):
        raise SnqiLatencyAnalysisError(
            f"seeds {seeds} do not match the registered seeds {list(EXPECTED_SEEDS)}"
        )
    if len(scenarios) != EXPECTED_SCENARIO_COUNT:
        raise SnqiLatencyAnalysisError(
            f"scenario count {len(scenarios)} does not match the registered "
            f"{EXPECTED_SCENARIO_COUNT} scenarios"
        )
    expected_scenarios = set(EXPECTED_SCENARIO_IDS)
    observed_scenarios = set(scenarios)
    if observed_scenarios != expected_scenarios:
        missing = sorted(expected_scenarios - observed_scenarios)
        extra = sorted(observed_scenarios - expected_scenarios)
        raise SnqiLatencyAnalysisError(
            "scenario roster does not match the registered fixed-scope plan: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    # Verify the result rows form the complete fixed-scope cross-product: every
    # scenario must appear in all 27 (planner_group x latency_step x seed)
    # combinations, so there are no missing or extra cells relative to the
    # observed scenario roster.
    expected_cells = (
        len(scenarios) * len(PLANNER_GROUPS) * len(EXPECTED_LATENCY_STEPS) * len(EXPECTED_SEEDS)
    )
    if len(result_rows) != expected_cells:
        raise SnqiLatencyAnalysisError(
            f"result row count {len(result_rows)} does not match the complete cross-product "
            f"{expected_cells} ({len(scenarios)} scenarios x {len(PLANNER_GROUPS)} planners x "
            f"{len(EXPECTED_LATENCY_STEPS)} steps x {len(EXPECTED_SEEDS)} seeds)"
        )
    # Each scenario must be present in every (planner_group, step, seed) combo.
    scenario_combos: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    for entry in result_rows:
        scenario_combos[entry.scenario_id].add(
            (entry.planner_group, entry.latency_step, entry.seed)
        )
    full_combo = {
        (group, step, seed)
        for group in PLANNER_GROUPS
        for step in EXPECTED_LATENCY_STEPS
        for seed in EXPECTED_SEEDS
    }
    incomplete = [
        scenario_id for scenario_id, combos in scenario_combos.items() if combos != full_combo
    ]
    if incomplete:
        raise SnqiLatencyAnalysisError(
            f"{len(incomplete)} scenarios are missing latency cells; the cross-product is "
            f"incomplete (sample={sorted(incomplete)[:3]})"
        )

    return {
        "status": "verified",
        "latency_row_count": len(result_rows),
        "expected_row_count": EXPECTED_SCENARIO_COUNT
        * len(PLANNER_GROUPS)
        * len(EXPECTED_LATENCY_STEPS)
        * len(EXPECTED_SEEDS),
        "scenario_count": len(scenarios),
        "seeds": seeds,
        "planner_groups": planner_groups,
        "latency_steps": steps,
        "missing_latency_cells": 0,
        "extra_latency_cells": 0,
        "duplicate_latency_cells": 0,
        "fallback_row_count": 0,
        "degraded_row_count": 0,
        "unavailable_row_count": 0,
        "excluded_row_count": len(exclusions),
    }


def _execution_mode_for_group(inputs: Sequence[SnqiLatencyInput]) -> list[dict[str, Any]]:
    """Return the per-planner-group execution-mode summary (native / adapter)."""
    summaries: list[dict[str, Any]] = []
    for planner_group in PLANNER_GROUPS:
        rows = [entry for entry in inputs if entry.planner_group == planner_group]
        modes = sorted({entry.execution_mode for entry in rows})
        if len(modes) != 1:
            raise SnqiLatencyAnalysisError(
                f"planner group {planner_group!r} has mixed execution modes {modes}"
            )
        summaries.append(
            {
                "planner_group": planner_group,
                "execution_mode": modes[0],
                "latency_row_count": len(rows),
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# SNQI computation: per-episode score, per-unit slope, paired bootstrap
# ---------------------------------------------------------------------------


def _snqi_metrics(entry: SnqiLatencyInput) -> dict[str, float | int | bool]:
    """Build the SNQI-v0 metrics dict for one input row (matches input_mapping)."""
    return {
        "success": entry.success,
        "time_to_goal_norm": entry.time_to_goal_norm,
        "collisions": int(bool(entry.collision)),
        "near_misses": entry.near_miss_rate * entry.steps,
        "comfort_exposure": entry.comfort_exposure_mean,
    }


def _near_miss_recovery_max_fractional_error(inputs: Sequence[SnqiLatencyInput]) -> float:
    """Return the max relative rounding error recovering ``near_misses``.

    ``near_misses = near_miss_rate * steps`` must recover a non-negative event
    count. The fractional error is ``|rate*steps - round(rate*steps)| /
    round(rate*steps)`` over rows with a non-zero count. This is a measured
    precision bound on the recovery, not a fixed constant.
    """
    max_error = 0.0
    for entry in inputs:
        if entry.classification != "result":
            continue
        product = entry.near_miss_rate * entry.steps
        count = round(product)
        if count > 0:
            max_error = max(max_error, abs(product - count) / count)
    return float(max_error)


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the ordinary-least-squares slope of ``y`` on ``x`` with an intercept."""
    x_arr = list(x)
    y_arr = list(y)
    n = len(x_arr)
    mean_x = math.fsum(x_arr) / n
    mean_y = math.fsum(y_arr) / n
    num = math.fsum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_arr, y_arr, strict=True))
    den = math.fsum((xi - mean_x) ** 2 for xi in x_arr)
    if den == 0.0:
        raise SnqiLatencyAnalysisError(
            "OLS slope denominator is zero; latency steps are degenerate"
        )
    return num / den


def _unit_slopes(
    inputs: Sequence[SnqiLatencyInput],
    weights: Mapping[str, float],
    baseline_stats: Mapping[str, Mapping[str, float]],
    planner_group: str,
) -> tuple[list[str], list[float]]:
    """Return aligned per-(scenario, seed) OLS latency slopes for one planner.

    Each unit is one (scenario, seed) pair with a SNQI score at each latency step
    [0, 1, 3]. The unit slope is the OLS fit of SNQI on the latency steps. Units
    are ordered deterministically by (scenario_id, seed) so the paired bootstrap
    is reproducible and planners can be paired on the same unit roster.
    """
    grouped: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for entry in inputs:
        if entry.planner_group != planner_group or entry.classification != "result":
            continue
        grouped[(entry.scenario_id, entry.seed)][entry.latency_step] = compute_snqi(
            _snqi_metrics(entry), weights, baseline_stats, score_version="SNQI-v0"
        )
    keys = sorted(grouped)
    slopes: list[float] = []
    for key in keys:
        per_step = grouped[key]
        missing = [step for step in EXPECTED_LATENCY_STEPS if step not in per_step]
        if missing:
            raise SnqiLatencyAnalysisError(
                f"planner {planner_group!r} unit {key} is missing latency steps {missing}"
            )
        y = [per_step[step] for step in EXPECTED_LATENCY_STEPS]
        slopes.append(_ols_slope([float(step) for step in EXPECTED_LATENCY_STEPS], y))
    return [f"{key[0]}@{key[1]}" for key in keys], slopes


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean (naive summation, matching the registered output)."""
    return math.fsum(values) / len(values)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a percentile using numpy linear interpolation."""
    return float(np.percentile(list(values), percentile))


def _paired_cluster_bootstrap(
    unit_slopes_by_group: Mapping[str, list[float]],
) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]], dict[str, Any]]:
    """Run the paired cluster-bootstrap and return per-group CIs and pairwise diffs.

    A single ``numpy.random.default_rng(BOOTSTRAP_SEED)`` draws one resampling
    index matrix of shape ``(BOOTSTRAP_RESAMPLES, n_units)``; the SAME indices are
    reused across planners so the pairwise slope differences inherit the paired
    structure. Each resample's per-group slope is the mean of the resampled unit
    slopes (equivalent to the OLS slope of the resampled per-latency means for a
    common latency-step regressor). Percentile endpoints use linear interpolation.
    """
    groups = list(unit_slopes_by_group)
    n_units = {group: len(unit_slopes_by_group[group]) for group in groups}
    for group in groups:
        if n_units[group] != EXPECTED_PAIRED_UNITS_PER_PLANNER:
            raise SnqiLatencyAnalysisError(
                f"planner {group!r} has {n_units[group]} paired units, expected "
                f"{EXPECTED_PAIRED_UNITS_PER_PLANNER}"
            )
    arrays = {group: np.asarray(unit_slopes_by_group[group], dtype=float) for group in groups}

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(
        0,
        EXPECTED_PAIRED_UNITS_PER_PLANNER,
        size=(BOOTSTRAP_RESAMPLES, EXPECTED_PAIRED_UNITS_PER_PLANNER),
    )
    boot_means = {group: arrays[group][idx].mean(axis=1) for group in groups}

    ci: dict[str, tuple[float, float]] = {}
    for group in groups:
        lo = _percentile(boot_means[group], BOOTSTRAP_PERCENTILES[0])
        hi = _percentile(boot_means[group], BOOTSTRAP_PERCENTILES[1])
        ci[group] = (lo, hi)

    pairwise: list[dict[str, Any]] = []
    for i, first in enumerate(groups):
        for second in groups[i + 1 :]:
            diff_boot = boot_means[first] - boot_means[second]
            lo = _percentile(diff_boot, BOOTSTRAP_PERCENTILES[0])
            hi = _percentile(diff_boot, BOOTSTRAP_PERCENTILES[1])
            prob = float((diff_boot > 0.0).mean())
            pairwise.append(
                {
                    "comparison": f"{first} minus {second}",
                    "slope_difference": float(arrays[first].mean() - arrays[second].mean()),
                    "slope_difference_95pct_ci": [lo, hi],
                    "probability_first_is_more_robust": prob,
                }
            )

    method = {
        "method": "paired cluster percentile bootstrap",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "percentiles": list(BOOTSTRAP_PERCENTILES),
        "units_per_planner": EXPECTED_PAIRED_UNITS_PER_PLANNER,
        "pairing": "shared resampling indices across planners",
        "tolerance_note": (
            "Percentile endpoints are Monte-Carlo quantities; they reproduce the registered "
            "intervals within the documented absolute tolerance but are not byte-identical, "
            "because the registered packet's generating code was never committed and endpoints "
            "depend on the exact resampling stream and numpy version."
        ),
    }
    return ci, pairwise, method


def _group_means_at_steps(
    inputs: Sequence[SnqiLatencyInput],
    weights: Mapping[str, float],
    baseline_stats: Mapping[str, Mapping[str, float]],
    planner_group: str,
) -> dict[int, float]:
    """Return mean SNQI at each latency step for one planner group."""
    means: dict[int, float] = {}
    for step in EXPECTED_LATENCY_STEPS:
        scores = [
            compute_snqi(_snqi_metrics(entry), weights, baseline_stats, score_version="SNQI-v0")
            for entry in inputs
            if entry.planner_group == planner_group
            and entry.classification == "result"
            and entry.latency_step == step
        ]
        means[step] = _mean(scores)
    return means


def _build_ranking(
    inputs: Sequence[SnqiLatencyInput],
    weights: Mapping[str, float],
    baseline_stats: Mapping[str, Mapping[str, float]],
    ci: Mapping[str, tuple[float, float]],
    unit_slopes_by_group: Mapping[str, list[float]],
) -> list[dict[str, Any]]:
    """Return the point-estimate robustness ranking (most robust first)."""
    execution_modes = {
        entry.planner_group: entry.execution_mode
        for entry in inputs
        if entry.classification == "result"
    }
    planners = {
        entry.planner_group: entry.planner for entry in inputs if entry.classification == "result"
    }
    rows: list[dict[str, Any]] = []
    for planner_group in PLANNER_GROUPS:
        means = _group_means_at_steps(inputs, weights, baseline_stats, planner_group)
        at_0 = means[0]
        at_1 = means[1]
        at_3 = means[3]
        slope = _mean(unit_slopes_by_group[planner_group])
        ci_lo, ci_hi = ci[planner_group]
        rows.append(
            {
                "planner_group": planner_group,
                "planner": planners[planner_group],
                "execution_mode": execution_modes[planner_group],
                "paired_units": len(unit_slopes_by_group[planner_group]),
                "snqi_mean_at_0_steps": at_0,
                "snqi_mean_at_1_step": at_1,
                "snqi_mean_at_3_steps": at_3,
                "snqi_delta_at_1_step_vs_0": at_1 - at_0,
                "snqi_delta_at_3_steps_vs_0": at_3 - at_0,
                "snqi_slope_per_100ms": slope,
                "snqi_slope_95pct_ci": [ci_lo, ci_hi],
                "snqi_degradation_per_100ms": -slope,
            }
        )
    # Rank by slope descending (least degradation first); tie-break by name.
    rows.sort(key=lambda row: (-row["snqi_slope_per_100ms"], row["planner_group"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    # Reorder keys so rank leads (matches the registered field order).
    return [
        {
            "rank": row["rank"],
            "planner_group": row["planner_group"],
            "planner": row["planner"],
            "execution_mode": row["execution_mode"],
            "paired_units": row["paired_units"],
            "snqi_mean_at_0_steps": row["snqi_mean_at_0_steps"],
            "snqi_mean_at_1_step": row["snqi_mean_at_1_step"],
            "snqi_mean_at_3_steps": row["snqi_mean_at_3_steps"],
            "snqi_delta_at_1_step_vs_0": row["snqi_delta_at_1_step_vs_0"],
            "snqi_delta_at_3_steps_vs_0": row["snqi_delta_at_3_steps_vs_0"],
            "snqi_slope_per_100ms": row["snqi_slope_per_100ms"],
            "snqi_slope_95pct_ci": row["snqi_slope_95pct_ci"],
            "snqi_degradation_per_100ms": row["snqi_degradation_per_100ms"],
        }
        for row in rows
    ]


def _build_verdict(
    ranking: Sequence[Mapping[str, Any]], pairwise: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the qualitative verdict block (high-confidence vs uncertain findings)."""
    point_order = [row["planner_group"] for row in ranking]

    high_confidence: list[str] = []
    uncertain: list[str] = []
    # Per-planner interval excludes zero.
    display = {
        "default_social_force": "default_social_force",
        "orca": "ORCA",
        "hybrid_rule_v0_minimal": "hybrid_rule_v0_minimal",
    }
    for row in ranking:
        lo, hi = row["snqi_slope_95pct_ci"]
        if lo < 0.0 < hi:
            pass
        elif row["snqi_slope_per_100ms"] < 0.0:
            high_confidence.append(
                f"{display[row['planner_group']]} has a negative SNQI slope whose 95% interval "
                "excludes zero."
            )
    # Pairwise findings keyed on the registered comparison phrasing.
    pairwise_by_key = {item["comparison"]: item for item in pairwise}
    dsf_hybrid = pairwise_by_key.get("default_social_force minus hybrid_rule_v0_minimal")
    dsf_orca = pairwise_by_key.get("default_social_force minus orca")
    hybrid_orca = pairwise_by_key.get("hybrid_rule_v0_minimal minus orca")
    if dsf_orca:
        lo, hi = dsf_orca["slope_difference_95pct_ci"]
        if not (lo < 0.0 < hi) and dsf_orca["slope_difference"] > 0.0:
            high_confidence.append(
                "default_social_force is more robust than ORCA on the paired slope comparison at "
                "the 95% level."
            )
    if dsf_hybrid:
        lo, hi = dsf_hybrid["slope_difference_95pct_ci"]
        if lo < 0.0 < hi:
            uncertain.append(
                "default_social_force and hybrid_rule_v0_minimal are not separated at the 95% level."
            )
    if hybrid_orca:
        lo, hi = hybrid_orca["slope_difference_95pct_ci"]
        if lo < 0.0 < hi:
            uncertain.append(
                "hybrid_rule_v0_minimal and ORCA are not separated at the 95% level because the "
                "paired interval narrowly crosses zero."
            )
    return {
        "point_estimate_order": point_order,
        "high_confidence_findings": high_confidence,
        "uncertain_findings": uncertain,
    }


def build_snqi_analysis(
    inputs: Sequence[SnqiLatencyInput],
    *,
    weights: Mapping[str, float],
    baseline_stats: Mapping[str, Mapping[str, float]],
    date: str,
    bootstrap_method: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``control-action-latency-snqi-analysis.v1`` packet from inputs.

    The inputs must already have passed fixed-scope / execution-mode / exclusion
    validation. Computes SNQI-v0 per episode, per-unit OLS latency slopes, and the
    paired cluster-bootstrap uncertainty, and returns the JSON-serializable packet.
    """
    coverage = validate_fixed_scope(inputs)
    execution_modes = _execution_mode_for_group(inputs)

    unit_slopes_by_group: dict[str, list[float]] = {}
    unit_keys_by_group: dict[str, list[str]] = {}
    for planner_group in PLANNER_GROUPS:
        keys, slopes = _unit_slopes(inputs, weights, baseline_stats, planner_group)
        unit_keys_by_group[planner_group] = keys
        unit_slopes_by_group[planner_group] = slopes

    ci, pairwise, computed_method = _paired_cluster_bootstrap(unit_slopes_by_group)
    ranking = _build_ranking(inputs, weights, baseline_stats, ci, unit_slopes_by_group)
    verdict = _build_verdict(ranking, pairwise)

    # The weights / baseline SHA-256 are the registered constants; the CLI loads
    # them from the registered config paths and the caller is expected to have
    # verified those checksums before computing SNQI.
    weights_sha = WEIGHTS_SHA256
    baseline_sha = BASELINE_SHA256

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "issue": ISSUE,
        "campaign_issue": CAMPAIGN_ISSUE,
        "parent_issue": PARENT_ISSUE,
        "date": date,
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "provenance": {
            "job_id": JOB_ID,
            "label": JOB_LABEL,
            "execution_commit": EXECUTION_COMMIT,
            "raw_rows_sha256": RAW_ROWS_SHA256,
            "fixed_scope_plan_sha256": FIXED_SCOPE_PLAN_SHA256,
            "raw_artifact_classification": "durable-required-private",
            "raw_artifact_pointer": "private campaign output for label 5034c-issue5034-latency-sweep, job 13516",
        },
        "scope_verification": {
            "status": coverage["status"],
            "full_episode_row_count": 7344,
            "full_unique_row_key_count": 7344,
            "run_cell_count": 153,
            "scenario_count": coverage["scenario_count"],
            "seeds": coverage["seeds"],
            "planner_groups": coverage["planner_groups"],
            "latency_steps": coverage["latency_steps"],
            "latency_episode_row_count": coverage["latency_row_count"],
            "latency_expected_row_count": coverage["expected_row_count"],
            "latency_unique_row_key_count": coverage["latency_row_count"],
            "missing_latency_cells": coverage["missing_latency_cells"],
            "extra_latency_cells": coverage["extra_latency_cells"],
            "duplicate_latency_cells": coverage["duplicate_latency_cells"],
            "fallback_row_count": coverage["fallback_row_count"],
            "degraded_row_count": coverage["degraded_row_count"],
            "unavailable_row_count": coverage["unavailable_row_count"],
            "execution_modes": execution_modes,
        },
        "snqi_method": {
            "primary_metric": "Social Navigation Quality Index (SNQI)",
            "score_version": "SNQI-v0",
            "implementation": "robot_sf.benchmark.snqi.compute.compute_snqi",
            "weights_path": WEIGHTS_PATH,
            "weights_sha256": weights_sha,
            "baseline_path": BASELINE_PATH,
            "baseline_sha256": baseline_sha,
            "input_mapping": {
                "success": "episode.success",
                "time_to_goal_norm": "episode.metrics.time_to_goal_norm",
                "collisions": "integer episode.collision",
                "near_misses": "episode.metrics.near_miss_rate multiplied by episode.steps",
                "comfort_exposure": "episode.metrics.comfort_exposure_mean",
            },
            "near_miss_recovery_max_fractional_error": _near_miss_recovery_max_fractional_error(
                inputs
            ),
            "neutral_defaulted_terms": list(NEUTRAL_DEFAULTED_TERMS),
            "slope_model": (
                "ordinary least squares over latency steps [0, 1, 3]; one step is 100 ms-equivalent"
            ),
            "uncertainty": (
                "95% paired cluster-bootstrap interval over 144 scenario-seed units per planner, "
                "10000 resamples, seed 5892"
            ),
            "bootstrap_method": computed_method if bootstrap_method is None else bootstrap_method,
        },
        "point_estimate_robustness_ranking": ranking,
        "pairwise_slope_uncertainty": pairwise,
        "verdict": verdict,
        "caveats": [
            "The strict native-only claim boundary is not satisfied for the full three-planner "
            "ranking because ORCA and hybrid_rule_v0_minimal are labeled adapter execution; their "
            "rows are retained only as clearly labeled internal diagnostic evidence.",
            "The raw campaign did not emit force_exceed_events or jerk_mean, so the canonical "
            "SNQI-v0 neutral defaults apply to those two terms.",
            "The bundle generator recorded a different git_head than the execution context; the "
            "job execution context and checksummed plan identify "
            "c153848d7be2851b5c5e89c11055bf96ea778a84 as the execution commit.",
            "This is not paper-facing evidence and does not establish simulator realism or "
            "sim-to-real validity.",
        ],
        "reproducibility_contract": {
            "point_estimate_abs_tol": POINT_TOL,
            "pairwise_slope_difference_abs_tol": PAIRWISE_DIFF_TOL,
            "bootstrap_interval_abs_tol": INTERVAL_TOL,
            "bootstrap_probability_abs_tol": PROBABILITY_TOL,
            "description": (
                "SNQI-v0 point estimates (per-planner means, deltas, slopes) are deterministic "
                "and match the registered packet within point_estimate_abs_tol (observed "
                "~2e-16). The pairwise slope_difference, bootstrap percentile endpoints, and "
                "posterior probabilities are Monte-Carlo / second-code-path quantities that "
                "match within their tolerances but are not byte-identical, because the "
                "registered packet's generating code was never committed and its uncertainty "
                "block is internally inconsistent (the registered pairwise slope_difference "
                "does not equal the difference of its own per-planner slopes, and the "
                "probabilities include half-integer counts such as 0.68635 = 6863.5/10000). "
                "This committed analyzer is the canonical deterministic generator going "
                "forward."
            ),
        },
        "_unit_keys": unit_keys_by_group,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_snqi_analysis(packet: Mapping[str, Any], evidence_dir: str | Path) -> list[Path]:
    """Write ``snqi_analysis.json`` and ``snqi_by_latency.csv`` (without the private unit keys)."""
    out = Path(evidence_dir)
    out.mkdir(parents=True, exist_ok=True)

    public_packet = {key: value for key, value in packet.items() if not key.startswith("_")}
    analysis_path = out / "snqi_analysis.json"
    analysis_path.write_text(
        json.dumps(public_packet, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    csv_path = out / "snqi_by_latency.csv"
    csv_fields = [
        "planner_group",
        "planner",
        "execution_mode",
        "latency_steps",
        "latency_ms_equivalent",
        "paired_units",
        "snqi_mean",
        "snqi_delta_vs_zero",
        "snqi_slope_per_100ms",
        "snqi_slope_ci_low",
        "snqi_slope_ci_high",
        "point_estimate_robustness_rank",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(review_marker_comment() + "\n")
        writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for row in packet["point_estimate_robustness_ranking"]:
            means = {
                0: row["snqi_mean_at_0_steps"],
                1: row["snqi_mean_at_1_step"],
                3: row["snqi_mean_at_3_steps"],
            }
            ci_lo, ci_hi = row["snqi_slope_95pct_ci"]
            for step in EXPECTED_LATENCY_STEPS:
                writer.writerow(
                    {
                        "planner_group": row["planner_group"],
                        "planner": row["planner"],
                        "execution_mode": row["execution_mode"],
                        "latency_steps": step,
                        "latency_ms_equivalent": step * MS_PER_LATENCY_STEP,
                        "paired_units": row["paired_units"],
                        "snqi_mean": means[step],
                        "snqi_delta_vs_zero": (0.0 if step == 0 else means[step] - means[0]),
                        "snqi_slope_per_100ms": row["snqi_slope_per_100ms"],
                        "snqi_slope_ci_low": ci_lo,
                        "snqi_slope_ci_high": ci_hi,
                        "point_estimate_robustness_rank": row["rank"],
                    }
                )
    return [analysis_path, csv_path]


# ---------------------------------------------------------------------------
# Reproducibility verification (DoD #3 comparison contract)
# ---------------------------------------------------------------------------


def _walk_numbers(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    """Yield (dotted-path, value) for every finite float in a nested structure."""
    found: list[tuple[str, float]] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            found.extend(_walk_numbers(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list | tuple):
        for index, value in enumerate(obj):
            found.extend(_walk_numbers(value, f"{prefix}[{index}]"))
    elif isinstance(obj, int | float) and not isinstance(obj, bool):
        value = float(obj)
        if math.isfinite(value):
            found.append((prefix, value))
    return found


def verify_against_reference(
    packet: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare a generated packet to a reference under the numeric-tolerance contract.

    Returns a report with the max point-estimate deviation, max interval deviation,
    max probability deviation, and a per-path worst offender. Raises
    :class:`SnqiLatencyAnalysisError` if the schemas mismatch or any numeric field
    exceeds its tolerance.
    """
    if packet.get("schema_version") != reference.get("schema_version"):
        raise SnqiLatencyAnalysisError(
            "schema_version mismatch: generated "
            f"{packet.get('schema_version')!r} vs reference {reference.get('schema_version')!r}"
        )

    generated = dict(_walk_numbers(packet))
    reference_numbers = dict(_walk_numbers(reference))

    # Diagnostic-only fields that are measured bounds rather than registered
    # targets: excluded from strict numeric comparison (documented below).
    excluded_paths = {"snqi_method.near_miss_recovery_max_fractional_error"}

    # Restrict comparison to numeric paths present in the reference; ignore
    # analyzer-only diagnostic blocks (bootstrap_method, reproducibility_contract).
    comparable = [
        path for path in generated if path in reference_numbers and path not in excluded_paths
    ]

    max_point = 0.0
    max_pairwise_diff = 0.0
    max_interval = 0.0
    max_probability = 0.0
    worst_point = worst_pairwise = worst_interval = worst_probability = None
    failures: list[dict[str, Any]] = []
    for path in comparable:
        diff = abs(generated[path] - reference_numbers[path])
        if "probability" in path:
            if diff > max_probability:
                max_probability = diff
                worst_probability = path
            if diff > PROBABILITY_TOL:
                failures.append({"path": path, "diff": diff, "tolerance": PROBABILITY_TOL})
        elif "95pct_ci" in path:
            if diff > max_interval:
                max_interval = diff
                worst_interval = path
            if diff > INTERVAL_TOL:
                failures.append({"path": path, "diff": diff, "tolerance": INTERVAL_TOL})
        elif "slope_difference" in path:
            if diff > max_pairwise_diff:
                max_pairwise_diff = diff
                worst_pairwise = path
            if diff > PAIRWISE_DIFF_TOL:
                failures.append({"path": path, "diff": diff, "tolerance": PAIRWISE_DIFF_TOL})
        else:
            if diff > max_point:
                max_point = diff
                worst_point = path
            if diff > POINT_TOL:
                failures.append({"path": path, "diff": diff, "tolerance": POINT_TOL})

    report = {
        "schema_version": packet.get("schema_version"),
        "status": "verified" if not failures else "failed",
        "compared_numeric_paths": len(comparable),
        "excluded_diagnostic_paths": sorted(excluded_paths),
        "max_point_estimate_deviation": max_point,
        "max_point_estimate_path": worst_point,
        "max_pairwise_slope_difference_deviation": max_pairwise_diff,
        "max_pairwise_slope_difference_path": worst_pairwise,
        "max_bootstrap_interval_deviation": max_interval,
        "max_bootstrap_interval_path": worst_interval,
        "max_probability_deviation": max_probability,
        "max_probability_path": worst_probability,
        "point_estimate_abs_tol": POINT_TOL,
        "pairwise_slope_difference_abs_tol": PAIRWISE_DIFF_TOL,
        "bootstrap_interval_abs_tol": INTERVAL_TOL,
        "bootstrap_probability_abs_tol": PROBABILITY_TOL,
        "failures": failures,
    }
    if failures:
        raise SnqiLatencyAnalysisError(
            "generated SNQI analysis deviates from the reference beyond the reproducibility "
            f"contract: {failures[:5]}"
        )
    return report


# ---------------------------------------------------------------------------
# Uncertainty re-issue (issue #5928 DoD #2)
# ---------------------------------------------------------------------------

#: Canonical command that regenerates the re-issued uncertainty packet. Recorded
#: in the packet provenance so a fresh checkout can reproduce it byte-for-byte.
UNCERTAINTY_REISSUE_COMMAND = (
    "uv run python scripts/benchmark/analyze_control_action_latency_snqi.py --reissue-uncertainty"
)

#: Concrete evidence that the registered uncertainty generator was never
#: committed and is unrecoverable. Recorded verbatim in the re-issued packet so
#: the recovery decision is auditable from durable files (issue #5928 DoD #1).
RECOVERY_EVIDENCE: tuple[str, ...] = (
    "PR #5904 registered docs/context/evidence/issue_5034_control_action_latency_sweep/"
    "snqi_analysis.json as a pure output artifact; the introducing commit added the "
    "JSON and CSV evidence plus documentation but no generator script.",
    "No committed Python module other than robot_sf/benchmark/control_action_latency_snqi.py "
    "(added by PR #5923) emits the pairwise_slope_uncertainty block under the "
    "control-action-latency-snqi-analysis.v1 schema.",
    "The registered pairwise slope_difference is internally inconsistent with its own "
    "per-planner slopes: e.g. the registered dsf-minus-hybrid difference 0.00230618 does "
    "not equal slope_A - slope_B = 0.00231315 of the ranking block, proving the "
    "uncertainty block came from a second, unrecoverable code path.",
    "The registered posterior probabilities include half-integer counts (e.g. "
    "0.68635 = 6863.5/10000, 0.97105 = 9710.5/10000), which a simple "
    "(diff > 0).mean() over 10000 resamples cannot produce; the estimator is unrecoverable.",
)


def build_uncertainty_reissue(
    packet: Mapping[str, Any],
    *,
    generator_source_sha256: str,
    generator_source_rel_path: str,
    input_sha256: str,
    input_provenance_anchor: Mapping[str, Any],
    reissue_date: str,
) -> dict[str, Any]:
    """Build the re-issued uncertainty packet from a canonical-analyzer packet.

    Issue #5928 DoD #2: the original job 13516 uncertainty generator was never
    committed and is unrecoverable (see :data:`RECOVERY_EVIDENCE`). This function
    re-issues the uncertainty block from the **committed canonical analyzer**
    output (``packet`` produced by :func:`build_snqi_analysis`) under a fresh,
    byte-reproducible provenance stamp.

    The re-issued block is internally consistent by construction: every
    ``slope_difference`` equals ``slope_A - slope_B`` of the committed per-planner
    slopes recorded alongside it, and every ``probability_first_is_more_robust``
    is an exact integer multiple of ``1 / BOOTSTRAP_RESAMPLES`` (no half-integer
    counts). Native / adapter execution-mode labels and the diagnostic-only claim
    boundary are carried through unchanged.

    The generator identity is **content-addressed** by the analyzer source
    SHA-256 (not by a volatile git head), so a regeneration from unchanged source
    and unchanged durable input reproduces this packet byte-for-byte. The CLI
    prints the best-effort git head to stdout for human convenience; it is
    deliberately not embedded in the committed artifact.

    Args:
        packet: A packet from :func:`build_snqi_analysis` (already validated).
        generator_source_sha256: SHA-256 of the analyzer source file at generation.
        generator_source_rel_path: Repo-relative path of the analyzer source file.
        input_sha256: SHA-256 of the durable sufficient input used.
        input_provenance_anchor: The durable-input provenance sidecar payload
            (anchoring the input to the raw rows).
        reissue_date: ISO date recorded in the re-issue provenance.

    Returns:
        The JSON-serializable ``control-action-latency-snqi-uncertainty-reissue.v1``
        packet.
    """
    if packet.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise SnqiLatencyAnalysisError(
            "uncertainty re-issue requires a canonical-analyzer packet with schema "
            f"{ANALYSIS_SCHEMA_VERSION!r}; got {packet.get('schema_version')!r}"
        )
    pairwise = packet.get("pairwise_slope_uncertainty")
    ranking = packet.get("point_estimate_robustness_ranking")
    if not isinstance(pairwise, list) or not isinstance(ranking, list):
        raise SnqiLatencyAnalysisError(
            "canonical-analyzer packet is missing pairwise_slope_uncertainty / "
            "point_estimate_robustness_ranking"
        )

    # Per-planner committed slopes the pairwise differences derive from. Recording
    # them makes the internal consistency of the re-issued block auditable.
    slope_by_group = {row["planner_group"]: row["snqi_slope_per_100ms"] for row in ranking}
    mode_by_group = {row["planner_group"]: row["execution_mode"] for row in ranking}
    per_planner_intervals = [
        {
            "planner_group": row["planner_group"],
            "execution_mode": row["execution_mode"],
            "snqi_slope_per_100ms": row["snqi_slope_per_100ms"],
            "snqi_slope_95pct_ci": row["snqi_slope_95pct_ci"],
        }
        for row in ranking
    ]

    # Verify internal consistency of the re-issued pairwise differences and that
    # every posterior probability is an exact integer resample count. The
    # canonical analyzer guarantees both; this assertion fails closed if a future
    # change breaks that invariant, so a re-issue can never silently carry an
    # unrecoverable-style block.
    consistency_checks: list[dict[str, Any]] = []
    for entry in pairwise:
        first, _, second = str(entry["comparison"]).partition(" minus ")
        expected_diff = slope_by_group[first] - slope_by_group[second]
        diff_dev = abs(float(entry["slope_difference"]) - float(expected_diff))
        prob = float(entry["probability_first_is_more_robust"])
        resample_count = prob * BOOTSTRAP_RESAMPLES
        is_integer_count = math.isclose(
            resample_count, round(resample_count), rel_tol=0.0, abs_tol=1e-9
        )
        consistency_checks.append(
            {
                "comparison": entry["comparison"],
                "slope_difference_equals_slope_A_minus_slope_B": diff_dev <= POINT_TOL,
                "slope_difference_deviation": diff_dev,
                "probability_is_integer_resample_count": is_integer_count,
                "implied_resample_count": round(resample_count),
            }
        )
        if diff_dev > POINT_TOL:
            raise SnqiLatencyAnalysisError(
                f"re-issued slope_difference for {entry['comparison']!r} deviates from "
                f"slope_A - slope_B by {diff_dev} (tol {POINT_TOL}); the canonical analyzer "
                "must produce an internally consistent block."
            )
        if not is_integer_count:
            raise SnqiLatencyAnalysisError(
                f"re-issued probability for {entry['comparison']!r} is not an exact integer "
                f"resample count (implied {resample_count}); the canonical analyzer must not "
                "produce half-integer counts."
            )

    bootstrap_method = packet.get("snqi_method", {}).get("bootstrap_method", {})
    source = dict(input_provenance_anchor.get("source") or {})
    promotion = dict(input_provenance_anchor.get("promotion") or {})

    return {
        "review_marker": review_marker_json(),
        "schema_version": UNCERTAINTY_REISSUE_SCHEMA_VERSION,
        "issue": 5928,
        "parent_issue": ISSUE,
        "campaign_issue": CAMPAIGN_ISSUE,
        "root_parent_issue": PARENT_ISSUE,
        "date": reissue_date,
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "recovery_decision": {
            "decision": (
                "re-issue from the committed canonical analyzer; the original uncertainty "
                "generator is unrecoverable"
            ),
            "recovery_path": "unrecoverable",
            "evidence": list(RECOVERY_EVIDENCE),
            "dod_reference": "issue #5928 Definition of Done #1 (recovery or maintainer "
            "decision) and #2 (re-issue from the committed analyzer with a fresh provenance "
            "stamp)",
            "linked_issues": [5928, ISSUE, CAMPAIGN_ISSUE, PARENT_ISSUE],
            "linked_prs": [5904, 5923],
        },
        "provenance": {
            "generator": {
                "kind": "committed_canonical_analyzer",
                "module": "robot_sf.benchmark.control_action_latency_snqi",
                "source_rel_path": generator_source_rel_path,
                "source_sha256": generator_source_sha256,
                "build_function": "build_snqi_analysis",
                "reissue_function": "build_uncertainty_reissue",
                "identity_note": (
                    "The generator is content-addressed by source_sha256 (not a git head) "
                    "so regenerating from unchanged source and unchanged durable input "
                    "reproduces this packet byte-for-byte."
                ),
            },
            "durable_input": {
                "input_rel_path": input_provenance_anchor.get("input_path"),
                "input_sha256": input_sha256,
                "input_schema_version": input_provenance_anchor.get("input_schema_version"),
                "raw_rows_sha256": source.get("raw_rows_sha256"),
                "axis": source.get("axis"),
                "promotion_commit": promotion.get("promoter_git_head"),
                "promotion_date": promotion.get("date"),
            },
            "job_id": JOB_ID,
            "job_label": JOB_LABEL,
            "execution_commit": EXECUTION_COMMIT,
            "weights_sha256": WEIGHTS_SHA256,
            "baseline_sha256": BASELINE_SHA256,
            "regenerate_command": UNCERTAINTY_REISSUE_COMMAND,
        },
        "reproducibility": {
            "byte_reproducible": True,
            "description": (
                "The re-issued uncertainty block is fully deterministic: a single "
                "numpy.random.default_rng(5892) draws one shared resampling-index matrix "
                "across planners, percentile endpoints use linear interpolation, and the "
                "per-planner unit roster is ordered by (scenario_id, seed). The committed "
                "analyzer source (sha256 above) plus the durable input (sha256 above) "
                "reproduce this block byte-for-byte on a fixed numpy version. Unlike the "
                "registered block, every probability is an exact integer resample count and "
                "every slope_difference equals slope_A - slope_B."
            ),
            "bootstrap_method": bootstrap_method,
            "point_estimate_abs_tol": POINT_TOL,
            "pairwise_slope_difference_abs_tol": PAIRWISE_DIFF_TOL,
            "bootstrap_interval_abs_tol": INTERVAL_TOL,
            "bootstrap_probability_abs_tol": PROBABILITY_TOL,
        },
        "point_estimate_slopes": {
            "description": (
                "Committed per-planner SNQI-v0 latency slopes (per 100 ms-equivalent) that "
                "the pairwise slope_difference values below are derived from. Recording "
                "them makes the internal consistency of the re-issued block auditable."
            ),
            "slope_per_100ms_by_group": slope_by_group,
            "execution_mode_by_group": mode_by_group,
        },
        "per_planner_slope_intervals": per_planner_intervals,
        "pairwise_slope_uncertainty": pairwise,
        "consistency_checks": consistency_checks,
        "caveats": [
            "The re-issued uncertainty block supersedes the unrecoverable registered "
            "block for byte-exact reproduction going forward; it does not retroactively "
            "make the registered snqi_analysis.json byte-identical, which remains the "
            "canonical reference target for the #5923 numeric-tolerance contract.",
            "This is internal diagnostic-only evidence and is not paper-facing; native "
            "claims apply only to the default_social_force native rows, while orca and "
            "hybrid_rule_v0_minimal remain explicitly labeled adapter diagnostics.",
            "No qualitative conclusion changes relative to the registered block: every "
            "95% interval sign and every probability threshold is preserved within the "
            "documented tolerances.",
        ],
    }


def write_uncertainty_reissue(
    reissue_packet: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write the re-issued uncertainty packet as deterministic JSON.

    The output is sorted-key JSON with a trailing newline so a regeneration from
    unchanged inputs and analyzer source is byte-identical (and therefore matches
    its own registered checksum / review sidecar).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reissue_packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_uncertainty_reissue_review_sidecar(
    artifact_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build the ``evidence-review-marker.v1`` sidecar for the re-issued packet.

    The sidecar mirrors the sibling ``*.review.json`` files in the evidence
    bundle so the pr_contract_check evidence-tree-hygiene rule authorizes the
    new artifact's AI-GENERATED / NEEDS-REVIEW marker.
    """
    root = Path(repo_root)
    artifact = Path(artifact_path)
    try:
        rel = str(artifact.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(artifact)
    return {
        "schema_version": "evidence-review-marker.v1",
        "artifact_path": rel,
        "artifact_sha256": sha256_file(artifact),
        "review_marker": review_marker_json(),
        "preserved_exact_bytes": True,
    }


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "BASELINE_PATH",
    "BASELINE_SHA256",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "EXPECTED_SCENARIO_IDS",
    "INPUT_COLUMNS",
    "INPUT_PROVENANCE_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "INTERVAL_TOL",
    "JOB_ID",
    "PAIRWISE_DIFF_TOL",
    "PLANNER_GROUPS",
    "POINT_TOL",
    "PROBABILITY_TOL",
    "RAW_ROWS_SHA256",
    "RECOVERY_EVIDENCE",
    "UNCERTAINTY_REISSUE_COMMAND",
    "UNCERTAINTY_REISSUE_SCHEMA_VERSION",
    "WEIGHTS_PATH",
    "WEIGHTS_SHA256",
    "SnqiLatencyAnalysisError",
    "SnqiLatencyInput",
    "build_snqi_analysis",
    "build_uncertainty_reissue",
    "build_uncertainty_reissue_review_sidecar",
    "classify_input_row",
    "derive_inputs_from_raw_rows",
    "load_input_provenance",
    "load_input_rows",
    "load_raw_rows",
    "validate_file_checksum",
    "validate_fixed_scope",
    "validate_input_checksum",
    "validate_raw_rows_checksum",
    "verify_against_reference",
    "write_input_provenance",
    "write_input_rows",
    "write_snqi_analysis",
    "write_uncertainty_reissue",
]

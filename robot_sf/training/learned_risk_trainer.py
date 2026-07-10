"""Learned-risk model v1 trainer core (issue #4617, parent #1472).

This module owns the *training-side* logic that the thin CLI entrypoint
``scripts/training/train_learned_risk_model.py`` composes. It is intentionally
separate from the pre-Slurm contract owners it depends on:

- :mod:`robot_sf.training.learned_risk_launch_packet` proves the launch packet;
- :mod:`robot_sf.training.learned_risk_campaign_readiness` folds the launch
  packet and durable trace manifest into one fail-closed launch decision.

The trainer adds three things on top of those owners:

1. a config contract (``learned-risk-model-training.v1``) whose labels/features
   must agree with the launch packet;
2. a dependency-light per-label logistic model exercised by a **CPU smoke** on a
   tiny synthetic fixture (no sklearn/torch, no private artifacts, no network);
3. a stable status-artifact JSON shape recording mode, training state,
   diagnostics, and an explicit claim boundary.

Boundary: this module never submits Slurm, fetches artifacts, publishes
checkpoints, or promotes a learned-risk claim. *Real* (non-smoke) mode is
fail-closed: it refuses to train unless the campaign-readiness gate is launch
ready, which stays blocked while the trace manifest (issue #2312 / #4586) is
unresolved. Hard guards remain authoritative; the learned output is
auxiliary-cost-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.errors import RobotSfError
from robot_sf.training.learned_risk_campaign_readiness import (
    CAMPAIGN_READY,
    evaluate_campaign_readiness,
)
from robot_sf.training.learned_risk_launch_packet import (
    LearnedRiskLaunchPacketError,
    validate_launch_packet,
)

TRAINING_SCHEMA_VERSION = "learned-risk-model-training.v1"
CANDIDATE_ID = "learned_risk_model_v1"
CLAIM_BOUNDARY = "launch_entrypoint_and_cpu_smoke_only_no_slurm_no_claim"
HARD_GUARD_ROLE = (
    "hard guards remain authoritative; learned risk is auxiliary_cost_only "
    "and may only rank otherwise-safe candidate commands"
)

# Training-state vocabulary written into the status artifact.
STATE_SMOKE_COMPLETED = "smoke_completed"
STATE_BLOCKED_TRACE_MANIFEST = "blocked_trace_manifest"
STATE_TRAINING_COMPLETED = "training_completed"
STATE_FAILED = "failed"

# The launch packet is the authority on labels; the training config must agree.
_REQUIRED_LABELS = ("collision", "near_miss", "low_progress")
# Fields every episode record must carry before it can be turned into a training
# row; mirrors the launch-packet trace_input_contract.required_episode_fields.
_REQUIRED_EPISODE_FIELDS = (
    "scenario_id",
    "seed",
    "candidate_id",
    "termination_reason",
    "metrics",
    "trajectory_features",
    "labels",
)


class LearnedRiskTrainerError(RobotSfError, ValueError):
    """Raised when the trainer config or its input traces are contract-invalid.

    This is reserved for operator/contract error (a malformed config, a config
    that disagrees with the launch packet, or trace rows missing required
    fields). A well-formed real run that is merely *not ready* does not raise; it
    is reported as a fail-closed ``blocked_trace_manifest`` status.
    """


def load_training_config(config_path: Path) -> dict[str, Any]:
    """Load and shape-check a learned-risk training config.

    Args:
        config_path: Path to the ``learned-risk-model-training.v1`` YAML config.

    Returns:
        Parsed config mapping.

    Raises:
        LearnedRiskTrainerError: If the file is missing, is not a mapping, or
            violates a config-level invariant (schema, candidate id, labels).
    """
    if not config_path.is_file():
        raise LearnedRiskTrainerError(f"training config is not a file: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LearnedRiskTrainerError("training config must be a YAML mapping")

    errors: list[str] = []
    if payload.get("schema_version") != TRAINING_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TRAINING_SCHEMA_VERSION!r}")
    if payload.get("candidate_id") != CANDIDATE_ID:
        errors.append(f"candidate_id must be {CANDIDATE_ID!r}")

    labels = payload.get("labels")
    if not isinstance(labels, list) or [str(v) for v in labels] != list(_REQUIRED_LABELS):
        errors.append(f"labels must be exactly {list(_REQUIRED_LABELS)} (launch-packet order)")

    features = payload.get("features")
    if not isinstance(features, list) or not all(
        isinstance(f, str) and f.strip() for f in features
    ):
        errors.append("features must be a non-empty list of non-empty strings")

    _require_non_output_path(payload, "output_root", errors)
    _require_non_output_path(payload, "status_artifact_path", errors)

    for key in ("launch_packet", "trace_manifest"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a non-empty path string")

    if errors:
        joined = "\n- ".join(errors)
        raise LearnedRiskTrainerError(f"learned-risk training config is invalid:\n- {joined}")
    return payload


def _require_non_output_path(config: dict[str, Any], key: str, errors: list[str]) -> None:
    """Reject config-declared artifact paths that live under worktree-local ``output/``.

    The launch-packet artifact contract forbids durable execution artifacts from
    depending on worktree-local ``output/``; the training config's declared
    ``output_root`` / ``status_artifact_path`` must honour the same rule. (A
    transient CPU smoke may still be redirected under ``output/`` via a CLI
    override; only the *config-declared* defaults are constrained here.)
    """
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty path string")
        return
    if "output" in Path(value).parts:
        errors.append(f"{key} must not depend on worktree-local output/: {value}")


def resolve_feature(record: dict[str, Any], feature: str) -> float:
    """Resolve a single feature value from an episode record.

    Dotted names (``metrics.min_distance``) are read as a nested path from the
    record root. Plain names (``min_rollout_clearance_m``) are read from the
    episode's ``trajectory_features`` mapping.

    Args:
        record: One episode record.
        feature: Feature key from the training config.

    Returns:
        The feature value as a float.

    Raises:
        LearnedRiskTrainerError: If the feature path is absent or non-numeric.
    """
    if "." in feature:
        node: Any = record
        for part in feature.split("."):
            if not isinstance(node, dict) or part not in node:
                raise LearnedRiskTrainerError(f"record missing feature path '{feature}'")
            node = node[part]
        value = node
    else:
        traj = record.get("trajectory_features")
        if not isinstance(traj, dict) or feature not in traj:
            raise LearnedRiskTrainerError(f"record missing trajectory feature '{feature}'")
        value = traj[feature]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise LearnedRiskTrainerError(f"feature '{feature}' is not numeric: {value!r}") from exc


def build_matrix(
    records: list[dict[str, Any]],
    features: list[str],
    labels: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Turn episode records into a feature matrix and per-label binary targets.

    Args:
        records: Episode records (each already schema-checked).
        features: Ordered feature keys.
        labels: Ordered label keys.

    Returns:
        ``(X, targets)`` where ``X`` has shape ``(n_rows, n_features)`` and
        ``targets`` maps each label to a ``(n_rows,)`` 0/1 array.

    Raises:
        LearnedRiskTrainerError: If a record is missing a required field, a
            feature, or a label.
    """
    if not records:
        raise LearnedRiskTrainerError("no trace records to train on")

    feature_rows: list[list[float]] = []
    target_rows: dict[str, list[int]] = {label: [] for label in labels}
    for index, record in enumerate(records):
        missing = [field for field in _REQUIRED_EPISODE_FIELDS if field not in record]
        if missing:
            raise LearnedRiskTrainerError(f"record {index} missing required fields: {missing}")
        record_labels = record.get("labels")
        if not isinstance(record_labels, dict):
            raise LearnedRiskTrainerError(f"record {index} 'labels' must be a mapping")
        feature_rows.append([resolve_feature(record, feature) for feature in features])
        for label in labels:
            if label not in record_labels:
                raise LearnedRiskTrainerError(f"record {index} missing label '{label}'")
            target_rows[label].append(int(bool(record_labels[label])))

    x = np.asarray(feature_rows, dtype=np.float64)
    targets = {label: np.asarray(values, dtype=np.float64) for label, values in target_rows.items()}
    return x, targets


def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance standardize columns, guarding zero-variance ones.

    Returns:
        The column-standardized matrix, same shape as ``x``.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (x - mean) / std


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    l2: float,
    epochs: int,
) -> np.ndarray:
    """Fit a logistic head with plain full-batch gradient descent.

    Returns:
        Weight vector of length ``n_features + 1`` (last entry is the bias). A
        deterministic zero init keeps the smoke reproducible without RNG state.
    """
    n_rows, n_features = x.shape
    design = np.hstack([x, np.ones((n_rows, 1))])
    weights = np.zeros(n_features + 1)
    for _ in range(epochs):
        logits = design @ weights
        preds = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (preds - y) / n_rows
        gradient[:-1] += l2 * weights[:-1]  # L2 on weights only, not the bias
        weights -= learning_rate * gradient
    return weights


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return per-row positive-class probabilities for a fitted logistic head."""
    design = np.hstack([x, np.ones((x.shape[0], 1))])
    return 1.0 / (1.0 + np.exp(-(design @ weights)))


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUROC; ``None`` when only one class is present (undefined).

    Returns:
        The AUROC in ``[0, 1]``, or ``None`` when the label has a single class.
    """
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied score groups so ties contribute 0.5.
    unique_scores = np.unique(scores)
    for value in unique_scores:
        tie_mask = scores == value
        if tie_mask.sum() > 1:
            ranks[tie_mask] = ranks[tie_mask].mean()
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auprc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Average-precision approximation; ``None`` when no positives exist.

    Returns:
        The average precision in ``[0, 1]``, or ``None`` when there are no
        positive rows.
    """
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    cumulative_tp = np.cumsum(y_sorted)
    precision = cumulative_tp / np.arange(1, len(y_sorted) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def _diagnostics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Compute the smoke diagnostics for one label head.

    Returns:
        Mapping with ``auroc``, ``auprc``, ``brier``, ``false_negative_rate``,
        and the positive/row counts. AUROC/AUPRC are ``None`` when undefined for
        the label's class balance rather than a misleading number.
    """
    predictions = (scores >= 0.5).astype(np.float64)
    positives = y_true == 1
    n_pos = int(positives.sum())
    false_negatives = int(((predictions == 0) & positives).sum())
    return {
        "auroc": _auroc(y_true, scores),
        "auprc": _auprc(y_true, scores),
        "brier": float(np.mean((scores - y_true) ** 2)),
        "false_negative_rate": (false_negatives / n_pos) if n_pos else None,
        "positive_count": n_pos,
        "row_count": len(y_true),
    }


def synthesize_smoke_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Fabricate a tiny deterministic synthetic trace set for the CPU smoke.

    The rows carry the full required-episode schema and a simple, separable
    feature→label relationship (plus noise) so each label head has both classes
    and well-defined diagnostics. Deterministic via the config training seed; no
    private artifacts, network, or Slurm involved.

    Returns:
        A list of synthetic episode records.
    """
    training = config.get("training") or {}
    seed = int(training.get("seed", 0))
    rows = int(training.get("smoke_rows", 48))
    rng = np.random.default_rng(seed)

    records: list[dict[str, Any]] = []
    for index in range(rows):
        clearance = float(rng.uniform(0.1, 3.0))
        ped_distance = float(rng.uniform(0.5, 6.0))
        progress = float(rng.uniform(0.0, 1.0))
        min_distance = float(clearance + rng.uniform(-0.1, 0.1))
        avg_speed = float(rng.uniform(0.1, 0.9))
        command_count = int(rng.integers(10, 25))
        # Simple risk relationships: tight clearance -> collision/near-miss,
        # low progress -> low_progress label. Noise keeps it non-degenerate.
        collision = clearance < 0.6 and rng.random() < 0.85
        near_miss = clearance < 1.2 and rng.random() < 0.8
        low_progress = progress < 0.4 and rng.random() < 0.85
        records.append(
            {
                "scenario_id": f"synthetic_smoke_{index % 4}",
                "seed": seed + index,
                "candidate_id": CANDIDATE_ID,
                "termination_reason": "collision" if collision else "max_steps",
                "metrics": {
                    "min_distance": min_distance,
                    "avg_speed": avg_speed,
                    "goal_progress": progress,
                },
                "trajectory_features": {
                    "candidate_command_count": command_count,
                    "min_rollout_clearance_m": clearance,
                    "mean_pedestrian_distance_m": ped_distance,
                    "route_progress_delta": progress,
                },
                "labels": {
                    "collision": bool(collision),
                    "near_miss": bool(near_miss),
                    "low_progress": bool(low_progress),
                },
            }
        )
    return records


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON episode records from ``path``.

    Returns:
        The parsed episode records (one per non-blank JSONL line).

    Raises:
        LearnedRiskTrainerError: If the file is missing or a line is not a JSON
            object.
    """
    if not path.is_file():
        raise LearnedRiskTrainerError(f"trace fixture is not a file: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LearnedRiskTrainerError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise LearnedRiskTrainerError(f"{path}:{line_number} must be a JSON object")
        records.append(record)
    return records


def train_smoke(
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit one logistic head per label on ``records`` and return smoke diagnostics.

    Args:
        config: Parsed training config.
        records: Schema-complete episode records (synthetic or fixture).

    Returns:
        Mapping with ``row_count`` and per-label ``diagnostics``.

    Raises:
        LearnedRiskTrainerError: If records violate the feature/label contract.
    """
    features = list(config["features"])
    labels = list(config["labels"])
    model = config.get("model") or {}
    training = config.get("training") or {}

    x_raw, targets = build_matrix(records, features, labels)
    x = _standardize(x_raw)

    per_label: dict[str, Any] = {}
    for label in labels:
        y = targets[label]
        if len(np.unique(y)) < 2:
            # Degenerate single-class label: skip the fit but still report a
            # constant-prediction diagnostic instead of a misleading model.
            scores = np.full(len(y), float(y.mean()))
        else:
            weights = _fit_logistic(
                x,
                y,
                learning_rate=float(model.get("learning_rate", 0.1)),
                l2=float(model.get("l2", 0.0)),
                epochs=int(training.get("max_epochs", 100)),
            )
            scores = _predict(x, weights)
        per_label[label] = _diagnostics(y, scores)
    return {"row_count": int(x.shape[0]), "diagnostics": per_label}


def build_status(  # noqa: PLR0913 - flat status contract; each field is a named artifact key
    *,
    config: dict[str, Any],
    config_path: Path,
    mode: str,
    training_state: str,
    output_root: str,
    status_artifact_path: str,
    row_count: int | None = None,
    diagnostics: dict[str, Any] | None = None,
    blockers: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Assemble the stable status-artifact mapping.

    The shape is intentionally flat and JSON-serialisable so downstream tooling
    and PR #4552's launch-packet contract can branch on ``training_state`` without
    reconciling multiple documents.

    Returns:
        The status mapping written to the status artifact.
    """
    diagnostics_available = sorted(diagnostics.keys()) if diagnostics else []
    status: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "config_path": str(config_path),
        "launch_packet": config.get("launch_packet"),
        "trace_manifest": config.get("trace_manifest"),
        "mode": mode,
        "training_state": training_state,
        "labels": list(config.get("labels", [])),
        "features": list(config.get("features", [])),
        "row_count": row_count,
        "diagnostics_available": diagnostics_available,
        "diagnostics": diagnostics or {},
        "output_root": output_root,
        "status_artifact_path": status_artifact_path,
        "claim_boundary": CLAIM_BOUNDARY,
        "hard_guard_role": HARD_GUARD_ROLE,
        "slurm_submission": False,
    }
    if blockers is not None:
        status["blockers"] = blockers
    if error is not None:
        status["error"] = error
    return status


def write_status(status: dict[str, Any], status_out: Path) -> None:
    """Write ``status`` as pretty, key-sorted JSON, creating parent dirs."""
    status_out.parent.mkdir(parents=True, exist_ok=True)
    status_out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_launch_packet_or_raise(config: dict[str, Any], repo_root: Path) -> None:
    """Validate the launch packet the config binds to.

    Raises:
        LearnedRiskTrainerError: If the launch packet fails validation, wrapping
            the underlying contract error so callers see one trainer error type.
    """
    packet_path = Path(config["launch_packet"])
    try:
        validate_launch_packet(packet_path, repo_root=repo_root)
    except LearnedRiskLaunchPacketError as exc:
        raise LearnedRiskTrainerError(f"launch packet failed validation: {exc}") from exc


def evaluate_real_mode_readiness(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Return the campaign-readiness report for a real (non-smoke) run.

    Wraps :func:`evaluate_campaign_readiness` with the config's launch packet and
    trace manifest so real mode is gated on the same fail-closed decision the
    #1472 campaign gate uses.
    """
    return evaluate_campaign_readiness(
        config["launch_packet"],
        config["trace_manifest"],
        repo_root=repo_root,
    )


def is_launch_ready(readiness: dict[str, Any]) -> bool:
    """Return True only when the campaign-readiness report is launch ready."""
    return readiness.get("campaign_state") == CAMPAIGN_READY


__all__ = [
    "CANDIDATE_ID",
    "CLAIM_BOUNDARY",
    "HARD_GUARD_ROLE",
    "STATE_BLOCKED_TRACE_MANIFEST",
    "STATE_FAILED",
    "STATE_SMOKE_COMPLETED",
    "STATE_TRAINING_COMPLETED",
    "TRAINING_SCHEMA_VERSION",
    "LearnedRiskTrainerError",
    "build_matrix",
    "build_status",
    "evaluate_real_mode_readiness",
    "is_launch_ready",
    "load_jsonl_records",
    "load_training_config",
    "resolve_feature",
    "synthesize_smoke_records",
    "train_smoke",
    "validate_launch_packet_or_raise",
    "write_status",
]

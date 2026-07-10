"""Diagnostic social preference label config and annotation utilities.

This module provides config validation via ``load_social_preference_label_config`` and an
annotation entry point ``annotate_episode_social_preferences`` that applies threshold bands
from the config to episode metric traces. The labels are diagnostic-only and must not be used
as RL rewards or planner control signals.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.errors import RobotSfError

SOCIAL_PREFERENCE_LABEL_SCHEMA_VERSION = "social-preference-labels.v1"
REQUIRED_LABEL_IDS = frozenset(
    {
        "clearance",
        "ttc_margin",
        "pedestrian_displacement",
        "path_blocking",
        "oscillation",
        "detour_burden",
        "recovery_smoothness",
    }
)
REQUIRED_LABEL_FIELDS = frozenset(
    {
        "id",
        "display_name",
        "description",
        "metric_family",
        "unit",
        "preferred_direction",
        "diagnostic_thresholds",
        "required_trace_fields",
        "candidate_metric_keys",
        "computation_status",
        "notes",
    }
)
ANNOTATION_METHODS = frozenset({"threshold_band", "manual", "not_available"})


class SocialPreferenceLabelConfigError(RobotSfError, ValueError):
    """Raised when a social preference label config violates the v1 contract."""


def load_social_preference_label_config(path: Path) -> dict[str, Any]:
    """Load and validate a social preference label YAML config.

    Returns:
        The validated YAML payload as a dictionary.
    """

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return validate_social_preference_label_config(payload)


def validate_social_preference_label_config(payload: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Validate the diagnostic social preference label config contract.

    Returns:
        The validated payload as a dictionary.
    """

    if not isinstance(payload, Mapping):
        raise SocialPreferenceLabelConfigError("config payload must be a mapping")
    _validate_top_level_contract(payload)
    labels = _validate_labels(payload.get("labels"), _allowed_direction_set(payload))
    _validate_required_label_ids(labels)
    return dict(payload)


def label_availability(label: Mapping[str, Any], available_fields: set[str]) -> str:
    """Return diagnostic availability for one label and a set of available trace fields."""

    if label.get("computation_status") == "not_available":
        return "not_available"
    required_fields = label.get("required_trace_fields")
    if not isinstance(required_fields, list):
        raise SocialPreferenceLabelConfigError("label required_trace_fields must be a list")
    missing_fields = [field for field in required_fields if field not in available_fields]
    return "not_available" if missing_fields else "diagnostic_available"


def _validate_top_level_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SOCIAL_PREFERENCE_LABEL_SCHEMA_VERSION:
        raise SocialPreferenceLabelConfigError(
            f"schema_version must be {SOCIAL_PREFERENCE_LABEL_SCHEMA_VERSION}"
        )

    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise SocialPreferenceLabelConfigError("claim_boundary is required")
    _require_boundary_language(claim_boundary)

    _validate_source_literature(payload.get("source_literature"))


def _allowed_direction_set(payload: Mapping[str, Any]) -> set[str]:
    allowed_directions = payload.get("allowed_preferred_directions")
    if not isinstance(allowed_directions, list) or not allowed_directions:
        raise SocialPreferenceLabelConfigError(
            "allowed_preferred_directions must be a non-empty list"
        )

    allowed_direction_set = set()
    for direction in allowed_directions:
        if not isinstance(direction, str) or not direction:
            raise SocialPreferenceLabelConfigError("preferred directions must be non-empty strings")
        allowed_direction_set.add(direction)
    return allowed_direction_set


def _validate_labels(labels: Any, allowed_directions: set[str]) -> list[Mapping[str, Any]]:
    if not isinstance(labels, list) or not labels:
        raise SocialPreferenceLabelConfigError("labels must be a non-empty list")

    validated_labels: list[Mapping[str, Any]] = []
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise SocialPreferenceLabelConfigError(f"label {index} must be a mapping")
        _validate_label_fields(index, label)
        _validate_label_direction(label, allowed_directions)
        _validate_label_thresholds(label)
        _validate_label_trace_fields(label)
        _validate_label_candidate_metrics(label)
        validated_labels.append(label)
    return validated_labels


def _validate_required_label_ids(labels: list[Mapping[str, Any]]) -> None:
    label_ids = [label["id"] for label in labels]
    duplicate_ids = sorted({label_id for label_id in label_ids if label_ids.count(label_id) > 1})
    if duplicate_ids:
        raise SocialPreferenceLabelConfigError(f"duplicate label ids: {', '.join(duplicate_ids)}")

    missing_required = sorted(REQUIRED_LABEL_IDS.difference(label_ids))
    if missing_required:
        raise SocialPreferenceLabelConfigError(
            f"missing required label ids: {', '.join(missing_required)}"
        )


def _require_boundary_language(claim_boundary: str) -> None:
    normalized = claim_boundary.casefold()
    required_phrases = ("diagnostic", "not a reward", "not calibrated")
    missing = [phrase for phrase in required_phrases if phrase not in normalized]
    if missing:
        raise SocialPreferenceLabelConfigError(
            "claim_boundary must explicitly state diagnostic, not-reward, and not-calibrated status"
        )


def _validate_source_literature(source_literature: Any) -> None:
    if not isinstance(source_literature, list) or not source_literature:
        raise SocialPreferenceLabelConfigError("source_literature must be a non-empty list")
    for entry in source_literature:
        if not isinstance(entry, Mapping):
            raise SocialPreferenceLabelConfigError("source_literature entries must be mappings")
        if entry.get("role") != "motivation_only":
            raise SocialPreferenceLabelConfigError(
                "source_literature entries must use role motivation_only"
            )
        if not entry.get("url"):
            raise SocialPreferenceLabelConfigError("source_literature entries require url")


def _validate_label_fields(index: int, label: Mapping[str, Any]) -> None:
    missing_fields = sorted(REQUIRED_LABEL_FIELDS.difference(label))
    if missing_fields:
        raise SocialPreferenceLabelConfigError(
            f"label {index} missing required fields: {', '.join(missing_fields)}"
        )

    label_id = label["id"]
    if not isinstance(label_id, str) or not _is_lowercase_snake_case(label_id):
        raise SocialPreferenceLabelConfigError(f"label {index} id must be lowercase snake_case")

    for field_name in (
        "display_name",
        "description",
        "metric_family",
        "unit",
        "computation_status",
        "notes",
    ):
        value = label[field_name]
        if not isinstance(value, str) or not value.strip():
            raise SocialPreferenceLabelConfigError(
                f"label {label_id} {field_name} must be a non-empty string"
            )


def _validate_label_direction(label: Mapping[str, Any], allowed_directions: set[str]) -> None:
    label_id = str(label["id"])
    preferred_direction = label["preferred_direction"]
    if preferred_direction not in allowed_directions:
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} preferred_direction {preferred_direction!r} is not allowed"
        )


def _validate_label_thresholds(label: Mapping[str, Any]) -> None:
    label_id = str(label["id"])
    diagnostic_thresholds = label["diagnostic_thresholds"]
    if not isinstance(diagnostic_thresholds, Mapping):
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} diagnostic_thresholds must be a mapping"
        )

    status = diagnostic_thresholds.get("status")
    if status != "placeholder_default_not_human_calibrated":
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} threshold status must be placeholder_default_not_human_calibrated"
        )


def _validate_label_trace_fields(label: Mapping[str, Any]) -> None:
    _require_string_list(str(label["id"]), label["required_trace_fields"], "required_trace_fields")


def _validate_label_candidate_metrics(label: Mapping[str, Any]) -> None:
    label_id = str(label["id"])
    candidate_metric_keys = label["candidate_metric_keys"]
    if not isinstance(candidate_metric_keys, list):
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} candidate_metric_keys must be a list"
        )
    for metric_key in candidate_metric_keys:
        if not isinstance(metric_key, str) or not metric_key:
            raise SocialPreferenceLabelConfigError(
                f"label {label_id} candidate_metric_keys must contain non-empty strings"
            )
    if not candidate_metric_keys and label.get("computation_status") != "not_available":
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} without candidate_metric_keys must be not_available"
        )


def _require_string_list(label_id: str, value: Any, field_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise SocialPreferenceLabelConfigError(f"label {label_id} {field_name} must be a list")
    for item in value:
        if not isinstance(item, str) or not item:
            raise SocialPreferenceLabelConfigError(
                f"label {label_id} {field_name} must contain non-empty strings"
            )


def _is_lowercase_snake_case(value: str) -> bool:
    if not value:
        return False
    return value[0].islower() and all(
        char.islower() or char.isdigit() or char == "_" for char in value
    )


# ---------------------------------------------------------------------------
# Threshold-band parsing helpers (private)
# ---------------------------------------------------------------------------

_THRESHOLDS_RE = re.compile(r"^(<|<=|>=|>|=)\s*([0-9eE.+\-]+)\s*[\-,]*(\s*[0-9eE.+\-]+)?")
_BAND_RE = re.compile(r"(\[|\()?\s*([0-9eE.+\-]+)\s*,\s*([0-9eE.+\-]+)\s*([)\]])?")


def _get_metric_value(
    episode: Mapping[str, Any], keys: Sequence[str]
) -> tuple[str | None, float | None]:
    """Search ``episode["metrics"]`` for the first key whose dot-path resolves to a number.

    Non-finite values (NaN, Inf) and non-numeric leaves are skipped so that a
    later candidate key can still supply a usable value.

    Returns:
        A ``(matched_key, numeric_value)`` tuple, or ``(None, None)`` if no
        candidate key resolves to a finite number.
    """

    metrics = episode.get("metrics")
    if not isinstance(metrics, Mapping):
        return None, None

    for key in keys:
        value = _resolve_dot_path(metrics, key)
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if _is_finite(numeric):
                return key, numeric

    return None, None


def _resolve_dot_path(root: Mapping[str, Any], dot_path: str) -> Any:
    """Resolve a dotted metric key into a value.

    Candidate keys in the YAML config may include a ``metrics.`` prefix even though
    ``root`` is already the episode's ``metrics`` mapping. A leading ``metrics.`` segment
    is stripped before resolution.

    Returns:
        The resolved Python object, or ``None`` if the path does not resolve.
    """

    parts = [p for p in dot_path.split(".") if p]
    if parts and parts[0] == "metrics":
        parts = parts[1:]
    cur: Any = root
    for part in parts:
        if isinstance(cur, Mapping):
            cur = cur.get(part)
        else:
            return None
    return cur


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


# ---------------------------------------------------------------------------
# Public annotation API
# ---------------------------------------------------------------------------


def get_episode_metric_names(episode: Mapping[str, Any]) -> set[str]:
    """Return a set of all leaf key-paths available under the episode's ``metrics``.

    Flattens nested dicts using dot-notation (e.g. ``social_acceptability.x``).
    """

    metrics = episode.get("metrics")
    if not isinstance(metrics, Mapping):
        return set()

    result: set[str] = set()
    _flatten_dict(metrics, "", result)
    return result


def _flatten_dict(d: Mapping[str, Any], prefix: str, out: set[str]) -> None:
    """Recursively add dot-notation leaf keys to ``out``."""

    for key, value in d.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            _flatten_dict(value, full, out)
        elif value is not None:
            out.add(full)


def annotate_episode_social_preferences(
    episode: Mapping[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    trace_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Annotate a single episode record with diagnostic social preference labels.

    Applies the threshold-band rules from the schema to the metrics embedded in
    ``episode``. Labels whose required trace fields or candidate metric keys are
    missing are marked ``"not_available"`` rather than silently defaulting.

    The return value is safe to serialise as JSON and append to a ``.jsonl`` file.

    Args:
        episode: An episode record containing at least a ``"metrics"`` mapping.
        schema: Pre-loaded, validated label schema (``dict``). Mutually exclusive
            with ``config_path``; ``schema`` takes precedence.
        config_path: Path to the YAML config when schema is not provided.
        trace_fields: Optional set of trace-field names available in the episode.
            If not provided, availability checks use the presence of candidate
            metric keys in the episode's ``"metrics"`` mapping.

    Returns:
        A dictionary with ``schema_version``, ``claim_boundary``, ``episode_id``,
        and a ``labels`` list describing each label's diagnostic annotation.
    """

    if schema is None:
        if config_path is None:
            raise ValueError("annotate_episode_social_preferences requires schema or config_path")
        schema = load_social_preference_label_config(Path(config_path))

    labels_cfg = schema["labels"]
    schema_version = schema.get("schema_version", SOCIAL_PREFERENCE_LABEL_SCHEMA_VERSION)
    claim_boundary = schema.get("claim_boundary", "")
    source_metrics_present = get_episode_metric_names(episode)
    trace_fields = set(trace_fields) if trace_fields else None

    annotations: list[dict[str, Any]] = []

    for label in labels_cfg:
        annotation = _annotate_one_label(
            label,
            episode,
            trace_fields=trace_fields,
            source_metrics=source_metrics_present,
        )
        annotations.append(annotation)

    episode_id = (
        episode.get("episode_id")
        or episode.get("seed")
        or episode.get("scenario_seed")
        or "unknown"
    )
    return {
        "schema_version": schema_version,
        "claim_boundary": claim_boundary,
        "episode_id": episode_id,
        "labels": annotations,
    }


def _annotate_one_label(
    label: Mapping[str, Any],
    episode: Mapping[str, Any],
    *,
    trace_fields: set[str] | None,
    source_metrics: set[str],
) -> dict[str, Any]:
    """Annotate a single label for a single episode.

    Returns:
        A dictionary with label_id, display_name, metric_family, value, annotation,
        method, reason, evidence, and unit.
    """

    label_id = str(label["id"])
    display_name = str(label.get("display_name", label_id))
    metric_family = str(label.get("metric_family", "unknown"))
    unit = str(label.get("unit", ""))
    computation_status = str(label.get("computation_status", ""))

    # Check structural availability first
    if trace_fields is not None:
        avail = label_availability(label, trace_fields)
        if avail == "not_available":
            return {
                "label_id": label_id,
                "display_name": display_name,
                "metric_family": metric_family,
                "value": None,
                "annotation": "not_available",
                "method": "not_available",
                "reason": "required trace fields missing",
                "evidence": {},
                "unit": unit,
            }

    candidate_keys = label.get("candidate_metric_keys")
    if not isinstance(candidate_keys, list) or not candidate_keys:
        return {
            "label_id": label_id,
            "display_name": display_name,
            "metric_family": metric_family,
            "value": None,
            "annotation": "not_available",
            "method": "not_available",
            "reason": "no candidate metric keys",
            "evidence": {},
            "unit": unit,
        }

    # Try to find a metric value; matched_key is the key that produced it.
    matched_key, metric_value = _get_metric_value(episode, candidate_keys)

    if metric_value is None:
        return {
            "label_id": label_id,
            "display_name": display_name,
            "metric_family": metric_family,
            "value": None,
            "annotation": "not_available",
            "method": "not_available",
            "reason": f"none of candidate keys present: {candidate_keys}",
            "evidence": {
                "candidate_keys_checked": candidate_keys,
            },
            "unit": unit,
        }

    # Apply threshold bands
    thresholds = label.get("diagnostic_thresholds", {})
    bands: dict[str, str] = thresholds.get("bands", {})
    band_result = _classify_value_to_band(metric_value, bands)

    return {
        "label_id": label_id,
        "display_name": display_name,
        "metric_family": metric_family,
        "value": metric_value,
        "annotation": band_result,
        "method": "threshold_band",
        "reason": f"threshold band {band_result}",
        "evidence": {
            "metric_key": matched_key,
            "metric_value": metric_value,
            "threshold_bands": bands,
            "computation_status": computation_status,
        },
        "unit": unit,
    }


def _classify_value_to_band(raw_value: float, bands: dict[str, str]) -> str:
    """Classify a metric value into a band name using the YAML threshold bands.

    Falls back to ``"uncertain"`` when no band matches or the bands dictionary is empty.

    Returns:
        The band name string (e.g. ``"acceptable"``, ``"poor"``, ``"caution"``, ``"uncertain"``).
    """

    if not bands:
        return "uncertain"

    matched_band = _match_value_to_band_spec(raw_value, bands)
    if matched_band is not None:
        return matched_band

    # Fallback: try numeric sorting to find nearest band
    return _fallback_band_classification(raw_value, bands)


def _match_value_to_band_spec(value: float, bands: dict[str, str]) -> str | None:
    """Try each band spec; return the first band name that contains ``value``.

    Returns:
        The band name or ``None`` if no band matches.
    """

    for band_name, spec in bands.items():
        spec_str = str(spec).strip()
        if _value_in_band_spec(value, spec_str):
            return band_name
    return None


def _value_in_band_spec(value: float, spec: str) -> bool:
    """Return True if ``value`` falls in the range described by ``spec``."""

    spec = spec.strip()

    # Simple comparisons: "< 0.0", "<= 0.0", ">= 0.5", "> 1.0"
    m = _THRESHOLDS_RE.match(spec)
    if m:
        op = m.group(1) or "<"
        threshold = float(m.group(2))
        return _apply_op(value, op, threshold)

    # Interval: "[0.0, 0.5)", "(0.0, 1.0]", "(0.0, 1.0)", "[0.0, 0.5]"
    im = _BAND_RE.match(spec)
    if im:
        lo_bracket = im.group(1) or "["
        lo = float(im.group(2))
        hi = float(im.group(3))
        hi_bracket = im.group(4) or "]"

        lo_ok = value > lo if lo_bracket == "(" else value >= lo
        hi_ok = value < hi if hi_bracket == ")" else value <= hi
        return lo_ok and hi_ok

    return False


def _apply_op(value: float, op: str, threshold: float) -> bool:
    """Evaluate a comparison operator.

    Returns:
        ``True`` if the operator holds for ``value`` and ``threshold``.
    """

    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "=":
        return value == threshold
    return False


def _fallback_band_classification(value: float, bands: dict[str, str]) -> str:
    """Heuristic fallback when the YAML bands use prose instead of numeric ranges.

    Returns:
        The best-guess band name string.
    """

    # Extract numeric thresholds and their band names to sort them
    numeric_bands: list[tuple[float, str]] = []
    for band_name, spec in bands.items():
        spec_str = str(spec).strip()
        extracted = _extract_threshold_number(spec_str)
        if extracted is not None:
            numeric_bands.append((extracted, band_name))

    if not numeric_bands:
        return "uncertain"

    numeric_bands.sort()

    # Find the highest band whose threshold the value still meets.
    for threshold, band_name in reversed(numeric_bands):
        if value >= threshold:
            return band_name

    # Below all thresholds — use the lowest band
    return numeric_bands[0][1]


def _extract_threshold_number(spec: str) -> float | None:
    """Extract a leading numeric threshold from a band spec string.

    Returns:
        The extracted float or ``None`` if not parseable.
    """

    spec = spec.replace("[", "").replace("(", "").replace("]", "").replace(")", "").strip()
    parts = spec.split(",")
    for part in parts:
        part = part.strip()
        part = part.lstrip("<>=").strip()
        try:
            return float(part)
        except (ValueError, TypeError):
            continue
    return None


def annotate_episodes_social_preferences(
    episodes: Sequence[Mapping[str, Any]],
    *,
    schema: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    trace_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Annotate a sequence of episode records with social preference labels.

    Returns:
        A list of annotation dicts, one per episode.
    """

    return [
        annotate_episode_social_preferences(
            ep, schema=schema, config_path=config_path, trace_fields=trace_fields
        )
        for ep in episodes
    ]


def build_label_summary(annotations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build a summary of social preference label results across a batch of annotations.

    Counts label annotation values and ``not_available`` reasons per label.

    Returns:
        A dictionary with schema_version, claim_boundary, total_episodes, per-label
        annotation counts, and not_available_reasons.
    """

    schema_version = ""
    claim_boundary = ""
    label_counts: dict[str, dict[str, int]] = {}
    not_available_reasons: dict[str, dict[str, int]] = {}

    for annotation in annotations:
        schema_version = annotation.get("schema_version", schema_version) or schema_version
        claim_boundary = annotation.get("claim_boundary", claim_boundary) or claim_boundary

        for label_result in annotation.get("labels", []):
            label_id = label_result.get("label_id", "unknown")
            annotation_value = label_result.get("annotation", "unknown")

            if label_id not in label_counts:
                label_counts[label_id] = {}

            label_counts[label_id][annotation_value] = (
                label_counts[label_id].get(annotation_value, 0) + 1
            )

            if annotation_value == "not_available":
                if label_id not in not_available_reasons:
                    not_available_reasons[label_id] = {}
                reason = label_result.get("reason", "unknown")
                not_available_reasons[label_id][reason] = (
                    not_available_reasons[label_id].get(reason, 0) + 1
                )

    total_episodes = len(annotations)
    summary = {
        "schema_version": schema_version,
        "claim_boundary": claim_boundary,
        "total_episodes": total_episodes,
        "labels": {},
        "not_available_reasons": not_available_reasons,
    }

    for label_id, counts in sorted(label_counts.items()):
        label_summary = {
            "count": _sum_counts(counts) if counts else 0,
            "annotation_counts": dict(sorted(counts.items())),
        }
        summary["labels"][label_id] = label_summary

    return summary


def _sum_counts(d: dict[str, int]) -> int:
    """Sum the values in a dictionary of counts.

    Returns:
        The total count across all keys.
    """

    return sum(d.values())

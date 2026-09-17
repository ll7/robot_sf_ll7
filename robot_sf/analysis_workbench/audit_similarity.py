"""Explainable compatible-case similarity for audit records.

Similarity is a retrieval aid, not causal evidence and not confirmation of a
finding.  Every result exposes the matched features, missing fields, and
compatibility decision so an agent suggestion can be reviewed or rejected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from robot_sf.analysis_workbench.audit_contracts import EpisodeRef, Finding, Signal

SIMILARITY_MODE_SAME_SCENARIO = "same_scenario_across_planners"
SIMILARITY_MODE_SAME_PLANNER = "same_planner_across_seeds"
SIMILARITY_MODE_SYMPTOM = "symptom"
SIMILARITY_MODE_ANOMALY = "anomaly_signature"
SIMILARITY_MODE_GEOMETRY = "geometry"
SIMILARITY_MODE_OUTCOME = "outcome"
SIMILARITY_MODE_METRIC = "metric_behaviour"
SIMILARITY_MODE_FINDING = "existing_finding"

SIMILARITY_MODES = (
    SIMILARITY_MODE_SAME_SCENARIO,
    SIMILARITY_MODE_SAME_PLANNER,
    SIMILARITY_MODE_SYMPTOM,
    SIMILARITY_MODE_ANOMALY,
    SIMILARITY_MODE_GEOMETRY,
    SIMILARITY_MODE_OUTCOME,
    SIMILARITY_MODE_METRIC,
    SIMILARITY_MODE_FINDING,
)
_MODE_ALIASES = {
    "same_scenario": SIMILARITY_MODE_SAME_SCENARIO,
    "same_scenario_across_planners": SIMILARITY_MODE_SAME_SCENARIO,
    "same_planner": SIMILARITY_MODE_SAME_PLANNER,
    "same_planner_across_seeds": SIMILARITY_MODE_SAME_PLANNER,
    "anomaly": SIMILARITY_MODE_ANOMALY,
    "metric": SIMILARITY_MODE_METRIC,
    "finding": SIMILARITY_MODE_FINDING,
}
COMPATIBLE = "compatible"
INCOMPATIBLE = "incompatible"
UNKNOWN_COMPATIBILITY = "unknown"
COMPATIBILITY_STATUSES = (COMPATIBLE, INCOMPATIBLE, UNKNOWN_COMPATIBILITY)


class SimilarityError(ValueError):
    """Raised for an unsupported mode or malformed case identity."""


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """One ranked retrieval result with an auditable explanation."""

    query_id: str
    candidate_id: str
    mode: str
    score: float
    compatibility: str
    reasons: tuple[str, ...] = ()
    features: Mapping[str, Any] = field(default_factory=dict)
    missingness: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate score and normalize feature mappings."""

        if self.compatibility not in COMPATIBILITY_STATUSES:
            raise SimilarityError(f"unknown compatibility status: {self.compatibility}")
        if not isinstance(self.score, (int, float)) or not 0.0 <= float(self.score) <= 1.0:
            raise SimilarityError("similarity score must be between 0 and 1")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "features", dict(self.features or {}))

    @property
    def compatibility_status(self) -> str:
        """Return the compatibility status under its descriptive alias."""

        return self.compatibility

    @property
    def candidate_episode_id(self) -> str:
        """Return the candidate ID under the episode-oriented alias."""

        return self.candidate_id

    @property
    def matched_features(self) -> Mapping[str, Any]:
        """Return the feature contributions used in the explanation."""

        return self.features

    @property
    def is_compatible(self) -> bool:
        """Return whether all required comparable identity fields agree."""

        return self.compatibility == COMPATIBLE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible explanation.

        Returns:
            Result fields including reasons, features, and missingness.
        """

        return {
            "query_id": self.query_id,
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "score": self.score,
            "compatibility": self.compatibility,
            "reasons": list(self.reasons),
            "features": dict(self.features),
            "missingness": list(self.missingness),
            "warnings": list(self.warnings),
        }

    explain = to_dict


def _normalize_mode(mode: str) -> str:
    normalized = _MODE_ALIASES.get(mode, mode)
    if normalized not in SIMILARITY_MODES:
        raise SimilarityError(f"similarity mode must be one of {SIMILARITY_MODES}")
    return normalized


def _case_mapping(case: Any) -> dict[str, Any]:
    if isinstance(case, EpisodeRef):
        payload = asdict(case)
        payload["case_id"] = case.episode_id
        return payload
    if isinstance(case, Signal):
        payload = asdict(case)
        payload["case_id"] = case.episode_id or case.signal_id
        return payload
    if isinstance(case, Finding):
        payload = asdict(case)
        payload["case_id"] = case.finding_id
        return payload
    if isinstance(case, Mapping):
        payload = dict(case)
        nested = payload.get("episode") or payload.get("episode_ref") or payload.get("primary")
        if isinstance(nested, EpisodeRef):
            payload = {**asdict(nested), **payload}
        elif isinstance(nested, Mapping):
            payload = {**dict(nested), **payload}
        return payload
    raise SimilarityError(f"unsupported case value: {type(case).__name__}")


def _case_id(case: Mapping[str, Any]) -> str:
    for name in ("case_id", "episode_id", "finding_id", "signal_id", "id"):
        value = case.get(name)
        if isinstance(value, str) and value:
            return value
    raise SimilarityError("case is missing a stable ID")


def _value(case: Mapping[str, Any], name: str) -> Any:
    value = case.get(name)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _compatibility(
    query: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Compare identity fields without treating absent values as equal.

    Returns:
        Compatibility status, matching reasons, and missing/mismatched fields.
    """

    reasons: list[str] = []
    missing: list[str] = []
    incompatible: list[str] = []
    required_fields = ("campaign_digest", "source_digest")
    optional_fields = ("config_digest", "environment_digest")
    for field_name in (*required_fields, *optional_fields):
        left = query.get(field_name)
        right = candidate.get(field_name)
        if not left or not right:
            missing.append(field_name)
        elif left != right:
            incompatible.append(field_name)
        else:
            reasons.append(f"{field_name} matches")
    if incompatible:
        return (
            INCOMPATIBLE,
            tuple(reasons),
            tuple(missing + [f"mismatch:{field}" for field in incompatible]),
        )
    required_missing = [field for field in missing if field in required_fields]
    if required_missing:
        return UNKNOWN_COMPATIBILITY, tuple(reasons), tuple(missing)
    return COMPATIBLE, tuple(reasons), ()


def _set_values(case: Mapping[str, Any], *names: str) -> set[str]:
    result: set[str] = set()
    for name in names:
        value = case.get(name)
        if isinstance(value, str) and value:
            result.add(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result.update(str(item) for item in value if item)
        elif isinstance(value, Mapping):
            result.update(str(key) for key, item in value.items() if item is not None)
    return result


def _numeric_mapping(case: Mapping[str, Any], *names: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in names:
        value = case.get(name)
        if not isinstance(value, Mapping):
            continue
        for key, item in value.items():
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                result[str(key)] = float(item)
    return result


def _similar_features(  # noqa: C901, PLR0912, PLR0915
    query: Mapping[str, Any], candidate: Mapping[str, Any], mode: str
) -> tuple[float, list[str], dict[str, Any], list[str]]:
    reasons: list[str] = []
    features: dict[str, Any] = {}
    missing: list[str] = []
    if mode == SIMILARITY_MODE_SAME_SCENARIO:
        left, right = query.get("scenario_id"), candidate.get("scenario_id")
        if not left or not right:
            missing.append("scenario_id")
            return 0.0, reasons, features, missing
        if left != right:
            return 0.0, ["scenario_id differs"], {"scenario_match": False}, missing
        reasons.append("scenario_id matches")
        features["scenario_match"] = True
        planner_match = query.get("planner_id") == candidate.get("planner_id")
        features["planner_match"] = planner_match
        if planner_match:
            reasons.append("planner_id also matches; cross-planner contrast unavailable")
        else:
            reasons.append("planner_id differs as requested for cross-planner comparison")
        return 0.85 if not planner_match else 0.65, reasons, features, missing
    if mode == SIMILARITY_MODE_SAME_PLANNER:
        left, right = query.get("planner_id"), candidate.get("planner_id")
        if not left or not right:
            missing.append("planner_id")
            return 0.0, reasons, features, missing
        if left != right:
            return 0.0, ["planner_id differs"], {"planner_match": False}, missing
        reasons.append("planner_id matches")
        features["planner_match"] = True
        if query.get("seed") == candidate.get("seed"):
            reasons.append("seed matches; realization contrast is limited")
            return 0.6, reasons, features, missing
        reasons.append("seed differs as requested for realization comparison")
        return 0.85, reasons, features, missing
    if mode in {SIMILARITY_MODE_SYMPTOM, SIMILARITY_MODE_ANOMALY, SIMILARITY_MODE_FINDING}:
        left = _set_values(query, "symptom", "symptoms", "tags", "reason_code", "signal_ids")
        right = _set_values(candidate, "symptom", "symptoms", "tags", "reason_code", "signal_ids")
        if not left or not right:
            missing.append("symptom/tags")
            return 0.0, reasons, features, missing
        overlap = left & right
        union = left | right
        score = len(overlap) / len(union) if union else 0.0
        features.update(
            {
                "shared_features": sorted(overlap),
                "query_features": sorted(left),
                "candidate_features": sorted(right),
            }
        )
        if overlap:
            reasons.append(f"shared features: {', '.join(sorted(overlap))}")
        else:
            reasons.append("no shared symptom/anomaly features")
        return score, reasons, features, missing
    if mode == SIMILARITY_MODE_GEOMETRY:
        left = _set_values(query, "geometry_signature", "geometry", "map_id", "scenario_id")
        right = _set_values(candidate, "geometry_signature", "geometry", "map_id", "scenario_id")
        if not left or not right:
            missing.append("geometry_signature")
            return 0.0, reasons, features, missing
        overlap = left & right
        score = len(overlap) / len(left | right)
        features["shared_geometry_features"] = sorted(overlap)
        reasons.append("geometry features overlap" if overlap else "geometry differs")
        return score, reasons, features, missing
    if mode in {SIMILARITY_MODE_OUTCOME, SIMILARITY_MODE_METRIC}:
        left = _numeric_mapping(query, "outcome", "metrics")
        right = _numeric_mapping(candidate, "outcome", "metrics")
        keys = sorted(set(left) & set(right))
        if not keys:
            missing.append("outcome/metrics")
            return 0.0, reasons, features, missing
        deltas = {key: abs(left[key] - right[key]) for key in keys}
        scale = sum(abs(left[key]) + abs(right[key]) + 1.0 for key in keys)
        score = max(0.0, 1.0 - sum(deltas.values()) / scale)
        features.update({"shared_metrics": keys, "absolute_deltas": deltas})
        reasons.append(f"compared {len(keys)} common numeric outcome/metric fields")
        absent = sorted((set(left) | set(right)) - set(keys))
        missing.extend(f"metric:{key}" for key in absent)
        return score, reasons, features, missing
    raise SimilarityError(f"unsupported similarity mode: {mode}")


def compatible_case_similarity(
    query: Any,
    candidate: Any,
    *,
    mode: str = SIMILARITY_MODE_SAME_SCENARIO,
) -> SimilarityResult:
    """Compare one candidate and return an explained, compatibility-aware result.

    Returns:
        A result whose score is an uncalibrated retrieval priority only.
    """

    normalized_mode = _normalize_mode(mode)
    query_mapping = _case_mapping(query)
    candidate_mapping = _case_mapping(candidate)
    compatibility, identity_reasons, identity_missing = _compatibility(
        query_mapping, candidate_mapping
    )
    score, reasons, features, feature_missing = _similar_features(
        query_mapping, candidate_mapping, normalized_mode
    )
    missingness = tuple(dict.fromkeys((*identity_missing, *feature_missing)))
    warnings: list[str] = []
    if compatibility == INCOMPATIBLE:
        score = 0.0
        warnings.append("identity mismatch: candidate is not a compatible peer")
    elif compatibility == UNKNOWN_COMPATIBILITY:
        warnings.append("compatibility is unknown because required identity fields are missing")
    return SimilarityResult(
        query_id=_case_id(query_mapping),
        candidate_id=_case_id(candidate_mapping),
        mode=normalized_mode,
        score=score,
        compatibility=compatibility,
        reasons=tuple(dict.fromkeys((*identity_reasons, *reasons))),
        features=features,
        missingness=missingness,
        warnings=tuple(warnings),
    )


def find_similar_cases(
    query: Any,
    candidates: Iterable[Any],
    *,
    mode: str = SIMILARITY_MODE_SAME_SCENARIO,
    include_incompatible: bool = False,
    include_unknown: bool = True,
    limit: int | None = None,
) -> list[SimilarityResult]:
    """Rank candidate cases with deterministic tie-breaking and explanations.

    Returns:
        Results sorted by compatibility class, score, and candidate ID.
    """

    results = [compatible_case_similarity(query, candidate, mode=mode) for candidate in candidates]
    if not include_incompatible:
        results = [item for item in results if item.compatibility != INCOMPATIBLE]
    if not include_unknown:
        results = [item for item in results if item.compatibility != UNKNOWN_COMPATIBILITY]
    rank = {COMPATIBLE: 0, UNKNOWN_COMPATIBILITY: 1, INCOMPATIBLE: 2}
    results.sort(key=lambda item: (rank[item.compatibility], -item.score, item.candidate_id))
    return results if limit is None else results[: max(0, limit)]


def similar_cases(*args: Any, **kwargs: Any) -> list[SimilarityResult]:
    """Alias for :func:`find_similar_cases`.

    Returns:
        Deterministically ranked similarity results.
    """

    return find_similar_cases(*args, **kwargs)


def filter_compatible_cases(query: Any, candidates: Iterable[Any], *, mode: str) -> list[Any]:
    """Return only candidates with a fully compatible identity.

    Returns:
        Original candidate values in deterministic similarity order.
    """

    candidate_list = list(candidates)
    results = find_similar_cases(query, candidate_list, mode=mode, include_unknown=False)
    by_id = {_case_id(_case_mapping(candidate)): candidate for candidate in candidate_list}
    return [by_id[item.candidate_id] for item in results if item.is_compatible]


__all__ = [
    "COMPATIBILITY_STATUSES",
    "COMPATIBLE",
    "INCOMPATIBLE",
    "SIMILARITY_MODES",
    "SIMILARITY_MODE_ANOMALY",
    "SIMILARITY_MODE_FINDING",
    "SIMILARITY_MODE_GEOMETRY",
    "SIMILARITY_MODE_METRIC",
    "SIMILARITY_MODE_OUTCOME",
    "SIMILARITY_MODE_SAME_PLANNER",
    "SIMILARITY_MODE_SAME_SCENARIO",
    "SIMILARITY_MODE_SYMPTOM",
    "UNKNOWN_COMPATIBILITY",
    "SimilarityError",
    "SimilarityResult",
    "compatible_case_similarity",
    "filter_compatible_cases",
    "find_similar_cases",
    "similar_cases",
]

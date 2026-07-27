"""Learned proposal and ranking models over failure archive metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.adversarial.archive import failure_archive_feature_rows
from robot_sf.adversarial.certification import CertificationStatus, certify_candidate
from robot_sf.adversarial.config import CandidateSpec, SearchSpaceConfig
from robot_sf.adversarial.disjoint_evaluation import family_invariant_distance
from robot_sf.adversarial.scenario_manifest import (
    AdversarialScenarioManifest,
    GeneratorInfo,
    SourceLineage,
    build_manifest,
)

# --- Issue #3275 frozen fit-only payload derivation ---------------------------
#
# The frozen same-planner contract (configs/adversarial/issue_3275_same_planner_
# contract.json) requires FailureArchiveProposalModel to be initialized from a
# FIT-ONLY payload: exactly the corrected, nominally eligible
# classic_group_crossing_medium / social_force records, excluding both
# non-nominal fit-family records and the five classic_cross_trap_medium / goal
# records. The helpers below derive that payload
# deterministically from the corrected recertification artifact
# (docs/context/evidence/issue_5305_certified_archive/recertification_issue_6139.
# json), validate its count and hash, and build the fit-only archive payload from
# the certified archive. They never execute planners and never fall back to
# synthetic data.

#: Schema tag for the issue #6103 frozen contract config.
ISSUE_3275_CONTRACT_SCHEMA = "issue_3275_same_planner_contract.v1"

#: Canonical JSON separators used for deterministic SHA-256 digests.
_CANON_SEPARATORS = (",", ":")


@dataclass(frozen=True)
class FitPayload:
    """A fit-only payload derived from the corrected recertification artifact."""

    entry_ids: tuple[str, ...]
    count: int
    entry_ids_sha256: str
    non_eligible_fit_entry_ids: tuple[str, ...]
    excluded_entry_ids: tuple[str, ...]
    fit_family: str
    fit_planner: str
    source: str
    recertification_sha256: str
    archive_payload: dict[str, Any] = field(default_factory=dict)


def _canonical_sha256(payload: Any) -> str:
    """Return a deterministic SHA-256 digest over canonical JSON of ``payload``."""
    encoded = json.dumps(payload, sort_keys=True, separators=_CANON_SEPARATORS).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from ``path`` fail-closed."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_raw_sha256_pinned_json(
    path: str | Path,
    *,
    expected_sha256: Any,
    label: str,
) -> tuple[dict[str, Any], str]:
    """Load one JSON object after verifying its frozen raw-byte digest."""
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"frozen contract missing {label} SHA-256")
    raw_bytes = Path(path).read_bytes()
    observed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: observed={observed_sha256} expected={expected_sha256}"
        )
    try:
        payload = json.loads(raw_bytes)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, observed_sha256


def _fit_family_records(
    recertification_data: dict[str, Any], *, fit_family: str
) -> list[dict[str, Any]]:
    """Return well-formed recertification records for one frozen fit family.

    The corrected eligibility decision is authoritative for #3275. A malformed
    fit-family row must therefore fail closed instead of being silently omitted
    from either the nominal fit set or its recorded exclusion set.
    """
    records = recertification_data.get("records")
    if not isinstance(records, list):
        raise ValueError("recertification artifact missing 'records' list")
    family_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or record.get("scenario_family") != fit_family:
            continue
        archive_id = record.get("archive_id")
        after = record.get("after")
        if not isinstance(archive_id, str) or not archive_id:
            raise ValueError(f"fit-family record has invalid archive_id: {archive_id!r}")
        if not isinstance(after, dict) or not isinstance(after.get("benchmark_eligibility"), str):
            raise ValueError(
                f"fit-family record {archive_id!r} has no corrected benchmark_eligibility"
            )
        family_records.append(record)
    if not family_records:
        raise ValueError(f"no fit records found for scenario_family={fit_family!r}")
    return family_records


def derive_fit_ids_from_recertification(
    recertification_data: dict[str, Any],
    *,
    fit_family: str,
    required_benchmark_eligibility: str = "eligible",
    expected_count: int | None = None,
    expected_ids_sha256: str | None = None,
) -> list[str]:
    """Derive and validate the frozen fit IDs from the recertification artifact.

    Selects only records whose ``scenario_family`` equals ``fit_family`` and
    whose corrected ``after.benchmark_eligibility`` equals
    ``required_benchmark_eligibility``. When ``expected_count`` and/or
    ``expected_ids_sha256`` are provided, the derived set must match exactly or
    the function fails closed. The SHA-256 is computed over the canonical JSON
    of the sorted fit IDs, matching the contract's ``entry_ids_sha256``.

    Args:
        recertification_data: Parsed recertification artifact (dict with a
            ``records`` list).
        fit_family: The frozen fit scenario family.
        required_benchmark_eligibility: Corrected eligibility tier required for
            a nominal fit record. ``stress_only`` and all other tiers are
            excluded when the frozen contract requires ``eligible``.
        expected_count: Optional required fit-record count.
        expected_ids_sha256: Optional required canonical SHA-256 of sorted IDs.

    Returns:
        The sorted list of fit archive IDs.
    """
    if not isinstance(required_benchmark_eligibility, str) or not required_benchmark_eligibility:
        raise ValueError("required_benchmark_eligibility must be a non-empty string")
    fit_ids = sorted(
        str(record["archive_id"])
        for record in _fit_family_records(recertification_data, fit_family=fit_family)
        if record["after"]["benchmark_eligibility"] == required_benchmark_eligibility
    )
    if not fit_ids:
        raise ValueError(
            "no nominal fit records found for "
            f"scenario_family={fit_family!r}, "
            f"benchmark_eligibility={required_benchmark_eligibility!r}"
        )
    if expected_count is not None and len(fit_ids) != expected_count:
        raise ValueError(
            f"fit count drift: derived={len(fit_ids)} expected={expected_count}; "
            "a changed fit count requires a new fit-set hash and a re-evaluated "
            "power/sensitivity contract"
        )
    observed_sha = _canonical_sha256(fit_ids)
    if expected_ids_sha256 is not None and observed_sha != expected_ids_sha256:
        raise ValueError(
            f"fit IDs SHA-256 drift: derived={observed_sha} expected={expected_ids_sha256}"
        )
    return fit_ids


def derive_non_eligible_fit_ids_from_recertification(
    recertification_data: dict[str, Any],
    *,
    fit_family: str,
    required_benchmark_eligibility: str,
) -> list[str]:
    """Return fit-family IDs excluded by the corrected eligibility requirement."""
    if not isinstance(required_benchmark_eligibility, str) or not required_benchmark_eligibility:
        raise ValueError("required_benchmark_eligibility must be a non-empty string")
    return sorted(
        str(record["archive_id"])
        for record in _fit_family_records(recertification_data, fit_family=fit_family)
        if record["after"]["benchmark_eligibility"] != required_benchmark_eligibility
    )


def derive_excluded_ids_from_recertification(
    recertification_data: dict[str, Any],
    *,
    excluded_family: str,
) -> list[str]:
    """Return the sorted archive IDs of the held-out-family exclusion set."""
    records = recertification_data.get("records")
    if not isinstance(records, list):
        raise ValueError("recertification artifact missing 'records' list")
    return sorted(
        str(record["archive_id"])
        for record in records
        if isinstance(record, dict) and record.get("scenario_family") == excluded_family
    )


def build_fit_archive_payload(
    archive_data: dict[str, Any],
    fit_ids: list[str],
) -> dict[str, Any]:
    """Filter a certified archive down to the frozen fit IDs (fit-only payload)."""
    entries = archive_data.get("entries")
    if not isinstance(entries, list):
        raise ValueError("archive missing 'entries' list")
    fit_set = set(fit_ids)
    fit_entries = [entry for entry in entries if entry.get("archive_id") in fit_set]
    missing = sorted(fit_set - {entry.get("archive_id") for entry in fit_entries})
    if missing:
        raise ValueError(f"fit IDs absent from archive: {missing}")
    return {
        "schema_version": archive_data.get("schema_version", "adversarial_failure_archive.v1"),
        "entries": fit_entries,
        "null_test_manifest": archive_data.get("null_test_manifest"),
        "summary": {"fit_only_filter": "issue_3275_same_planner_contract"},
    }


def derive_fit_payload_from_recertification(  # noqa: PLR0913
    recertification_data: dict[str, Any],
    archive_data: dict[str, Any],
    *,
    fit_family: str,
    fit_planner: str,
    excluded_family: str,
    required_benchmark_eligibility: str = "eligible",
    expected_count: int | None = None,
    expected_ids_sha256: str | None = None,
    expected_non_eligible_count: int | None = None,
    expected_non_eligible_ids_sha256: str | None = None,
) -> FitPayload:
    """Derive the full fit-only payload for the frozen #3275 contract."""
    fit_ids = derive_fit_ids_from_recertification(
        recertification_data,
        fit_family=fit_family,
        required_benchmark_eligibility=required_benchmark_eligibility,
        expected_count=expected_count,
        expected_ids_sha256=expected_ids_sha256,
    )
    non_eligible_fit_ids = derive_non_eligible_fit_ids_from_recertification(
        recertification_data,
        fit_family=fit_family,
        required_benchmark_eligibility=required_benchmark_eligibility,
    )
    if (
        expected_non_eligible_count is not None
        and len(non_eligible_fit_ids) != expected_non_eligible_count
    ):
        raise ValueError(
            "non-eligible fit count drift: "
            f"derived={len(non_eligible_fit_ids)} expected={expected_non_eligible_count}"
        )
    observed_non_eligible_sha = _canonical_sha256(non_eligible_fit_ids)
    if (
        expected_non_eligible_ids_sha256 is not None
        and observed_non_eligible_sha != expected_non_eligible_ids_sha256
    ):
        raise ValueError(
            "non-eligible fit IDs SHA-256 drift: "
            f"derived={observed_non_eligible_sha} "
            f"expected={expected_non_eligible_ids_sha256}"
        )
    excluded_ids = derive_excluded_ids_from_recertification(
        recertification_data,
        excluded_family=excluded_family,
    )
    overlap = sorted(set(fit_ids) & set(excluded_ids))
    if overlap:
        raise ValueError(f"fit/excluded ID overlap detected: {overlap}")
    archive_payload = build_fit_archive_payload(archive_data, fit_ids)
    return FitPayload(
        entry_ids=tuple(fit_ids),
        count=len(fit_ids),
        entry_ids_sha256=_canonical_sha256(fit_ids),
        non_eligible_fit_entry_ids=tuple(non_eligible_fit_ids),
        excluded_entry_ids=tuple(excluded_ids),
        fit_family=fit_family,
        fit_planner=fit_planner,
        source="docs/context/evidence/issue_5305_certified_archive/recertification_issue_6139.json",
        recertification_sha256=str(recertification_data.get("recertification_sha256", "")),
        archive_payload=archive_payload,
    )


def load_issue_3275_contract(path_or_data: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load and schema-check the frozen #3275 contract config."""
    if isinstance(path_or_data, dict):
        data = path_or_data
    else:
        data = _load_json(path_or_data)
    schema = data.get("schema_version")
    if schema != ISSUE_3275_CONTRACT_SCHEMA:
        raise ValueError(
            f"unexpected contract schema_version: {schema!r}; "
            f"expected {ISSUE_3275_CONTRACT_SCHEMA!r}"
        )
    return data


def attach_robot_geometry(
    payload: FitPayload,
    recertification_data: dict[str, Any],
) -> None:
    """Attach recertification route metadata for legacy diagnostic consumers.

    The #3275 ranker does not call this helper: ``CandidateSpec.start/goal``
    already are robot-route endpoints, and the frozen scorer normalizes them
    with the pinned shared search space. The attached ``robot`` block is only
    retained for callers that inspect recertification metadata; it cannot
    influence frozen scores.
    """
    records = recertification_data.get("records")
    if not isinstance(records, list):
        raise ValueError("recertification artifact missing 'records' list")
    reconstruction_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        archive_id = record.get("archive_id")
        reconstruction = record.get("reconstruction")
        if isinstance(archive_id, str) and isinstance(reconstruction, dict):
            reconstruction_by_id[archive_id] = reconstruction
    for entry in payload.archive_payload.get("entries", []):
        archive_id = entry.get("archive_id")
        reconstruction = reconstruction_by_id.get(str(archive_id))
        if reconstruction is None:
            raise ValueError(
                f"fit entry {archive_id} has no reconstruction in the "
                "recertification artifact; cannot attach robot geometry"
            )
        entry["robot"] = {
            "robot_start": reconstruction.get("robot_start"),
            "robot_goal": reconstruction.get("robot_goal"),
            "map_file": reconstruction.get("map_file"),
        }


def validate_fit_payload_integrity(
    payload: FitPayload,
    *,
    expected_planner: str,
    expected_planner_config_sha256: str,
) -> dict[str, Any]:
    """Validate planner/family integrity of a fit payload; return any drift.

    Every fit entry must belong to the fit family, target the frozen planner, and
    carry the frozen planner config SHA-256. Any drift is returned as a dict of
    lists (empty dict means clean) so callers can fail closed.
    """
    drift: dict[str, list[str]] = {
        "family_drift": [],
        "planner_drift": [],
        "planner_config_sha256_drift": [],
    }
    for entry in payload.archive_payload.get("entries", []):
        archive_id = str(entry.get("archive_id"))
        if entry.get("scenario_family") != payload.fit_family:
            drift["family_drift"].append(archive_id)
        provenance = entry.get("provenance")
        if not isinstance(provenance, dict):
            drift["planner_drift"].append(archive_id)
            drift["planner_config_sha256_drift"].append(archive_id)
            continue
        if provenance.get("target_planner") != expected_planner:
            drift["planner_drift"].append(archive_id)
        if provenance.get("config_sha256") != expected_planner_config_sha256:
            drift["planner_config_sha256_drift"].append(archive_id)
    return {key: sorted(values) for key, values in drift.items() if values}


class FailureArchiveProposalModel:
    """A deterministic ranking and proposal model over failure archive metadata."""

    def __init__(
        self,
        archive_path_or_data: str | Path | dict[str, Any] | None = None,
        search_space: SearchSpaceConfig | None = None,
        *,
        fit_entry_ids: Any = None,
        feature_view: str = "absolute",
    ) -> None:
        """Initialize the FailureArchiveProposalModel.

        Args:
            archive_path_or_data: Filepath or parsed dictionary of archive entries.
            search_space: Search space bounds for normalizing distance metrics.
            fit_entry_ids: Optional frozen fit-entry ID allowlist (issue #3275).
                When provided, every entry whose ``archive_id`` is not in this set
                is dropped before ranking, so excluded and held-out-family records
                cannot influence scores or rank order even if the full archive is
                supplied. Negative regression: scores/ranks are identical whether
                the input archive contains only the fit IDs or also the excluded
                records.
            feature_view: ``"absolute"`` (legacy) or ``"family_invariant"`` (the
                frozen #3275 view normalized by the shared search-space contract).
        """
        self.archive_path_or_data = archive_path_or_data
        self.search_space = search_space
        self.fit_entry_ids: set[str] | None = (
            {str(identifier) for identifier in fit_entry_ids} if fit_entry_ids is not None else None
        )
        self.feature_view = feature_view
        self.excluded_entry_ids: list[str] = []
        self.entries: list[dict[str, Any]] = []
        self.state = "active"
        self.state_reason = "archive_loaded"

        # Load archive data
        if archive_path_or_data is None:
            self.state = "blocked"
            self.state_reason = "missing_archive"
            return

        try:
            load_state, load_reason, raw_entries = self._load_archive_payload(archive_path_or_data)
        except (ValueError, TypeError, json.JSONDecodeError, OSError):
            self.state = "blocked"
            self.state_reason = "archive_load_error"
            return
        if load_state == "blocked":
            self.state = "blocked"
            self.state_reason = load_reason
            return
        self.entries = self._apply_fit_filter(raw_entries)
        if self.state != "blocked":
            self.state_reason = "archive_loaded"

    @staticmethod
    def _load_archive_payload(
        archive_path_or_data: str | Path | dict[str, Any],
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Read and structurally validate an archive payload (path or dict).

        Returns a ``(state, reason, entries)`` triple. ``state`` is ``"blocked"``
        with a precise ``reason`` for any structural problem, or ``"active"``
        with the raw entry list ready for fit-only filtering.
        """
        if isinstance(archive_path_or_data, (str, Path)):
            path = Path(archive_path_or_data)
            if not path.exists() or path.stat().st_size == 0:
                return "blocked", "missing_or_empty_archive_file", []
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = archive_path_or_data

        if not isinstance(data, dict) or "entries" not in data:
            return "blocked", "malformed_archive_payload", []
        try:
            failure_archive_feature_rows(data)
        except ValueError as exc:
            return "blocked", f"invalid_failure_archive_schema: {exc}", []
        raw_entries = data.get("entries", [])
        if not isinstance(raw_entries, list):
            return "blocked", "malformed_archive_entries", []
        if not raw_entries:
            return "blocked", "empty_archive_entries", []
        if any(
            not isinstance(entry, dict) or not isinstance(entry.get("candidate", {}), dict)
            for entry in raw_entries
        ):
            return "blocked", "missing_candidate_metadata", []
        return "active", "archive_loaded", raw_entries

    def _apply_fit_filter(self, raw_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only entries whose ``archive_id`` is in the frozen fit allowlist.

        When :attr:`fit_entry_ids` is set, every other entry is dropped and
        recorded in :attr:`excluded_entry_ids`, so excluded and held-out-family
        records cannot influence scores or rank order. If any frozen fit ID is
        absent from the archive, the model fails closed. This is the structural
        guarantee behind the issue #6103 negative regression: scores/ranks are
        identical whether the input contains only the fit IDs or also the
        excluded records.
        """
        if self.fit_entry_ids is None:
            return list(raw_entries)
        kept: list[dict[str, Any]] = []
        excluded: list[str] = []
        for entry in raw_entries:
            archive_id = entry.get("archive_id")
            if archive_id in self.fit_entry_ids:
                kept.append(entry)
            else:
                excluded.append(str(archive_id))
        self.excluded_entry_ids = sorted(excluded)
        missing = sorted(self.fit_entry_ids - {entry.get("archive_id") for entry in kept})
        if missing:
            self.state = "blocked"
            self.state_reason = f"fit_entry_ids_missing_from_archive:{','.join(missing)}"
            return []
        return kept

    def _entry_distance(self, candidate: CandidateSpec, entry: dict[str, Any]) -> float:
        """Distance from ``candidate`` to one archive entry under the active view."""
        if self.feature_view != "family_invariant":
            return self._distance(candidate, entry.get("candidate", {}))
        anchor = entry.get("candidate", {})
        if self.search_space is None:
            raise ValueError(
                "family_invariant feature view requires the frozen shared search space"
            )
        return family_invariant_distance(candidate, anchor, self.search_space)

    def get_tabular_view(self) -> list[dict[str, Any]]:
        """Build a tabular feature view from archive entries."""
        return failure_archive_feature_rows(
            {"schema_version": "adversarial_failure_archive.v1", "entries": self.entries}
        )

    def _get_candidate_value(self, cand_dict: dict[str, Any], name: str) -> float | None:
        """Helper to safely extract a scalar feature from candidate dict."""
        if name.startswith("start_") or name.startswith("goal_"):
            parts = name.split("_")
            pose = cand_dict.get(parts[0], {})
            return pose.get(parts[1]) if isinstance(pose, dict) else None
        return cand_dict.get(name)

    def _get_feature_scale(self, name: str) -> float:
        """Helper to get normalization scale for a feature."""
        if self.search_space is not None:
            range_cfg = getattr(self.search_space, name, None)
            if range_cfg is not None and hasattr(range_cfg, "max") and hasattr(range_cfg, "min"):
                span = range_cfg.max - range_cfg.min
                if span > 0.0:
                    return span

        # Calculate scale from entries
        vals = []
        for entry in self.entries:
            val = self._get_candidate_value(entry.get("candidate", {}), name)
            if val is not None:
                vals.append(float(val))
        if vals:
            span = max(vals) - min(vals)
            if span > 0.0:
                return span
        return 1.0

    def _distance(self, c1: CandidateSpec, c2_dict: dict[str, Any]) -> float:
        """Calculate a normalized L1 distance between CandidateSpec and an archive candidate dict."""
        c2_start = c2_dict.get("start", {})
        features = {
            "start_x": (c1.start.x, c2_start.get("x")),
            "start_y": (c1.start.y, c2_start.get("y")),
            "goal_x": (c1.goal.x, c2_dict.get("goal", {}).get("x")),
            "goal_y": (c1.goal.y, c2_dict.get("goal", {}).get("y")),
            "spawn_time_s": (c1.spawn_time_s, c2_dict.get("spawn_time_s")),
            "pedestrian_speed_mps": (c1.pedestrian_speed_mps, c2_dict.get("pedestrian_speed_mps")),
            "pedestrian_delay_s": (c1.pedestrian_delay_s, c2_dict.get("pedestrian_delay_s")),
        }

        total_dist = 0.0
        for name, (v1, v2) in features.items():
            if v1 is not None and v2 is not None:
                scale = self._get_feature_scale(name)
                total_dist += abs(float(v1) - float(v2)) / scale

        return total_dist

    def score_candidate(
        self, candidate: CandidateSpec, strategy: str = "nearest_neighbor"
    ) -> float:
        """Calculate a ranking score for a candidate based on the archive."""
        if not self.entries:
            return 0.0

        distances = []
        for entry in self.entries:
            d = self._entry_distance(candidate, entry)
            distances.append((d, entry))

        if not distances:
            return 0.0

        if strategy == "nearest_neighbor":
            min_dist = min(d for d, _ in distances)
            return -min_dist

        elif strategy == "objective_weighted":
            epsilon = 0.1
            total_score = 0.0
            for d, entry in distances:
                obj = entry.get("objective_value")
                if obj is None:
                    obj = 0.0
                total_score += float(obj) / (d + epsilon)
            return total_score

        else:
            min_dist = min(d for d, _ in distances)
            return -min_dist

    def rank_candidates(
        self,
        candidates: list[CandidateSpec],
        strategy: str = "nearest_neighbor",
    ) -> list[tuple[CandidateSpec, float]]:
        """Rank candidates using the specified strategy.

        Returns:
            A list of tuples (candidate, score) sorted by score descending.
        """
        if not candidates:
            return []

        if self.state == "blocked" or not self.entries:
            return [(c, 0.0) for c in candidates]

        scored_candidates = []
        for candidate in candidates:
            score = self.score_candidate(candidate, strategy)
            scored_candidates.append((candidate, score))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        return scored_candidates

    def emit_manifest(
        self,
        candidate: CandidateSpec,
        *,
        source: SourceLineage | None = None,
        generator_seed: int = 0,
        candidate_index: int = 0,
    ) -> AdversarialScenarioManifest:
        """Emit an AdversarialScenarioManifest for a given candidate."""
        gen_info = GeneratorInfo(
            family="learned_proposal_model",
            generator_id="FailureArchiveProposalModel",
            seed=generator_seed,
            candidate_index=candidate_index,
        )
        return build_manifest(
            candidate,
            source=source,
            generator=gen_info,
            search_space=self.search_space,
        )

    def certify_candidate(
        self,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        require_certification: bool = False,
    ) -> CertificationStatus:
        """Certify a candidate using existing certification helpers."""
        return certify_candidate(
            candidate,
            scenario_yaml_path=scenario_yaml_path,
            require_certification=require_certification,
        )

    @classmethod
    def from_frozen_contract(  # noqa: C901
        cls,
        contract_path_or_data: str | Path | dict[str, Any],
        *,
        repo_root: str | Path | None = None,
        feature_view: str = "family_invariant",
    ) -> tuple[FailureArchiveProposalModel, dict[str, Any]]:
        """Build the fit-only model from the frozen #3275 contract.

        Derives the fit IDs from the corrected recertification artifact,
        validates count/hash/planner/family, ignores legacy robot geometry
        from the reconstruction, and constructs the model from the fit-only
        payload with the family-invariant feature view. Returns the model and a
        JSON-safe provenance dict for the contract checker.
        """
        if feature_view != "family_invariant":
            raise ValueError("the frozen #3275 contract requires the family_invariant feature view")
        contract = load_issue_3275_contract(contract_path_or_data)
        root = Path(repo_root) if repo_root is not None else Path.cwd()
        source = contract["source_lineage"]
        recertification_path = root / source["corrected_recertification_path"]
        recertification_bytes = recertification_path.read_bytes()
        expected_artifact_sha = source.get("corrected_recertification_artifact_sha256")
        if not isinstance(expected_artifact_sha, str) or not expected_artifact_sha:
            raise ValueError(
                "frozen contract is missing corrected recertification artifact SHA-256"
            )
        observed_artifact_sha = hashlib.sha256(recertification_bytes).hexdigest()
        if observed_artifact_sha != expected_artifact_sha:
            raise ValueError(
                "corrected recertification artifact SHA-256 mismatch: "
                f"observed={observed_artifact_sha} expected={expected_artifact_sha}"
            )
        try:
            recert = json.loads(recertification_bytes)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load corrected recertification artifact: {exc}") from exc
        if not isinstance(recert, dict):
            raise ValueError("corrected recertification artifact must be a JSON object")
        archive, observed_archive_sha = _load_raw_sha256_pinned_json(
            root / source["pre_correction_archive_path"],
            expected_sha256=source.get("pre_correction_archive_sha256"),
            label="pre-correction archive",
        )
        expected_recert_sha = source["corrected_recertification_sha256"]
        if recert.get("recertification_sha256") != expected_recert_sha:
            raise ValueError(
                "recertification_sha256 mismatch: file="
                f"{recert.get('recertification_sha256')} contract={expected_recert_sha}"
            )
        fit_cfg = contract["fit"]
        excl_cfg = contract["exclusions"]
        planner_cfg = contract["target_planner"]
        evaluation_cfg = contract["evaluation"]
        search_space_file = evaluation_cfg.get("search_space_path")
        expected_search_space_sha256 = evaluation_cfg.get("search_space_sha256")
        if not isinstance(search_space_file, str) or not search_space_file:
            raise ValueError(
                "frozen contract evaluation.search_space_path must be a non-empty string"
            )
        if not isinstance(expected_search_space_sha256, str) or not expected_search_space_sha256:
            raise ValueError(
                "frozen contract evaluation.search_space_sha256 must be a non-empty string"
            )
        search_space_path = root / search_space_file
        search_space_bytes = search_space_path.read_bytes()
        observed_search_space_sha256 = hashlib.sha256(search_space_bytes).hexdigest()
        if observed_search_space_sha256 != expected_search_space_sha256:
            raise ValueError(
                "frozen contract search-space SHA-256 mismatch: "
                f"observed={observed_search_space_sha256} "
                f"expected={expected_search_space_sha256}"
            )
        search_space = SearchSpaceConfig.from_file(search_space_path)
        map_file = evaluation_cfg.get("map_file")
        expected_map_sha256 = evaluation_cfg.get("map_file_sha256")
        if not isinstance(map_file, str) or not map_file:
            raise ValueError("frozen contract evaluation.map_file must be a non-empty string")
        if not isinstance(expected_map_sha256, str) or not expected_map_sha256:
            raise ValueError(
                "frozen contract evaluation.map_file_sha256 must be a non-empty string"
            )
        map_path = root / map_file
        if not map_path.is_file():
            raise ValueError(f"frozen contract evaluation map is missing: {map_file}")
        observed_map_sha256 = hashlib.sha256(map_path.read_bytes()).hexdigest()
        if observed_map_sha256 != expected_map_sha256:
            raise ValueError(
                "frozen contract evaluation map SHA-256 mismatch: "
                f"observed={observed_map_sha256} expected={expected_map_sha256}"
            )
        payload = derive_fit_payload_from_recertification(
            recert,
            archive,
            fit_family=fit_cfg["scenario_family"],
            fit_planner=fit_cfg["target_planner"],
            excluded_family=excl_cfg["scenario_family"],
            required_benchmark_eligibility=fit_cfg["required_benchmark_eligibility"],
            expected_count=fit_cfg["count"],
            expected_ids_sha256=fit_cfg["entry_ids_sha256"],
            expected_non_eligible_count=fit_cfg["excluded_from_nominal_fit_count"],
            expected_non_eligible_ids_sha256=fit_cfg["excluded_from_nominal_fit_entry_ids_sha256"],
        )
        planner_drift = validate_fit_payload_integrity(
            payload,
            expected_planner=planner_cfg["id"],
            expected_planner_config_sha256=planner_cfg["config_sha256"],
        )
        if planner_drift:
            raise ValueError(f"frozen fit payload has planner/family drift: {planner_drift}")
        model = cls(
            payload.archive_payload,
            search_space=search_space,
            fit_entry_ids=payload.entry_ids,
            feature_view=feature_view,
        )
        provenance = {
            "contract_schema_version": contract["schema_version"],
            "fit_count": payload.count,
            "fit_entry_ids_sha256": payload.entry_ids_sha256,
            "non_eligible_fit_count": len(payload.non_eligible_fit_entry_ids),
            "excluded_count": len(payload.excluded_entry_ids),
            "recertification_sha256": payload.recertification_sha256,
            "recertification_artifact_sha256": observed_artifact_sha,
            "pre_correction_archive_sha256": observed_archive_sha,
            "fit_only_initialized": model.state == "active",
            "model_state": model.state,
            "model_entry_count": len(model.entries),
            "excluded_entry_ids_dropped": model.excluded_entry_ids,
            "planner_drift": planner_drift,
            "feature_view": feature_view,
            "feature_semantics": "robot_route_and_controls_normalized_by_shared_search_space",
            "search_space_file": search_space_file,
            "search_space_sha256": observed_search_space_sha256,
            "evaluation_map_file": map_file,
            "evaluation_map_sha256": observed_map_sha256,
        }
        return model, provenance

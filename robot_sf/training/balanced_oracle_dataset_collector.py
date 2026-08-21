"""Collector for balanced non-learning oracle imitation datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.map_runner import map_runner
from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.errors import RobotSfError
from robot_sf.training.action_bin_accounting import compute_action_bin_accounting
from robot_sf.training.oracle_imitation_launch_packet import (
    load_launch_packet,
    validate_launch_packet,
)

EPISODE_ID_RE = re.compile(r"^(?P<split>[a-z_]+)__(?P<scenario>.+)__seed(?P<seed>\d+)$")
_SCHEMA_VERSION = "balanced-oracle-dataset-manifest.v1"
_PLAN_SCHEMA_VERSION = "balanced-oracle-collection-plan.v1"
_SPLITS = ("train", "validation", "evaluation")
_VALID_EXECUTION_MODES = frozenset({"native", "adapter", "mixed"})
_PACKET_FINGERPRINT_FIELDS = (
    "dataset_id",
    "source_candidate",
    "source_candidate_config",
    "scenario_source",
    "scenario_ids",
    "seeds_by_split",
    "episode_ids_by_split",
    "hard_slice_assignment",
    "relabeling_policy",
    "exclusion_rules",
)
_DIFFERENTIAL_REMEDY_CATEGORIES = (
    "scenario_roster_change",
    "minimum_change_with_scientific_justification",
    "budget_or_sampling_change",
    "collector_or_eligibility_defect_fix",
)
_CHECK_STATUSES = frozenset(
    {
        "eligible_complete",
        "blocked_scientific_yield",
        "blocked_integrity_or_lineage",
        "inconclusive_missing_input",
    }
)
_YIELD_STAT_FIELDS = (
    "declared",
    "attempted",
    "completed",
    "usable",
    "nondegenerate",
    "excluded",
    "failed",
    "missing",
    "usable_transitions",
    "target_minimum",
    "shortfall",
)
_YIELD_TOTAL_FIELDS = _YIELD_STAT_FIELDS[:-2] + ("target_minimum_usable_transitions",)


class BalancedDatasetCollectionError(RobotSfError, ValueError):
    """Raised when balanced oracle dataset collection fails or violates gates."""


def _git_sha(repo_root: Path) -> str | None:
    """Return the current HEAD git SHA of the repo, or None when git is unavailable."""
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy values into JSON-serializable Python types.

    Returns:
        The value with numpy arrays/scalars converted to native Python types.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(payload: Any) -> str:
    """Return a stable SHA-256 over the canonical JSON encoding of ``payload``."""
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _contains_degraded_marker(value: Any) -> bool:
    """Return whether nested runtime metadata reports fallback/degraded execution."""
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in {"fallback", "degraded", "fallback_or_degraded"} and bool(item):
                return True
            if key.endswith("fallback_count") and isinstance(item, (int, float)) and item > 0:
                return True
            if key == "readiness_status" and str(item).lower() in {"fallback", "degraded"}:
                return True
            if key == "availability_status" and str(item).lower() in {
                "failed",
                "not_available",
                "partial-failure",
            }:
                return True
            if _contains_degraded_marker(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_degraded_marker(item) for item in value)
    return False


def _is_explicit_true(value: Any) -> bool:
    """Return whether a marker is a real boolean true, not a truthy string."""
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _is_strict_integer(value: Any) -> bool:
    """Return whether ``value`` is an integer identity field, excluding booleans."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _has_explicit_invalid_marker(episode: dict[str, Any]) -> bool:
    """Return whether an episode carries an explicit invalid-result marker."""
    if _is_explicit_true(episode.get("invalid")):
        return True
    if str(episode.get("status", "")).strip().lower() == "invalid":
        return True
    provenance = episode.get("provenance")
    if not isinstance(provenance, dict):
        return False
    record = provenance.get("record")
    if not isinstance(record, dict):
        return False
    if str(record.get("status", "")).strip().lower() == "invalid":
        return True
    exclusion = record.get("scenario_exclusion")
    return isinstance(exclusion, dict) and (
        str(exclusion.get("status", "")).strip().lower() == "invalid"
    )


def _trajectory_step_count(episode: dict[str, Any]) -> int | None:
    """Return one aligned trajectory length, or ``None`` for a malformed row."""
    fields = ("actions", "observations", "positions", "rewards", "terminated", "truncated")
    lengths: list[int] = []
    for field in fields:
        value = episode.get(field)
        if value is None or isinstance(value, (str, bytes, dict)):
            return None
        try:
            lengths.append(len(value))
        except (TypeError, ValueError):
            return None
    if len(set(lengths)) != 1:
        return None
    return lengths[0]


def _episode_exclusion_reason(episode: dict[str, Any]) -> tuple[str | None, int]:
    """Classify one row and return its reason plus a safe step count.

    Returns:
        Tuple of the exclusion reason, or ``None`` for usable rows, and step count.
    """
    provenance = episode.get("provenance")
    raw_steps = _trajectory_step_count(episode)
    steps = raw_steps or 0
    if bool(episode.get("leakage_invalid", False)):
        return "leakage_invalid", steps
    if _has_explicit_invalid_marker(episode):
        return "invalid", steps
    if bool(episode.get("failed", False)):
        return "failed", steps
    if bool(episode.get("fallback", False) or episode.get("degraded", False)):
        return "fallback", steps
    if (
        bool(episode.get("provenance_incomplete", False))
        or not isinstance(provenance, dict)
        or not provenance
    ):
        return "provenance_incomplete", steps

    if raw_steps is None:
        return "invalid", 0
    if raw_steps <= 1:
        return "one-step", raw_steps
    return None, raw_steps


def _file_sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _packet_fingerprint_payload(packet: dict[str, Any]) -> dict[str, Any]:
    """Return the scientific/execution fields that define a launch packet."""
    return {field: copy.deepcopy(packet.get(field)) for field in _PACKET_FINGERPRINT_FIELDS}


def compute_packet_fingerprint(packet: dict[str, Any]) -> str:
    """Return a stable fingerprint for the packet's scientific/execution inputs."""
    return _canonical_sha256(_packet_fingerprint_payload(packet))


def _validate_exhausted_attempt(attempt: Any, index: int) -> tuple[str, dict[str, Any]]:
    """Validate one exhausted-attempt digest and its complete comparison payload.

    Returns:
        Tuple of the verified digest and the payload used to compute it.
    """
    if isinstance(attempt, str):
        attempt = {"fingerprint": attempt}
    if not isinstance(attempt, dict):
        raise BalancedDatasetCollectionError(
            f"Exhausted attempt {index} must be a mapping with a packet fingerprint"
        )
    attempt_fingerprint = attempt.get("packet_fingerprint", attempt.get("fingerprint"))
    if not isinstance(attempt_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", attempt_fingerprint
    ):
        raise BalancedDatasetCollectionError(
            f"Exhausted attempt {index} must carry a lowercase 64-character packet fingerprint"
        )
    reference_payload = attempt.get(
        "packet_fingerprint_payload", attempt.get("fingerprint_payload")
    )
    if not isinstance(reference_payload, dict):
        raise BalancedDatasetCollectionError(
            f"Exhausted attempt {index} must carry a complete packet fingerprint payload"
        )
    missing_fields = [
        field for field in _PACKET_FINGERPRINT_FIELDS if field not in reference_payload
    ]
    if missing_fields:
        raise BalancedDatasetCollectionError(
            f"Exhausted attempt {index} fingerprint payload is missing fields: {missing_fields}"
        )
    reference_fingerprint = _canonical_sha256(_packet_fingerprint_payload(reference_payload))
    if reference_fingerprint != attempt_fingerprint:
        raise BalancedDatasetCollectionError(
            f"Exhausted attempt {index} packet fingerprint does not match its payload"
        )
    return attempt_fingerprint, reference_payload


def _changed_packet_fields(proposed: dict[str, Any], reference: dict[str, Any] | None) -> list[str]:
    """Return exact packet fingerprint fields that differ from a reference payload."""
    if reference is None:
        return ["unknown_without_reference_payload"]
    return [
        field for field in _PACKET_FINGERPRINT_FIELDS if proposed.get(field) != reference.get(field)
    ]


def validate_packet_difference(
    packet: dict[str, Any], exhausted_attempts: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Reject a packet whose fingerprint matches an exhausted attempt.

    Attempt records must carry ``packet_fingerprint`` or ``fingerprint`` plus a complete
    ``packet_fingerprint_payload`` to make changed fields auditable.

    Returns:
        A deterministic comparison report, unless an unchanged attempt is rejected.
    """
    proposed_payload = _packet_fingerprint_payload(packet)
    proposed_fingerprint = _canonical_sha256(proposed_payload)
    attempts = exhausted_attempts or []
    comparisons: list[dict[str, Any]] = []

    for index, attempt in enumerate(attempts):
        attempt_fingerprint, reference_payload = _validate_exhausted_attempt(attempt, index)
        if proposed_fingerprint == attempt_fingerprint:
            raise BalancedDatasetCollectionError(
                "Unchanged exhausted packet rejected before collection submission: "
                f"fingerprint={proposed_fingerprint}, attempt={index}"
            )
        comparisons.append(
            {
                "attempt_index": index,
                "attempt_fingerprint": attempt_fingerprint,
                "changed_fields": _changed_packet_fields(proposed_payload, reference_payload),
            }
        )

    return {
        "status": "changed" if comparisons else "not_checked",
        "proposed_fingerprint": proposed_fingerprint,
        "fingerprint_fields": list(_PACKET_FINGERPRINT_FIELDS),
        "comparisons": comparisons,
    }


def parse_episode_id(episode_id: str, split: str | None = None) -> tuple[str, str, int]:
    """Parse launch-packet episode ID into (split, scenario_id, seed).

    Args:
        episode_id: Raw episode ID string.
        split: Expected split name.

    Returns:
        Tuple of (split, scenario_id, seed).
    """
    match = EPISODE_ID_RE.match(episode_id)
    if match is None:
        raise BalancedDatasetCollectionError(f"Invalid episode ID format: {episode_id!r}")
    parsed_split = match.group("split")
    if split is not None and parsed_split != split:
        raise BalancedDatasetCollectionError(
            f"Episode ID split mismatch for {episode_id!r}: expected {split!r}, got {parsed_split!r}"
        )
    return parsed_split, match.group("scenario"), int(match.group("seed"))


def validate_split_and_episode_invariants(packet: dict[str, Any]) -> None:  # noqa: C901
    """Validate split seed overlap and duplicate episode ID invariants."""
    seeds_by_split = packet.get("seeds_by_split", {})
    if isinstance(seeds_by_split, dict):
        for i, left in enumerate(_SPLITS):
            left_seeds = set(seeds_by_split.get(left, []))
            for right in _SPLITS[i + 1 :]:
                right_seeds = set(seeds_by_split.get(right, []))
                overlap = sorted(left_seeds & right_seeds)
                if overlap:
                    raise BalancedDatasetCollectionError(
                        f"Seed overlap detected between {left} and {right}: {overlap}"
                    )

    scenario_ids = packet.get("scenario_ids", [])
    allowed_scenarios = set(scenario_ids) if isinstance(scenario_ids, list) else set()
    episodes_by_split = packet.get("episode_ids_by_split", {})
    if isinstance(episodes_by_split, dict):
        all_ids: set[str] = set()
        for split in _SPLITS:
            split_ids = episodes_by_split.get(split, [])
            if not isinstance(split_ids, list) or not split_ids:
                raise BalancedDatasetCollectionError(
                    f"episode_ids_by_split.{split} must be a non-empty list"
                )
            for ep_id in split_ids:
                if ep_id in all_ids:
                    raise BalancedDatasetCollectionError(
                        f"Duplicate episode ID detected in launch packet: {ep_id!r}"
                    )
                parsed_split, scenario_id, seed = parse_episode_id(str(ep_id), split)
                if parsed_split != split:
                    raise BalancedDatasetCollectionError(
                        f"Episode ID {ep_id!r} does not belong to split {split!r}"
                    )
                if allowed_scenarios and scenario_id not in allowed_scenarios:
                    raise BalancedDatasetCollectionError(
                        f"Episode ID {ep_id!r} references undeclared scenario {scenario_id!r}"
                    )
                split_seeds = set(seeds_by_split.get(split, []))
                if split_seeds and seed not in split_seeds:
                    raise BalancedDatasetCollectionError(
                        f"Episode ID {ep_id!r} uses seed {seed} outside seeds_by_split.{split}"
                    )
                all_ids.add(ep_id)


def _yield_ledger_integrity_error(  # noqa: C901, PLR0912 - ordered fail-closed guards
    manifest: dict[str, Any],
    ledger: dict[str, Any],
    strata: Any,
    yield_gates: Any,
) -> str | None:
    """Return a reason when a yield ledger is structurally inconsistent."""
    if ledger.get("schema_version") != "yield-ledger.v1":
        return "Yield ledger schema_version is missing or unsupported"
    if not isinstance(yield_gates, dict):
        return "Manifest yield_gates must be a mapping"
    gate_status = yield_gates.get("status")
    if gate_status not in {"pass", "fail"}:
        return "Manifest yield_gates.status must be 'pass' or 'fail'"

    min_transitions = yield_gates.get("min_usable_transitions")
    min_episodes = yield_gates.get("min_episodes_per_stratum")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (min_transitions, min_episodes)
    ):
        return "Manifest yield gate thresholds must be non-negative integers"

    scenario_ids = manifest.get("scenario_ids")
    if (
        not isinstance(scenario_ids, list)
        or not scenario_ids
        or not all(isinstance(value, str) and value for value in scenario_ids)
    ):
        return "Manifest scenario_ids must be a non-empty list of strings"
    if not isinstance(strata, dict):
        return "Yield ledger strata must be a mapping"
    expected_splits = set(_SPLITS)
    actual_splits = set(strata)
    if actual_splits != expected_splits:
        missing_splits = sorted(expected_splits - actual_splits)
        unexpected_splits = sorted(actual_splits - expected_splits)
        return (
            "Yield ledger strata keys are inconsistent"
            f" (missing={missing_splits}, unexpected={unexpected_splits})"
        )
    expected_scenarios = set(scenario_ids)

    totals = ledger.get("totals")
    if not isinstance(totals, dict):
        return "Yield ledger totals must be a mapping"
    for field in _YIELD_TOTAL_FIELDS:
        value = totals.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return f"Yield ledger totals.{field} must be a non-negative integer"
    if totals["target_minimum_usable_transitions"] != min_transitions:
        return "Yield ledger transition threshold does not match yield_gates"

    sums = dict.fromkeys(_YIELD_STAT_FIELDS, 0)
    for split in _SPLITS:
        scenarios = strata.get(split)
        if not isinstance(scenarios, dict):
            return f"Yield ledger is missing strata mapping for {split}"
        missing_scenarios = [scenario for scenario in scenario_ids if scenario not in scenarios]
        if missing_scenarios:
            return f"Yield ledger is missing scenarios for {split}: {missing_scenarios}"
        unexpected_scenarios = sorted(set(scenarios) - expected_scenarios)
        if unexpected_scenarios:
            return f"Yield ledger has unexpected scenarios for {split}: {unexpected_scenarios}"
        for scenario_id in scenario_ids:
            stats = scenarios[scenario_id]
            if not isinstance(stats, dict):
                return f"Yield ledger stats for {split}/{scenario_id} must be a mapping"
            for field in _YIELD_STAT_FIELDS:
                value = stats.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    return f"Yield ledger {split}/{scenario_id}.{field} is invalid"
                sums[field] += value
            reason_counts = stats.get("reason_counts")
            if not isinstance(reason_counts, dict) or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in reason_counts.values()
            ):
                return f"Yield ledger {split}/{scenario_id}.reason_counts is invalid"
            if stats["nondegenerate"] != stats["usable"]:
                return (
                    f"Yield ledger {split}/{scenario_id} nondegenerate count must match usable "
                    "under the one-step exclusion contract"
                )
            expected_shortfall = (
                max(0, stats["target_minimum"] - stats["usable"]) if split == "train" else 0
            )
            if stats["shortfall"] != expected_shortfall:
                return f"Yield ledger {split}/{scenario_id}.shortfall is inconsistent"
            if stats["target_minimum"] != (min_episodes if split == "train" else 0):
                return f"Yield ledger {split}/{scenario_id}.target_minimum is inconsistent"
            if stats["usable"] > stats["completed"] or stats["completed"] > stats["attempted"]:
                return f"Yield ledger {split}/{scenario_id} count ordering is inconsistent"
            if stats["failed"] > stats["excluded"]:
                return f"Yield ledger {split}/{scenario_id}.failed exceeds excluded"
            if sum(reason_counts.values()) != stats["excluded"]:
                return f"Yield ledger {split}/{scenario_id}.reason_counts do not sum to excluded"
            if stats["attempted"] + stats["missing"] != stats["declared"]:
                return f"Yield ledger {split}/{scenario_id} does not account for attempted IDs"
            if stats["usable"] + stats["excluded"] != stats["attempted"]:
                return f"Yield ledger {split}/{scenario_id} does not account for attempted rows"

    differential_remedy = ledger.get("differential_remedy")
    if not isinstance(differential_remedy, dict):
        return "Yield ledger differential_remedy must be a mapping"
    if differential_remedy.get("category") not in _DIFFERENTIAL_REMEDY_CATEGORIES:
        return "Yield ledger differential_remedy category is unsupported"
    if differential_remedy.get("selected") is not False:
        return "Yield ledger differential_remedy must remain unselected"
    for field in ("selection_status", "rationale"):
        if not isinstance(differential_remedy.get(field), str) or not differential_remedy[field]:
            return f"Yield ledger differential_remedy.{field} must be a non-empty string"
    for field in ("fields_to_change", "evidence_required"):
        value = differential_remedy.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            return f"Yield ledger differential_remedy.{field} must be a list of strings"

    for field in _YIELD_STAT_FIELDS[:-2]:
        if sums[field] != totals[field]:
            return f"Yield ledger totals.{field} does not match per-stratum sums"
    if gate_status == "pass":
        if manifest.get("eligibility_status") != "training_ready":
            return "Passed yield gates require training_ready eligibility_status"
        if any(strata["train"][scenario]["shortfall"] for scenario in scenario_ids):
            return "Passed yield gates contain a training-stratum shortfall"
        if totals["usable_transitions"] < min_transitions:
            return "Passed yield gates do not satisfy the transition threshold"
    elif manifest.get("eligibility_status") == "training_ready":
        return "Failed yield gates cannot have training_ready eligibility_status"
    return None


def check_yield_status(  # noqa: C901, PLR0912 - ordered fail-closed verdict guards
    output_root: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Check the yield status of an existing manifest. Deterministic and side-effect-free.

    Returns exactly one of: eligible_complete, blocked_scientific_yield,
    blocked_integrity_or_lineage, or inconclusive_missing_input.

    Returns:
        Dictionary with check_status, reason, and manifest_path fields.
    """
    manifest_path = Path(output_root) / "balanced_oracle_dataset_manifest.json"

    if not manifest_path.is_file():
        return {
            "check_status": "inconclusive_missing_input",
            "reason": "No manifest found at expected path",
            "manifest_path": str(manifest_path),
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "check_status": "inconclusive_missing_input",
            "reason": f"Manifest is unavailable or invalid: {type(exc).__name__}",
            "manifest_path": str(manifest_path),
        }

    if not isinstance(manifest, dict):
        return {
            "check_status": "inconclusive_missing_input",
            "reason": "Manifest must contain a JSON object",
            "manifest_path": str(manifest_path),
        }

    ledger = manifest.get("yield_ledger")
    if not isinstance(ledger, dict) or not isinstance(ledger.get("lineage"), dict):
        return {
            "check_status": "inconclusive_missing_input",
            "reason": "Manifest lacks the required yield ledger and lineage summary",
            "manifest_path": str(manifest_path),
        }

    identity_defects = manifest.get("identity_defects", [])
    if not isinstance(identity_defects, list):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Manifest identity_defects must be a list",
            "manifest_path": str(manifest_path),
        }
    if identity_defects:
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": f"Identity defects detected: {identity_defects[:5]}",
            "manifest_path": str(manifest_path),
        }

    ledger_identity_defects = ledger.get("identity_defects")
    if not isinstance(ledger_identity_defects, list):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Yield ledger identity_defects must be a list",
            "manifest_path": str(manifest_path),
        }
    if ledger_identity_defects:
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": f"Yield ledger identity defects detected: {ledger_identity_defects[:5]}",
            "manifest_path": str(manifest_path),
        }

    lineage = ledger["lineage"]
    required_lineage = ("source_candidate", "config_path", "source_packet_sha256", "commit")
    missing_lineage = [field for field in required_lineage if not lineage.get(field)]
    if missing_lineage:
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Missing lineage fields: " + ", ".join(missing_lineage),
            "manifest_path": str(manifest_path),
        }
    if not isinstance(lineage["source_candidate"], str) or not isinstance(
        lineage["config_path"], str
    ):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Lineage source_candidate and config_path must be strings",
            "manifest_path": str(manifest_path),
        }
    if re.fullmatch(r"[0-9a-f]{64}", str(lineage["source_packet_sha256"])) is None:
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Lineage source_packet_sha256 is malformed",
            "manifest_path": str(manifest_path),
        }
    if re.fullmatch(r"[0-9a-f]{40}", str(lineage["commit"])) is None:
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Lineage commit is malformed",
            "manifest_path": str(manifest_path),
        }
    if lineage["source_candidate"] != manifest.get("source_candidate"):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Lineage source_candidate does not match manifest metadata",
            "manifest_path": str(manifest_path),
        }
    if lineage["source_packet_sha256"] != manifest.get("source_packet_sha256"):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Lineage source_packet_sha256 does not match manifest metadata",
            "manifest_path": str(manifest_path),
        }

    missing = manifest.get("missing_episode_ids", [])
    if not isinstance(missing, list):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Manifest missing_episode_ids must be a list",
            "manifest_path": str(manifest_path),
        }
    if missing:
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": f"Missing episode IDs: {missing[:10]}",
            "manifest_path": str(manifest_path),
        }

    exclusions = manifest.get("exclusions", [])
    integrity_reasons = {
        "leakage_invalid",
        "provenance_incomplete",
        "duplicate_episode_id",
        "unexpected_episode_id",
    }
    if not isinstance(exclusions, list) or any(
        not isinstance(exclusion, dict) for exclusion in exclusions
    ):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Manifest exclusions must be a list of mappings",
            "manifest_path": str(manifest_path),
        }
    if any(ex.get("reason") in integrity_reasons for ex in exclusions):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Integrity-invalid episode rows detected",
            "manifest_path": str(manifest_path),
        }

    stored_fingerprint = manifest.get("packet_fingerprint")
    if (
        not isinstance(stored_fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", stored_fingerprint) is None
    ):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Missing or malformed packet fingerprint",
            "manifest_path": str(manifest_path),
        }

    fingerprint_fields = manifest.get("packet_fingerprint_fields")
    if fingerprint_fields != list(_PACKET_FINGERPRINT_FIELDS):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Packet fingerprint fields are missing or unsupported",
            "manifest_path": str(manifest_path),
        }
    fingerprint_payload = manifest.get("packet_fingerprint_payload")
    if not isinstance(fingerprint_payload, dict) or any(
        field not in fingerprint_payload for field in _PACKET_FINGERPRINT_FIELDS
    ):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Packet fingerprint payload is missing required fields",
            "manifest_path": str(manifest_path),
        }
    if _canonical_sha256(_packet_fingerprint_payload(fingerprint_payload)) != stored_fingerprint:
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Packet fingerprint does not match its payload",
            "manifest_path": str(manifest_path),
        }

    for field in (
        "dataset_id",
        "source_candidate",
        "source_candidate_config",
        "scenario_ids",
        "seeds_by_split",
        "episode_ids_by_split",
        "hard_slice_assignment",
        "relabeling_policy",
        "exclusion_rules",
    ):
        if field not in manifest or manifest[field] != fingerprint_payload.get(field):
            return {
                "check_status": "blocked_integrity_or_lineage",
                "reason": f"Manifest {field} does not match packet fingerprint payload",
                "manifest_path": str(manifest_path),
            }

    if config_path:
        try:
            current_fingerprint = compute_packet_fingerprint(load_launch_packet(config_path))
            current_source_packet_sha256 = _file_sha256(config_path)
        except (OSError, ValueError) as exc:
            return {
                "check_status": "inconclusive_missing_input",
                "reason": f"Packet input is unavailable or invalid: {type(exc).__name__}",
                "manifest_path": str(manifest_path),
            }
        if current_fingerprint != stored_fingerprint:
            return {
                "check_status": "blocked_integrity_or_lineage",
                "reason": "Packet fingerprint mismatch",
                "manifest_path": str(manifest_path),
            }
        if current_source_packet_sha256 != lineage["source_packet_sha256"]:
            return {
                "check_status": "blocked_integrity_or_lineage",
                "reason": "Source packet SHA-256 mismatch",
                "manifest_path": str(manifest_path),
            }

    yield_gates = manifest.get("yield_gates")
    strata = ledger.get("strata")
    if not isinstance(yield_gates, dict) or not isinstance(strata, dict):
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": "Manifest lacks a valid yield-gates or per-stratum ledger mapping",
            "manifest_path": str(manifest_path),
        }
    ledger_integrity_error = _yield_ledger_integrity_error(manifest, ledger, strata, yield_gates)
    if ledger_integrity_error is not None:
        return {
            "check_status": "blocked_integrity_or_lineage",
            "reason": ledger_integrity_error,
            "manifest_path": str(manifest_path),
        }

    if yield_gates.get("status") != "pass":
        shortfalls: list[str] = []
        for split, scenarios in strata.items():
            for scenario_id, stats in scenarios.items():
                if int(stats.get("shortfall", 0)) > 0:
                    shortfalls.append(
                        f"{split}/{scenario_id}: usable={stats.get('usable', 0)}, "
                        f"minimum={stats.get('target_minimum', 0)}, "
                        f"shortfall={stats['shortfall']}"
                    )
        return {
            "check_status": "blocked_scientific_yield",
            "reason": "Yield gates failed" + (": " + "; ".join(shortfalls) if shortfalls else ""),
            "manifest_path": str(manifest_path),
        }

    return {
        "check_status": "eligible_complete",
        "manifest_path": str(manifest_path),
    }


class _CaptureEnv:
    """Transparent environment proxy that records the exact policy I/O trajectory."""

    def __init__(self, env: Any, sink: dict[str, Any]) -> None:
        """Wire the proxy to the real environment and the recording sink."""
        self._env = env
        self._sink = sink
        self._pending_observation: Any | None = None

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped environment.

        Returns:
            The attribute fetched from the wrapped environment.
        """
        return getattr(self._env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Forward reset to the environment and record the initial observation.

        Returns:
            The ``(observation, info)`` tuple returned by the environment.
        """
        observation, info = self._env.reset(*args, **kwargs)
        self._sink["initial_observation"] = copy.deepcopy(observation)
        self._pending_observation = copy.deepcopy(observation)
        return observation, info

    def step(self, action: Any) -> Any:
        """Forward step to the environment and record the transition into the sink.

        Returns:
            The ``(observation, reward, terminated, truncated, info)`` step tuple.
        """
        if self._pending_observation is None:
            raise BalancedDatasetCollectionError("CaptureEnv.step called before reset")
        policy_observation = copy.deepcopy(self._pending_observation)
        observation, reward, terminated, truncated, info = self._env.step(action)
        self._sink["actions"].append(np.asarray(action, dtype=np.float32).copy())
        self._sink["observations"].append(policy_observation)
        self._sink["rewards"].append(float(reward))
        self._sink["terminated"].append(bool(terminated))
        self._sink["truncated"].append(bool(truncated))
        self._pending_observation = copy.deepcopy(observation)
        return observation, reward, terminated, truncated, info


class BalancedOracleCollector:
    """Orchestrates balanced oracle dataset collection, preflight planning, and validation."""

    def __init__(
        self,
        config_path: Path,
        *,
        output_root: Path,
        candidate_registry: Path | None = None,
        repo_root: Path | None = None,
        min_usable_transitions: int = 10000,
        min_episodes_per_stratum: int = 10,
    ) -> None:
        """Initialize BalancedOracleCollector.

        Args:
            config_path: Path to launch packet YAML.
            output_root: Output directory for dataset artifacts and manifests.
            candidate_registry: Path to candidate registry YAML.
            repo_root: Repository root path.
            min_usable_transitions: Minimum required usable training transitions.
            min_episodes_per_stratum: Minimum required usable episodes per stratum.
        """
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.config_path = (
            config_path.resolve()
            if config_path.is_absolute()
            else (self.repo_root / config_path).resolve()
        )
        self.output_root = (
            output_root.resolve()
            if output_root.is_absolute()
            else (self.repo_root / output_root).resolve()
        )
        self.candidate_registry = (
            candidate_registry.resolve()
            if candidate_registry and candidate_registry.is_absolute()
            else (
                (self.repo_root / candidate_registry).resolve()
                if candidate_registry
                else (self.repo_root / "docs/context/policy_search/candidate_registry.yaml")
            )
        )
        self.min_usable_transitions = min_usable_transitions
        self.min_episodes_per_stratum = min_episodes_per_stratum

        self.packet_validation = validate_launch_packet(self.config_path, repo_root=self.repo_root)
        self.packet = load_launch_packet(self.config_path)
        validate_split_and_episode_invariants(self.packet)

        self.dataset_id = str(self.packet["dataset_id"])
        self.source_candidate = str(self.packet["source_candidate"])
        self.scenario_ids = list(self.packet.get("scenario_ids", []))
        self.episodes_by_split = dict(self.packet.get("episode_ids_by_split", {}))
        self.seeds_by_split = dict(self.packet.get("seeds_by_split", {}))

    def _public_git_sha(self) -> str:
        """Return the resolved public git SHA, enforcing the packet's ``generating_commit``."""
        current_sha = _git_sha(self.repo_root)
        if current_sha is None:
            raise BalancedDatasetCollectionError("Cannot resolve the current public Git SHA")
        configured = self.packet.get("generating_commit")
        if configured in (None, "", "current"):
            return current_sha
        configured_sha = str(configured)
        if configured_sha != current_sha:
            raise BalancedDatasetCollectionError(
                "Launch packet generating_commit does not match the executing checkout: "
                f"packet={configured_sha}, checkout={current_sha}"
            )
        return current_sha

    def _compute_packet_fingerprint(self) -> str:
        """Compute a SHA-256 fingerprint of scientific/execution packet fields.

        Returns:
            Lowercase hex SHA-256 digest of the canonical packet payload.
        """
        return compute_packet_fingerprint(self.packet)

    def packet_fingerprint_payload(self) -> dict[str, Any]:
        """Return the auditable fields used by this collector's packet fingerprint."""
        return _packet_fingerprint_payload(self.packet)

    def validate_packet_difference(
        self, exhausted_attempts: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Validate this packet against exhausted-attempt fingerprint records.

        Returns:
            Deterministic comparison report, unless an unchanged attempt is rejected.
        """
        return validate_packet_difference(self.packet, exhausted_attempts)

    def _repo_relative(self, path: Path) -> str:
        """Return ``path`` relative to the repo root, or as-is when it lies outside."""
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.as_posix()

    def build_preflight_plan(
        self, *, exhausted_attempts: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Build a deterministic launch plan without performing simulation.

        An optional exhausted-attempt ledger is checked before any plan is written. An
        unchanged packet raises ``BalancedDatasetCollectionError`` and cannot proceed to
        a collection command.

        Returns:
            Dictionary containing the deterministic launch plan.
        """
        packet_difference = self.validate_packet_difference(exhausted_attempts)
        self.output_root.mkdir(parents=True, exist_ok=True)
        npz_filename = "expert_traj_v1.npz"
        manifest_destination = self.output_root / "balanced_oracle_dataset_manifest.json"

        from scripts.validation.run_policy_search_candidate import (  # noqa: PLC0415
            load_candidate_definition,
        )

        public_sha = self._public_git_sha()
        candidate_entry, candidate_payload, _candidate_config, candidate_config_path = (
            load_candidate_definition(self.candidate_registry, self.source_candidate)
        )
        stratum_counts = dict.fromkeys(self.scenario_ids, 0)
        for episode_id in self.episodes_by_split.get("train", []):
            _split, scenario_id, _seed = parse_episode_id(str(episode_id), "train")
            stratum_counts[scenario_id] = stratum_counts.get(scenario_id, 0) + 1

        plan = {
            "schema_version": _PLAN_SCHEMA_VERSION,
            "git_commit": public_sha,
            "dataset_id": self.dataset_id,
            "source_candidate": self.source_candidate,
            "source_candidate_algorithm": str(candidate_payload.get("algo", "")),
            "source_candidate_config": self._repo_relative(candidate_config_path),
            "source_candidate_entry_sha256": _canonical_sha256(candidate_entry),
            "candidate_registry": self._repo_relative(self.candidate_registry),
            "candidate_registry_sha256": _file_sha256(self.candidate_registry),
            "config_path": self._repo_relative(self.config_path),
            "config_sha256": _file_sha256(self.config_path),
            "packet_fingerprint": self._compute_packet_fingerprint(),
            "packet_fingerprint_fields": list(_PACKET_FINGERPRINT_FIELDS),
            "packet_difference": packet_difference,
            "output_npz_path": npz_filename,
            "manifest_destination": manifest_destination.name,
            "scenarios": self.scenario_ids,
            "planned_strata": self.scenario_ids,
            "planned_train_episodes_per_stratum": stratum_counts,
            "planned_episodes_by_split": {
                split: len(self.episodes_by_split.get(split, [])) for split in _SPLITS
            },
            "planned_seeds_by_split": {
                split: list(self.seeds_by_split.get(split, [])) for split in _SPLITS
            },
            "gates": {
                "min_usable_transitions": self.min_usable_transitions,
                "min_episodes_per_stratum": self.min_episodes_per_stratum,
            },
            "exclusion_rules": [
                "one-step trajectories (steps <= 1)",
                "malformed or explicitly invalid trajectory records",
                "failed or crashed trajectories",
                "fallback or degraded policy execution",
                "leakage invalid or seed overlap trajectories",
            ],
            "packet_validation_status": self.packet_validation.get("status", "valid"),
        }

        inadequate = {
            scenario_id: count
            for scenario_id, count in stratum_counts.items()
            if count < self.min_episodes_per_stratum
        }
        if inadequate:
            raise BalancedDatasetCollectionError(
                "Launch packet cannot satisfy the per-stratum episode gate: "
                f"{inadequate}; required={self.min_episodes_per_stratum}"
            )
        plan["plan_identity_sha256"] = _canonical_sha256(plan)

        plan_path = self.output_root / "balanced_oracle_collection_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return plan

    def _build_yield_ledger(
        self,
        raw_episodes: list[dict[str, Any]],
        usable_episodes: list[dict[str, Any]],
        exclusions: list[dict[str, Any]],
        missing_episode_ids: list[str],
        stratum_counts: dict[str, dict[str, int]],
        stratum_transitions: dict[str, dict[str, int]],
        identity_defects: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the per-stratum yield ledger with declared/attempted/completed/usable counts.

        Returns:
            Yield ledger dictionary with strata, totals, lineage, and differential_remedy.
        """
        strata: dict[str, dict[str, dict[str, Any]]] = {}

        for split in _SPLITS:
            strata[split] = {}
            for sc_id in self.scenario_ids:
                declared_ids = [
                    ep_id
                    for ep_id in self.episodes_by_split.get(split, [])
                    if parse_episode_id(str(ep_id), split)[1] == sc_id
                ]
                declared = len(declared_ids)

                raw_for_stratum = [
                    ep
                    for ep in raw_episodes
                    if ep.get("split") == split and ep.get("scenario_id") == sc_id
                ]
                attempted = len(raw_for_stratum)
                completed = sum(
                    1
                    for ep in raw_for_stratum
                    if not (
                        isinstance(ep.get("provenance"), dict)
                        and ep["provenance"].get("collection_error")
                    )
                )

                usable = stratum_counts.get(split, {}).get(sc_id, 0)
                excluded_for_stratum = [
                    ex for ex in exclusions if ex["split"] == split and ex["scenario_id"] == sc_id
                ]
                excluded_count = len(excluded_for_stratum)

                failed = sum(1 for ex in excluded_for_stratum if ex["reason"] == "failed")

                missing = 0
                for ep_id in missing_episode_ids:
                    parsed = parse_episode_id(str(ep_id))
                    if parsed[0] == split and parsed[1] == sc_id:
                        missing += 1

                usable_transitions = stratum_transitions.get(split, {}).get(sc_id, 0)

                reason_counts: dict[str, int] = {}
                for ex in excluded_for_stratum:
                    reason = ex["reason"]
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

                target = self.min_episodes_per_stratum if split == "train" else 0
                shortfall = max(0, target - usable) if split == "train" else 0

                strata[split][sc_id] = {
                    "declared": declared,
                    "attempted": attempted,
                    "completed": completed,
                    "usable": usable,
                    # The current collector's nondegeneracy rule is steps > 1; those rows
                    # are already excluded before usable counts are accumulated.
                    "nondegenerate": usable,
                    "excluded": excluded_count,
                    "failed": failed,
                    "missing": missing,
                    "usable_transitions": usable_transitions,
                    "reason_counts": reason_counts,
                    "target_minimum": target,
                    "shortfall": shortfall,
                }

        total_declared = sum(len(self.episodes_by_split.get(split, [])) for split in _SPLITS)
        total_attempted = len(raw_episodes)
        total_completed = sum(
            1
            for ep in raw_episodes
            if not (
                isinstance(ep.get("provenance"), dict) and ep["provenance"].get("collection_error")
            )
        )
        total_usable = len(usable_episodes)
        total_excluded = len(exclusions)
        total_failed = sum(1 for ex in exclusions if ex["reason"] == "failed")
        total_missing = len(missing_episode_ids)
        total_usable_transitions = sum(
            stratum_transitions.get(split, {}).get(sc_id, 0)
            for split in _SPLITS
            for sc_id in self.scenario_ids
        )

        return {
            "schema_version": "yield-ledger.v1",
            "strata": strata,
            "identity_defects": list(identity_defects or []),
            "totals": {
                "declared": total_declared,
                "attempted": total_attempted,
                "completed": total_completed,
                "usable": total_usable,
                # See the per-stratum note above: usable rows are nondegenerate
                # under the existing one-step exclusion contract.
                "nondegenerate": total_usable,
                "excluded": total_excluded,
                "failed": total_failed,
                "missing": total_missing,
                "usable_transitions": total_usable_transitions,
                "target_minimum_usable_transitions": self.min_usable_transitions,
            },
            "lineage": {
                "source_candidate": self.source_candidate,
                "config_path": self._repo_relative(self.config_path),
                "source_packet_sha256": _file_sha256(self.config_path),
                "commit": _git_sha(self.repo_root),
                "source_candidate_config": str(self.packet.get("source_candidate_config", "")),
                "job_id": self.packet.get("job_id"),
                "artifact_uri": self.packet.get("artifact_uri"),
            },
            "differential_remedy": self._build_differential_remedy(strata),
        }

    def _build_differential_remedy(
        self, strata: dict[str, dict[str, dict[str, Any]]]
    ) -> dict[str, Any]:
        """Build exactly one differential remedy category and rationale without selecting it.

        Returns:
            Dictionary with category and rationale strings.
        """
        worst_shortfall = 0
        worst_stratum: tuple[str, str] | None = None

        for split, scenarios in strata.items():
            for sc_id, stats in scenarios.items():
                if stats["shortfall"] > worst_shortfall:
                    worst_shortfall = stats["shortfall"]
                    worst_stratum = (split, sc_id)

        if worst_shortfall > 0 and worst_stratum is not None:
            split, sc_id = worst_stratum
            stats = strata[split][sc_id]
            return {
                "category": "budget_or_sampling_change",
                "selected": False,
                "selection_status": "pending_maintainer_ruling",
                "rationale": (
                    f"Stratum {split}/{sc_id} has usable={stats['usable']}, "
                    f"below target minimum of {self.min_episodes_per_stratum}. "
                    f"Shortfall is {worst_shortfall}. "
                    "A budget or sampling change is a diagnostic candidate only; it "
                    "must not be applied without a maintainer ruling and fresh evidence."
                ),
                "fields_to_change": ["seeds_by_split", "episode_ids_by_split"],
                "evidence_required": [
                    "per-stratum yield ledger",
                    "changed packet fingerprint",
                    "maintainer ruling before submission",
                ],
            }

        return {
            "category": "collector_or_eligibility_defect_fix",
            "selected": False,
            "selection_status": "not_required_pending_new_failure",
            "rationale": "All strata meet or exceed the target minimum; no remedy is selected.",
            "fields_to_change": [],
            "evidence_required": [],
        }

    def _build_decision_packet(
        self,
        yield_ledger: dict[str, Any],
        check_status: str,
        fingerprint: str,
        packet_difference: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the public-safe decision packet for diagnostic/blocker-resolution.

        Returns:
            Decision packet dictionary with check_status, observed shortfalls, and yield gates.
        """
        observed: dict[str, dict[str, int]] = {}
        for split, scenarios in yield_ledger["strata"].items():
            for sc_id, stats in scenarios.items():
                if stats["shortfall"] > 0:
                    observed[f"{split}/{sc_id}"] = {
                        "usable": stats["usable"],
                        "minimum": stats["target_minimum"],
                        "shortfall": stats["shortfall"],
                    }

        candidate_categories = [
            {
                "category": "scenario_roster_change",
                "fields_to_change": ["scenario_source", "scenario_ids", "episode_ids_by_split"],
                "evidence_required": ["scientific justification and maintainer ruling"],
            },
            {
                "category": "minimum_change_with_scientific_justification",
                "fields_to_change": ["min_episodes_per_stratum", "min_usable_transitions"],
                "evidence_required": ["pre-specified threshold rationale and maintainer ruling"],
            },
            {
                "category": "budget_or_sampling_change",
                "fields_to_change": ["seeds_by_split", "episode_ids_by_split"],
                "evidence_required": ["changed packet fingerprint and maintainer ruling"],
            },
            {
                "category": "collector_or_eligibility_defect_fix",
                "fields_to_change": ["exclusion_rules", "collector implementation"],
                "evidence_required": ["reproducible defect and focused regression test"],
            },
        ]

        return {
            "schema_version": "yield-decision-packet.v1",
            "check_status": check_status,
            "dataset_id": self.dataset_id,
            "packet_fingerprint": fingerprint,
            "packet_fingerprint_fields": list(_PACKET_FINGERPRINT_FIELDS),
            "packet_difference": packet_difference,
            "observed": observed,
            "differential_remedy": yield_ledger["differential_remedy"],
            "candidate_remedy_categories": candidate_categories,
            "evidence_required_before_submission": [
                "current per-stratum ledger",
                "complete source, commit, and artifact lineage",
                "changed packet fingerprint",
                "maintainer ruling selecting one differential remedy or stopping the lane",
            ],
            "claim_boundary": "diagnostic/blocker-resolution only; not dataset success or scientific evidence",
            "yield_gates": {
                "status": "pass" if check_status == "eligible_complete" else "fail",
                "min_usable_transitions": self.min_usable_transitions,
                "min_episodes_per_stratum": self.min_episodes_per_stratum,
            },
        }

    def _capture_episode(  # noqa: PLR0913
        self,
        scenario: dict[str, Any],
        *,
        seed: int,
        split: str,
        episode_id: str,
        algo: str,
        algo_config: dict[str, Any],
        scenario_path: Path,
        horizon: int,
        dt: float,
    ) -> dict[str, Any]:
        """Run one packet episode through the proven job-13520 capture seam.

        Returns:
            Captured episode fields plus exact benchmark-record provenance.
        """
        sink: dict[str, Any] = {
            "actions": [],
            "observations": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
        }
        original_factory = map_runner.make_robot_env
        episode_module = map_runner._map_runner_episode_module
        original_episode_factory = episode_module.make_robot_env

        def capture_factory(*args: Any, **kwargs: Any) -> _CaptureEnv:
            """Wrap the original robot-env factory to return a recording proxy.

            Returns:
                A :class:`_CaptureEnv` wrapping the constructed environment.
            """
            return _CaptureEnv(original_factory(*args, **kwargs), sink)

        map_runner.make_robot_env = capture_factory
        try:
            record = _run_map_episode(
                scenario,
                seed,
                horizon=horizon,
                dt=dt,
                record_forces=False,
                snqi_weights=None,
                snqi_baseline=None,
                algo=algo,
                scenario_path=scenario_path,
                algo_config=algo_config,
                benchmark_track=None,
                record_simulation_step_trace=True,
            )
        finally:
            map_runner.make_robot_env = original_factory
            episode_module.make_robot_env = original_episode_factory

        trace_steps = (
            record.get("algorithm_metadata", {}).get("simulation_step_trace", {}).get("steps", [])
        )
        step_count = len(sink["actions"])
        if not (
            step_count
            == len(sink["observations"])
            == len(sink["rewards"])
            == len(sink["terminated"])
            == len(sink["truncated"])
            == len(trace_steps)
        ):
            raise BalancedDatasetCollectionError(
                f"Captured trajectory fields are misaligned for {episode_id!r}"
            )

        metadata = record.get("algorithm_metadata", {})
        planner_runtime = metadata.get("planner_runtime", {})
        kinematics = metadata.get("planner_kinematics", {})
        pedestrian_model = record.get("pedestrian_model", {})
        execution_mode = str(kinematics.get("execution_mode", "unknown"))
        fallback_count = int(planner_runtime.get("fallback_count", 0) or 0)
        pedestrian_status = str(pedestrian_model.get("fallback_degraded_status", "unknown"))
        fallback = (
            execution_mode not in _VALID_EXECUTION_MODES
            or fallback_count > 0
            or _contains_degraded_marker(planner_runtime)
        )
        degraded = pedestrian_status != "native" or _contains_degraded_marker(record)
        failed = str(record.get("status", "failed")) != "success"
        actual_scenario = str(record.get("scenario_id") or "").strip()
        try:
            actual_seed = int(record.get("seed", -1))
        except (TypeError, ValueError):
            actual_seed = -1
        _declared_split, declared_scenario, declared_seed = parse_episode_id(episode_id, split)
        identity_missing = not actual_scenario or actual_seed < 0
        leakage_invalid = (
            identity_missing or actual_scenario != declared_scenario or actual_seed != declared_seed
        )

        return {
            "episode_id": episode_id,
            "scenario_id": actual_scenario or declared_scenario,
            "seed": actual_seed if actual_seed >= 0 else declared_seed,
            "split": split,
            "actions": sink["actions"],
            "observations": sink["observations"],
            "positions": [step["robot"]["position"] for step in trace_steps],
            "rewards": sink["rewards"],
            "terminated": sink["terminated"],
            "truncated": sink["truncated"],
            "failed": failed,
            "fallback": fallback,
            "degraded": degraded,
            "leakage_invalid": leakage_invalid,
            "provenance": {
                "record": record,
                "execution_mode": execution_mode,
                "fallback_count": fallback_count,
                "pedestrian_fallback_degraded_status": pedestrian_status,
                "identity_missing": identity_missing,
            },
        }

    def collect_source_episodes(
        self,
        *,
        horizon: int = 500,
        dt: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Collect every predeclared packet episode from the registered source candidate.

        Returns:
            One terminal captured or failed-provenance row per packet episode.
        """
        from scripts.training.collect_oracle_imitation_candidate_traces import (  # noqa: PLC0415
            build_split_scenarios,
        )
        from scripts.validation.run_policy_search_candidate import (  # noqa: PLC0415
            _group_scenarios_by_config_overrides,
            load_candidate_definition,
        )

        _entry, candidate_payload, candidate_config, candidate_config_path = (
            load_candidate_definition(self.candidate_registry, self.source_candidate)
        )
        default_algo = str(candidate_payload.get("algo", "")).strip().lower()
        if not default_algo:
            raise BalancedDatasetCollectionError(
                f"Registered source candidate {self.source_candidate!r} has no algorithm"
            )
        scenario_source = Path(str(self.packet["scenario_source"]))
        if not scenario_source.is_absolute():
            scenario_source = (self.repo_root / scenario_source).resolve()

        collected: list[dict[str, Any]] = []
        for split in _SPLITS:
            scenarios = build_split_scenarios(self.packet, split=split, repo_root=self.repo_root)
            groups = _group_scenarios_by_config_overrides(
                scenarios,
                candidate_payload=candidate_payload,
                candidate_config=candidate_config,
                default_algo=default_algo,
                config_anchor=candidate_config_path.parent,
            )
            entries: dict[str, dict[str, Any]] = {}
            for group in groups.values():
                for scenario in group["scenarios"]:
                    seeds = scenario.get("seeds")
                    if not isinstance(seeds, list) or len(seeds) != 1:
                        raise BalancedDatasetCollectionError(
                            f"Scenario in split {split!r} must declare exactly one seed"
                        )
                    episode_id = str(
                        scenario.get("metadata", {}).get("oracle_imitation_episode_id", "")
                    )
                    entries[episode_id] = {
                        "scenario": scenario,
                        "seed": int(seeds[0]),
                        "algo": str(group["algo"]),
                        "algo_config": dict(group["config"]),
                    }

            for episode_id in self.episodes_by_split[split]:
                entry = entries.get(str(episode_id))
                if entry is None:
                    raise BalancedDatasetCollectionError(
                        f"No runnable scenario entry for packet episode {episode_id!r}"
                    )
                try:
                    collected.append(
                        self._capture_episode(
                            entry["scenario"],
                            seed=entry["seed"],
                            split=split,
                            episode_id=str(episode_id),
                            algo=entry["algo"],
                            algo_config=entry["algo_config"],
                            scenario_path=scenario_source,
                            horizon=horizon,
                            dt=dt,
                        )
                    )
                except (
                    AssertionError,
                    KeyError,
                    OSError,
                    RobotSfError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:  # preserve a terminal row for fail-closed accounting
                    _parsed_split, scenario_id, seed = parse_episode_id(str(episode_id), split)
                    collected.append(
                        {
                            "episode_id": str(episode_id),
                            "scenario_id": scenario_id,
                            "seed": seed,
                            "split": split,
                            "actions": [],
                            "observations": [],
                            "positions": [],
                            "rewards": [],
                            "terminated": [],
                            "truncated": [],
                            "failed": True,
                            "fallback": False,
                            "degraded": False,
                            "leakage_invalid": False,
                            "provenance": {"collection_error": f"{type(exc).__name__}: {exc}"},
                        }
                    )
        return collected

    def _filter_episodes(
        self, raw_episodes: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split raw episodes into usable and excluded sets by leakage/gate reasons.

        Returns:
            Tuple of (usable episodes, exclusion records).
        """
        usable: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []

        for ep in raw_episodes:
            ep_id = str(ep.get("episode_id", ""))
            sc_id = str(ep.get("scenario_id", ""))
            raw_seed = ep.get("seed", -1)
            seed = int(raw_seed) if _is_strict_integer(raw_seed) else -1
            split = str(ep.get("split", ""))
            reason, steps = _episode_exclusion_reason(ep)

            if reason is not None:
                exclusions.append(
                    {
                        "episode_id": ep_id,
                        "scenario_id": sc_id,
                        "seed": seed,
                        "split": split,
                        "reason": reason,
                        "steps": steps,
                    }
                )
            else:
                usable.append(ep)
        return usable, exclusions

    def _validate_collected_identities(  # noqa: C901
        self, raw_episodes: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Verify packet identities and retain duplicate/unexpected defects for diagnosis.

        Returns:
            Missing predeclared IDs and identity-defect records. Defective rows are marked
            invalid so they remain in raw provenance but cannot become usable evidence.
        """
        expected: dict[str, tuple[str, str, int]] = {}
        for split in _SPLITS:
            for episode_id in self.episodes_by_split[split]:
                parsed_split, scenario_id, seed = parse_episode_id(str(episode_id), split)
                expected[str(episode_id)] = (parsed_split, scenario_id, seed)

        counts: dict[str, int] = {}
        for episode in raw_episodes:
            raw_episode_id = episode.get("episode_id", "")
            episode_id = raw_episode_id if isinstance(raw_episode_id, str) else ""
            counts[episode_id] = counts.get(episode_id, 0) + 1

        seen: set[str] = set()
        defects: list[dict[str, Any]] = []

        def mark_identity_defect(episode: dict[str, Any], kind: str) -> None:
            """Mark a row invalid while preserving a mapping provenance record."""
            episode["leakage_invalid"] = True
            provenance = episode.get("provenance")
            if not isinstance(provenance, dict):
                provenance = {}
                episode["provenance"] = provenance
            provenance["identity_defect"] = kind

        for episode in raw_episodes:
            raw_episode_id = episode.get("episode_id", "")
            episode_id = raw_episode_id if isinstance(raw_episode_id, str) else ""
            duplicate = counts.get(episode_id, 0) > 1
            seen.add(episode_id)
            identity = expected.get(episode_id)
            if identity is None:
                defects.append({"kind": "unexpected_episode_id", "episode_id": episode_id})
                mark_identity_defect(episode, "unexpected_episode_id")
                continue
            split, scenario_id, seed = identity
            mismatches: list[str] = []
            if not isinstance(episode.get("split"), str) or episode["split"] != split:
                mismatches.append("split")
            if (
                not isinstance(episode.get("scenario_id"), str)
                or episode["scenario_id"] != scenario_id
            ):
                mismatches.append("scenario_id")
            raw_seed = episode.get("seed", -1)
            observed_seed = int(raw_seed) if _is_strict_integer(raw_seed) else None
            if observed_seed != seed:
                mismatches.append("seed")
            if duplicate:
                defects.append({"kind": "duplicate_episode_id", "episode_id": episode_id})
                mark_identity_defect(episode, "duplicate_episode_id")
            if mismatches:
                episode["leakage_invalid"] = True
                provenance = episode.get("provenance")
                if not isinstance(provenance, dict):
                    provenance = {}
                    episode["provenance"] = provenance
                provenance["identity_mismatches"] = mismatches
                provenance["expected_identity"] = {
                    "split": split,
                    "scenario_id": scenario_id,
                    "seed": seed,
                }
                defects.append(
                    {
                        "kind": "identity_mismatch",
                        "episode_id": episode_id,
                        "fields": mismatches,
                    }
                )
        return sorted(set(expected) - seen), defects

    def _write_raw_provenance(self, raw_episodes: list[dict[str, Any]]) -> Path:
        """Write raw per-episode provenance as JSONL and return its path.

        Returns:
            Path to the written JSONL provenance file.
        """
        path = self.output_root / "raw_episode_provenance.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for episode in raw_episodes:
                handle.write(json.dumps(_jsonable(episode), sort_keys=True) + "\n")
        return path

    def _assert_yield_gates(
        self,
        *,
        missing_episode_ids: list[str],
        leakage_detected: bool,
        usable_train_transitions: int,
        stratum_counts: dict[str, dict[str, int]],
        usable_split_counts: dict[str, int],
    ) -> None:
        """Raise before artifact creation when any promotion gate fails."""
        if missing_episode_ids:
            raise BalancedDatasetCollectionError(
                "Insufficient yield: collection did not produce every predeclared "
                f"packet episode: {missing_episode_ids[:10]}"
            )
        if leakage_detected:
            raise BalancedDatasetCollectionError(
                "Leakage-invalid episode identity detected; refusing to materialize a "
                "training candidate"
            )
        if usable_train_transitions < self.min_usable_transitions:
            raise BalancedDatasetCollectionError(
                f"Insufficient yield: usable training transitions ({usable_train_transitions}) "
                f"< required minimum ({self.min_usable_transitions})"
            )
        for sc_id in self.scenario_ids:
            count = stratum_counts["train"].get(sc_id, 0)
            if count < self.min_episodes_per_stratum:
                raise BalancedDatasetCollectionError(
                    f"Insufficient yield for training stratum {sc_id!r}: "
                    f"usable episodes ({count}) < required minimum "
                    f"({self.min_episodes_per_stratum})"
                )
        empty_splits = [split for split, count in usable_split_counts.items() if count == 0]
        if empty_splits:
            raise BalancedDatasetCollectionError(
                "Insufficient yield: no usable episodes for split(s): " + ", ".join(empty_splits)
            )

    def collect_dataset(
        self,
        *,
        episodes_override: list[dict[str, Any]] | None = None,
        exhausted_attempts: list[dict[str, Any]] | None = None,
        allow_insufficient_yield: bool = False,
        cli_command: str | None = None,
        horizon: int = 500,
        dt: float = 0.1,
    ) -> dict[str, Any]:
        """Collect balanced dataset, materialize NPZ, and write manifest.

        Args:
            episodes_override: Optional list of episode dictionaries for testing.
            exhausted_attempts: Optional prior-attempt fingerprint records. An unchanged
                packet is rejected before collection starts.
            allow_insufficient_yield: Whether to bypass minimum yield gates.
            cli_command: Explicit CLI command string to record in manifest.
            horizon: Maximum simulation steps per episode.
            dt: Simulation step duration in seconds.

        Returns:
            Manifest dictionary.
        """
        packet_difference = self.validate_packet_difference(exhausted_attempts)
        self.output_root.mkdir(parents=True, exist_ok=True)
        raw_episodes = (
            episodes_override
            if episodes_override is not None
            else self.collect_source_episodes(horizon=horizon, dt=dt)
        )
        self._write_raw_provenance(raw_episodes)
        missing_episode_ids, identity_defects = self._validate_collected_identities(raw_episodes)
        raw_provenance_path = self._write_raw_provenance(raw_episodes)
        usable_episodes, exclusions = self._filter_episodes(raw_episodes)

        stratum_counts: dict[str, dict[str, int]] = {
            split: dict.fromkeys(self.scenario_ids, 0) for split in _SPLITS
        }
        stratum_transitions: dict[str, dict[str, int]] = {
            split: dict.fromkeys(self.scenario_ids, 0) for split in _SPLITS
        }

        usable_train_transitions = 0
        train_usable_episodes = [ep for ep in usable_episodes if ep["split"] == "train"]

        for ep in usable_episodes:
            split = ep["split"]
            sc_id = ep["scenario_id"]
            steps = len(ep["actions"])
            stratum_counts[split][sc_id] = stratum_counts[split].get(sc_id, 0) + 1
            stratum_transitions[split][sc_id] = stratum_transitions[split].get(sc_id, 0) + steps
            if split == "train":
                usable_train_transitions += steps

        usable_split_counts = {
            split: sum(1 for episode in usable_episodes if episode["split"] == split)
            for split in _SPLITS
        }
        leakage_detected = bool(identity_defects) or any(
            bool(episode.get("leakage_invalid")) for episode in raw_episodes
        )

        packet_fingerprint = self._compute_packet_fingerprint()
        yield_ledger = self._build_yield_ledger(
            raw_episodes,
            usable_episodes,
            exclusions,
            missing_episode_ids,
            stratum_counts,
            stratum_transitions,
            identity_defects,
        )

        if not allow_insufficient_yield:
            self._assert_yield_gates(
                missing_episode_ids=missing_episode_ids,
                leakage_detected=leakage_detected,
                usable_train_transitions=usable_train_transitions,
                stratum_counts=stratum_counts,
                usable_split_counts=usable_split_counts,
            )

        train_actions_list = [
            np.asarray(ep["actions"], dtype=np.float32) for ep in train_usable_episodes
        ]
        train_weights, bin_summary = compute_action_bin_accounting(train_actions_list)
        weight_by_episode: dict[str, np.ndarray] = {}
        cursor = 0
        for episode in train_usable_episodes:
            steps = len(episode["actions"])
            weight_by_episode[str(episode["episode_id"])] = train_weights[cursor : cursor + steps]
            cursor += steps
        bin_summary["weights_sha256"] = hashlib.sha256(train_weights.tobytes()).hexdigest()
        bin_summary["weight_mean"] = float(np.mean(train_weights)) if len(train_weights) else 0.0
        bin_summary["weight_min"] = float(np.min(train_weights)) if len(train_weights) else 0.0
        bin_summary["weight_max"] = float(np.max(train_weights)) if len(train_weights) else 0.0

        npz_filename = "expert_traj_v1.npz"
        npz_path = self.output_root / npz_filename
        _write_expert_traj_npz(
            npz_path,
            usable_episodes,
            self.dataset_id,
            self.source_candidate,
            action_weights=weight_by_episode,
        )

        sha256_npz = _file_sha256(npz_path)
        sha256_raw_provenance = _file_sha256(raw_provenance_path)
        public_sha = self._public_git_sha()
        gates_passed = not missing_episode_ids and all(
            [
                usable_train_transitions >= self.min_usable_transitions,
                all(
                    stratum_counts["train"].get(sc_id, 0) >= self.min_episodes_per_stratum
                    for sc_id in self.scenario_ids
                ),
                all(usable_split_counts[split] > 0 for split in _SPLITS),
                not leakage_detected,
            ]
        )

        if missing_episode_ids or identity_defects or leakage_detected:
            check_status = "blocked_integrity_or_lineage"
        elif not gates_passed:
            check_status = "blocked_scientific_yield"
        else:
            check_status = "eligible_complete"

        decision_packet = self._build_decision_packet(
            yield_ledger, check_status, packet_fingerprint, packet_difference
        )

        cmd_str = cli_command or " ".join(sys.argv)
        manifest_path = self.output_root / "balanced_oracle_dataset_manifest.json"
        bc_smoke_cmd = (
            f"uv run python scripts/validation/run_oracle_imitation_bc_smoke.py "
            f"--dataset-path {npz_path}"
        )

        sha256_inventory = {
            npz_filename: sha256_npz,
            raw_provenance_path.name: sha256_raw_provenance,
        }
        artifact_paths = {
            "dataset_npz": npz_filename,
            "raw_episode_provenance": raw_provenance_path.name,
        }
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": public_sha,
            "exact_public_sha": public_sha,
            "generating_commit": public_sha,
            "dataset_id": self.dataset_id,
            "source_candidate": self.source_candidate,
            "source_candidate_config": str(self.packet["source_candidate_config"]),
            "scenario_ids": list(self.scenario_ids),
            "seeds_by_split": copy.deepcopy(self.seeds_by_split),
            "episode_ids_by_split": copy.deepcopy(self.episodes_by_split),
            "hard_slice_assignment": copy.deepcopy(self.packet.get("hard_slice_assignment", [])),
            "relabeling_policy": copy.deepcopy(self.packet.get("relabeling_policy")),
            "exclusion_rules": copy.deepcopy(self.packet.get("exclusion_rules", [])),
            "provenance": str(self.packet.get("provenance", "")),
            "source_packet_sha256": _file_sha256(self.config_path),
            "candidate_registry_sha256": _file_sha256(self.candidate_registry),
            "artifact_paths": artifact_paths,
            "checksums": sha256_inventory,
            "sha256_inventory": sha256_inventory,
            "dataset_sha256": sha256_npz,
            "command": cmd_str,
            "exclusions": exclusions,
            "missing_episode_ids": missing_episode_ids,
            "eligibility_status": (
                "training_ready" if gates_passed else "diagnostic_insufficient_yield"
            ),
            "yield_gates": {
                "status": "pass" if gates_passed else "fail",
                "min_usable_transitions": self.min_usable_transitions,
                "min_episodes_per_stratum": self.min_episodes_per_stratum,
            },
            "balance_summary": {
                "action_bin_accounting": bin_summary,
                "stratum_counts": stratum_counts,
                "stratum_transitions": stratum_transitions,
                "usable_train_episodes": len(train_usable_episodes),
                "usable_train_transitions": usable_train_transitions,
                "total_usable_episodes": len(usable_episodes),
                "total_excluded_episodes": len(exclusions),
            },
            "private_artifact_registry_candidate": (
                {
                    "dataset_id": self.dataset_id,
                    "uri": f"private-artifact://oracle-imitation/{self.dataset_id}/{npz_filename}",
                    "sha256": sha256_npz,
                    "splits": {
                        split: {
                            "episode_ids": [
                                ep["episode_id"] for ep in usable_episodes if ep["split"] == split
                            ]
                        }
                        for split in _SPLITS
                    },
                }
                if gates_passed
                else None
            ),
            "bc_loader_smoke_command": bc_smoke_cmd,
            "manifest_path": str(manifest_path),
            "npz_path": str(npz_path),
            "raw_provenance_path": str(raw_provenance_path),
            "yield_ledger": yield_ledger,
            "packet_fingerprint": packet_fingerprint,
            "packet_fingerprint_fields": list(_PACKET_FINGERPRINT_FIELDS),
            "packet_fingerprint_payload": self.packet_fingerprint_payload(),
            "packet_difference": packet_difference,
            "identity_defects": identity_defects,
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        decision_packet_path = self.output_root / "yield_decision_packet.json"
        decision_packet_path.write_text(
            json.dumps(decision_packet, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return manifest


def _write_expert_traj_npz(
    path: Path,
    episodes: list[dict[str, Any]],
    dataset_id: str,
    source_candidate: str,
    *,
    action_weights: dict[str, np.ndarray],
) -> None:
    """Materialize the expert-trajectory NPZ with per-step arrays and split metadata."""
    episode_count = len(episodes)
    per_episode: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "observations",
            "actions",
            "positions",
            "rewards",
            "return_to_go",
            "terminated",
            "truncated",
            "action_balance_weights",
        )
    }
    episode_ids: list[np.ndarray] = []
    scenario_ids: list[np.ndarray] = []
    seeds: list[np.ndarray] = []
    split_tags: list[np.ndarray] = []

    splits_mapping: dict[str, list[str]] = {split: [] for split in _SPLITS}

    for ep in episodes:
        ep_id = str(ep["episode_id"])
        sc_id = str(ep["scenario_id"])
        seed = int(ep["seed"])
        split = str(ep["split"])
        steps = len(ep["actions"])

        splits_mapping.setdefault(split, []).append(ep_id)

        ep_obs = ep.get("observations", [])
        ep_act = ep.get("actions", [])
        ep_rew = ep.get("rewards", [0.05] * steps)
        ep_pos = ep.get("positions", [np.zeros(2, dtype=np.float32)] * steps)
        ep_terminated = ep.get("terminated", [False] * steps)
        ep_truncated = ep.get("truncated", [False] * steps)
        if not all(
            len(values) == steps
            for values in (ep_obs, ep_act, ep_rew, ep_pos, ep_terminated, ep_truncated)
        ):
            raise BalancedDatasetCollectionError(
                f"Trajectory fields are misaligned for episode {ep_id!r}"
            )
        weights = action_weights.get(ep_id, np.ones(steps, dtype=np.float32))
        if len(weights) != steps:
            raise BalancedDatasetCollectionError(
                f"Action balance weights are misaligned for episode {ep_id!r}"
            )

        running_rtg = 0.0
        rtg_vals: list[float] = []
        for r in reversed(ep_rew):
            running_rtg += float(r)
            rtg_vals.append(running_rtg)
        rtg_vals.reverse()

        per_episode["observations"].append(np.asarray(ep_obs, dtype=object))
        per_episode["actions"].append(np.asarray(ep_act, dtype=np.float32))
        per_episode["positions"].append(np.asarray(ep_pos, dtype=np.float32))
        per_episode["rewards"].append(np.asarray(ep_rew, dtype=np.float32))
        per_episode["return_to_go"].append(np.asarray(rtg_vals, dtype=np.float32))
        per_episode["terminated"].append(np.asarray(ep_terminated, dtype=bool))
        per_episode["truncated"].append(np.asarray(ep_truncated, dtype=bool))
        per_episode["action_balance_weights"].append(np.asarray(weights, dtype=np.float32))

        episode_ids.append(np.asarray([ep_id], dtype=object))
        scenario_ids.append(np.asarray([sc_id], dtype=object))
        seeds.append(np.asarray([seed], dtype=np.int64))
        split_tags.append(np.asarray([split], dtype=object))

    def ragged(values: list[np.ndarray]) -> np.ndarray:
        """Pack a list of per-episode arrays into a single object-dtype ndarray.

        Returns:
            Object-dtype ndarray holding the per-episode arrays.
        """
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array

    scenario_coverage = {
        scenario_id: sum(str(ep["scenario_id"]) == scenario_id for ep in episodes)
        for scenario_id in sorted({str(ep["scenario_id"]) for ep in episodes})
    }
    observation_keys = sorted(
        {
            str(key)
            for episode in episodes
            for observation in episode.get("observations", [])
            if isinstance(observation, dict)
            for key in observation
        }
    )

    metadata = {
        "dataset_id": dataset_id,
        "source_policy_id": source_candidate,
        "dataset_schema": "trajectory_dataset.v2.decision_transformer_preflight",
        "splits": {split: {"episode_ids": ids} for split, ids in splits_mapping.items()},
        "scenario_coverage": scenario_coverage,
        "observation_contract": {"keys": observation_keys},
        "action_contract": {"fields": ["acceleration", "angular_velocity"]},
        "data_collection_only": True,
        "training_performed": False,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=ragged(per_episode["observations"]),
        actions=ragged(per_episode["actions"]),
        positions=ragged(per_episode["positions"]),
        rewards=ragged(per_episode["rewards"]),
        return_to_go=ragged(per_episode["return_to_go"]),
        terminated=ragged(per_episode["terminated"]),
        truncated=ragged(per_episode["truncated"]),
        episode_ids=np.asarray(episode_ids, dtype=object),
        scenario_ids=np.asarray(scenario_ids, dtype=object),
        seeds=np.asarray(seeds, dtype=object),
        splits=np.asarray(split_tags, dtype=object),
        action_balance_weights=ragged(per_episode["action_balance_weights"]),
        episode_count=np.array(episode_count),
        metadata=np.array(metadata, dtype=object),
    )


__all__ = [
    "BalancedDatasetCollectionError",
    "BalancedOracleCollector",
    "check_yield_status",
    "compute_packet_fingerprint",
    "parse_episode_id",
    "validate_packet_difference",
    "validate_split_and_episode_invariants",
]

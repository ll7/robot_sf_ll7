"""Append-only, replay-verified adversarial challenge corpus (issue #9652).

The corpus stores immutable discovery evidence and append-only planner observations.
Solved status is derived on request; a later success never deletes a challenge or its
original failure evidence. Unknown feasibility and incomplete execution evidence remain
explicit states.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.adversarial.bundle import compute_effective_scenario_hash
from robot_sf.benchmark.algorithm_metadata import canonical_algorithm_name
from robot_sf.benchmark.episode_input_identity import (
    EPISODE_INPUT_IDENTITY_SCHEMA,
    scenario_semantic_sha256,
)
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.termination_reason import (
    TERMINATION_REASONS,
    outcome_contradictions,
    status_from_termination_reason,
)
from robot_sf.cli_scenarios import validate_scenario_payload

CORPUS_SCHEMA_VERSION = "adversarial-counterexample-corpus.v1"
CASE_SCHEMA_VERSION = "adversarial-counterexample.v1"
ATTEMPT_SCHEMA_VERSION = "adversarial-counterexample-admission-attempt.v1"
EVALUATION_SCHEMA_VERSION = "adversarial-counterexample-planner-evaluation.v1"
EVALUATION_REPLAY_RECEIPT_SCHEMA_VERSION = "adversarial-planner-replay-receipt.v2"
LEGACY_EVALUATION_REPLAY_RECEIPT_SCHEMA_VERSION = "adversarial-planner-replay-receipt.v1"
SLICE_SCHEMA_VERSION = "adversarial-counterexample-slice.v1"
CASE_INPUT_IDENTITY_SCHEMA_VERSION = "adversarial-case-input-identity.v1"
CASE_ADMISSION_REPLAY_SCHEMA_VERSION = "adversarial-case-admission-replay.v2"
LEGACY_CASE_ADMISSION_REPLAY_SCHEMA_VERSION = "adversarial-case-admission-replay.v1"
CASE_ADMISSIBILITY_EVIDENCE_SCHEMA_VERSION = "adversarial-case-admissibility-evidence.v1"
ISSUE_9645_SUMMARY_SCHEMA = "issue_9645_bounded_pilot_summary.v1"
ISSUE_9645_REPLAY_SCHEMA = "issue_9645_replay_validation_collection.v1"
ISSUE_9645_BUNDLE_SCHEMA = "evidence_bundle.v1"
ISSUE_9656_SUMMARY_SCHEMA = "benchmark-hard-case-slice.v1"
SEARCH_MANIFEST_SCHEMA = "adversarial-search-manifest.v1"
_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_SCHEMA_PATH = _ROOT / "robot_sf/benchmark/schemas/adversarial-counterexample-corpus.v1.json"


class CorpusError(ValueError):
    """Raised when corpus or source evidence violates the versioned contract."""


def new_corpus() -> dict[str, Any]:
    """Return an empty corpus with stable, append-only collections."""
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "claim_boundary": (
            "diagnostic challenge memory; not a safety claim, feasibility oracle, "
            "or proof of search-space coverage"
        ),
        "search_runs": [],
        "cases": [],
        "historical_candidates": [],
        "historical_candidate_imports": [],
        "admission_attempts": [],
        "planner_evaluations": [],
    }


def compute_case_effective_scenario_hash(
    scenario: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    map_assets: Sequence[Mapping[str, Any]],
) -> str:
    """Hash runtime scenario/route content together with resolved map input bytes.

    Asset paths are deliberately excluded from the identity; their roles and content
    digests are included. A path move with identical geometry is therefore a duplicate,
    while a map or registry change produces a new case identity.
    """
    asset_identity = sorted(
        ({"role": item.get("role"), "sha256": item.get("sha256")} for item in map_assets),
        key=lambda item: str(item["role"]),
    )
    payload = {
        "schema_version": CASE_INPUT_IDENTITY_SCHEMA_VERSION,
        "scenario_route_sha256": compute_effective_scenario_hash(scenario, route_payload),
        "map_assets": asset_identity,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _historical_map_asset_snapshot(
    scenario: Mapping[str, Any], source_revision: str
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Resolve and pin a historical scenario's registry and map blobs at its replay commit."""
    map_id = scenario.get("map_id")
    if not isinstance(map_id, str) or not map_id.strip():
        raise CorpusError("#1501 historical scenario must reference a registered map_id")
    registry_relative = "maps/registry.yaml"
    registry_bytes = _read_git_blob(
        source_revision, registry_relative, "historical replay map registry blob is unavailable"
    )
    try:
        registry = yaml.safe_load(registry_bytes)
    except yaml.YAMLError as exc:
        raise CorpusError("historical replay map registry is malformed") from exc
    row = _map_registry_entry(registry, map_id)
    if not isinstance(row, Mapping):
        raise CorpusError(f"historical replay map registry has no entry for {map_id!r}")
    declared_path = row.get("path") or row.get("map_file")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise CorpusError("historical replay map registry entry has no path")
    map_relative_path = PurePosixPath(declared_path)
    if map_relative_path.is_absolute() or ".." in map_relative_path.parts:
        raise CorpusError("historical replay map registry path is unsafe")
    if map_relative_path.parts and map_relative_path.parts[0] == "maps":
        map_relative = map_relative_path.as_posix()
    else:
        map_relative = (PurePosixPath("maps") / map_relative_path).as_posix()
    map_bytes = _read_git_blob(
        source_revision, map_relative, "historical replay map blob is unavailable"
    )
    map_sha256 = hashlib.sha256(map_bytes).hexdigest()
    declared_sha256 = row.get("source_sha256")
    if not _is_sha256(declared_sha256) or declared_sha256 != map_sha256:
        raise CorpusError("historical replay map bytes do not match the pinned registry digest")

    # The canonical structural validator uses the checked-out registry. Admit only when it
    # resolves to the same bytes as the exact replay revision.
    current_registry = _ROOT / registry_relative
    current_map = _ROOT / map_relative
    if (
        not current_registry.is_file()
        or current_registry.read_bytes() != registry_bytes
        or not current_map.is_file()
        or current_map.read_bytes() != map_bytes
    ):
        raise CorpusError("current map registry or map differs from the exact replay revision")

    assets = [
        {
            "role": "map_registry",
            "path": "pending",
            "source_path": registry_relative,
            "source_revision": source_revision,
            "sha256": hashlib.sha256(registry_bytes).hexdigest(),
        },
        {
            "role": "map",
            "path": "pending",
            "source_path": map_relative,
            "source_revision": source_revision,
            "sha256": map_sha256,
        },
    ]
    return assets, {"map_registry": registry_bytes, "map": map_bytes}


def _read_git_blob(revision: str, relative_path: str, error_message: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{revision}:{relative_path}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise CorpusError(error_message) from exc


def _map_registry_entry(document: Any, map_id: str) -> Mapping[str, Any] | None:
    if not isinstance(document, Mapping):
        return None
    entries = document.get("maps", document)
    if isinstance(entries, Mapping):
        row = entries.get(map_id)
        if isinstance(row, str):
            return {"path": row}
        return row if isinstance(row, Mapping) else None
    if isinstance(entries, list):
        for row in entries:
            if isinstance(row, Mapping) and (row.get("map_id") or row.get("id")) == map_id:
                return row
    return None


def validate_corpus(corpus: Mapping[str, Any], *, corpus_root: str | Path | None = None) -> None:
    """Validate corpus shape and stable case/evaluation references."""
    if not isinstance(corpus, Mapping):
        raise CorpusError("corpus root must be an object")
    try:
        schema = json.loads(_CORPUS_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"corpus schema could not be loaded: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(corpus)),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "$"
        raise CorpusError(f"invalid corpus at {location}: {first.message}")
    case_ids = [case["case_id"] for case in corpus["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise CorpusError("case_id values must be unique")
    root = Path(corpus_root).resolve() if corpus_root is not None else None
    for case in corpus["cases"]:
        if case["case_id"] != f"case-{case['effective_scenario_sha256']}":
            raise CorpusError("case_id must be the full canonical effective-scenario SHA-256")
        errors = _validate_case_record(case, corpus_root=root)
        if errors:
            raise CorpusError(f"case {case['case_id']} is not admissible: " + "; ".join(errors))
    case_id_set = set(case_ids)
    for evaluation in corpus["planner_evaluations"]:
        if evaluation["case_id"] not in case_id_set:
            raise CorpusError(
                f"planner evaluation references absent case {evaluation['case_id']!r}"
            )
    _validate_search_run_evidence(corpus["search_runs"], corpus_root=root)
    _validate_all_case_admissibility_evidence(
        corpus["cases"], corpus["planner_evaluations"], corpus_root=root
    )
    _validate_historical_candidate_registry(corpus)


def _validate_search_run_evidence(
    search_runs: Sequence[Mapping[str, Any]], *, corpus_root: Path | None
) -> None:
    """Verify every persisted search-run artifact receipt against corpus custody."""
    if not search_runs:
        return
    if corpus_root is None:
        raise CorpusError("corpus_root is required to validate persisted search-run evidence")

    run_ids = [run["run_id"] for run in search_runs]
    if len(run_ids) != len(set(run_ids)):
        raise CorpusError("search-run run_id values must be unique")

    root = corpus_root.resolve()
    known_artifact_digests: dict[str, str] = {}
    for run in search_runs:
        _validate_one_search_run_evidence(run, root, known_artifact_digests)


def _validate_one_search_run_evidence(
    run: Mapping[str, Any], root: Path, known_artifact_digests: dict[str, str]
) -> None:
    artifacts_in_run: dict[str, tuple[str, str]] = {}
    for collection in ("source_files", "manifest_files"):
        _validate_search_run_file_collection(
            run[collection],
            collection,
            root,
            artifacts_in_run=artifacts_in_run,
            known_artifact_digests=known_artifact_digests,
        )
    _validate_search_run_bundle_receipts(
        run.get("bundle_receipts", []),
        root,
        artifacts_in_run=artifacts_in_run,
        known_artifact_digests=known_artifact_digests,
    )


def _validate_search_run_file_collection(
    receipts: Sequence[Mapping[str, Any]],
    collection: str,
    root: Path,
    *,
    artifacts_in_run: dict[str, tuple[str, str]],
    known_artifact_digests: dict[str, str],
) -> None:
    seen_paths: set[str] = set()
    for receipt in receipts:
        relative = receipt.get("path")
        digest = receipt.get("sha256")
        if not _safe_bundle_relative_path(relative) or not _is_sha256(digest):
            raise CorpusError(f"search-run {collection} receipt has an unsafe path or digest")
        if relative in seen_paths:
            raise CorpusError(f"search-run {collection} contains duplicate path: {relative}")
        seen_paths.add(relative)
        _register_search_run_artifact(
            relative,
            digest,
            collection,
            artifacts_in_run=artifacts_in_run,
            known_artifact_digests=known_artifact_digests,
        )
        _verify_search_run_artifact(root, relative, digest)


def _validate_search_run_bundle_receipts(
    receipts: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    artifacts_in_run: dict[str, tuple[str, str]],
    known_artifact_digests: dict[str, str],
) -> None:
    bundle_source_paths: set[str] = set()
    bundle_paths: set[str] = set()
    for receipt in receipts:
        source_path = receipt.get("source_path")
        relative = receipt.get("path")
        digest = receipt.get("sha256")
        if (
            not _safe_bundle_relative_path(source_path)
            or not _safe_bundle_relative_path(relative)
            or not _is_sha256(digest)
        ):
            raise CorpusError("search-run bundle receipt has an unsafe identity, path, or digest")
        if source_path in bundle_source_paths:
            raise CorpusError(
                f"search-run bundle receipts contain duplicate source identity: {source_path}"
            )
        if relative in bundle_paths:
            raise CorpusError(f"search-run bundle receipts contain duplicate path: {relative}")
        bundle_source_paths.add(source_path)
        bundle_paths.add(relative)
        _register_search_run_artifact(
            relative,
            digest,
            "bundle_receipts",
            artifacts_in_run=artifacts_in_run,
            known_artifact_digests=known_artifact_digests,
        )
        _verify_search_run_artifact(root, relative, digest)


def _register_search_run_artifact(
    relative: str,
    digest: str,
    collection: str,
    *,
    artifacts_in_run: dict[str, tuple[str, str]],
    known_artifact_digests: dict[str, str],
) -> None:
    previous = artifacts_in_run.get(relative)
    if previous is not None:
        previous_collection, previous_digest = previous
        intentional_manifest_overlap = {previous_collection, collection} == {
            "source_files",
            "manifest_files",
        } and previous_digest == digest
        if not intentional_manifest_overlap:
            reason = "conflicting" if previous_digest != digest else "duplicate"
            raise CorpusError(f"search-run {reason} artifact receipt identity: {relative}")
    else:
        artifacts_in_run[relative] = (collection, digest)

    known_digest = known_artifact_digests.get(relative)
    if known_digest is not None and known_digest != digest:
        raise CorpusError(f"search-run artifact receipts conflict on digest: {relative}")
    known_artifact_digests[relative] = digest


def _verify_search_run_artifact(root: Path, relative: str, expected_sha256: str) -> None:
    """Resolve a stored search receipt inside corpus custody and verify its digest."""
    try:
        source = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
        source.relative_to(root)
    except ValueError as exc:
        raise CorpusError(f"search-run artifact path escapes corpus root: {relative}") from exc
    except (OSError, RuntimeError) as exc:
        raise CorpusError(f"search-run artifact is missing or unresolvable: {relative}") from exc
    if not source.is_file():
        raise CorpusError(f"search-run artifact is missing or not a file: {relative}")
    try:
        actual_sha256 = _sha256_file(source)
    except OSError as exc:
        raise CorpusError(f"search-run artifact cannot be read: {relative}") from exc
    if actual_sha256 != expected_sha256:
        raise CorpusError(f"search-run artifact digest mismatch: {relative}")


def _validate_all_case_admissibility_evidence(
    cases: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    *,
    corpus_root: Path | None,
) -> None:
    for case in cases:
        errors = _validate_case_admissibility_evidence(case, evaluations, corpus_root=corpus_root)
        if errors:
            raise CorpusError(
                f"case {case['case_id']} has invalid admissibility evidence: " + "; ".join(errors)
            )


def _validate_historical_candidate_registry(corpus: Mapping[str, Any]) -> None:
    candidates = corpus.get("historical_candidates", [])
    imports = corpus.get("historical_candidate_imports", [])
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CorpusError("historical candidate_id values must be unique")
    import_ids = [receipt["import_id"] for receipt in imports]
    if len(import_ids) != len(set(import_ids)):
        raise CorpusError("historical candidate import_id values must be unique")
    known_candidate_ids = set(candidate_ids)
    imports_by_id, source_identity_by_import = _historical_candidate_import_indexes(
        imports, known_candidate_ids
    )
    known_cases = {case["case_id"] for case in corpus.get("cases", [])}
    successful_attempts = {
        (attempt.get("attempt_id"), attempt.get("source_id"))
        for attempt in corpus.get("admission_attempts", [])
        if attempt.get("decision") in {"admitted", "duplicate"}
    }
    for candidate in candidates:
        _validate_historical_candidate_registry_row(
            candidate,
            imports_by_id,
            source_identity_by_import,
            known_cases,
            successful_attempts,
        )


def _historical_candidate_import_indexes(
    imports: Sequence[Mapping[str, Any]], known_candidate_ids: set[str]
) -> tuple[dict[str, set[str]], dict[str, Mapping[str, Any]]]:
    imports_by_id = {}
    source_identity_by_import = {}
    for receipt in imports:
        source_identity = receipt["source_identity"]
        expected_id = hashlib.sha256(_stable_json(source_identity).encode("utf-8")).hexdigest()
        if receipt["import_id"] != expected_id:
            raise CorpusError("historical candidate import ID does not bind its source identity")
        candidate_ids = set(receipt["candidate_ids"])
        if receipt["candidate_count"] != len(candidate_ids) or not candidate_ids.issubset(
            known_candidate_ids
        ):
            raise CorpusError("historical candidate import receipt has absent candidate rows")
        imports_by_id[receipt["import_id"]] = candidate_ids
        source_identity_by_import[receipt["import_id"]] = source_identity
    return imports_by_id, source_identity_by_import


def _validate_historical_candidate_registry_row(
    candidate: Mapping[str, Any],
    imports_by_id: Mapping[str, set[str]],
    source_identity_by_import: Mapping[str, Mapping[str, Any]],
    known_cases: set[str],
    successful_attempts: set[tuple[Any, Any]],
) -> None:
    import_id = candidate["source_provenance"]["import_id"]
    if import_id not in imports_by_id or candidate["candidate_id"] not in imports_by_id[import_id]:
        raise CorpusError("historical candidate references an absent import receipt")
    identity = {
        **source_identity_by_import[import_id],
        "source_case_id": candidate["source_case_id"],
        "source_record_sha256": candidate["source_record_sha256"],
    }
    expected_id = hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()
    if candidate["candidate_id"] != expected_id:
        raise CorpusError("historical candidate ID does not bind its source alias and row digest")
    if candidate.get("candidate_status") == "admitted":
        _validate_promoted_historical_candidate(candidate, known_cases, successful_attempts)


def _validate_promoted_historical_candidate(
    candidate: Mapping[str, Any],
    known_cases: set[str],
    successful_attempts: set[tuple[Any, Any]],
) -> None:
    if candidate.get("promoted_case_id") not in known_cases:
        raise CorpusError("promoted historical candidate references an absent case")
    if candidate.get("source_candidate_status") != "pending_exact_replay":
        raise CorpusError("promoted historical candidate lost its original pending status")
    if (candidate.get("promotion_attempt_id"), candidate.get("candidate_id")) not in (
        successful_attempts
    ):
        raise CorpusError("promoted historical candidate has no successful admission attempt")


def load_corpus(path: str | Path, *, create: bool = False) -> dict[str, Any]:
    """Load and validate a corpus JSON file; optionally create an empty one."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        if not create:
            raise CorpusError(f"corpus file does not exist: {corpus_path}")
        corpus = new_corpus()
        save_corpus(corpus_path, corpus)
        return corpus
    try:
        value = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"could not read corpus {corpus_path}: {exc}") from exc
    validate_corpus(value, corpus_root=corpus_path.parent)
    return value


def save_corpus(path: str | Path, corpus: Mapping[str, Any]) -> None:
    """Atomically write a validated corpus with stable JSON formatting."""
    validate_corpus(corpus, corpus_root=Path(path).resolve().parent)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(corpus, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    text += "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(text)
            handle.flush()
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def import_issue9645_packet(
    payload_root: str | Path,
    corpus: dict[str, Any],
    *,
    corpus_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Import #9645's zero-discovery run and replay-verified #1501 case.

    ``payload_root`` is the ``payload`` directory from the durable #9645 evidence
    bundle. The pilot's zero discoveries are recorded independently of the historical
    case, which came from #1501 and remains marked as historical lineage.
    """
    root = Path(corpus_root).resolve()
    validate_corpus(corpus, corpus_root=root)
    payload = Path(payload_root).resolve()
    if (payload / "payload").is_dir():
        payload = payload / "payload"
    root.mkdir(parents=True, exist_ok=True)

    try:
        pilot = _verify_issue9645_pilot(payload)
    except (CorpusError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        updated, receipt = _record_attempt(
            corpus,
            source_kind="issue_9645_packet",
            source_id="issue_9645_bounded_pilot",
            decision="rejected",
            blockers=[f"pilot_evidence_invalid:{type(exc).__name__}:{exc}"],
            candidate_identity=None,
            near_duplicate_report=_unassessed_near_duplicates(),
        )
        return updated, receipt

    try:
        _persist_pilot_evidence(payload, root, pilot)
    except (CorpusError, OSError) as exc:
        updated, receipt = _record_attempt(
            corpus,
            source_kind="issue_9645_packet",
            source_id="issue_9645_bounded_pilot",
            decision="rejected",
            blockers=[f"pilot_evidence_persistence_failed:{type(exc).__name__}:{exc}"],
            candidate_identity=None,
            near_duplicate_report=_unassessed_near_duplicates(),
        )
        return updated, receipt
    corpus = _append_unique(
        corpus,
        "search_runs",
        pilot,
        key="run_id",
    )
    try:
        case, source_files, observations = _build_issue9645_historical_case(payload)
    except (
        CorpusError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        yaml.YAMLError,
    ) as exc:
        updated, receipt = _record_attempt(
            corpus,
            source_kind="issue_9645_historical_replay",
            source_id="issue_1501/failure_0002",
            decision="rejected",
            blockers=[f"historical_case_invalid:{type(exc).__name__}:{exc}"],
            candidate_identity=None,
            near_duplicate_report=_unassessed_near_duplicates(),
        )
        return updated, receipt

    existing = next(
        (
            item
            for item in corpus["cases"]
            if item["effective_scenario_sha256"] == case["effective_scenario_sha256"]
        ),
        None,
    )
    duplicate_case_id = existing["case_id"] if existing else None
    near_report = _near_duplicate_report(case, corpus["cases"])
    if existing is None:
        try:
            _materialize_case_artifacts(case, source_files, root)
        except (CorpusError, OSError) as exc:
            updated, receipt = _record_attempt(
                corpus,
                source_kind="issue_9645_historical_replay",
                source_id="issue_1501/failure_0002",
                decision="rejected",
                blockers=[f"case_evidence_persistence_failed:{type(exc).__name__}:{exc}"],
                candidate_identity=case["effective_scenario_sha256"],
                near_duplicate_report=near_report,
            )
            return updated, receipt
        corpus["cases"].append(case)
        corpus["cases"].sort(key=lambda item: item["case_id"])
        stored_case_id = case["case_id"]
    else:
        _merge_source_evidence(existing, case["source_evidence"])
        stored_case_id = existing["case_id"]

    stored_case = existing if existing is not None else case
    replay_receipts = []
    for index, observation in enumerate(observations, start=1):
        observation["case_id"] = stored_case_id
        observation["effective_scenario_sha256"] = case["effective_scenario_sha256"]
        observation["replay_receipt"] = create_planner_replay_receipt(
            observation,
            stored_case,
            artifact_path=(f"cases/{stored_case_id}/source_evidence/replay_{index}.jsonl"),
            corpus_root=root,
        )
        replay_receipts.append(observation["replay_receipt"])
    if existing is None:
        _bind_historical_replay_artifact_receipts(stored_case, replay_receipts)
    for observation in observations:
        corpus = append_planner_evaluation(corpus, observation, corpus_root=root)

    corpus, receipt = _record_attempt(
        corpus,
        source_kind="issue_9645_historical_replay",
        source_id="issue_1501/failure_0002",
        decision="admitted" if existing is None else "duplicate",
        blockers=[],
        candidate_identity=case["effective_scenario_sha256"],
        duplicate_case_id=duplicate_case_id,
        near_duplicate_report=near_report,
    )
    receipt["case_id"] = stored_case_id
    receipt["pilot_new_discoveries"] = 0
    receipt["pilot_run_id"] = pilot["run_id"]
    return corpus, receipt


def _bind_historical_replay_artifact_receipts(
    case: dict[str, Any], artifact_receipts: list[dict[str, Any]]
) -> None:
    """Bind historical inventory rows to their artifact receipts without upgrading claims."""
    case_receipt = case.get("replay_receipt")
    if not isinstance(case_receipt, dict):
        raise CorpusError("historical case replay receipt is missing")
    inventory = case_receipt.get("replay_artifacts")
    if not isinstance(inventory, list) or len(inventory) != len(artifact_receipts):
        raise CorpusError("historical replay inventory and artifact receipt counts differ")
    case_receipt["artifact_receipts"] = artifact_receipts
    for inventory_row, artifact_receipt in zip(inventory, artifact_receipts, strict=True):
        inventory_row.update(
            {
                "path": artifact_receipt["artifact_path"],
                "sha256": artifact_receipt["artifact_sha256"],
                "run_id": artifact_receipt["run_id"],
                "selected_event_identity": artifact_receipt["selected_event_identity"],
            }
        )


def import_issue9656_candidates(
    summary_path: str | Path,
    materialized_root: str | Path,
    evidence_bundle_root: str | Path,
    campaign_root: str | Path,
    corpus: dict[str, Any],
    *,
    corpus_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append #9656 historical rows as pending candidates, never admitted cases.

    The evidence bundle is verified before any corpus or artifact mutation. Each source
    alias is retained verbatim while the corpus candidate receives its own content-derived
    ID. Imported replay matrices are path-normalized into candidate custody with the
    original input digests retained separately.
    """
    root = Path(corpus_root).resolve()
    validate_corpus(corpus, corpus_root=root)
    summary_file = Path(summary_path).resolve()
    materialized = Path(materialized_root).resolve()
    bundle_root = Path(evidence_bundle_root).resolve()
    campaign = Path(campaign_root).resolve()
    summary, summary_receipts = _verify_issue9656_source_summary(summary_file, bundle_root)
    materialized_manifest_path = materialized / "manifest.json"
    materialized_manifest = _read_json_object(materialized_manifest_path)
    source_rows = _validate_issue9656_materialization(
        summary, materialized, materialized_manifest, campaign
    )
    materialized_manifest_sha = _sha256_file(materialized_manifest_path)

    import_identity = {
        "source_issue": 9656,
        "source_row_binding": "verified_episode_file_and_line_sha256",
        "summary_sha256": summary_receipts["summary_sha256"],
        "evidence_bundle_manifest_sha256": summary_receipts["manifest_sha256"],
        "evidence_checksums_sha256": summary_receipts["checksums_sha256"],
        "materialized_manifest_sha256": materialized_manifest_sha,
    }
    import_id = hashlib.sha256(_stable_json(import_identity).encode("utf-8")).hexdigest()
    existing_import = next(
        (
            item
            for item in corpus.get("historical_candidate_imports", [])
            if item.get("import_id") == import_id
        ),
        None,
    )
    if existing_import is not None:
        if existing_import.get("source_identity") != import_identity:
            raise CorpusError("historical candidate import ID conflicts with stored provenance")
        expected_candidate_ids = sorted(
            hashlib.sha256(
                _stable_json(
                    {
                        **import_identity,
                        "source_case_id": source_case["case_id"],
                        "source_record_sha256": source_case["source_record"]["record_sha256"],
                    }
                ).encode("utf-8")
            ).hexdigest()
            for source_case, _materialized_case, *_artifacts in source_rows
        )
        if existing_import.get("candidate_ids") != expected_candidate_ids:
            raise CorpusError("duplicate #9656 import does not contain every source alias")
        expected_candidate_set = set(expected_candidate_ids)
        existing_candidates = [
            item
            for item in corpus.get("historical_candidates", [])
            if item["candidate_id"] in expected_candidate_set
        ]
        _verify_issue9656_duplicate_materialized_cases(source_rows, existing_candidates)
        _verify_existing_issue9656_candidate_artifacts(
            root,
            existing_import,
            existing_candidates,
        )
        return corpus, {**existing_import, "decision": "duplicate"}

    candidates: list[dict[str, Any]] = []
    staging_root = root / ".historical_candidate_import_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{import_id}.", dir=staging_root))
    promoted_dirs: list[Path] = []
    try:
        import_artifacts = staging / "historical_candidate_imports" / import_id
        import_artifacts.mkdir(parents=True)
        for payload_receipt in summary_receipts["payload_files"]:
            relative = payload_receipt["path"]
            source_payload_file = _safe_materialized_path(
                bundle_root / "payload", relative, label="evidence payload file"
            )
            _copy_verified_source_file(
                source_payload_file,
                import_artifacts / "payload" / Path(*PurePosixPath(relative).parts),
                expected_sha256=payload_receipt["sha256"],
            )
        _copy_verified_source_file(
            bundle_root / "evidence_bundle_manifest.json",
            import_artifacts / "evidence_bundle_manifest.json",
            expected_sha256=summary_receipts["manifest_sha256"],
        )
        _copy_verified_source_file(
            bundle_root / "checksums.sha256",
            import_artifacts / "checksums.sha256",
            expected_sha256=summary_receipts["checksums_sha256"],
        )
        _copy_verified_source_file(
            materialized_manifest_path,
            import_artifacts / "materialized_manifest.json",
            expected_sha256=materialized_manifest_sha,
        )

        candidates = [
            _stage_issue9656_candidate(
                source_row,
                staging=staging,
                import_id=import_id,
                import_identity=import_identity,
                summary_receipts=summary_receipts,
                materialized_manifest_sha256=materialized_manifest_sha,
            )
            for source_row in source_rows
        ]

        import_record = _issue9656_candidate_import_record(
            summary,
            import_id=import_id,
            source_identity=import_identity,
            summary_receipts=summary_receipts,
            materialized_manifest_sha256=materialized_manifest_sha,
            candidates=candidates,
        )
        import_record["artifact_paths"] = {
            "payload_root": f"historical_candidate_imports/{import_id}/payload",
            "evidence_bundle_manifest": f"historical_candidate_imports/{import_id}/evidence_bundle_manifest.json",
            "checksums": f"historical_candidate_imports/{import_id}/checksums.sha256",
            "materialized_manifest": f"historical_candidate_imports/{import_id}/materialized_manifest.json",
        }

        prospective = dict(corpus)
        prospective["historical_candidates"] = sorted(
            [*corpus.get("historical_candidates", []), *candidates],
            key=lambda item: item["candidate_id"],
        )
        prospective["historical_candidate_imports"] = sorted(
            [*corpus.get("historical_candidate_imports", []), import_record],
            key=lambda item: item["import_id"],
        )
        validate_corpus(prospective, corpus_root=root)
        _promote_issue9656_candidate_artifacts(
            staging,
            root,
            import_id=import_id,
            candidate_ids=[candidate["candidate_id"] for candidate in candidates],
            promoted_dirs=promoted_dirs,
        )
        corpus["historical_candidates"] = prospective["historical_candidates"]
        corpus["historical_candidate_imports"] = prospective["historical_candidate_imports"]
        return corpus, {**import_record, "decision": "imported"}
    except BaseException:
        for promoted in reversed(promoted_dirs):
            shutil.rmtree(promoted, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def admit_case_record(
    case_record: Mapping[str, Any],
    corpus: dict[str, Any],
    *,
    corpus_root: str | Path,
    artifact_root: str,
    source_kind: str,
    source_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Admit a complete, replay-verified case record from corpus-staged artifacts.

    ``artifact_root`` is a relative directory under the corpus root that contains every
    scenario, route, map, and replay artifact referenced by the case. The whole source
    directory is copied into immutable case custody before the case is appended.
    """
    root = Path(corpus_root).resolve()
    validate_corpus(corpus, corpus_root=root)
    if not source_kind.strip() or not source_id.strip():
        raise CorpusError("case admission source kind and ID must be non-empty")
    incoming = dict(case_record) if isinstance(case_record, Mapping) else {}
    identity = incoming.get("effective_scenario_sha256")
    try:
        artifact_root_path = _resolve_corpus_directory(artifact_root, root)
        errors = _validate_case_record(incoming, corpus_root=root)
        errors.extend(_validate_case_current_target_revision(incoming))
        if errors:
            raise CorpusError("case record rejected: " + "; ".join(errors))
        _case_artifacts_within_root(incoming, root, artifact_root_path)
    except (CorpusError, OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        corpus, receipt = _record_attempt(
            corpus,
            source_kind=source_kind,
            source_id=source_id,
            decision="rejected",
            blockers=[f"case_record_invalid:{type(exc).__name__}:{exc}"],
            candidate_identity=identity if isinstance(identity, str) else None,
            near_duplicate_report=_unassessed_near_duplicates(),
        )
        return corpus, receipt

    existing = next(
        (
            item
            for item in corpus["cases"]
            if item["effective_scenario_sha256"] == incoming["effective_scenario_sha256"]
        ),
        None,
    )
    near_report = _near_duplicate_report(incoming, corpus["cases"])
    if existing is not None:
        _merge_source_evidence(existing, incoming["source_evidence"])
        corpus, receipt = _record_attempt(
            corpus,
            source_kind=source_kind,
            source_id=source_id,
            decision="duplicate",
            blockers=[],
            candidate_identity=incoming["effective_scenario_sha256"],
            duplicate_case_id=existing["case_id"],
            near_duplicate_report=near_report,
        )
        receipt["case_id"] = existing["case_id"]
        validate_corpus(corpus, corpus_root=root)
        return corpus, receipt

    case_id = incoming["case_id"]
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    final_dir = cases_dir / case_id
    if final_dir.exists():
        corpus, receipt = _record_attempt(
            corpus,
            source_kind=source_kind,
            source_id=source_id,
            decision="rejected",
            blockers=["case_artifact_destination_exists_without_corpus_record"],
            candidate_identity=incoming["effective_scenario_sha256"],
            near_duplicate_report=near_report,
        )
        return corpus, receipt

    staging = Path(tempfile.mkdtemp(prefix=f".{case_id}.", dir=cases_dir))
    try:
        source_bundle = staging / "inputs" / "source_bundle"
        _copy_tree_without_symlinks(artifact_root_path, source_bundle)
        _rewrite_case_artifact_paths(
            incoming,
            corpus_root=root,
            artifact_root=artifact_root_path,
            new_root=f"cases/{case_id}/inputs/source_bundle",
        )
        incoming["source_evidence"]["admission_artifact_root"] = artifact_root
        staging.replace(final_dir)
        incoming["source_evidence"]["corpus_files"] = _case_file_inventory(final_dir, root)
        errors = _validate_case_record(incoming, corpus_root=root)
        if errors:
            raise CorpusError("materialized case record rejected: " + "; ".join(errors))
        corpus["cases"].append(incoming)
        corpus["cases"].sort(key=lambda item: item["case_id"])
        corpus, receipt = _record_attempt(
            corpus,
            source_kind=source_kind,
            source_id=source_id,
            decision="admitted",
            blockers=[],
            candidate_identity=incoming["effective_scenario_sha256"],
            near_duplicate_report=near_report,
        )
        receipt["case_id"] = case_id
        validate_corpus(corpus, corpus_root=root)
        return corpus, receipt
    except BaseException:
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def promote_historical_candidate(
    candidate_id: str,
    case_record: Mapping[str, Any],
    corpus: dict[str, Any],
    *,
    corpus_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Promote one pending #9656 candidate after exact replay and admission checks."""
    root = Path(corpus_root).resolve()
    validate_corpus(corpus, corpus_root=root)
    candidate = next(
        (
            item
            for item in corpus.get("historical_candidates", [])
            if item["candidate_id"] == candidate_id
        ),
        None,
    )
    if candidate is None:
        return _record_attempt(
            corpus,
            source_kind="issue_9656_historical_candidate",
            source_id=candidate_id,
            decision="rejected",
            blockers=["historical_candidate_missing"],
            candidate_identity=None,
            near_duplicate_report=_unassessed_near_duplicates(),
        )
    blockers = _historical_candidate_promotion_blockers(candidate, case_record, root, corpus=corpus)
    if blockers:
        return _record_attempt(
            corpus,
            source_kind="issue_9656_historical_candidate",
            source_id=candidate_id,
            decision="rejected",
            blockers=blockers,
            candidate_identity=(
                case_record.get("effective_scenario_sha256")
                if isinstance(case_record, Mapping)
                and isinstance(case_record.get("effective_scenario_sha256"), str)
                else None
            ),
            near_duplicate_report=_unassessed_near_duplicates(),
        )
    case_for_admission = copy.deepcopy(dict(case_record))
    binding = case_for_admission["discovery"]["historical_candidate_binding"]
    binding["historical_source_replay"] = {
        "source_replay_status": candidate["source_replay_status"],
        "raw_episode_artifact_custody": "digest_only_not_copied_from_campaign_output",
        "raw_episode_artifact_used_as_admission_evidence": False,
        "local_ignored_output_used_as_admission_evidence": False,
        "admission_replay_revision": case_for_admission["replay_receipt"]["replay_revision"],
        "admission_replay_matches_target_revision": True,
    }
    case_for_admission["source_evidence"]["historical_candidate_promotion"] = {
        "candidate_id": candidate["candidate_id"],
        **binding["historical_source_replay"],
    }
    corpus, receipt = admit_case_record(
        case_for_admission,
        corpus,
        corpus_root=root,
        artifact_root=f"historical_candidates/{candidate_id}",
        source_kind="issue_9656_historical_candidate",
        source_id=candidate_id,
    )
    if receipt["decision"] in {"admitted", "duplicate"}:
        candidate["source_candidate_status"] = candidate["candidate_status"]
        candidate["candidate_status"] = "admitted"
        candidate["promoted_case_id"] = receipt["case_id"]
        candidate["promotion_attempt_id"] = receipt["attempt_id"]
        validate_corpus(corpus, corpus_root=root)
    return corpus, receipt


def _historical_candidate_promotion_blockers(
    candidate: Mapping[str, Any],
    case_record: Mapping[str, Any],
    corpus_root: Path,
    *,
    corpus: Mapping[str, Any],
) -> list[str]:
    blockers = []
    if candidate.get("candidate_status") != "pending_exact_replay":
        blockers.append("candidate_is_not_pending_exact_replay")
    provenance = candidate.get("source_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("source_identity_binding_status") != "verified"
    ):
        blockers.append("candidate_source_provenance_is_not_verified")
    blockers.extend(_historical_candidate_source_blockers(candidate, corpus, corpus_root))
    if not isinstance(case_record, Mapping):
        return [*blockers, "completed_case_record_missing"]
    discovery = case_record.get("discovery")
    binding = discovery.get("historical_candidate_binding") if isinstance(discovery, dict) else None
    expected_binding = {
        "candidate_id": candidate.get("candidate_id"),
        "source_issue": 9656,
        "source_case_id": candidate.get("source_case_id"),
        "source_record_sha256": candidate.get("source_record_sha256"),
        "source_replay_status": candidate.get("source_replay_status"),
    }
    if not isinstance(binding, dict) or any(
        binding.get(key) != value for key, value in expected_binding.items()
    ):
        blockers.append("case_record_does_not_bind_historical_candidate")
    target = case_record.get("target_planner")
    target = target if isinstance(target, dict) else {}
    candidate_target = candidate.get("target_planner")
    candidate_target = candidate_target if isinstance(candidate_target, dict) else {}
    if (
        case_record.get("scenario_id") != candidate.get("scenario_id")
        or case_record.get("scenario_seed") != candidate.get("scenario_seed")
        or target.get("planner_id") != candidate_target.get("planner_id")
        or target.get("config_identity") != candidate_target.get("config_hash")
    ):
        blockers.append("case_record_scenario_or_planner_differs_from_candidate")

    try:
        candidate_matrix_path = _resolve_corpus_artifact(
            candidate.get("artifact_paths", {}).get("replay_matrix"), corpus_root
        )
        candidate_config_path = _resolve_corpus_artifact(
            candidate.get("artifact_paths", {}).get("planner_config"), corpus_root
        )
        case_paths = _case_input_paths(case_record, corpus_root)
        candidate_matrix = _load_yaml_object(candidate_matrix_path, "candidate replay matrix")
        candidate_rows = candidate_matrix.get("scenarios")
        case_document = _load_yaml_object(case_paths["scenario"], "promoted scenario")
        case_rows = case_document.get("scenarios")
        if (
            not isinstance(candidate_rows, list)
            or len(candidate_rows) != 1
            or not isinstance(candidate_rows[0], dict)
            or not isinstance(case_rows, list)
            or len(case_rows) != 1
            or not isinstance(case_rows[0], dict)
            or _historical_candidate_scenario_projection(candidate_rows[0]) != case_rows[0]
        ):
            blockers.append("case_scenario_differs_from_candidate_replay_matrix")
        candidate_config = yaml.safe_load(candidate_config_path.read_text(encoding="utf-8"))
        if candidate_config != target.get("configuration_snapshot"):
            blockers.append("case_planner_config_differs_from_candidate_snapshot")
        candidate_map_assets = candidate.get("replay_inputs", {}).get("map_assets", [])
        candidate_map_digests = sorted(
            item.get("stored_sha256") for item in candidate_map_assets if isinstance(item, dict)
        )
        case_map_digests = sorted(
            item.get("sha256")
            for item in case_record.get("inputs", {}).get("map_assets", [])
            if isinstance(item, dict) and item.get("role") == "map"
        )
        if candidate_map_digests != case_map_digests:
            blockers.append("case_map_bytes_differ_from_candidate_materialization")
    except (CorpusError, OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        blockers.append(f"candidate_input_binding_invalid:{type(exc).__name__}:{exc}")
    blockers.extend(_validate_case_record(case_record, corpus_root=corpus_root))
    return sorted(set(blockers))


def _historical_candidate_source_blockers(
    candidate: Mapping[str, Any], corpus: Mapping[str, Any], corpus_root: Path
) -> list[str]:
    """Recheck source-bound #9656 records and all imported input bytes before promotion."""
    provenance = candidate.get("source_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    import_id = provenance.get("import_id")
    if not isinstance(import_id, str) or not import_id:
        return ["candidate_import_id_missing"]
    try:
        summary_row, source_record = _load_pinned_historical_source_row(
            candidate, corpus, corpus_root, provenance, import_id
        )
        errors = _historical_candidate_metadata_errors(
            candidate, provenance, summary_row, source_record
        )
        errors.extend(_verify_historical_candidate_artifacts(candidate, corpus_root))
        errors.extend(
            _validate_materialized_historical_case(
                candidate, corpus_root, source_record, provenance
            )
        )
    except (CorpusError, OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        return [f"candidate_source_artifact_invalid:{type(exc).__name__}:{exc}"]
    return errors


def _load_pinned_historical_source_row(
    candidate: Mapping[str, Any],
    corpus: Mapping[str, Any],
    corpus_root: Path,
    provenance: Mapping[str, Any],
    import_id: str,
) -> tuple[dict[str, Any], Any]:
    imports = corpus.get("historical_candidate_imports", [])
    import_record = next(
        (item for item in imports if isinstance(item, dict) and item.get("import_id") == import_id),
        None,
    )
    if not isinstance(import_record, dict) or candidate.get(
        "candidate_id"
    ) not in import_record.get("candidate_ids", []):
        raise CorpusError("candidate import receipt is missing or does not name this row")
    imported_files = import_record.get("source_files", [])
    if not isinstance(imported_files, list):
        raise CorpusError("candidate import receipt has no source file inventory")
    for receipt in imported_files:
        if not isinstance(receipt, dict):
            raise CorpusError("candidate import receipt contains a malformed source file")
        _verify_corpus_artifact(corpus_root, receipt.get("stored_path"), receipt.get("sha256"))
    summary_receipt = next(
        (
            item
            for item in imported_files
            if isinstance(item, dict) and item.get("path") == "payload/summary.json"
        ),
        None,
    )
    summary_path_ref = candidate.get("artifact_paths", {}).get("import_summary")
    if (
        not isinstance(summary_receipt, dict)
        or summary_path_ref != summary_receipt.get("stored_path")
        or provenance.get("summary_sha256") != summary_receipt.get("sha256")
    ):
        raise CorpusError("candidate summary reference differs from its import receipt")
    summary_path = _resolve_corpus_artifact(summary_path_ref, corpus_root)
    summary_digest = provenance.get("summary_sha256")
    if not _is_sha256(summary_digest) or _sha256_file(summary_path) != summary_digest:
        raise CorpusError("candidate import summary digest differs from source provenance")
    summary = _read_json_object(summary_path)
    source_case_alias = candidate.get("source_case_id")
    row = next(
        (
            item
            for item in summary.get("cases", [])
            if isinstance(item, dict) and item.get("case_id") == source_case_alias
        ),
        None,
    )
    if not isinstance(row, dict):
        raise CorpusError("candidate_import_summary_does_not_contain_source_case")
    return row, row.get("source_record")


def _historical_candidate_metadata_errors(
    candidate: Mapping[str, Any],
    provenance: Mapping[str, Any],
    summary_row: Mapping[str, Any],
    source_record: Any,
) -> list[str]:
    source_record = source_record if isinstance(source_record, Mapping) else {}
    raw_custody = provenance.get("raw_episode_artifact_custody")
    errors = []
    if (
        not isinstance(raw_custody, dict)
        or raw_custody.get("status") != "digest_only_not_copied_from_campaign_output"
        or raw_custody.get("episode_file") != source_record.get("episode_file")
        or raw_custody.get("episode_file_sha256") != source_record.get("episode_file_sha256")
        or raw_custody.get("raw_episode_artifact_used_as_admission_evidence") is not False
        or raw_custody.get("local_ignored_output_used_as_admission_evidence") is not False
        or raw_custody.get("admission_requires") != "exact_current_revision_replay"
    ):
        errors.append("candidate_raw_episode_custody_boundary_is_missing_or_inaccurate")
    if (
        candidate.get("source_issue") != 9656
        or candidate.get("source_record_sha256") != source_record.get("record_sha256")
        or candidate.get("scenario_id") != summary_row.get("scenario_id")
        or candidate.get("scenario_seed") != summary_row.get("seed")
        or candidate.get("target_planner", {}).get("planner_id") != summary_row.get("planner_key")
        or candidate.get("source_replay_status") != "not_attempted"
        or candidate.get("candidate_status") != "pending_exact_replay"
    ):
        errors.append("candidate_metadata_differs_from_checksum_pinned_summary_row")
    return errors


def _verify_historical_candidate_artifacts(
    candidate: Mapping[str, Any], corpus_root: Path
) -> list[str]:
    provenance = candidate.get("source_provenance", {})
    replay_inputs = candidate.get("replay_inputs", {})
    paths = candidate.get("artifact_paths", {})
    expected_artifacts = (
        ("source_case", provenance.get("source_case_file_sha256")),
        ("source_matrix", replay_inputs.get("source_matrix_sha256")),
        ("source_planner_config", replay_inputs.get("source_planner_config_sha256")),
        ("replay_matrix", replay_inputs.get("normalized_matrix_sha256")),
        ("planner_config", replay_inputs.get("normalized_planner_config_sha256")),
    )
    errors = []
    for name, digest in expected_artifacts:
        try:
            _verify_corpus_artifact(corpus_root, paths.get(name), digest)
        except CorpusError as exc:
            errors.append(f"candidate_{name}_artifact_invalid:{exc}")
    assets = replay_inputs.get("map_assets", [])
    for asset in assets:
        if isinstance(asset, dict):
            try:
                _verify_corpus_artifact(
                    corpus_root, asset.get("stored_path"), asset.get("stored_sha256")
                )
            except CorpusError as exc:
                errors.append(f"candidate_map_artifact_invalid:{exc}")
    return errors


def _validate_materialized_historical_case(
    candidate: Mapping[str, Any],
    corpus_root: Path,
    source_record: Any,
    provenance: Mapping[str, Any],
) -> list[str]:
    if not isinstance(source_record, dict):
        return ["candidate_materialized_source_record_missing"]
    paths = candidate.get("artifact_paths", {})
    source_case = _read_json_object(_resolve_corpus_artifact(paths.get("source_case"), corpus_root))
    source = source_case.get("source")
    planner = source_case.get("planner")
    scenario = source_case.get("scenario")
    materialized_source_record = source_case.get("source_record")
    target = candidate.get("target_planner", {})
    if (
        source_case.get("case_id") != candidate.get("source_case_id")
        or not isinstance(source, dict)
        or source.get("record_sha256") != candidate.get("source_record_sha256")
        or not isinstance(planner, dict)
        or planner.get("key") != target.get("planner_id")
        or not isinstance(scenario, dict)
        or scenario.get("scenario_id") != candidate.get("scenario_id")
        or scenario.get("seed") != candidate.get("scenario_seed")
        or not isinstance(materialized_source_record, dict)
        or materialized_source_record.get("scenario_id") != candidate.get("scenario_id")
        or materialized_source_record.get("seed") != candidate.get("scenario_seed")
        or materialized_source_record.get("algo") != target.get("planner_id")
        or materialized_source_record.get("algorithm_metadata", {}).get("config_hash")
        != target.get("config_hash")
        or materialized_source_record.get("algorithm_metadata", {}).get("canonical_algorithm")
        != target.get("canonical_algorithm")
    ):
        return ["candidate_materialized_case_differs_from_checksum_pinned_source"]
    metadata = materialized_source_record.get("algorithm_metadata")
    expected_config = metadata.get("config_hash") if isinstance(metadata, dict) else None
    if target.get("config_hash") != expected_config:
        return ["candidate_planner_config_differs_from_checksum_pinned_source_row"]
    if provenance.get("source_case_file_sha256") != _sha256_file(
        _resolve_corpus_artifact(paths.get("source_case"), corpus_root)
    ):
        return ["candidate_materialized_case_digest_differs_from_source_provenance"]
    return []


def _historical_candidate_scenario_projection(candidate_row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize #9656 replay-only aliases before comparing runtime scenario fields."""
    projected = dict(candidate_row)
    scenario_id = projected.pop("id", None)
    projected.pop("algo", None)
    projected.pop("map_file", None)
    name = projected.get("name")
    if scenario_id is not None:
        if name is not None and name != scenario_id:
            return {"_invalid_scenario_alias": True}
        projected["name"] = scenario_id
    return projected


def _resolve_corpus_directory(relative: Any, root: Path) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise CorpusError("case artifact root must be a non-empty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise CorpusError("case artifact root path is unsafe")
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise CorpusError("case artifact root escapes corpus root") from exc
    if not path.is_dir():
        raise CorpusError("case artifact root is missing")
    return path


def _case_artifacts_within_root(case: Mapping[str, Any], root: Path, artifact_root: Path) -> None:
    paths = _case_input_paths(case, root)
    referenced = [
        path
        for key, path in paths.items()
        if key in {"scenario", "route"} or key.endswith("_asset")
    ]
    receipt = case["replay_receipt"]
    for replay in receipt.get("replay_artifacts", []):
        for key in ("path", "provenance_path"):
            value = replay.get(key)
            if value is not None:
                referenced.append(_resolve_corpus_artifact(value, root))
    for replay in receipt.get("artifact_receipts", []):
        referenced.append(_resolve_corpus_artifact(replay.get("artifact_path"), root))
    for path in referenced:
        try:
            path.resolve().relative_to(artifact_root.resolve())
        except ValueError as exc:
            raise CorpusError("case input or replay evidence is outside artifact_root") from exc


def _copy_tree_without_symlinks(source: Path, destination: Path) -> None:
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_symlink():
            raise CorpusError("case artifact bundle contains a symlink")
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
        else:
            raise CorpusError("case artifact bundle contains a non-regular file")


def _rewrite_case_artifact_paths(
    case: dict[str, Any],
    *,
    corpus_root: Path,
    artifact_root: Path,
    new_root: str,
) -> None:
    def rewrite(relative: Any) -> str:
        path = _resolve_corpus_artifact(relative, corpus_root)
        try:
            within = path.relative_to(artifact_root).as_posix()
        except ValueError as exc:
            raise CorpusError("case artifact path is outside its artifact root") from exc
        return (PurePosixPath(new_root) / PurePosixPath(within)).as_posix()

    inputs = case["inputs"]
    inputs["scenario_path"] = rewrite(inputs["scenario_path"])
    inputs["route_overrides_path"] = rewrite(inputs["route_overrides_path"])
    for asset in inputs["map_assets"]:
        asset["path"] = rewrite(asset["path"])
    receipt = case["replay_receipt"]
    for replay in receipt.get("replay_artifacts", []):
        for key in ("path", "provenance_path"):
            if replay.get(key) is not None:
                replay[key] = rewrite(replay[key])
    for replay in receipt.get("artifact_receipts", []):
        replay["artifact_path"] = rewrite(replay["artifact_path"])
    case["source_evidence"]["corpus_files"] = []


def _stage_issue9656_candidate(
    source_row: tuple[dict[str, Any], dict[str, Any], Path, Path, Path, dict[str, Any]],
    *,
    staging: Path,
    import_id: str,
    import_identity: Mapping[str, Any],
    summary_receipts: Mapping[str, Any],
    materialized_manifest_sha256: str,
) -> dict[str, Any]:
    source_case, materialized_case, source_case_path, source_matrix, source_config, source_map = (
        source_row
    )
    source_record = source_case["source_record"]
    candidate_identity = {
        **import_identity,
        "source_case_id": source_case["case_id"],
        "source_record_sha256": source_record["record_sha256"],
    }
    candidate_id = hashlib.sha256(_stable_json(candidate_identity).encode("utf-8")).hexdigest()
    candidate_dir = staging / "historical_candidates" / candidate_id
    (candidate_dir / "source").mkdir(parents=True)
    (candidate_dir / "replay_input").mkdir()
    (candidate_dir / "maps").mkdir()

    source_case_sha = _sha256_file(source_case_path)
    if source_case_sha != source_map.get("_materialized_case_sha256"):
        raise CorpusError("#9656 materialized case changed after source validation")
    source_matrix_sha = _sha256_file(source_matrix)
    source_config_sha = _sha256_file(source_config)
    _copy_verified_source_file(
        source_case_path, candidate_dir / "source_case.json", expected_sha256=source_case_sha
    )
    _copy_verified_source_file(
        source_matrix,
        candidate_dir / "source" / "replay_matrix.yaml",
        expected_sha256=source_case["replay_input"]["scenario_matrix_sha256"],
    )
    _copy_verified_source_file(
        source_config,
        candidate_dir / "source" / "planner_config.yaml",
        expected_sha256=source_case["replay_input"]["planner_config_sha256"],
    )
    normalized_matrix, map_receipt = _normalize_issue9656_replay_matrix(
        source_matrix, candidate_dir, source_map
    )
    matrix_path = candidate_dir / "replay_input" / "replay_matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(normalized_matrix, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    config_path = candidate_dir / "replay_input" / "planner_config.yaml"
    _copy_verified_source_file(
        source_config,
        config_path,
        expected_sha256=source_case["replay_input"]["planner_config_sha256"],
    )
    return _issue9656_candidate_record(
        source_case,
        materialized_case,
        {
            "candidate_id": candidate_id,
            "import_id": import_id,
            "summary_receipts": summary_receipts,
            "materialized_manifest_sha256": materialized_manifest_sha256,
            "source_case_sha256": source_case_sha,
            "source_matrix_sha256": source_matrix_sha,
            "source_config_sha256": source_config_sha,
            "normalized_matrix_sha256": _sha256_file(matrix_path),
            "normalized_config_sha256": _sha256_file(config_path),
            "map_receipt": map_receipt,
            "source_binding": source_map["_source_binding"],
        },
    )


def _issue9656_candidate_record(
    source_case: Mapping[str, Any],
    materialized_case: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    source_record = source_case["source_record"]
    source = materialized_case.get("source", {})
    summary_receipts = context["summary_receipts"]
    source_binding = context["source_binding"]
    replay_status = source_case["replay"]["status"]
    candidate_status = {
        "not_attempted": "pending_exact_replay",
        "unavailable_model_artifact": "blocked_unavailable_model_artifact",
        "mismatch_different_revision": "blocked_replay_revision_mismatch",
    }[replay_status]
    if source_binding["status"] != "verified":
        candidate_status = "blocked_source_provenance_mismatch"
    return {
        "schema_version": "adversarial-historical-candidate.v1",
        "candidate_id": context["candidate_id"],
        "source_issue": 9656,
        "source_case_id": source_case["case_id"],
        "source_record_sha256": source_record["record_sha256"],
        "source_provenance": {
            "import_id": context["import_id"],
            "source_row_binding": "verified_episode_file_and_line_sha256",
            "summary_sha256": summary_receipts["summary_sha256"],
            "materialized_manifest_sha256": context["materialized_manifest_sha256"],
            "source_bundle_sha256": source.get("bundle_sha256"),
            "campaign_id": source.get("campaign_id"),
            "campaign_source_revision": source.get("campaign_source_revision"),
            "row_git_hash": source.get("row_git_hash"),
            "summary_source_revision": source_binding["summary_source_revision"],
            "episode_git_hash": source_binding["episode_git_hash"],
            "source_identity_binding_status": source_binding["status"],
            "source_identity_binding_issues": source_binding["issues"],
            "episode_file": source_record["episode_file"],
            "episode_file_sha256": source_record["episode_file_sha256"],
            "line_number": source_record["line_number"],
            "materialized_planner_config_hash": source.get("planner_config_hash"),
            "episode_planner_config_hash": source_binding["episode_planner_config_hash"],
            "episode_scenario_algo_config_hash": source_binding[
                "episode_scenario_algo_config_hash"
            ],
            "raw_planner_alias": source_case["planner_key"],
            "episode_canonical_algorithm": source_binding["episode_canonical_algorithm"],
            "materialized_canonical_algorithm": source_binding["materialized_canonical_algorithm"],
            "source_case_file_sha256": context["source_case_sha256"],
            "source_replay_matrix_sha256": context["source_matrix_sha256"],
            "source_planner_config_sha256": context["source_config_sha256"],
            "raw_episode_artifact_custody": {
                "status": "digest_only_not_copied_from_campaign_output",
                "episode_file": source_record["episode_file"],
                "episode_file_sha256": source_record["episode_file_sha256"],
                "raw_episode_artifact_used_as_admission_evidence": False,
                "local_ignored_output_used_as_admission_evidence": False,
                "admission_requires": "exact_current_revision_replay",
            },
        },
        "candidate_status": candidate_status,
        "source_replay_status": replay_status,
        "source_replay": dict(source_case["replay"]),
        "scenario_id": source_case["scenario_id"],
        "scenario_family": source_case["scenario_family"],
        "scenario_seed": source_case["seed"],
        "benchmark_eligible": source_case["benchmark_eligible"],
        "target_planner": {
            "planner_id": source_case["planner_key"],
            "canonical_algorithm": source_binding["canonical_algorithm"],
            "config_hash": (
                source_binding["episode_planner_config_hash"]
                if source_binding["status"] == "verified"
                else None
            ),
        },
        "criticality": dict(source_case["criticality"]),
        "replay_inputs": {
            "source_matrix_sha256": context["source_matrix_sha256"],
            "normalized_matrix_sha256": context["normalized_matrix_sha256"],
            "source_planner_config_sha256": context["source_config_sha256"],
            "normalized_planner_config_sha256": context["normalized_config_sha256"],
            "map_assets": context["map_receipt"],
        },
        "artifact_paths": {
            "source_case": f"historical_candidates/{context['candidate_id']}/source_case.json",
            "source_matrix": f"historical_candidates/{context['candidate_id']}/source/replay_matrix.yaml",
            "source_planner_config": f"historical_candidates/{context['candidate_id']}/source/planner_config.yaml",
            "replay_matrix": f"historical_candidates/{context['candidate_id']}/replay_input/replay_matrix.yaml",
            "planner_config": f"historical_candidates/{context['candidate_id']}/replay_input/planner_config.yaml",
            "import_summary": f"historical_candidate_imports/{context['import_id']}/payload/summary.json",
            "import_materialized_manifest": f"historical_candidate_imports/{context['import_id']}/materialized_manifest.json",
        },
    }


def _verify_issue9656_source_summary(
    summary_file: Path, bundle_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_summary = (bundle_root / "payload" / "summary.json").resolve()
    if summary_file != expected_summary:
        raise CorpusError("#9656 summary must be the payload/summary.json in its evidence bundle")
    manifest_path = bundle_root / "evidence_bundle_manifest.json"
    checksum_path = bundle_root / "checksums.sha256"
    bundle_manifest = _read_json_object(manifest_path)
    if bundle_manifest.get("schema_version") != ISSUE_9645_BUNDLE_SCHEMA:
        raise CorpusError("#9656 evidence bundle manifest schema is invalid")
    entries = bundle_manifest.get("files")
    totals = bundle_manifest.get("totals")
    if not isinstance(entries, list) or not isinstance(totals, dict):
        raise CorpusError("#9656 evidence bundle inventory is incomplete")
    files_by_path = _validated_bundle_file_map(entries)
    total_bytes = sum(item["size_bytes"] for item in files_by_path.values())
    if totals.get("file_count") != len(files_by_path) or totals.get("total_bytes") != total_bytes:
        raise CorpusError("#9656 evidence bundle totals disagree with its file inventory")
    checksums = _parse_sha256_receipt(checksum_path, label="#9656 bundle checksums")
    expected_checksums = {
        f"payload/{relative}": record["sha256"] for relative, record in files_by_path.items()
    }
    if checksums != expected_checksums:
        raise CorpusError("#9656 checksum sidecar disagrees with the evidence manifest")
    for relative, record in files_by_path.items():
        _verify_bundle_payload_file(bundle_root / "payload", relative, record)
    summary = _read_json_object(summary_file)
    if summary.get("schema_version") != ISSUE_9656_SUMMARY_SCHEMA:
        raise CorpusError("unsupported #9656 hard-case summary schema")
    summary_entry = files_by_path.get("summary.json")
    if summary_entry is None:
        raise CorpusError("#9656 evidence bundle does not bind summary.json")
    return summary, {
        "summary_sha256": summary_entry["sha256"],
        "manifest_sha256": _sha256_file(manifest_path),
        "checksums_sha256": _sha256_file(checksum_path),
        "payload_files": [
            {"path": relative, "sha256": item["sha256"]}
            for relative, item in sorted(files_by_path.items())
        ],
    }


def _validate_issue9656_campaign_source(
    summary: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, str]:
    source = summary.get("source")
    materialized_source = manifest.get("source")
    required = (
        "source_revision",
        "source_campaign_id",
        "bundle_sha256",
        "matrix_path",
        "matrix_sha256",
    )
    if not isinstance(source, dict) or not isinstance(materialized_source, dict):
        raise CorpusError("#9656 source summary or materialized manifest lacks source identity")
    revision = source.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not isinstance(source.get("source_campaign_id"), str)
        or not source["source_campaign_id"].strip()
        or not _is_sha256(source.get("bundle_sha256"))
        or not _safe_bundle_relative_path(source.get("matrix_path"))
        or not _is_sha256(source.get("matrix_sha256"))
    ):
        raise CorpusError("#9656 source summary has invalid campaign identity fields")
    if any(materialized_source.get(key) != source.get(key) for key in required):
        raise CorpusError("#9656 materialized manifest source identity differs from summary")
    historical_matrix = _issue9656_historical_git_blob(revision, source["matrix_path"])
    if hashlib.sha256(historical_matrix).hexdigest() != source["matrix_sha256"]:
        raise CorpusError("#9656 source matrix digest does not match its campaign revision")
    return {key: source[key] for key in required}


def _issue9656_historical_git_blob(revision: str, relative_path: str) -> bytes:
    """Read one path from a pinned local Git commit, without consulting the worktree file."""
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not _safe_bundle_relative_path(relative_path)
    ):
        raise CorpusError("#9656 historical Git blob identity is invalid")
    try:
        result = subprocess.run(
            ["git", "show", f"{revision}:{relative_path}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CorpusError(
            f"#9656 historical Git blob is unavailable: {revision}:{relative_path}"
        ) from exc
    return result.stdout


def _validate_issue9656_materialization(
    summary: Mapping[str, Any],
    materialized: Path,
    manifest: Mapping[str, Any],
    campaign_root: Path,
) -> list[tuple[dict[str, Any], dict[str, Any], Path, Path, Path, dict[str, Any]]]:
    if manifest.get("schema_version") != ISSUE_9656_SUMMARY_SCHEMA:
        raise CorpusError("#9656 materialized manifest schema differs from the source summary")
    summary_source = _validate_issue9656_campaign_source(summary, manifest)
    summary_cases = summary.get("cases")
    manifest_cases = manifest.get("cases")
    selection = summary.get("selection")
    if not isinstance(summary_cases, list) or not isinstance(manifest_cases, list):
        raise CorpusError("#9656 summary or materialized manifest has no case rows")
    if not isinstance(selection, dict) or selection.get("case_count") != len(summary_cases):
        raise CorpusError("#9656 selection count does not match its candidate rows")
    rows_by_id, manifest_by_id = _issue9656_rows_by_alias(summary_cases, manifest_cases, selection)
    replay = _issue9656_replay_summary(summary)
    status_counts: Counter[str] = Counter()
    anomalies: Counter[str] = Counter()
    rows = []
    for case_id, summary_case in rows_by_id.items():
        manifest_case = manifest_by_id[case_id]
        status = _validate_issue9656_summary_manifest_row(summary_case, manifest_case)
        status_counts[status] += 1
        anomalies.update(summary_case["criticality"]["anomalies"])
        rows.append(
            _validate_issue9656_materialized_case(
                summary_case, manifest_case, materialized, campaign_root, summary_source
            )
        )
    if dict(status_counts) != replay["status_counts"]:
        raise CorpusError("#9656 row replay statuses do not reconcile with the summary totals")
    if dict(anomalies) != summary.get("criticality_anomaly_counts"):
        raise CorpusError("#9656 row criticality anomalies do not reconcile with summary totals")
    _validate_issue9656_setup_accounting(summary)
    return rows


def _issue9656_rows_by_alias(
    summary_cases: list[Any], manifest_cases: list[Any], selection: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if any(not isinstance(case, dict) for case in summary_cases + manifest_cases):
        raise CorpusError("#9656 case lists contain a malformed row")
    summary_ids = [case["case_id"] for case in summary_cases]
    manifest_ids = [case["case_id"] for case in manifest_cases]
    if (
        any(not isinstance(case_id, str) for case_id in summary_ids + manifest_ids)
        or len(summary_ids) != len(set(summary_ids))
        or summary_ids != selection.get("case_ids")
        or set(summary_ids) != set(manifest_ids)
        or len(manifest_ids) != len(set(manifest_ids))
    ):
        raise CorpusError("#9656 stable source case aliases disagree across the manifests")
    if not all(_is_issue9656_source_alias(case_id) for case_id in summary_ids):
        raise CorpusError("#9656 source aliases must retain their case- plus 16-hex form")
    return (
        {case["case_id"]: case for case in summary_cases},
        {case["case_id"]: case for case in manifest_cases},
    )


def _issue9656_replay_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    replay = summary.get("replay")
    if not isinstance(replay, dict) or not isinstance(replay.get("status_counts"), dict):
        raise CorpusError("#9656 replay status accounting is absent")
    return replay


def _is_issue9656_source_alias(value: str) -> bool:
    return (
        len(value) == 21
        and value.startswith("case-")
        and all(character in "0123456789abcdef" for character in value[5:])
    )


def _validate_issue9656_summary_manifest_row(
    summary_case: Mapping[str, Any], manifest_case: Mapping[str, Any]
) -> str:
    projected_manifest_case = {
        key: value for key, value in manifest_case.items() if key not in {"case_file", "replay"}
    }
    projected_summary_case = {key: value for key, value in summary_case.items() if key != "replay"}
    replay_row = summary_case.get("replay")
    manifest_replay = manifest_case.get("replay")
    binding_keys = ("attempted", "status", "source_revision", "replay_revision", "comparison")
    if (
        projected_manifest_case != projected_summary_case
        or not isinstance(replay_row, dict)
        or not isinstance(manifest_replay, dict)
        or any(manifest_replay.get(key) != replay_row.get(key) for key in binding_keys)
    ):
        raise CorpusError("#9656 materializer changed source summary row")
    status = replay_row.get("status")
    supported_statuses = {
        "not_attempted",
        "unavailable_model_artifact",
        "mismatch_different_revision",
    }
    if (
        not isinstance(status, str)
        or status not in supported_statuses
        or replay_row.get("attempted") is not (status == "mismatch_different_revision")
    ):
        raise CorpusError("#9656 replay status or attempt flag is unsupported")
    criticality = summary_case.get("criticality")
    source_record = summary_case.get("source_record")
    replay_input = summary_case.get("replay_input")
    if (
        not isinstance(criticality, dict)
        or not isinstance(criticality.get("anomalies"), list)
        or not isinstance(source_record, dict)
        or not _is_sha256(source_record.get("record_sha256"))
        or not isinstance(replay_input, dict)
        or replay_input.get("status") != "materialized"
    ):
        raise CorpusError("#9656 candidate row lacks required source evidence")
    return status


def _validate_issue9656_materialized_case(
    summary_case: dict[str, Any],
    manifest_case: dict[str, Any],
    materialized: Path,
    campaign_root: Path,
    summary_source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, dict[str, Any]]:
    case_id = summary_case["case_id"]
    replay_input = summary_case["replay_input"]
    case_file = _safe_materialized_path(
        materialized, manifest_case.get("case_file"), label="case file"
    )
    materialized_case = _read_json_object(case_file)
    if not _issue9656_case_document_matches(summary_case, manifest_case, materialized_case):
        raise CorpusError(f"#9656 materialized case does not bind summary row {case_id}")
    source_row = _verify_issue9656_source_row(campaign_root, summary_case, materialized_case)
    source_binding = _issue9656_source_identity_binding(
        summary_case, materialized_case, source_row, summary_source
    )
    matrix_path = _safe_materialized_path(
        case_file.parent, replay_input.get("scenario_matrix_path"), label="scenario matrix"
    )
    config_path = _safe_materialized_path(
        case_file.parent, replay_input.get("planner_config_path"), label="planner config"
    )
    matrix_sha = replay_input.get("scenario_matrix_sha256")
    config_sha = replay_input.get("planner_config_sha256")
    if _sha256_file(matrix_path) != matrix_sha or _sha256_file(config_path) != config_sha:
        raise CorpusError(f"#9656 materialized replay input digest differs for {case_id}")
    map_info = _issue9656_map_asset(matrix_path, summary_case, summary_source["source_revision"])
    map_info["_source_binding"] = source_binding
    map_info["_materialized_case_sha256"] = _sha256_file(case_file)
    return summary_case, materialized_case, case_file, matrix_path, config_path, map_info


def _issue9656_case_document_matches(
    summary_case: Mapping[str, Any],
    manifest_case: Mapping[str, Any],
    materialized_case: Mapping[str, Any],
) -> bool:
    source_record = summary_case["source_record"]
    source = materialized_case.get("source")
    planner = materialized_case.get("planner")
    scenario = materialized_case.get("scenario")
    if not all(isinstance(item, dict) for item in (source, planner, scenario)):
        return False
    return (
        materialized_case.get("schema_version") == "benchmark-hard-case.v1"
        and materialized_case.get("case_id") == summary_case.get("case_id")
        and all(source.get(key) == value for key, value in source_record.items())
        and materialized_case.get("criticality") == summary_case.get("criticality")
        and materialized_case.get("replay") == manifest_case.get("replay")
        and materialized_case.get("replay_input") == summary_case.get("replay_input")
        and planner.get("key") == summary_case.get("planner_key")
        and scenario.get("scenario_id") == summary_case.get("scenario_id")
        and scenario.get("seed") == summary_case.get("seed")
    )


def _verify_issue9656_source_row(
    campaign_root: Path,
    summary_case: Mapping[str, Any],
    materialized_case: Mapping[str, Any],
) -> dict[str, Any]:
    source_ref = summary_case["source_record"]
    episode_path = _safe_materialized_path(
        campaign_root, source_ref.get("episode_file"), label="source episode file"
    )
    if _sha256_file(episode_path) != source_ref.get("episode_file_sha256"):
        raise CorpusError(f"#9656 source episode file digest differs for {summary_case['case_id']}")
    line_number = source_ref.get("line_number")
    if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number < 1:
        raise CorpusError(
            f"#9656 source episode line number is invalid for {summary_case['case_id']}"
        )
    lines = episode_path.read_bytes().splitlines(keepends=True)
    if line_number > len(lines):
        raise CorpusError(f"#9656 source episode row is absent for {summary_case['case_id']}")
    raw_line = lines[line_number - 1]
    if hashlib.sha256(raw_line).hexdigest() != source_ref.get("record_sha256"):
        raise CorpusError(f"#9656 source episode row digest differs for {summary_case['case_id']}")
    try:
        row = json.loads(raw_line)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(
            f"#9656 source episode row is malformed for {summary_case['case_id']}"
        ) from exc
    if (
        not isinstance(row, dict)
        or row != materialized_case.get("source_record")
        or row.get("scenario_id") != summary_case.get("scenario_id")
        or row.get("seed") != summary_case.get("seed")
        or row.get("algo") != summary_case.get("planner_key")
    ):
        raise CorpusError("#9656 materialized source row differs from campaign evidence")
    return row


def _issue9656_source_identity_binding(
    summary_case: Mapping[str, Any],
    materialized_case: Mapping[str, Any],
    source_row: Mapping[str, Any],
    summary_source: Mapping[str, Any],
) -> dict[str, Any]:
    source = materialized_case.get("source")
    planner = materialized_case.get("planner")
    if not isinstance(source, dict) or not isinstance(planner, dict):
        raise CorpusError("#9656 materialized source identity is malformed")
    metadata = source_row.get("algorithm_metadata")
    planner_metadata = planner.get("algorithm_metadata")
    scenario_params = source_row.get("scenario_params")
    if not isinstance(metadata, dict) or not isinstance(planner_metadata, dict):
        metadata = {}
        planner_metadata = {}
    if not isinstance(scenario_params, dict):
        scenario_params = {}
    summary_revision = summary_source["source_revision"]
    episode_git_hash = source_row.get("git_hash")
    # The materializer defines planner_config_hash from algorithm_metadata.config_hash;
    # scenario_params.algo_config_hash is retained separately because the source can differ.
    episode_config_hash = metadata.get("config_hash")
    canonical_algorithm = canonical_algorithm_name(str(summary_case["planner_key"]))
    expected = {
        "campaign_source_revision": summary_revision,
        "row_git_hash": episode_git_hash,
        "episode_git_hash": summary_revision,
        "planner_config_hash": episode_config_hash,
        "materialized_planner_config_hash": episode_config_hash,
        "campaign_id": summary_source["source_campaign_id"],
        "bundle_sha256": summary_source["bundle_sha256"],
        "scenario_matrix": summary_source["matrix_path"],
        "scenario_matrix_sha256": summary_source["matrix_sha256"],
        "planner_key": summary_case["planner_key"],
        "episode_planner_key": summary_case["planner_key"],
        "materialized_planner_key": summary_case["planner_key"],
        "episode_canonical_algorithm": canonical_algorithm,
        "materialized_canonical_algorithm": canonical_algorithm,
    }
    actual = {
        "campaign_source_revision": source.get("campaign_source_revision"),
        "row_git_hash": source.get("row_git_hash"),
        "episode_git_hash": episode_git_hash,
        "episode_planner_config_hash": episode_config_hash,
        "planner_config_hash": source.get("planner_config_hash"),
        "materialized_planner_config_hash": planner_metadata.get("config_hash"),
        "campaign_id": source.get("campaign_id"),
        "bundle_sha256": source.get("bundle_sha256"),
        "scenario_matrix": source.get("scenario_matrix"),
        "scenario_matrix_sha256": source.get("scenario_matrix_sha256"),
        "planner_key": source.get("planner_key"),
        "episode_planner_key": source_row.get("algo"),
        "materialized_planner_key": planner.get("key"),
        "episode_canonical_algorithm": metadata.get("canonical_algorithm"),
        "materialized_canonical_algorithm": planner_metadata.get("canonical_algorithm"),
    }
    issues = sorted(field for field, value in expected.items() if actual[field] != value)
    for field in ("episode_git_hash", "episode_planner_config_hash"):
        value = actual[field]
        if not isinstance(value, str) or not value.strip():
            issues = sorted({*issues, field})
    return {
        "status": "verified" if not issues else "blocked",
        "issues": issues,
        "summary_source_revision": summary_revision,
        "episode_git_hash": episode_git_hash,
        "episode_planner_config_hash": episode_config_hash,
        "episode_scenario_algo_config_hash": scenario_params.get("algo_config_hash"),
        "canonical_algorithm": canonical_algorithm,
        "episode_canonical_algorithm": metadata.get("canonical_algorithm"),
        "materialized_canonical_algorithm": planner_metadata.get("canonical_algorithm"),
    }


def _validate_issue9656_setup_accounting(summary: Mapping[str, Any]) -> None:
    replay_budget = summary.get("replay_budget_accounting")
    failed_attempts = summary.get("failed_replay_attempts")
    if (
        not isinstance(replay_budget, dict)
        or not isinstance(failed_attempts, dict)
        or not isinstance(failed_attempts.get("attempts"), list)
        or not isinstance(replay_budget.get("failed_setup_jobs"), int)
        or isinstance(replay_budget.get("failed_setup_jobs"), bool)
        or replay_budget["failed_setup_jobs"] < 0
    ):
        raise CorpusError("#9656 failed replay setup accounting is absent or malformed")
    if any(
        not isinstance(attempt, dict)
        or not isinstance(attempt.get("return_code"), int)
        or isinstance(attempt.get("return_code"), bool)
        or attempt.get("return_code") == 0
        or attempt.get("replay_row_count") != 0
        for attempt in failed_attempts["attempts"]
    ):
        raise CorpusError("#9656 setup failures must remain explicit zero-row failed attempts")


def _safe_materialized_path(root: Path, relative: Any, *, label: str) -> Path:
    if not _safe_bundle_relative_path(relative):
        raise CorpusError(f"#9656 {label} path is unsafe")
    resolved = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CorpusError(f"#9656 {label} path escapes materialized root") from exc
    if not resolved.is_file():
        raise CorpusError(f"#9656 {label} is missing: {relative}")
    return resolved


def _issue9656_map_asset(
    matrix_path: Path, summary_case: Mapping[str, Any], campaign_source_revision: str
) -> dict[str, Any]:
    map_path, relative = _issue9656_map_path(matrix_path, summary_case)
    if not map_path.is_file():
        raise CorpusError(f"#9656 current map file is unavailable: {relative.as_posix()}")
    try:
        current_bytes = map_path.read_bytes()
    except OSError as exc:
        raise CorpusError(f"#9656 current map file is unavailable: {relative.as_posix()}") from exc
    historical_bytes = _issue9656_historical_git_blob(campaign_source_revision, relative.as_posix())
    if current_bytes != historical_bytes:
        raise CorpusError(f"#9656 map differs from campaign source revision: {relative.as_posix()}")
    return {
        "source_path": relative.as_posix(),
        "source_sha256": hashlib.sha256(historical_bytes).hexdigest(),
        "source_revision": campaign_source_revision,
        "_source_bytes": historical_bytes,
        "_repo_relative": relative.as_posix(),
    }


def _issue9656_map_path(matrix_path: Path, summary_case: Mapping[str, Any]) -> tuple[Path, Path]:
    try:
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CorpusError(f"#9656 replay matrix could not be parsed: {exc}") from exc
    scenarios = matrix.get("scenarios") if isinstance(matrix, dict) else None
    if not isinstance(scenarios, list) or len(scenarios) != 1 or not isinstance(scenarios[0], dict):
        raise CorpusError(
            f"#9656 replay matrix must contain one scenario for {summary_case['case_id']}"
        )
    scenario = scenarios[0]
    if (
        scenario.get("id") != summary_case.get("scenario_id")
        or scenario.get("seeds") != [summary_case.get("seed")]
        or scenario.get("algo") != summary_case.get("planner_key")
    ):
        raise CorpusError(f"#9656 replay matrix identity differs for {summary_case['case_id']}")
    map_value = scenario.get("map_file")
    if not isinstance(map_value, str) or not map_value.strip():
        raise CorpusError(f"#9656 replay matrix has no map file for {summary_case['case_id']}")
    map_path = (matrix_path.parent / map_value).resolve()
    repo_root = next(
        (ancestor for ancestor in map_path.parents if (ancestor / "maps" / "svg_maps").is_dir()),
        None,
    )
    if repo_root is None:
        raise CorpusError("#9656 materialized input is not below a repository containing maps")
    try:
        relative = map_path.relative_to(repo_root)
    except ValueError as exc:
        raise CorpusError(
            "#9656 candidate map path must resolve inside its source repository"
        ) from exc
    if (
        not relative.parts
        or relative.parts[0] != "maps"
        or not relative.as_posix().startswith("maps/svg_maps/")
    ):
        raise CorpusError("#9656 candidate map path must resolve to a repository map file")
    return map_path, relative


def _normalize_issue9656_replay_matrix(
    source_matrix: Path, candidate_dir: Path, map_info: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        matrix = yaml.safe_load(source_matrix.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CorpusError(f"#9656 source replay matrix could not be parsed: {exc}") from exc
    if not isinstance(matrix, dict) or not isinstance(matrix.get("scenarios"), list):
        raise CorpusError("#9656 source replay matrix must be an object with scenarios")
    scenario = matrix["scenarios"][0]
    repo_relative = str(map_info["_repo_relative"])
    source_map = map_info.get("_source_bytes")
    if not isinstance(source_map, bytes):
        raise CorpusError("#9656 historical map bytes are unavailable")
    map_output = candidate_dir / Path(*PurePosixPath(repo_relative).parts)
    map_output.parent.mkdir(parents=True, exist_ok=True)
    map_output.write_bytes(source_map)
    if _sha256_file(map_output) != map_info["source_sha256"]:
        raise CorpusError("#9656 copied historical map digest differs")
    scenario["map_file"] = (Path("..") / Path(*PurePosixPath(repo_relative).parts)).as_posix()
    matrix["map_search_paths"] = ["../maps/svg_maps"]
    map_receipt = [
        {
            "source_path": map_info["source_path"],
            "source_revision": map_info["source_revision"],
            "source_sha256": map_info["source_sha256"],
            "stored_path": f"historical_candidates/{candidate_dir.name}/{repo_relative}",
            "stored_sha256": _sha256_file(map_output),
        }
    ]
    return matrix, map_receipt


def _issue9656_candidate_import_record(
    summary: Mapping[str, Any],
    *,
    import_id: str,
    source_identity: Mapping[str, Any],
    summary_receipts: Mapping[str, Any],
    materialized_manifest_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    replay_counts = Counter(candidate["source_replay_status"] for candidate in candidates)
    candidate_status_counts = Counter(candidate["candidate_status"] for candidate in candidates)
    return {
        "schema_version": "adversarial-historical-candidate-import.v1",
        "import_id": import_id,
        "source_issue": 9656,
        "source_identity": dict(source_identity),
        "source_files": [
            *[
                {
                    "path": f"payload/{item['path']}",
                    "stored_path": f"historical_candidate_imports/{import_id}/payload/{item['path']}",
                    "sha256": item["sha256"],
                }
                for item in summary_receipts["payload_files"]
            ],
            {
                "path": "evidence_bundle_manifest.json",
                "stored_path": f"historical_candidate_imports/{import_id}/evidence_bundle_manifest.json",
                "sha256": summary_receipts["manifest_sha256"],
            },
            {
                "path": "checksums.sha256",
                "stored_path": f"historical_candidate_imports/{import_id}/checksums.sha256",
                "sha256": summary_receipts["checksums_sha256"],
            },
            {
                "path": "materialized_manifest.json",
                "stored_path": f"historical_candidate_imports/{import_id}/materialized_manifest.json",
                "sha256": materialized_manifest_sha256,
            },
        ],
        "candidate_ids": sorted(candidate["candidate_id"] for candidate in candidates),
        "candidate_count": len(candidates),
        "source_rows_verified": len(candidates),
        "candidate_status_counts": dict(sorted(candidate_status_counts.items())),
        "source_replay_status_counts": dict(sorted(replay_counts.items())),
        "criticality_anomaly_counts": dict(
            sorted((summary.get("criticality_anomaly_counts") or {}).items())
        ),
        "failed_replay_attempts": summary["failed_replay_attempts"],
        "failed_setup_jobs": summary["replay_budget_accounting"]["failed_setup_jobs"],
        "evidence_tier": summary.get("evidence_tier"),
        "claim_boundary": summary.get("claim_boundary"),
    }


def _copy_verified_source_file(source: Path, destination: Path, *, expected_sha256: str) -> None:
    if (
        not source.is_file()
        or not _is_sha256(expected_sha256)
        or _sha256_file(source) != expected_sha256
    ):
        raise CorpusError(f"source artifact digest differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256_file(destination) != expected_sha256:
        raise CorpusError(f"copied artifact digest differs: {destination}")


def _verify_existing_issue9656_candidate_artifacts(
    corpus_root: Path,
    import_record: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    if len(candidates) != import_record.get("candidate_count"):
        raise CorpusError("duplicate #9656 import is missing candidate registry rows")
    for source_file in import_record.get("source_files", []):
        if not isinstance(source_file, dict):
            raise CorpusError("duplicate #9656 import contains a malformed source receipt")
        _verify_corpus_artifact(
            corpus_root, source_file.get("stored_path"), source_file.get("sha256")
        )
    for candidate in candidates:
        paths = candidate.get("artifact_paths", {})
        provenance = candidate.get("source_provenance", {})
        inputs = candidate.get("replay_inputs", {})
        for field, digest in (
            ("source_case", provenance.get("source_case_file_sha256")),
            ("source_matrix", inputs.get("source_matrix_sha256")),
            ("source_planner_config", inputs.get("source_planner_config_sha256")),
            ("replay_matrix", inputs.get("normalized_matrix_sha256")),
            ("planner_config", inputs.get("normalized_planner_config_sha256")),
        ):
            _verify_corpus_artifact(corpus_root, paths.get(field), digest)
        for map_asset in inputs.get("map_assets", []):
            if not isinstance(map_asset, dict):
                raise CorpusError("duplicate #9656 import contains a malformed map receipt")
            _verify_corpus_artifact(
                corpus_root,
                map_asset.get("stored_path"),
                map_asset.get("stored_sha256"),
            )


def _verify_issue9656_duplicate_materialized_cases(
    source_rows: Sequence[tuple[Any, ...]], candidates: Sequence[Mapping[str, Any]]
) -> None:
    candidates_by_alias = {candidate["source_case_id"]: candidate for candidate in candidates}
    if len(candidates_by_alias) != len(source_rows):
        raise CorpusError("duplicate #9656 import has missing materialized-case receipts")
    for source_row in source_rows:
        source_case, _materialized_case, case_path, *_artifacts = source_row
        candidate = candidates_by_alias.get(source_case["case_id"])
        stored_sha256 = (
            candidate.get("source_provenance", {}).get("source_case_file_sha256")
            if candidate is not None
            else None
        )
        if not _is_sha256(stored_sha256) or _sha256_file(case_path) != stored_sha256:
            raise CorpusError(
                "duplicate #9656 import materialized case differs from its verified source"
            )


def _verify_corpus_artifact(root: Path, relative: Any, expected_sha256: Any) -> None:
    if not _safe_bundle_relative_path(relative) or not _is_sha256(expected_sha256):
        raise CorpusError("stored #9656 artifact receipt contains an unsafe path or digest")
    source = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise CorpusError("stored #9656 artifact path escapes corpus root") from exc
    if not source.is_file() or _sha256_file(source) != expected_sha256:
        raise CorpusError(f"stored #9656 artifact digest differs: {relative}")


def _promote_issue9656_candidate_artifacts(
    staging: Path,
    root: Path,
    *,
    import_id: str,
    candidate_ids: Sequence[str],
    promoted_dirs: list[Path],
) -> None:
    source_import = staging / "historical_candidate_imports" / import_id
    source_candidates = staging / "historical_candidates"
    final_import = root / "historical_candidate_imports" / import_id
    final_candidates_root = root / "historical_candidates"
    final_import.parent.mkdir(parents=True, exist_ok=True)
    final_candidates_root.mkdir(parents=True, exist_ok=True)
    destinations = [
        (source_candidates / candidate_id, final_candidates_root / candidate_id)
        for candidate_id in candidate_ids
    ]
    destinations.append((source_import, final_import))
    if any(destination.exists() for _source, destination in destinations):
        raise CorpusError(
            "#9656 candidate artifact destination already exists without its import receipt"
        )
    try:
        for source, destination in destinations:
            source.replace(destination)
            promoted_dirs.append(destination)
    except BaseException:
        for promoted in reversed(promoted_dirs):
            shutil.rmtree(promoted, ignore_errors=True)
        promoted_dirs.clear()
        raise


def create_planner_replay_receipt(
    observation: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    artifact_path: str,
    corpus_root: str | Path,
) -> dict[str, Any]:
    """Build a receipt only when one stored episode artifact matches the observation."""
    item = dict(observation)
    root = Path(corpus_root).resolve()
    artifact = _resolve_corpus_artifact(artifact_path, root)
    episode_sha256 = _sha256_file(artifact)
    if item.get("episode_sha256") is None:
        item["episode_sha256"] = episode_sha256
    record = _read_single_jsonl_record(artifact)
    inputs = case.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    episode_input_identity = provenance.get("case_input_identity")
    if isinstance(episode_input_identity, Mapping):
        input_binding = dict(episode_input_identity)
        run_id = input_binding.get("run_id")
    elif item.get("input_binding_status") == "unknown_historical":
        input_binding = {
            "schema_version": EPISODE_INPUT_IDENTITY_SCHEMA,
            "status": "unknown_historical",
            "run_id": item.get("run_id"),
            "reason": "stored episode predates direct scenario/route/map input binding",
        }
        run_id = item.get("run_id")
    else:
        raise CorpusError("replay receipt rejected: episode input identity is unavailable")
    receipt = {
        "schema_version": EVALUATION_REPLAY_RECEIPT_SCHEMA_VERSION,
        "verification_status": "artifact_projection_match",
        "run_id": run_id,
        "input_binding": input_binding,
        "artifact_path": PurePosixPath(artifact_path).as_posix(),
        "artifact_sha256": episode_sha256,
        "case_id": item.get("case_id"),
        "effective_scenario_sha256": item.get("effective_scenario_sha256"),
        "scenario_id": case.get("scenario_id"),
        "scenario_seed": case.get("scenario_seed"),
        "scenario_input_sha256": inputs.get("scenario_sha256"),
        "route_overrides_sha256": inputs.get("route_overrides_sha256"),
        "map_assets": _map_asset_identity(inputs.get("map_assets")),
        "planner_id": item.get("planner_id"),
        "planner_config_identity": item.get("planner_config_identity"),
        "source_revision": item.get("source_revision"),
        "episode_sha256": episode_sha256,
        "outcome": item.get("outcome"),
        "termination_reason": item.get("termination_reason"),
        "metrics": item.get("metrics"),
        "selected_event_identity": _selected_event_identity(record),
        "execution_mode": item.get("execution_mode"),
        "readiness_status": item.get("readiness_status"),
        "availability_status": item.get("availability_status"),
        "fallback_or_degraded": item.get("fallback_or_degraded"),
    }
    errors = _validate_replay_artifact(item, case, receipt, root)
    if errors:
        raise CorpusError("replay receipt rejected: " + "; ".join(errors))
    return receipt


def create_case_admission_replay_receipt(
    observation: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    artifact_path: str,
    corpus_root: str | Path,
    target_revision: str | None = None,
) -> dict[str, Any]:
    """Create one exact-revision admission receipt from a stored replay artifact.

    The artifact must already be a one-row episode JSONL under ``corpus_root``. This
    validates and records existing evidence; it does not execute a simulator. The
    target is resolved independently from this module's checkout HEAD. An optional
    ``target_revision`` is only an expected-value check and cannot override that HEAD.
    """
    revision = _current_target_revision()
    if revision is None:
        raise CorpusError(
            "admission replay rejected: independent current target revision is unavailable"
        )
    if target_revision is not None and target_revision != revision:
        raise CorpusError(
            "admission replay target revision differs from the independently resolved current target"
        )
    if observation.get("source_revision") != revision:
        raise CorpusError(
            "admission replay revision must match the independently resolved current target"
        )
    evaluation_receipt = create_planner_replay_receipt(
        observation,
        case,
        artifact_path=artifact_path,
        corpus_root=corpus_root,
    )
    if evaluation_receipt.get("input_binding", {}).get("status") != "bound":
        raise CorpusError("admission replay rejected: direct case input binding is unavailable")
    projection = {
        "scenario_id": case.get("scenario_id"),
        "seed": case.get("scenario_seed"),
        "planner_id": observation.get("planner_id"),
        "planner_config_identity": observation.get("planner_config_identity"),
        "source_revision": observation.get("source_revision"),
        "outcome": observation.get("outcome"),
        "termination_reason": observation.get("termination_reason"),
        "metrics": observation.get("metrics"),
        "selected_event_identity": evaluation_receipt["selected_event_identity"],
    }
    inputs = case.get("inputs", {})
    inputs = inputs if isinstance(inputs, dict) else {}
    return {
        "schema_version": CASE_ADMISSION_REPLAY_SCHEMA_VERSION,
        "verification_status": "exact_current_revision_match",
        "target_revision": revision,
        "replay_revision": revision,
        "target_and_replay_revision_match": True,
        "replay_count": 1,
        "selected_projection": projection,
        "selected_projection_sha256": hashlib.sha256(
            _stable_json(projection).encode("utf-8")
        ).hexdigest(),
        "replay_artifacts": [
            {
                "path": PurePosixPath(artifact_path).as_posix(),
                "sha256": evaluation_receipt["artifact_sha256"],
                "run_id": evaluation_receipt["run_id"],
                "selected_event_identity": evaluation_receipt["selected_event_identity"],
            }
        ],
        "artifact_receipts": [evaluation_receipt],
        "input_binding_status": "bound",
        "effective_scenario_sha256": case.get("effective_scenario_sha256"),
        "scenario_input_sha256": inputs.get("scenario_sha256"),
        "route_overrides_sha256": inputs.get("route_overrides_sha256"),
        "map_assets": _map_asset_identity(inputs.get("map_assets")),
        "planner_id": observation.get("planner_id"),
        "planner_config_identity": observation.get("planner_config_identity"),
    }


def append_planner_evaluation(
    corpus: dict[str, Any],
    observation: Mapping[str, Any],
    *,
    corpus_root: str | Path | None = None,
) -> dict[str, Any]:
    """Append a planner result only when its case, planner, and input binding are explicit."""
    validate_corpus(
        corpus,
        corpus_root=Path(corpus_root).resolve() if corpus_root is not None else None,
    )
    item = dict(observation)
    if item.get("evidence_status") == "complete" and not isinstance(
        item.get("replay_receipt"), dict
    ):
        raise CorpusError("planner evaluation rejected: replay_receipt_missing")
    errors = _validate_evaluation(item)
    if errors:
        raise CorpusError("planner evaluation rejected: " + "; ".join(errors))
    case = next((entry for entry in corpus["cases"] if entry["case_id"] == item["case_id"]), None)
    if case is None:
        raise CorpusError(f"unknown case_id: {item['case_id']}")
    if item["effective_scenario_sha256"] != case["effective_scenario_sha256"]:
        raise CorpusError("planner evaluation input hash does not match the corpus case")
    if item.get("evidence_status") == "complete":
        if corpus_root is None:
            raise CorpusError("complete planner evaluations require a replay artifact corpus_root")
        replay_errors = _validate_replay_artifact(
            item, case, item["replay_receipt"], Path(corpus_root).resolve()
        )
        if replay_errors:
            raise CorpusError(
                "planner evaluation replay evidence rejected: " + "; ".join(replay_errors)
            )
    item["schema_version"] = EVALUATION_SCHEMA_VERSION
    item["evaluation_id"] = _evaluation_digest(item)
    corpus["planner_evaluations"] = _append_unique(
        corpus, "planner_evaluations", item, key="evaluation_id"
    )["planner_evaluations"]
    validate_corpus(
        corpus,
        corpus_root=Path(corpus_root).resolve() if corpus_root is not None else None,
    )
    return corpus


def recompute_planner_status(
    corpus: Mapping[str, Any],
    *,
    planner_id: str,
    planner_config_identity: str,
    corpus_root: str | Path | None = None,
) -> dict[str, Any]:
    """Derive solved/unsolved/mixed/unknown status for every case and planner config."""
    validate_corpus(
        corpus,
        corpus_root=Path(corpus_root).resolve() if corpus_root is not None else None,
    )
    if not planner_id.strip() or not planner_config_identity.strip():
        raise CorpusError("planner_id and planner_config_identity must be non-empty")
    result = []
    for case in sorted(corpus["cases"], key=lambda entry: entry["case_id"]):
        rows = [
            item
            for item in corpus["planner_evaluations"]
            if item["case_id"] == case["case_id"]
            and item["planner_id"] == planner_id
            and item["planner_config_identity"] == planner_config_identity
        ]
        states = [
            _evaluation_state(
                row,
                case=case,
                corpus_root=Path(corpus_root).resolve() if corpus_root is not None else None,
            )
            for row in rows
        ]
        valid_states = [state for state in states if state["status"] in {"solved", "unsolved"}]
        invalid_rows = [state for state in states if state["status"] == "unknown"]
        distinct = {state["status"] for state in valid_states}
        if invalid_rows:
            status = "unknown"
        elif len(distinct) > 1:
            status = "mixed"
        elif distinct:
            status = next(iter(distinct))
        else:
            status = "unknown"
        result.append(
            {
                "case_id": case["case_id"],
                "effective_scenario_sha256": case["effective_scenario_sha256"],
                "planner_id": planner_id,
                "planner_config_identity": planner_config_identity,
                "status": status,
                "observation_count": len(rows),
                "valid_observation_count": len(valid_states),
                "unknown_observation_count": len(invalid_rows),
                "reason_codes": sorted(
                    {reason for state in states for reason in state["reason_codes"]}
                ),
            }
        )
    return {
        "schema_version": "adversarial-counterexample-status-report.v1",
        "planner_id": planner_id,
        "planner_config_identity": planner_config_identity,
        "case_count": len(result),
        "status_counts": {
            name: sum(row["status"] == name for row in result)
            for name in ("solved", "unsolved", "mixed", "unknown")
        },
        "cases": result,
    }


def export_regression_slice(
    corpus: Mapping[str, Any], *, corpus_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Write a deterministic, self-contained replay slice for all admitted cases."""
    validate_corpus(corpus, corpus_root=Path(corpus_root).resolve())
    base = Path(corpus_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise CorpusError(f"slice output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    cases = sorted(corpus["cases"], key=lambda entry: entry["case_id"])
    matrix_entries: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    try:
        routes_dir = staging / "routes"
        configs_dir = staging / "planner_configs"
        maps_dir = staging / "maps"
        results_dir = staging / "results"
        routes_dir.mkdir()
        configs_dir.mkdir()
        maps_dir.mkdir()
        results_dir.mkdir()
        for case in cases:
            scenario, manifest_case = _export_case_slice(case, base, staging)
            matrix_entries.append(scenario)
            manifest_cases.append(manifest_case)

        matrix_path = staging / "replay_matrix.yaml"
        matrix_path.write_text(
            yaml.safe_dump({"scenarios": matrix_entries}, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        _validate_exported_slice(matrix_path, manifest_cases, staging)
        manifest = {
            "schema_version": SLICE_SCHEMA_VERSION,
            "source_corpus_schema_version": corpus["schema_version"],
            "claim_boundary": (
                "replay inputs only; execution must be evaluated under the declared planner, "
                "configuration, simulator revision, and benchmark profile"
            ),
            "case_count": len(manifest_cases),
            "replay_matrix_path": "replay_matrix.yaml",
            "replay_matrix_sha256": _sha256_file(matrix_path),
            "cases": manifest_cases,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _export_case_slice(
    case: Mapping[str, Any], base: Path, staging: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_input = _case_input_paths(case, base)
    scenario_path = case_input["scenario"]
    route_path = case_input["route"]
    scenario_document = _load_yaml_object(scenario_path, "scenario")
    scenarios = scenario_document.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 1 or not isinstance(scenarios[0], dict):
        raise CorpusError(f"{case['case_id']}: scenario input must contain one scenario")
    source_scenario = dict(scenarios[0])
    route_payload = _load_yaml_object(route_path, "route overrides")
    route_relative = f"routes/{case['case_id']}.yaml"
    config_relative = f"planner_configs/{case['case_id']}.yaml"
    exported_assets = _export_case_map_assets(case, case_input, staging)
    exported_map_relative = next(
        (item["path"] for item in exported_assets if item["role"] == "map"), None
    )
    scenario, identity_mapping = _map_slice_scenario_identity(
        case,
        source_scenario,
        route_payload,
        route_relative,
        case["inputs"]["map_assets"],
        exported_map_relative,
    )
    route_output = staging / route_relative
    shutil.copyfile(route_path, route_output)
    config_output = _write_slice_planner_config(case, staging, config_relative)
    command_parts = _slice_replay_command(case, case_input, config_relative)
    manifest_case = {
        "case_id": case["case_id"],
        "scenario_id": case["scenario_id"],
        "seed": case["scenario_seed"],
        "effective_scenario_sha256": case["effective_scenario_sha256"],
        "replay_input_binding_status": _case_replay_input_binding_status(case),
        "identity_mapping": identity_mapping,
        "planner": case["target_planner"],
        "scenario_source_sha256": _sha256_file(scenario_path),
        "route_overrides_sha256": _sha256_file(route_output),
        "map_assets": [
            {"role": item["role"], "path": item["path"], "sha256": item["sha256"]}
            for item in exported_assets
        ],
        "planner_config_path": config_relative,
        "planner_config_sha256": _sha256_file(config_output),
        "replay_command": " ".join(command_parts),
    }
    return scenario, manifest_case


def _export_case_map_assets(
    case: Mapping[str, Any], case_input: Mapping[str, Path], staging: Path
) -> list[dict[str, Any]]:
    exported = []
    for asset in case["inputs"]["map_assets"]:
        source_asset = case_input[f"{asset['role']}_asset"]
        asset_relative = _slice_map_asset_path(asset)
        destination = staging / Path(*PurePosixPath(asset_relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_asset, destination)
        if _sha256_file(destination) != asset["sha256"]:
            raise CorpusError(f"{case['case_id']}: exported map asset checksum differs")
        exported.append({**asset, "path": asset_relative})
    return exported


def _write_slice_planner_config(
    case: Mapping[str, Any], staging: Path, config_relative: str
) -> Path:
    snapshot = case["target_planner"].get("configuration_snapshot")
    if not isinstance(snapshot, dict):
        raise CorpusError(f"{case['case_id']}: planner configuration snapshot is missing")
    config_path = staging / config_relative
    config_path.write_text(
        yaml.safe_dump(snapshot, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    return config_path


def _slice_replay_command(
    case: Mapping[str, Any], case_input: Mapping[str, Path], config_relative: str
) -> list[str]:
    return [
        *(
            ["ROBOT_SF_MAP_REGISTRY=maps/registry.yaml"]
            if "map_registry_asset" in case_input
            else []
        ),
        "uv run robot_sf_bench run",
        "--matrix replay_matrix.yaml",
        f"--out results/{case['case_id']}.jsonl",
        f"--algo {shlex.quote(case['target_planner']['planner_id'])}",
        f"--algo-config {shlex.quote(config_relative)}",
        f"--scenario-id {shlex.quote(case['scenario_id'])}",
        "--no-video",
    ]


def _validate_exported_slice(
    matrix_path: Path, manifest_cases: list[dict[str, Any]], staging: Path
) -> None:
    exported_matrix = _load_yaml_object(matrix_path, "exported replay matrix")
    exported_rows = exported_matrix.get("scenarios")
    if not isinstance(exported_rows, list) or len(exported_rows) != len(manifest_cases):
        raise CorpusError("exported replay matrix does not preserve the case rows")
    for scenario, manifest_case in zip(exported_rows, manifest_cases, strict=True):
        route_doc = _load_yaml_object(
            staging / str(scenario.get("route_overrides_file")), "exported route overrides"
        )
        mapping = manifest_case["identity_mapping"]
        actual_hash = compute_case_effective_scenario_hash(
            scenario, route_doc, mapping["map_assets"]
        )
        receipts = [
            {
                "role": item["role"],
                "path": item["path"],
                "source_path": item["source_path"],
                "sha256": item["sha256"],
            }
            for item in mapping["map_assets"]
        ]
        _validate_case_map_assets(scenario, matrix_path, receipts, staging)
        if actual_hash != mapping["exported_effective_scenario_sha256"]:
            raise CorpusError(
                f"{manifest_case['case_id']}: exported scenario identity mapping is invalid"
            )


def _map_slice_scenario_identity(
    case: Mapping[str, Any],
    source_scenario: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    route_relative: str,
    map_assets: Sequence[Mapping[str, Any]],
    exported_map_relative: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_hash = compute_case_effective_scenario_hash(source_scenario, route_payload, map_assets)
    if source_hash != case["effective_scenario_sha256"]:
        raise CorpusError(f"{case['case_id']}: source scenario identity changed")
    scenario = dict(source_scenario)
    source_route = scenario.get("route_overrides_file")
    source_map_file = scenario.get("map_file")
    scenario["route_overrides_file"] = route_relative
    scenario["seeds"] = [int(case["scenario_seed"])]
    if not scenario.get("map_id"):
        if not exported_map_relative:
            raise CorpusError(f"{case['case_id']}: exported map file path is missing")
        scenario["map_file"] = exported_map_relative
    exported_hash = compute_case_effective_scenario_hash(scenario, route_payload, map_assets)
    mapping = {
        "schema_version": "adversarial-slice-identity-mapping.v1",
        "verification_status": "source_and_export_hashes_verified",
        "normalization_fields": [
            field
            for field, before, after in (
                ("route_overrides_file", source_route, route_relative),
                ("map_file", source_map_file, scenario.get("map_file")),
                ("seeds", source_scenario.get("seeds"), scenario.get("seeds")),
            )
            if before != after
        ],
        "source_effective_scenario_sha256": source_hash,
        "exported_effective_scenario_sha256": exported_hash,
        "source_route_overrides_file": source_route,
        "exported_route_overrides_file": route_relative,
        "source_map_file": source_map_file,
        "exported_map_file": scenario.get("map_file"),
        "map_assets": [{**asset, "path": _slice_map_asset_path(asset)} for asset in map_assets],
    }
    return scenario, mapping


def _slice_map_asset_path(asset: Mapping[str, Any]) -> str:
    role = asset.get("role")
    source_path = asset.get("source_path")
    if role == "map_registry":
        return "maps/registry.yaml"
    if role != "map" or not isinstance(source_path, str):
        raise CorpusError("slice map asset role or source path is invalid")
    source_relative = PurePosixPath(source_path)
    if source_relative.is_absolute() or ".." in source_relative.parts:
        raise CorpusError("slice map asset source path is unsafe")
    parts = (
        source_relative.parts[1:]
        if source_relative.parts[:1] == ("maps",)
        else source_relative.parts
    )
    return PurePosixPath("maps").joinpath(*parts).as_posix()


def _verify_issue9645_pilot(payload: Path) -> dict[str, Any]:
    summary = _read_json_object(payload / "summary.json")
    metadata = _read_json_object(payload / "run_metadata.json")
    row_status = _read_json_object(payload / "row_status.json")
    bundle_receipts = _verify_issue9645_bundle(
        payload,
        source_revision=str(metadata.get("experiment_source_commit") or ""),
        source_file_hashes=metadata.get("source_file_sha256"),
        required_payload_paths=_pilot_accounting_paths(payload),
    )
    _validate_pilot_summary(summary, metadata)
    _validate_pilot_design(metadata)
    manifests_by_identity = _verify_pilot_manifests(payload, metadata)
    _verify_pilot_candidate_rows(payload, row_status)
    return _pilot_search_run(summary, metadata, manifests_by_identity, payload, bundle_receipts)


def _pilot_accounting_paths(payload: Path) -> list[str]:
    source_manifests = sorted((payload / "source_manifests").glob("*.json"))
    return [
        "summary.json",
        "run_metadata.json",
        "row_status.json",
        "candidate_evaluations.csv",
        "source_hashes.sha256",
        "report_provenance.json",
        "inputs/crossing_ttc.yaml",
        "inputs/issue_9645_pilot_space.v1.yaml",
        *[path.relative_to(payload).as_posix() for path in source_manifests],
    ]


def _verify_issue9645_bundle(
    payload: Path,
    *,
    source_revision: str,
    source_file_hashes: Any,
    required_payload_paths: Sequence[str],
) -> list[dict[str, str]]:
    """Verify the bundle's outer receipt and hashes for every consumed payload file."""
    bundle_root = payload.parent
    manifest_path, checksums_path, files_by_path = _read_issue9645_bundle_inventory(
        bundle_root, source_revision
    )
    _verify_issue9645_payload_receipts(
        payload,
        source_revision=source_revision,
        source_file_hashes=source_file_hashes,
        required_payload_paths=required_payload_paths,
        files_by_path=files_by_path,
    )
    return [
        {
            "source_path": "evidence_bundle_manifest.json",
            "path": "packet_receipts/evidence_bundle_manifest.json",
            "sha256": _sha256_file(manifest_path),
        },
        {
            "source_path": "checksums.sha256",
            "path": "packet_receipts/checksums.sha256",
            "sha256": _sha256_file(checksums_path),
        },
    ]


def _read_issue9645_bundle_inventory(
    bundle_root: Path, source_revision: str
) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    manifest_path = bundle_root / "evidence_bundle_manifest.json"
    checksums_path = bundle_root / "checksums.sha256"
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != ISSUE_9645_BUNDLE_SCHEMA:
        raise CorpusError("#9645 evidence bundle manifest schema is invalid")
    if (
        manifest.get("bundle_name") != "issue_9645_bounded_falsification_2026-09-24"
        or manifest.get("commit") != source_revision
    ):
        raise CorpusError("#9645 evidence bundle identity differs from the source run")
    bundle_files = manifest.get("files")
    totals = manifest.get("totals")
    if not isinstance(bundle_files, list) or not isinstance(totals, dict):
        raise CorpusError("#9645 evidence bundle file inventory is incomplete")
    files_by_path = _validated_bundle_file_map(bundle_files)
    total_bytes = sum(item["size_bytes"] for item in files_by_path.values())
    if totals.get("file_count") != len(files_by_path) or totals.get("total_bytes") != total_bytes:
        raise CorpusError("#9645 evidence bundle totals disagree with its file inventory")

    checksums = _parse_sha256_receipt(checksums_path, label="bundle checksums")
    expected_checksums = {
        f"payload/{relative}": item["sha256"] for relative, item in files_by_path.items()
    }
    if checksums != expected_checksums:
        raise CorpusError("#9645 bundle checksum file disagrees with its manifest")
    return manifest_path, checksums_path, files_by_path


def _verify_issue9645_payload_receipts(
    payload: Path,
    *,
    source_revision: str,
    source_file_hashes: Any,
    required_payload_paths: Sequence[str],
    files_by_path: Mapping[str, Mapping[str, Any]],
) -> None:
    source_hashes = _parse_sha256_receipt(payload / "source_hashes.sha256", label="source hashes")
    if not isinstance(source_file_hashes, dict) or source_hashes != source_file_hashes:
        raise CorpusError("#9645 source hashes disagree with run metadata")
    for relative in required_payload_paths:
        record = files_by_path.get(relative)
        if record is None:
            raise CorpusError(f"#9645 consumed file is absent from the bundle manifest: {relative}")
        _verify_bundle_payload_file(payload, relative, record)

    report = _read_json_object(payload / "report_provenance.json")
    if (
        report.get("schema_version") != "issue_9645_report_build_provenance.v1"
        or report.get("source_search_commit") != source_revision
    ):
        raise CorpusError("#9645 report provenance does not bind the source search revision")
    comparison_record = files_by_path.get("pilot_comparison.json")
    if comparison_record is None:
        raise CorpusError("#9645 pilot comparison is absent from the bundle manifest")
    _verify_bundle_payload_file(payload, "pilot_comparison.json", comparison_record)
    if report.get("comparison_sha256") != comparison_record["sha256"]:
        raise CorpusError("#9645 report provenance does not bind the pilot comparison")


def _validated_bundle_file_map(entries: list[Any]) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CorpusError("#9645 bundle file inventory contains a malformed entry")
        relative = entry.get("path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if (
            not _safe_bundle_relative_path(relative)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not _is_sha256(digest)
            or relative in files
        ):
            raise CorpusError("#9645 bundle file inventory contains invalid identity or checksum")
        files[relative] = {"size_bytes": size, "sha256": digest}
    if not files:
        raise CorpusError("#9645 bundle file inventory is empty")
    return files


def _safe_bundle_relative_path(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        return False
    path = PurePosixPath(value)
    return (
        value not in {"", "."}
        and not path.is_absolute()
        and path.as_posix() == value
        and ".." not in path.parts
    )


def _parse_sha256_receipt(path: Path, *, label: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CorpusError(f"could not read #9645 {label}: {exc}") from exc
    records: dict[str, str] = {}
    for line in lines:
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not _is_sha256(fields[0]):
            raise CorpusError(f"#9645 {label} contains a malformed checksum line")
        relative = fields[1].strip()
        if not _safe_bundle_relative_path(relative) or relative in records:
            raise CorpusError(f"#9645 {label} contains an unsafe or duplicate path")
        records[relative] = fields[0]
    if not records:
        raise CorpusError(f"#9645 {label} is empty")
    return records


def _verify_bundle_payload_file(payload: Path, relative: str, record: Mapping[str, Any]) -> None:
    if not _safe_bundle_relative_path(relative):
        raise CorpusError("#9645 consumed payload path is unsafe")
    source = (payload / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        source.relative_to(payload.resolve())
    except ValueError as exc:
        raise CorpusError("#9645 consumed payload path escapes the packet") from exc
    if (
        not source.is_file()
        or source.stat().st_size != record.get("size_bytes")
        or _sha256_file(source) != record.get("sha256")
    ):
        raise CorpusError(f"#9645 consumed payload checksum differs: {relative}")


def _validate_pilot_summary(summary: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != ISSUE_9645_SUMMARY_SCHEMA:
        raise CorpusError("unsupported #9645 summary schema")
    if metadata.get("schema_version") != "issue_9645_execution_provenance.v1":
        raise CorpusError("unsupported #9645 execution provenance schema")
    if summary.get("source_revision") != metadata.get("experiment_source_commit"):
        raise CorpusError("#9645 summary and run metadata revisions differ")
    outcomes = summary.get("candidate_outcomes")
    budget = summary.get("pilot_budget")
    if not isinstance(outcomes, dict) or outcomes.get("new_counterexamples_discovered") != 0:
        raise CorpusError("#9645 must explicitly report zero newly discovered counterexamples")
    if outcomes.get("new_counterexamples_admitted_to_regression_corpus") != 0:
        raise CorpusError("#9645 must explicitly report zero newly admitted pilot cases")
    if (
        not isinstance(budget, dict)
        or budget.get("attempted") != 64
        or budget.get("completed") != 64
    ):
        raise CorpusError("#9645 bounded pilot accounting must be 64 attempted and completed")
    if budget.get("failed") != 0 or budget.get("invalid") != 0:
        raise CorpusError("#9645 pilot failed/invalid accounting differs from its durable receipt")
    if summary.get("evaluation_budget_consumed", {}).get("pilot_search_candidates") != 64:
        raise CorpusError("#9645 pilot candidate budget is not bound")


def _validate_pilot_design(metadata: Mapping[str, Any]) -> None:
    if (
        metadata.get("issue") != 9645
        or metadata.get("objective", {}).get("name") != "constraints_first_lexicographic_v1"
    ):
        raise CorpusError("#9645 issue/objective identity is incomplete")
    if metadata.get("pilot_design", {}).get("fixed_scenario_seed") != 123:
        raise CorpusError("#9645 fixed environment seed is missing")
    expected_seeds = {1101, 2202}
    if set(metadata.get("pilot_design", {}).get("sampler_rng_seeds", [])) != expected_seeds:
        raise CorpusError("#9645 sampler seeds do not match its bounded design")
    if set(metadata.get("pilot_design", {}).get("sampler_methods", [])) != {"random", "optuna_tpe"}:
        raise CorpusError("#9645 sampler methods do not match its bounded design")


def _verify_pilot_manifests(
    payload: Path, metadata: Mapping[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    manifest_receipts = metadata.get("manifest_files")
    if not isinstance(manifest_receipts, list) or len(manifest_receipts) != 4:
        raise CorpusError("#9645 must bind all four per-seed/per-sampler manifests")
    manifests_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    source_manifest_paths = sorted((payload / "source_manifests").glob("*.json"))
    if len(source_manifest_paths) != 4:
        raise CorpusError("#9645 packet must contain its four exact search manifests")
    for manifest_path in source_manifest_paths:
        identity, manifest_record = _verify_one_pilot_manifest(manifest_path, manifest_receipts)
        manifests_by_identity[identity] = manifest_record
    if len(manifests_by_identity) != 4:
        raise CorpusError("#9645 manifest sampler/seed identities are duplicated")
    return manifests_by_identity


def _verify_one_pilot_manifest(
    manifest_path: Path, manifest_receipts: Sequence[Mapping[str, Any]]
) -> tuple[tuple[str, int], dict[str, Any]]:
    actual_hash = _sha256_file(manifest_path)
    matches = [entry for entry in manifest_receipts if entry.get("sha256") == actual_hash]
    if len(matches) != 1:
        raise CorpusError(f"#9645 search manifest hash is not bound: {manifest_path.name}")
    manifest = _read_json_object(manifest_path)
    config = manifest.get("config")
    manifest_summary = manifest.get("summary")
    if manifest.get("schema_version") != SEARCH_MANIFEST_SCHEMA or not isinstance(config, dict):
        raise CorpusError(f"#9645 manifest schema/config is invalid: {manifest_path.name}")
    sampler = (
        "optuna_tpe" if "tpe" in manifest_path.name or "optuna" in manifest_path.name else "random"
    )
    seed = int(config.get("seed", -1))
    if (
        config.get("objective") != "constraints_first_lexicographic_v1"
        or config.get("policy") != "goal"
        or config.get("budget") != 16
        or seed not in {1101, 2202}
    ):
        raise CorpusError(f"#9645 manifest search settings differ: {manifest_path.name}")
    if not isinstance(manifest_summary, dict) or any(
        manifest_summary.get(key) != expected
        for key, expected in (
            ("num_candidates", 16),
            ("num_failed_evaluations", 0),
            ("num_invalid_candidates", 0),
        )
    ):
        raise CorpusError(f"#9645 manifest accounting differs: {manifest_path.name}")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 16:
        raise CorpusError(f"#9645 manifest candidate rows are incomplete: {manifest_path.name}")
    if any(not _pilot_candidate_is_success(candidate) for candidate in candidates):
        raise CorpusError(f"#9645 manifest contains a critical or failed row: {manifest_path.name}")
    return (sampler, seed), {
        "path": f"source_manifests/{manifest_path.name}",
        "sha256": actual_hash,
        "candidate_count": len(candidates),
    }


def _pilot_candidate_is_success(candidate: Any) -> bool:
    return (
        isinstance(candidate, dict)
        and not candidate.get("error")
        and candidate.get("objective_value") == 0.0
        and isinstance(candidate.get("failure_attribution"), dict)
        and candidate["failure_attribution"].get("primary_failure") == "success"
    )


def _verify_pilot_candidate_rows(payload: Path, row_status: Mapping[str, Any]) -> None:
    with (payload / "candidate_evaluations.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise CorpusError("#9645 candidate accounting table must contain all 64 rows")
    for row in rows:
        if (
            row.get("row_status") != "successful_evidence"
            or row.get("counts_as_execution_success_evidence") != "True"
            or row.get("execution_mode") != "native"
            or row.get("readiness_status") != "native"
            or row.get("availability_status") != "available"
            or row.get("collision_event") != "False"
            or row.get("route_complete") != "True"
            or row.get("timeout_event") != "False"
            or row.get("near_miss_count") != "0"
            or row.get("objective_value") != "0.0"
        ):
            raise CorpusError("#9645 candidate table has an incomplete or critical row")
    if row_status.get("counts_as_success_evidence_total") != 64:
        raise CorpusError("#9645 row-status receipt disagrees with the candidate table")
    if row_status.get("evidence_tier") != "diagnostic_only":
        raise CorpusError("#9645 diagnostic evidence tier is missing")


def _pilot_search_run(
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    manifests_by_identity: Mapping[tuple[str, int], Mapping[str, Any]],
    payload: Path,
    bundle_receipts: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": "adversarial-counterexample-search-run.v1",
        "run_id": "issue_9645_bounded_pilot",
        "source_issue": 9645,
        "source_revision": summary["source_revision"],
        "objective": metadata["objective"],
        "search_space": {
            "path": metadata.get("effective_search_space_path"),
            "sha256": metadata.get("effective_search_space_sha256"),
        },
        "planner": summary.get("planner"),
        "fixed_environment_seed": 123,
        "sampler_seeds": [1101, 2202],
        "sampler_methods": ["random", "optuna_tpe"],
        "budget_per_method_seed": 16,
        "attempted_candidates": 64,
        "completed_candidates": 64,
        "failed_candidates": 0,
        "invalid_candidates": 0,
        "new_counterexamples_discovered": 0,
        "new_counterexamples_admitted": 0,
        "all_objective_values": [0.0],
        "terminal_decision": summary.get("terminal_decision"),
        "evidence_tier": "diagnostic_only",
        "source_files": _copy_pilot_evidence(payload),
        "manifest_files": [
            manifests_by_identity[key]
            for key in sorted(manifests_by_identity, key=lambda item: (item[1], item[0]))
        ],
        "bundle_receipts": bundle_receipts,
    }


def _copy_pilot_evidence(payload: Path) -> list[dict[str, str]]:
    """Return exact checksums for the minimal #9645 pilot accounting packet."""
    relative_paths = [
        "summary.json",
        "run_metadata.json",
        "candidate_evaluations.csv",
        "row_status.json",
        "source_hashes.sha256",
        "report_provenance.json",
        "pilot_comparison.json",
        "inputs/crossing_ttc.yaml",
        "inputs/issue_9645_pilot_space.v1.yaml",
        "source_manifests/random_seed_1101.json",
        "source_manifests/random_seed_2202.json",
        "source_manifests/tpe_seed_1101.json",
        "source_manifests/tpe_seed_2202.json",
    ]
    # The historical case subdirectory is copied with the admitted case below; run-level
    # accounting stays in the evidence inventory and keeps the explicit zero result.
    return [{"path": name, "sha256": _sha256_file(payload / name)} for name in relative_paths]


def _persist_pilot_evidence(payload: Path, corpus_root: Path, pilot: dict[str, Any]) -> None:
    """Copy the accounting rows/manifests into corpus custody and bind their hashes."""
    evidence_dir = corpus_root / "evidence" / "issue_9645_pilot"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    persisted_paths = _persist_pilot_payload_files(payload, evidence_dir, corpus_root, pilot)
    pilot["source_files"] = [
        {
            "path": persisted_paths[source["path"]],
            "sha256": source["sha256"],
        }
        for source in pilot["source_files"]
    ]
    pilot["manifest_files"] = [
        {
            **manifest,
            "path": persisted_paths[manifest["path"]],
        }
        for manifest in pilot["manifest_files"]
    ]
    pilot["bundle_receipts"] = _persist_pilot_bundle_receipts(
        payload, evidence_dir, corpus_root, pilot["bundle_receipts"]
    )
    pilot["evidence_bundle_root"] = "evidence/issue_9645_pilot"


def _persist_pilot_payload_files(
    payload: Path,
    evidence_dir: Path,
    corpus_root: Path,
    pilot: Mapping[str, Any],
) -> dict[str, str]:
    expected_hashes = {item["path"]: item["sha256"] for item in pilot["source_files"]}
    expected_hashes.update({item["path"]: item["sha256"] for item in pilot["manifest_files"]})
    persisted: dict[str, str] = {}
    for relative, expected in sorted(expected_hashes.items()):
        source = payload / relative
        destination = evidence_dir / relative
        _copy_verified_file(source, destination, expected, f"pilot evidence {relative}")
        persisted[relative] = destination.resolve().relative_to(corpus_root.resolve()).as_posix()
    return persisted


def _persist_pilot_bundle_receipts(
    payload: Path,
    evidence_dir: Path,
    corpus_root: Path,
    receipts: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    persisted = []
    for receipt in receipts:
        source_path = receipt["source_path"]
        relative = receipt["path"]
        source = payload.parent / source_path
        destination = evidence_dir / relative
        _copy_verified_file(source, destination, receipt["sha256"], f"bundle receipt {source_path}")
        persisted.append(
            {
                "source_path": source_path,
                "path": destination.resolve().relative_to(corpus_root.resolve()).as_posix(),
                "sha256": receipt["sha256"],
            }
        )
    return persisted


def _copy_verified_file(source: Path, destination: Path, expected_sha256: str, label: str) -> None:
    if not source.is_file() or _sha256_file(source) != expected_sha256:
        raise CorpusError(f"{label} does not match its verified source checksum")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise CorpusError(f"corpus path conflicts with {label}")
    else:
        shutil.copyfile(source, destination)
    if _sha256_file(destination) != expected_sha256:
        raise CorpusError(f"corpus copy changed {label}")


def _build_issue9645_historical_case(
    payload: Path,
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]]]:
    metadata = _read_json_object(payload / "run_metadata.json")
    _verify_issue9645_bundle(
        payload,
        source_revision=str(metadata.get("experiment_source_commit") or ""),
        source_file_hashes=metadata.get("source_file_sha256"),
        required_payload_paths=[
            "historical_issue_1501_failure_0002.json",
            "historical_issue_1501_failure_0002/scenario.yaml",
            "historical_issue_1501_failure_0002/route_overrides.yaml",
            "historical_issue_1501_failure_0002/replay_1.jsonl",
            "historical_issue_1501_failure_0002/replay_1.provenance.json",
            "historical_issue_1501_failure_0002/replay_2.jsonl",
            "historical_issue_1501_failure_0002/replay_2.provenance.json",
            "path_normalization.json",
            "replay_validation.json",
        ],
    )
    context = _load_historical_case_context(payload)
    replay = _verify_historical_replay_pair(payload, context)
    case, observations = _materialize_historical_case(context, replay, payload)
    errors = _validate_case_record(case)
    if errors:
        raise CorpusError("case record rejected: " + "; ".join(errors))
    source_files = {
        **replay["source_files"],
        "historical_sources": context["historical_sources"],
        "map_assets": context["map_asset_bytes"],
    }
    return case, source_files, observations


def _load_historical_case_context(payload: Path) -> dict[str, Any]:
    replay_validation = _read_json_object(payload / "replay_validation.json")
    historical = _read_json_object(payload / "historical_issue_1501_failure_0002.json")
    archived, run_context, result, candidate = _validated_historical_packet(
        replay_validation, historical
    )
    scenario_context = _validated_historical_scenario(
        payload, candidate, replay_revision=str(result.get("regeneration_commit") or "")
    )
    archive_hash, archive_path = _validated_historical_archive(historical, archived)
    return {
        "replay_validation": replay_validation,
        "historical": historical,
        "archived": archived,
        "run_context": run_context,
        "result": result,
        "candidate": candidate,
        "archive_hash": archive_hash,
        "archive_path": archive_path,
        "historical_sources": _historical_source_snapshot(historical),
        **scenario_context,
    }


def _validated_historical_packet(
    replay_validation: Mapping[str, Any], historical: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if replay_validation.get("schema_version") != ISSUE_9645_REPLAY_SCHEMA:
        raise CorpusError("unsupported #9645 replay validation schema")
    if historical.get("schema_version") != "issue_1501_archived_failure_replay.v1":
        raise CorpusError("unsupported #1501 archived-case schema")
    archived = historical.get("archived_entry")
    run_context = historical.get("archived_run_context")
    result = replay_validation.get("historical_issue_1501_case")
    if not all(isinstance(value, dict) for value in (archived, run_context, result)):
        raise CorpusError("#1501 archived case, run context, or replay receipt is missing")
    _validate_historical_replay_claim(result)
    _validate_historical_search_identity(archived, run_context)
    candidate = archived.get("candidate")
    if not isinstance(candidate, dict):
        raise CorpusError("#1501 candidate parameters are missing")
    return archived, run_context, result, candidate


def _validate_historical_replay_claim(result: Mapping[str, Any]) -> None:
    if result.get("historical_case_id") != "issue_1501/failure_0002":
        raise CorpusError("#1501 case identity differs from the requested historical failure")
    if result.get("archived_failure_type") != "collision" or result.get("planner") != "goal":
        raise CorpusError("#1501 archive does not identify a goal-planner collision")
    if result.get("replay_count") != 2 or result.get("replays_identical") is not True:
        raise CorpusError("#1501 case does not contain two matching current replays")
    if result.get("dynamic_task_feasibility", "").split("_")[0] != "unknown":
        raise CorpusError("#1501 dynamic feasibility must remain unknown")
    if (
        result.get("current_scenario_certification", {}).get("classification")
        != "hard_but_solvable"
    ):
        raise CorpusError("#1501 static certificate summary is absent")


def _validate_historical_search_identity(
    archived: Mapping[str, Any], run_context: Mapping[str, Any]
) -> None:
    if run_context.get("commit") != "4bd5fb412d2bf023d26f674833dd379aedf8c17e":
        raise CorpusError("#1501 historical source revision is not pinned")
    if (
        archived.get("source_candidate_index") != 8
        or archived.get("source_manifest") != "crossing_ttc/goal/optuna/manifest.json"
    ):
        raise CorpusError("#1501 archived candidate source identity is incomplete")
    if run_context.get("seed") != 42 or run_context.get("budget_per_sampler") != 32:
        raise CorpusError("#1501 search seed/budget provenance is incomplete")


def _validated_historical_scenario(
    payload: Path,
    candidate: Mapping[str, Any],
    *,
    replay_revision: str,
) -> dict[str, Any]:
    historical_dir = payload / "historical_issue_1501_failure_0002"
    scenario_path = historical_dir / "scenario.yaml"
    route_path = historical_dir / "route_overrides.yaml"
    scenario_doc = _load_yaml_object(scenario_path, "scenario")
    scenario_rows = scenario_doc.get("scenarios")
    if (
        not isinstance(scenario_rows, list)
        or len(scenario_rows) != 1
        or not isinstance(scenario_rows[0], dict)
    ):
        raise CorpusError("#1501 scenario input must contain one scenario")
    scenario = scenario_rows[0]
    route_doc = _load_yaml_object(route_path, "route overrides")
    scenario_id = str(scenario.get("name") or scenario.get("id") or "").strip()
    seed = _scenario_seed(scenario)
    if seed != 320 or seed != candidate.get("scenario_seed"):
        raise CorpusError("#1501 scenario seed does not match archived candidate parameters")
    metadata_candidate = (scenario.get("metadata") or {}).get("adversarial_candidate")
    if not _candidate_parameters_match(metadata_candidate, candidate):
        raise CorpusError("#1501 materialized scenario differs from archived candidate parameters")
    if scenario.get("route_overrides_file") != "route_overrides.yaml":
        raise CorpusError("#1501 scenario route binding is missing or unexpected")
    scenario_errors = _validate_scenario_structure(scenario_path)
    if scenario_errors:
        raise CorpusError("#1501 scenario is structurally invalid: " + "; ".join(scenario_errors))
    if not _is_full_git_revision(replay_revision):
        raise CorpusError("#1501 replay revision for map inputs is missing")
    map_assets, map_asset_bytes = _historical_map_asset_snapshot(scenario, replay_revision)
    return {
        "historical_dir": historical_dir,
        "scenario_path": scenario_path,
        "route_path": route_path,
        "scenario": scenario,
        "route_doc": route_doc,
        "scenario_id": scenario_id,
        "seed": seed,
        "identity_hash": compute_case_effective_scenario_hash(scenario, route_doc, map_assets),
        "scenario_hash": _sha256_file(scenario_path),
        "route_hash": _sha256_file(route_path),
        "map_assets": map_assets,
        "map_asset_bytes": map_asset_bytes,
    }


def _candidate_parameters_match(materialized: Any, archived: Mapping[str, Any]) -> bool:
    return isinstance(materialized, dict) and all(
        materialized.get(key) == value for key, value in archived.items()
    )


def _validated_historical_archive(
    historical: Mapping[str, Any], archived: Mapping[str, Any]
) -> tuple[str, Path]:
    archive_hash = str(historical.get("source_archive_sha256") or "")
    archive_path = _ROOT / str(historical.get("source_archive") or "")
    if not archive_path.is_file() or _sha256_file(archive_path) != archive_hash:
        raise CorpusError("#1501 tracked archive is unavailable or its checksum differs")
    archive_data = _read_json_object(archive_path)
    archive_entry = next(
        (
            item
            for item in archive_data.get("entries", [])
            if isinstance(item, dict) and item.get("archive_id") == "failure_0002"
        ),
        None,
    )
    if archive_entry != archived:
        raise CorpusError("#9645 archived entry differs from the tracked #1501 archive")
    return archive_hash, archive_path


def _verify_historical_replay_pair(payload: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    path_normalization = _read_json_object(payload / "path_normalization.json")
    if path_normalization.get("schema_version") != "evidence_path_normalization.v1":
        raise CorpusError("#9645 path-normalization receipt is missing or unsupported")
    normalization_rows = {
        item.get("path"): item
        for item in path_normalization.get("records", [])
        if isinstance(item, dict)
    }
    result = context["result"]
    expected_hashes = result.get("replay_files_sha256")
    if not isinstance(expected_hashes, list) or len(expected_hashes) != 2:
        raise CorpusError("#1501 replay receipt must bind both replay artifact hashes")
    source_files: dict[str, Any] = {
        "scenario": context["scenario_path"],
        "route": context["route_path"],
        "historical_packet": payload / "historical_issue_1501_failure_0002.json",
        "replay_validation": payload / "replay_validation.json",
        "path_normalization": payload / "path_normalization.json",
        "source_archive": context["archive_path"],
        "source_run_report": _ROOT / "docs/context/issue_1501_adversarial_smoke_run.md",
    }
    replay_rows = [
        _verify_one_historical_replay(
            index, context, normalization_rows, expected_hashes[index - 1]
        )
        for index in (1, 2)
    ]
    run_ids = [row["run_id"] for row in replay_rows]
    if any(not isinstance(run_id, str) or not run_id.strip() for run_id in run_ids):
        raise CorpusError("#1501 replay provenance is missing a distinct run ID")
    if len(set(run_ids)) != len(run_ids):
        raise CorpusError("#1501 replay provenance reuses a run ID")
    projections = [row["projection"] for row in replay_rows]
    if projections[0] != projections[1]:
        raise CorpusError("#1501 current replay records disagree on selected event/metric identity")
    _validate_historical_summary(result, projections[0])
    source_files.update(
        {key: value for row in replay_rows for key, value in row["source_files"].items()}
    )
    target = replay_rows[0]["episode"].get("algorithm_metadata")
    if not isinstance(target, dict) or not target.get("config_hash"):
        raise CorpusError("#1501 planner configuration identity is missing")
    return {
        "source_files": source_files,
        "projection": projections[0],
        "run_ids": run_ids,
        "expected_hashes": expected_hashes,
        "normalization_rows": normalization_rows,
        "target_config_identity": target["config_hash"],
        "target_config": target.get("config", {}),
        "identity_hash": context["identity_hash"],
    }


def _verify_one_historical_replay(
    index: int,
    context: Mapping[str, Any],
    normalization_rows: Mapping[str, Any],
    expected_source_hash: str,
) -> dict[str, Any]:
    relative = f"historical_issue_1501_failure_0002/replay_{index}"
    replay_file = context["historical_dir"] / f"replay_{index}.jsonl"
    provenance_file = context["historical_dir"] / f"replay_{index}.provenance.json"
    normalized_replay = normalization_rows.get(f"{relative}.jsonl")
    normalized_provenance = normalization_rows.get(f"{relative}.provenance.json")
    if not isinstance(normalized_replay, dict) or not isinstance(normalized_provenance, dict):
        raise CorpusError(f"#1501 replay {index} normalization is not recorded")
    _validate_replay_normalization(
        index,
        normalized_replay,
        normalized_provenance,
        replay_file,
        provenance_file,
        expected_source_hash,
    )
    episode = _read_single_jsonl_record(replay_file)
    provenance = _read_json_object(provenance_file)
    run = provenance.get("run")
    run_id = run.get("run_id") if isinstance(run, Mapping) else None
    projection = _historical_replay_projection(episode)
    _validate_replay_provenance(
        provenance,
        episode,
        replay_file,
        context["scenario_path"],
        expected_revision=str(context["result"].get("regeneration_commit")),
        expected_source_artifact_sha256=expected_source_hash,
    )
    if projection["outcome"].get("collision_event") is not True:
        raise CorpusError(f"#1501 replay {index} did not reproduce the archived collision event")
    return {
        "episode": episode,
        "projection": projection,
        "run_id": run_id,
        "source_files": {
            f"replay_{index}": replay_file,
            f"replay_{index}_provenance": provenance_file,
        },
    }


def _validate_replay_normalization(
    index: int,
    normalized_replay: Mapping[str, Any],
    normalized_provenance: Mapping[str, Any],
    replay_file: Path,
    provenance_file: Path,
    expected_source_hash: str,
) -> None:
    replay_ok = (
        normalized_replay.get("source_sha256_before_path_normalization") == expected_source_hash
        and normalized_replay.get("normalized_sha256") == _sha256_file(replay_file)
        and normalized_replay.get("field_rewrites") == {"scenario_params.route_overrides_file": 1}
    )
    provenance_ok = (
        _is_sha256(normalized_provenance.get("source_sha256_before_path_normalization"))
        and normalized_provenance.get("normalized_sha256") == _sha256_file(provenance_file)
        and normalized_provenance.get("field_rewrites") == {"run.invocation": 1}
    )
    if not replay_ok or not provenance_ok:
        raise CorpusError(f"#1501 replay {index} normalized artifact binding differs")


def _validate_historical_summary(result: Mapping[str, Any], projection: Mapping[str, Any]) -> None:
    if result.get("current_replay_outcome", {}).get("outcome") != projection["outcome"]:
        raise CorpusError("#1501 durable outcome summary differs from replay records")
    reported_metrics = result.get("current_replay_metrics")
    aliases = {
        "success": ("success",),
        "collisions": ("collisions", "total_collision_count"),
        "total_collision_count": ("total_collision_count",),
        "ped_collision_count": ("ped_collision_count",),
        "near_misses": ("near_misses",),
        "min_clearance": ("min_clearance", "min_clearance_m"),
    }
    if not isinstance(reported_metrics, dict) or any(
        not any(reported_metrics.get(alias) == value for alias in aliases[key])
        for key, value in projection["metrics"].items()
    ):
        raise CorpusError("#1501 selected metric summary differs from replay records")


def _materialize_historical_case(
    context: Mapping[str, Any], replay: Mapping[str, Any], payload: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _historical_search_source(context)
    replay_receipt = _historical_replay_receipt(context, replay)
    case = _historical_case_record(context, replay, source, replay_receipt, payload)
    observations = [
        {
            **_historical_failure_observation(replay, replay_receipt, run_id=replay["run_ids"][0]),
            "episode_sha256": _sha256_file(replay["source_files"]["replay_1"]),
        },
        {
            **_historical_failure_observation(replay, replay_receipt, run_id=replay["run_ids"][1]),
            "episode_sha256": _sha256_file(replay["source_files"]["replay_2"]),
        },
    ]
    return case, observations


def _historical_search_source(context: Mapping[str, Any]) -> dict[str, Any]:
    historical = context["historical"]
    archived = context["archived"]
    run_context = context["run_context"]
    return {
        "kind": "historical_adversarial_search_archive",
        "origin_issue": 1501,
        "origin_case_id": "issue_1501/failure_0002",
        "historical_source_revision": run_context["commit"],
        "historical_manifest": {
            "declared_path": archived["source_manifest"],
            "status": "not_archived",
            "sha256": None,
        },
        "search_settings": {
            "sampler": "optuna_tpe",
            "objective": "worst_case_snqi",
            "objective_direction": "maximize",
            "budget_per_sampler": 32,
            "seed": 42,
            "candidate_index": archived["source_candidate_index"],
            "search_space_path": "configs/adversarial/crossing_ttc_space.yaml",
            "search_space_sha256": "e90353f9653173cc351117bfc874c1e7d5933d32f1f892f1b264d8148c767f34",
            "scenario_template_path": "configs/scenarios/templates/crossing_ttc.yaml",
            "scenario_template_sha256": "4718aa6ee78d8013f251e20f1c78173ac4052a79e63ec80e19dd2347866b4d8f",
            "objective_source_path": "robot_sf/adversarial/objectives.py",
            "objective_source_sha256": "9c2752b64d57af36f281f59592716d50ca61d9b99dadb76fcfdf9d06096bf0c8",
            "launcher_path": "SLURM/Auxme/adversarial_smoke_1501.sl",
        },
        "archive": {
            "path": historical["source_archive"],
            "sha256": context["archive_hash"],
            "archive_id": archived["archive_id"],
            "objective_value": archived["objective_value"],
            "objective_recomputed": False,
        },
        "source_run_report": {
            "path": "docs/context/issue_1501_adversarial_smoke_run.md",
            "source_candidate_count": run_context["historical_source_candidate_count"],
            "valid_failures": run_context["historical_valid_failures"],
            "valid_non_failures": run_context["historical_valid_non_failures"],
            "invalid_candidates": run_context["historical_invalid_candidates"],
        },
    }


def _historical_replay_receipt(
    context: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    result = context["result"]
    projection = replay["projection"]
    normalization = replay["normalization_rows"]
    expected_hashes = replay["expected_hashes"]
    source_files = replay["source_files"]
    replay_artifacts = []
    for index in (1, 2):
        normalized_provenance = normalization[
            f"historical_issue_1501_failure_0002/replay_{index}.provenance.json"
        ]
        replay_artifacts.append(
            {
                "path": f"source_evidence/replay_{index}.jsonl",
                "run_id": replay["run_ids"][index - 1],
                "selected_event_identity": projection["selected_event_identity"],
                "sha256": _sha256_file(source_files[f"replay_{index}"]),
                "normalized_bundle_sha256": _sha256_file(source_files[f"replay_{index}"]),
                "source_artifact_sha256_before_path_normalization": expected_hashes[index - 1],
                "path_normalization": "scenario_params.route_overrides_file",
                "provenance_path": f"source_evidence/replay_{index}.provenance.json",
                "provenance_sha256_before_path_normalization": normalized_provenance[
                    "source_sha256_before_path_normalization"
                ],
                "provenance_sha256_normalized": _sha256_file(
                    source_files[f"replay_{index}_provenance"]
                ),
                "provenance_path_normalization": "run.invocation",
            }
        )
    return {
        "schema_version": CASE_ADMISSION_REPLAY_SCHEMA_VERSION,
        "verification_status": "repeated_current_revision_match",
        "input_binding_status": "unknown_historical",
        "input_binding_limitation": (
            "historical replay rows predate direct scenario, route, and map input binding; "
            "their current-revision projection match does not prove exact case-input replay"
        ),
        "historical_origin_match": "not_verifiable_original_raw_episode_absent",
        "historical_original_raw_episode_available": False,
        "target_revision": str(result["regeneration_commit"]),
        "replay_revision": str(result["regeneration_commit"]),
        "target_and_replay_revision_match": True,
        "effective_scenario_sha256": context["identity_hash"],
        "scenario_input_sha256": context["scenario_hash"],
        "route_overrides_sha256": context["route_hash"],
        "map_assets": _map_asset_identity(context["map_assets"]),
        "planner_id": projection["planner_id"],
        "planner_config_identity": projection["planner_config_identity"],
        "replay_count": 2,
        "replay_signature_sha256_reported": result.get("replay_signature_sha256"),
        "selected_projection_sha256": hashlib.sha256(
            _stable_json(projection).encode("utf-8")
        ).hexdigest(),
        "selected_projection": projection,
        "replay_artifacts": replay_artifacts,
        "limitations": list(result.get("limitations", [])),
    }


def _historical_failure_observation(
    replay: Mapping[str, Any], replay_receipt: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    projection = replay["projection"]
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "case_id": "pending",
        "effective_scenario_sha256": replay["identity_hash"],
        "planner_id": "goal",
        "planner_config_identity": replay["target_config_identity"],
        "source_revision": replay["projection"]["source_revision"],
        "episode_sha256": "pending",
        "run_id": run_id,
        "input_binding_status": "unknown_historical",
        "execution_mode": "native",
        "readiness_status": "native",
        "availability_status": "available",
        "fallback_or_degraded": False,
        "evidence_status": "complete",
        "error": None,
        "outcome": projection["outcome"],
        "termination_reason": projection["termination_reason"],
        "metrics": projection["metrics"],
        "evidence_role": "replay_verified_discovery_failure",
        "replay_verification_status": replay_receipt["verification_status"],
    }


def _historical_case_record(
    context: Mapping[str, Any],
    replay: Mapping[str, Any],
    source: Mapping[str, Any],
    replay_receipt: Mapping[str, Any],
    payload: Path,
) -> dict[str, Any]:
    archived = context["archived"]
    scenario = context["scenario"]
    identity_hash = context["identity_hash"]
    scenario_hash = context["scenario_hash"]
    route_hash = context["route_hash"]
    config_identity = replay["target_config_identity"]
    feasibility = {
        "verdict": "admissible_feasibility_unknown",
        "dynamic_task_status": "unknown",
        "static_certificate": {
            "schema_version": "scenario_cert.v1",
            "classification": "hard_but_solvable",
            "benchmark_eligibility": "eligible",
            "reasons": ["low_static_clearance_margin"],
            "claim_boundary": "static route certificate only; dynamic task feasibility remains unknown",
        },
    }
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": f"case-{identity_hash}",
        "effective_scenario_sha256": identity_hash,
        "scenario_id": context["scenario_id"],
        "scenario_seed": context["seed"],
        "scenario_template": {
            "scenario_id": str(scenario.get("metadata", {}).get("archetype", "unknown")),
            "map_id": scenario.get("map_id"),
        },
        "inputs": {
            "scenario_path": "pending",
            "scenario_sha256": scenario_hash,
            "route_overrides_path": "pending",
            "route_overrides_sha256": route_hash,
            "map_assets": [dict(item) for item in context["map_assets"]],
        },
        "structural_validation": {
            "status": "valid",
            "validator": "robot_sf.cli_scenarios.validate_scenario_payload",
            "schema_path": "robot_sf/benchmark/schemas/scenarios.schema.json",
            "schema_sha256": _sha256_file(
                _ROOT / "robot_sf/benchmark/schemas/scenarios.schema.json"
            ),
            "scenario_sha256": scenario_hash,
            "route_overrides_sha256": route_hash,
            "error_count": 0,
        },
        "discovery": {
            "round_id": "issue_1501_crossing_ttc_smoke",
            "discovery_issue": 1501,
            "imported_by_issue": 9652,
            "origin_case_id": "issue_1501/failure_0002",
            "candidate_parameters": context["candidate"],
            "objective": {
                "name": "worst_case_snqi",
                "direction": "maximize",
                "archived_value": archived["objective_value"],
                "value_recomputed": False,
            },
            "failure_attribution": archived["failure_attribution"],
            "search_source": source,
        },
        "admissibility": feasibility,
        "target_planner": {
            "planner_id": "goal",
            "config_identity": config_identity,
            "configuration_snapshot": replay["target_config"],
            "configuration_snapshot_source": "episode.algorithm_metadata.config",
        },
        "replay_receipt": replay_receipt,
        "source_evidence": {
            "source_archive_path": context["historical"]["source_archive"],
            "source_archive_sha256": context["archive_hash"],
            "historical_packet_sha256": _sha256_file(
                payload / "historical_issue_1501_failure_0002.json"
            ),
            "replay_validation_sha256": _sha256_file(payload / "replay_validation.json"),
            "historical_raw_episode_status": "not_archived",
            "source_run_manifest_status": "not_archived; exact settings are pinned by launch receipt/source revision",
            "corpus_files": [],
        },
        "mechanism_group": archived.get("cluster_key"),
        "near_duplicate_report": _unassessed_near_duplicates(),
    }


def _historical_source_snapshot(historical: Mapping[str, Any]) -> dict[str, bytes]:
    """Pin the #1501 objective, search space, template, and launcher source bytes."""
    commit = str(historical.get("archived_run_context", {}).get("commit"))
    sources = {
        "objective_source.py": (
            "robot_sf/adversarial/objectives.py",
            "9c2752b64d57af36f281f59592716d50ca61d9b99dadb76fcfdf9d06096bf0c8",
        ),
        "search_space.yaml": (
            "configs/adversarial/crossing_ttc_space.yaml",
            "e90353f9653173cc351117bfc874c1e7d5933d32f1f892f1b264d8148c767f34",
        ),
        "scenario_template.yaml": (
            "configs/scenarios/templates/crossing_ttc.yaml",
            "4718aa6ee78d8013f251e20f1c78173ac4052a79e63ec80e19dd2347866b4d8f",
        ),
        "launcher.sl": ("SLURM/Auxme/adversarial_smoke_1501.sl", None),
    }
    result: dict[str, bytes] = {}
    for name, (relative, expected_sha) in sources.items():
        try:
            content = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise CorpusError(f"historical source snapshot unavailable: {relative}: {exc}") from exc
        digest = hashlib.sha256(content).hexdigest()
        if expected_sha is not None and digest != expected_sha:
            raise CorpusError(f"historical source snapshot hash differs: {relative}")
        result[name] = content
    return result


def _validate_replay_provenance(
    provenance: Mapping[str, Any],
    episode: Mapping[str, Any],
    replay_file: Path,
    scenario_file: Path,
    *,
    expected_revision: str,
    expected_source_artifact_sha256: str,
) -> None:
    if provenance.get("schema_version") != "benchmark_result_provenance.v1":
        raise CorpusError("historical replay provenance schema is invalid")
    run = provenance.get("run")
    inputs = provenance.get("inputs")
    identity = provenance.get("campaign_identity")
    rows = provenance.get("rows")
    raw = provenance.get("raw_artifacts")
    if not all(isinstance(value, dict) for value in (run, inputs, identity)):
        raise CorpusError("historical replay provenance is incomplete")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise CorpusError("historical replay must have exactly one provenance row")
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise CorpusError("historical replay raw artifact binding is incomplete")
    row = rows[0]
    _validate_provenance_execution_identity(run, row, identity, episode, expected_revision)
    _validate_provenance_input_identity(
        inputs, raw[0], episode, scenario_file, expected_source_artifact_sha256
    )
    _validate_native_replay_execution(episode)


def _validate_provenance_execution_identity(
    run: Mapping[str, Any],
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    episode: Mapping[str, Any],
    expected_revision: str,
) -> None:
    current_revision = run.get("repo_commit")
    if current_revision != expected_revision or row.get("repo_commit") != expected_revision:
        raise CorpusError("historical target execution revision does not match replay revision")
    if (
        episode.get("git_hash") != expected_revision
        or episode.get("provenance", {}).get("git_hash") != expected_revision
    ):
        raise CorpusError("historical replay episode revision differs from its receipt")
    if identity.get("algorithm") != "goal" or episode.get("algo") != "goal":
        raise CorpusError("historical replay planner identity differs")
    if row.get("episode_id") != episode.get("episode_id") or row.get("scenario_id") != episode.get(
        "scenario_id"
    ):
        raise CorpusError("historical replay episode identity differs from provenance")
    if row.get("seed") != episode.get("seed") or episode.get("seed") != 320:
        raise CorpusError("historical replay seed differs from provenance")


def _validate_provenance_input_identity(
    inputs: Mapping[str, Any],
    raw_artifact: Mapping[str, Any],
    episode: Mapping[str, Any],
    scenario_file: Path,
    expected_source_artifact_sha256: str,
) -> None:
    if raw_artifact.get("sha256") != expected_source_artifact_sha256:
        raise CorpusError("historical replay source checksum differs from normalization receipt")
    scenario_input = inputs.get("scenario_matrix")
    if not isinstance(scenario_input, dict) or scenario_input.get("sha256") != _sha256_file(
        scenario_file
    ):
        raise CorpusError("historical replay scenario input checksum differs from provenance")


def _validate_native_replay_execution(episode: Mapping[str, Any]) -> None:
    planner = episode.get("algorithm_metadata")
    planner_identity = planner.get("planner_kinematics") if isinstance(planner, dict) else None
    if (
        not isinstance(planner_identity, dict)
        or planner_identity.get("execution_mode") != "native"
        or planner.get("status") != "ok"
        or episode.get("integrity", {}).get("effective_view", {}).get("degraded") is not False
    ):
        raise CorpusError("historical replay is not a native, non-degraded planner execution")


def _historical_replay_projection(episode: Mapping[str, Any]) -> dict[str, Any]:
    outcome = episode.get("outcome")
    metrics = episode.get("metrics")
    if not isinstance(outcome, dict) or not isinstance(metrics, dict):
        raise CorpusError("historical replay outcome or metrics are missing")
    fields = ("collision_event", "route_complete", "timeout_event")
    if any(not isinstance(outcome.get(key), bool) for key in fields):
        raise CorpusError("historical replay outcome flags must be explicit booleans")
    selected_metrics = {
        key: _require_finite_metric(metrics, key)
        for key in (
            "success",
            "collisions",
            "total_collision_count",
            "ped_collision_count",
            "near_misses",
            "min_clearance",
        )
    }
    termination = str(episode.get("termination_reason") or "")
    contradictions = outcome_contradictions(
        termination_reason=termination,
        outcome=outcome,
        metrics=metrics,
    )
    if contradictions:
        raise CorpusError(
            "historical replay record is internally contradictory: " + "; ".join(contradictions)
        )
    if termination != "collision" or episode.get("status") != "collision":
        raise CorpusError("historical replay did not terminate with the archived collision")
    return {
        "scenario_id": episode.get("scenario_id"),
        "seed": episode.get("seed"),
        "planner_id": episode.get("algo"),
        "planner_config_identity": episode.get("algorithm_metadata", {}).get("config_hash"),
        "source_revision": episode.get("git_hash"),
        "termination_reason": termination,
        "outcome": {key: outcome[key] for key in fields},
        "metrics": selected_metrics,
        "selected_event_identity": _selected_event_identity(episode),
    }


def _validate_case_record(case: Mapping[str, Any], *, corpus_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    if case.get("schema_version") != CASE_SCHEMA_VERSION:
        errors.append("unsupported case schema")
    for field in ("case_id", "effective_scenario_sha256", "scenario_id"):
        if not isinstance(case.get(field), str) or not case[field]:
            errors.append(f"{field} is missing")
    errors.extend(_validate_case_execution_contract(case))
    errors.extend(_validate_case_discovery(case))
    errors.extend(_validate_case_inputs(case))
    if corpus_root is not None:
        try:
            _case_input_paths(case, corpus_root)
        except (CorpusError, OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            errors.append(f"materialized case inputs are invalid: {exc}")
        errors.extend(_validate_case_corpus_evidence(case, corpus_root))
        errors.extend(_validate_case_admission_replay(case, corpus_root))
    return errors


def _validate_case_current_target_revision(case: Mapping[str, Any]) -> list[str]:
    """Require an independently resolved checkout HEAD before admission as current."""
    receipt = case.get("replay_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("verification_status") not in {
        "exact_current_revision_match",
        "repeated_current_revision_match",
    }:
        return []
    current_revision = _current_target_revision()
    if current_revision is None:
        return ["independent current target revision is unavailable"]
    if (
        receipt.get("target_revision") != current_revision
        or receipt.get("replay_revision") != current_revision
    ):
        return [
            "admission replay does not match the independently resolved current target revision"
        ]
    return []


def _validate_case_execution_contract(case: Mapping[str, Any]) -> list[str]:
    errors = _validate_case_structure_admissibility(case)
    receipt = case.get("replay_receipt")
    receipt_valid = isinstance(receipt, dict) and receipt.get("verification_status") in {
        "repeated_current_revision_match",
        "exact_current_revision_match",
    }
    if not receipt_valid:
        errors.append("current-revision replay verification is missing")
    else:
        errors.extend(_validate_case_replay_binding(case, receipt))
        errors.extend(_validate_target_planner_replay_failure(receipt))
    target = case.get("target_planner")
    target_valid = (
        isinstance(target, dict)
        and isinstance(target.get("planner_id"), str)
        and bool(target.get("planner_id"))
        and isinstance(target.get("config_identity"), str)
        and bool(target.get("config_identity"))
        and isinstance(target.get("configuration_snapshot"), dict)
    )
    if not target_valid:
        errors.append("target planner/configuration identity is incomplete")
    elif receipt_valid and (
        receipt.get("planner_id") != target.get("planner_id")
        or receipt.get("planner_config_identity") != target.get("config_identity")
    ):
        errors.append("replay receipt planner/configuration differs from target identity")
    return errors


def _validate_target_planner_replay_failure(receipt: Mapping[str, Any]) -> list[str]:
    """Require a verified canonical target collision or timeout before admission.

    This is the supported v1 criticality boundary. Other metric-extreme cases need
    a separately versioned objective and replay-bound threshold contract.
    """
    projection = receipt.get("selected_projection")
    if not isinstance(projection, Mapping):
        return ["target planner replay projection is missing"]
    outcome = projection.get("outcome")
    metrics = projection.get("metrics")
    termination_reason = projection.get("termination_reason")
    if not isinstance(outcome, Mapping) or not isinstance(metrics, Mapping):
        return ["target planner replay outcome or metrics are missing"]
    outcome_fields = ("collision_event", "route_complete", "timeout_event")
    if any(not isinstance(outcome.get(field), bool) for field in outcome_fields):
        return ["target planner replay outcome flags must be explicit booleans"]
    if not isinstance(termination_reason, str) or termination_reason not in TERMINATION_REASONS:
        return ["target planner replay termination reason is unsupported"]
    contradictions = outcome_contradictions(
        termination_reason=termination_reason,
        outcome=outcome,
        metrics=metrics,
    )
    if contradictions:
        return [
            "target planner replay outcome contradicts canonical flags or metrics: "
            + "; ".join(contradictions)
        ]
    collision_failure = (
        outcome["collision_event"] is True
        and outcome["timeout_event"] is False
        and termination_reason == "collision"
    )
    timeout_failure = (
        outcome["timeout_event"] is True
        and outcome["collision_event"] is False
        and termination_reason in {"truncated", "max_steps"}
    )
    if outcome["route_complete"] is not False or not (collision_failure or timeout_failure):
        return ["target planner replay does not verify canonical collision/timeout noncompletion"]
    return []


def _validate_case_structure_admissibility(case: Mapping[str, Any]) -> list[str]:
    errors = []
    structural = case.get("structural_validation")
    if not isinstance(structural, dict) or structural.get("status") != "valid":
        errors.append("structural validation is not valid")
    elif (
        not isinstance(structural.get("validator"), str)
        or not structural["validator"].strip()
        or not _is_sha256(structural.get("schema_sha256"))
        or structural.get("error_count") != 0
    ):
        errors.append("structural validation evidence is incomplete")
    if not isinstance(case.get("scenario_seed"), int) or isinstance(
        case.get("scenario_seed"), bool
    ):
        errors.append("scenario seed must be an integer")
    admissibility = case.get("admissibility")
    if not isinstance(admissibility, dict) or admissibility.get("verdict") not in {
        "admissible_feasibility_unknown",
        "empirically_feasible",
        "planner_specific_failure",
    }:
        errors.append("admissibility verdict must be explicit and non-excluding")
    return errors


def _validate_case_admissibility_evidence(
    case: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    *,
    corpus_root: Path | None,
) -> list[str]:
    admissibility = case.get("admissibility")
    if not isinstance(admissibility, Mapping):
        return ["admissibility record is missing"]
    verdict = admissibility.get("verdict")
    evidence = admissibility.get("evidence_receipt")
    if verdict == "admissible_feasibility_unknown":
        return _validate_unknown_admissibility_evidence(evidence)
    if verdict not in {"empirically_feasible", "planner_specific_failure"}:
        return []
    if not isinstance(evidence, Mapping):
        return ["positive feasibility verdict requires an evidence receipt"]
    return _validate_positive_admissibility_evidence(
        case, evaluations, verdict, evidence, corpus_root=corpus_root
    )


def _validate_unknown_admissibility_evidence(evidence: Any) -> list[str]:
    if evidence is not None and (
        not isinstance(evidence, Mapping)
        or evidence.get("verdict") != "admissible_feasibility_unknown"
    ):
        return ["admissibility evidence receipt conflicts with unknown verdict"]
    return []


def _validate_positive_admissibility_evidence(
    case: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    verdict: str,
    evidence: Mapping[str, Any],
    *,
    corpus_root: Path | None,
) -> list[str]:
    errors = _admissibility_receipt_binding_errors(case, verdict, evidence)
    evaluation, evaluation_errors = _resolve_admissibility_evaluation(case, evaluations, evidence)
    errors.extend(evaluation_errors)
    if evaluation is None:
        return errors
    errors.extend(
        _validate_admissibility_evaluation(case, evaluation, verdict, corpus_root=corpus_root)
    )
    return errors


def _admissibility_receipt_binding_errors(
    case: Mapping[str, Any], verdict: str, evidence: Mapping[str, Any]
) -> list[str]:
    expected_fields = {
        "schema_version": CASE_ADMISSIBILITY_EVIDENCE_SCHEMA_VERSION,
        "case_id": case.get("case_id"),
        "effective_scenario_sha256": case.get("effective_scenario_sha256"),
        "verdict": verdict,
    }
    return [
        f"admissibility evidence receipt {field} does not bind the case"
        for field, expected in expected_fields.items()
        if evidence.get(field) != expected
    ]


def _resolve_admissibility_evaluation(
    case: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, list[str]]:
    evaluation_id = evidence.get("evaluation_id")
    if not _is_sha256(evaluation_id):
        return None, ["admissibility evidence receipt evaluation_id is invalid"]
    matches = [item for item in evaluations if item.get("evaluation_id") == evaluation_id]
    if len(matches) != 1:
        return None, ["admissibility evidence receipt must identify exactly one evaluation"]
    evaluation = matches[0]
    errors = []
    if evaluation.get("evaluation_id") != _evaluation_digest(evaluation):
        errors.append("admissibility evidence evaluation digest is invalid")
    if evaluation.get("case_id") != case.get("case_id") or evaluation.get(
        "effective_scenario_sha256"
    ) != case.get("effective_scenario_sha256"):
        errors.append("admissibility evidence evaluation does not bind the case inputs")
    replay_receipt = case.get("replay_receipt")
    replay_revision = (
        replay_receipt.get("replay_revision") if isinstance(replay_receipt, Mapping) else None
    )
    if evaluation.get("source_revision") != replay_revision:
        errors.append("admissibility evidence evaluation is not from the admitted replay revision")
    return evaluation, errors


def _validate_admissibility_evaluation(
    case: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    verdict: str,
    *,
    corpus_root: Path | None,
) -> list[str]:
    errors = []
    if corpus_root is None:
        errors.append("positive feasibility verdict requires verified replay artifacts")
    else:
        state = _evaluation_state(evaluation, case=case, corpus_root=corpus_root)
        if state.get("status") != "solved":
            errors.append(
                "admissibility evidence evaluation is not a valid successful replay"
                + (
                    f": {', '.join(state.get('reason_codes', []))}"
                    if state.get("reason_codes")
                    else ""
                )
            )
    if verdict == "planner_specific_failure":
        errors.extend(_planner_specific_failure_evidence_errors(case, evaluation))
    return errors


def _planner_specific_failure_evidence_errors(
    case: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> list[str]:
    target = case.get("target_planner")
    target = target if isinstance(target, Mapping) else {}
    replay_receipt = case.get("replay_receipt")
    projection = (
        replay_receipt.get("selected_projection") if isinstance(replay_receipt, Mapping) else None
    )
    outcome = projection.get("outcome") if isinstance(projection, Mapping) else None
    errors = []
    if evaluation.get("planner_id") == target.get("planner_id"):
        errors.append("planner-specific failure evidence must use a reference planner")
    if (
        not isinstance(outcome, Mapping)
        or outcome.get("route_complete") is not False
        or not (outcome.get("collision_event") is True or outcome.get("timeout_event") is True)
    ):
        errors.append("planner-specific failure verdict lacks a target-planner failure")
    return errors


def _validate_case_replay_binding(case: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[str]:
    errors = _validate_replay_revision_and_count(receipt)
    replay_revision = receipt.get("replay_revision")
    projection = receipt.get("selected_projection")
    if not isinstance(projection, dict) or not _valid_projection_digest(receipt, projection):
        errors.append("selected replay projection or digest is missing")
    elif not _projection_binds_case(case, receipt, projection, replay_revision):
        errors.append("selected replay projection does not bind case and target planner")
    elif not _projection_events_are_consistent(projection, receipt, replay_revision):
        errors.append("selected replay event identity is invalid or contradictory")
    errors.extend(_validate_case_replay_inputs(case, receipt))
    return errors


def _validate_replay_revision_and_count(receipt: Mapping[str, Any]) -> list[str]:
    errors = []
    target_revision = receipt.get("target_revision")
    replay_revision = receipt.get("replay_revision")
    if (
        receipt.get("target_and_replay_revision_match") is not True
        or not _is_full_git_revision(target_revision)
        or replay_revision != target_revision
    ):
        errors.append("target and replay revisions do not match")
    minimum_count = (
        2 if receipt.get("verification_status") == "repeated_current_revision_match" else 1
    )
    replay_count = receipt.get("replay_count")
    if (
        isinstance(replay_count, bool)
        or not isinstance(replay_count, int)
        or replay_count < minimum_count
    ):
        errors.append("replay count is incomplete for its verification status")
    artifacts = receipt.get("replay_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < minimum_count:
        errors.append("replay artifact inventory is incomplete")
    return errors


def _valid_projection_digest(receipt: Mapping[str, Any], projection: Mapping[str, Any]) -> bool:
    digest = receipt.get("selected_projection_sha256")
    return (
        _is_sha256(digest)
        and hashlib.sha256(_stable_json(projection).encode("utf-8")).hexdigest() == digest
    )


def _projection_binds_case(
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    projection: Mapping[str, Any],
    replay_revision: Any,
) -> bool:
    return (
        projection.get("scenario_id") == case.get("scenario_id")
        and projection.get("seed") == case.get("scenario_seed")
        and projection.get("planner_id") == receipt.get("planner_id")
        and projection.get("planner_config_identity") == receipt.get("planner_config_identity")
        and projection.get("source_revision") == replay_revision
        and isinstance(projection.get("outcome"), dict)
        and isinstance(projection.get("metrics"), dict)
        and isinstance(projection.get("selected_event_identity"), dict)
    )


def _projection_events_are_consistent(
    projection: Mapping[str, Any], receipt: Mapping[str, Any], replay_revision: Any
) -> bool:
    event_identity = projection["selected_event_identity"]
    outcome = projection["outcome"]
    return (
        event_identity.get("scenario_id") == projection.get("scenario_id")
        and event_identity.get("seed") == projection.get("seed")
        and event_identity.get("planner_id") == receipt.get("planner_id")
        and event_identity.get("source_revision") == replay_revision
        and event_identity.get("invalid_run") is False
        and event_identity.get("exact_events")
        == {
            "collision": outcome.get("collision_event"),
            "goal_reached": outcome.get("route_complete"),
            "timeout": outcome.get("timeout_event"),
        }
    )


def _validate_case_replay_inputs(case: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[str]:
    errors = []
    if receipt.get("effective_scenario_sha256") != case.get("effective_scenario_sha256"):
        errors.append("replay receipt does not bind effective scenario identity")
    inputs = case.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    if receipt.get("scenario_input_sha256") != inputs.get("scenario_sha256"):
        errors.append("replay receipt does not bind scenario input digest")
    if receipt.get("route_overrides_sha256") != inputs.get("route_overrides_sha256"):
        errors.append("replay receipt does not bind route input digest")
    if receipt.get("map_assets") != _map_asset_identity(inputs.get("map_assets")):
        errors.append("replay receipt does not bind map asset digests")
    return errors


def _validate_case_discovery(case: Mapping[str, Any]) -> list[str]:
    errors = []
    discovery = case.get("discovery")
    if not isinstance(discovery, dict):
        errors.append("discovery provenance, candidate parameters, or objective is missing")
        return errors
    candidate_parameters = discovery.get("candidate_parameters")
    objective = discovery.get("objective")
    search_source = discovery.get("search_source")
    if not isinstance(candidate_parameters, dict) or not candidate_parameters:
        errors.append("discovery candidate parameters are missing")
    if (
        not isinstance(objective, dict)
        or not isinstance(objective.get("name"), str)
        or not objective.get("name")
        or objective.get("direction") not in {"maximize", "minimize"}
    ):
        errors.append("discovery objective identity or direction is missing")
    if (
        not isinstance(search_source, dict)
        or not isinstance(search_source.get("kind"), str)
        or not search_source.get("kind")
        or not (
            _is_full_git_revision(search_source.get("historical_source_revision"))
            or _is_full_git_revision(search_source.get("source_revision"))
        )
        or not isinstance(search_source.get("search_settings"), dict)
    ):
        errors.append("discovery search source revision or settings are incomplete")
    criticality = discovery.get("criticality", discovery.get("failure_attribution"))
    if not isinstance(criticality, dict) and not (
        isinstance(objective, dict)
        and any(key in objective for key in ("criticality", "archived_value", "value"))
    ):
        errors.append("discovery criticality or failure attribution is missing")
    return errors


def _validate_case_inputs(case: Mapping[str, Any]) -> list[str]:
    errors = []
    inputs = case.get("inputs")
    if (
        not isinstance(inputs, dict)
        or not _is_sha256(inputs.get("scenario_sha256"))
        or not _is_sha256(inputs.get("route_overrides_sha256"))
    ):
        errors.append("materialized scenario/route digest is missing")
    assets = inputs.get("map_assets") if isinstance(inputs, dict) else None
    if not isinstance(assets, list) or not assets:
        errors.append("resolved map asset digests are missing")
    else:
        roles = [item.get("role") for item in assets if isinstance(item, dict)]
        if len(roles) != len(assets) or len(roles) != len(set(roles)):
            errors.append("resolved map asset roles are malformed or duplicated")
        if set(roles) not in ({"map"}, {"map", "map_registry"}):
            errors.append("resolved map asset inventory is incomplete")
        for item in assets:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or not isinstance(item.get("source_path"), str)
                or not item["source_path"]
                or not _is_sha256(item.get("sha256"))
            ):
                errors.append("resolved map asset receipt is incomplete")
                break
    return errors


def _map_asset_identity(assets: Any) -> list[dict[str, str]]:
    if not isinstance(assets, list):
        return []
    return sorted(
        [
            {"role": item["role"], "sha256": item["sha256"]}
            for item in assets
            if isinstance(item, Mapping)
            and isinstance(item.get("role"), str)
            and isinstance(item.get("sha256"), str)
        ],
        key=lambda item: item["role"],
    )


def _validate_case_corpus_evidence(case: Mapping[str, Any], corpus_root: Path) -> list[str]:
    source_evidence = case.get("source_evidence")
    files = source_evidence.get("corpus_files") if isinstance(source_evidence, dict) else None
    if not isinstance(files, list) or not files:
        return ["corpus custody inventory is missing"]
    for receipt in files:
        if (
            not isinstance(receipt, dict)
            or not _safe_bundle_relative_path(receipt.get("path"))
            or not _is_sha256(receipt.get("sha256"))
        ):
            return ["corpus custody inventory receipt is malformed"]
        try:
            _verify_corpus_artifact(corpus_root, receipt["path"], receipt["sha256"])
        except CorpusError as exc:
            return [f"corpus custody inventory verification failed: {exc}"]
    return []


def _validate_case_admission_replay(case: Mapping[str, Any], corpus_root: Path) -> list[str]:
    receipt = case.get("replay_receipt")
    if not isinstance(receipt, Mapping):
        return ["admission replay receipt is missing"]
    if receipt.get("schema_version") == CASE_ADMISSION_REPLAY_SCHEMA_VERSION:
        return _validate_new_admission_replay(case, receipt, corpus_root)
    if receipt.get("schema_version") == LEGACY_CASE_ADMISSION_REPLAY_SCHEMA_VERSION:
        if isinstance(receipt.get("artifact_receipts"), list):
            return _validate_legacy_v1_admission_replay(case, receipt, corpus_root)
        # Historical v1 receipts without row-level receipts keep the old validator.
        return _validate_legacy_admission_replay(receipt, corpus_root)
    return _validate_legacy_admission_replay(receipt, corpus_root)


def _validate_new_admission_replay(
    case: Mapping[str, Any], receipt: Mapping[str, Any], corpus_root: Path
) -> list[str]:
    artifact_receipts = receipt.get("artifact_receipts")
    replay_artifacts = receipt.get("replay_artifacts")
    replay_count = receipt.get("replay_count")
    if (
        not isinstance(artifact_receipts, list)
        or len(artifact_receipts) != replay_count
        or not artifact_receipts
        or not isinstance(replay_artifacts, list)
        or len(replay_artifacts) != replay_count
    ):
        return ["admission replay artifact receipts are incomplete"]
    errors = []
    run_ids: list[str] = []
    artifact_paths: list[str] = []
    selected_projection = receipt.get("selected_projection")
    input_binding_statuses: list[str] = []
    for inventory_row, artifact_receipt in zip(replay_artifacts, artifact_receipts, strict=True):
        pair_errors, run_id, artifact_path, status = _validate_admission_replay_pair(
            case,
            inventory_row,
            artifact_receipt,
            selected_projection,
            corpus_root,
        )
        errors.extend(pair_errors)
        if run_id is not None:
            run_ids.append(run_id)
        if artifact_path is not None:
            artifact_paths.append(artifact_path)
        if status is not None:
            input_binding_statuses.append(status)
    errors.extend(
        _validate_admission_replay_set(receipt, run_ids, artifact_paths, input_binding_statuses)
    )
    return errors


def _validate_legacy_v1_admission_replay(
    case: Mapping[str, Any], receipt: Mapping[str, Any], corpus_root: Path
) -> list[str]:
    """Read v1 row receipts while treating their missing direct input binding as unknown."""
    artifacts = receipt.get("replay_artifacts")
    artifact_receipts = receipt.get("artifact_receipts")
    count = receipt.get("replay_count")
    if (
        not isinstance(artifacts, list)
        or not isinstance(artifact_receipts, list)
        or len(artifacts) != count
        or len(artifact_receipts) != count
        or not artifacts
    ):
        return ["legacy admission replay artifact inventory is incomplete"]
    projection = receipt.get("selected_projection")
    if not isinstance(projection, Mapping):
        return ["legacy admission selected projection is missing"]
    errors = []
    artifact_paths: list[str] = []
    for inventory_row, artifact_receipt in zip(artifacts, artifact_receipts, strict=True):
        pair_errors, artifact_path = _validate_legacy_v1_admission_pair(
            case, inventory_row, artifact_receipt, projection, corpus_root
        )
        errors.extend(pair_errors)
        if artifact_path is not None:
            artifact_paths.append(artifact_path)
    if len(artifact_paths) != len(set(artifact_paths)):
        errors.append("legacy admission replay artifact paths must be unique")
    return errors


def _validate_legacy_v1_admission_pair(
    case: Mapping[str, Any],
    inventory_row: Any,
    artifact_receipt: Any,
    projection: Mapping[str, Any],
    corpus_root: Path,
) -> tuple[list[str], str | None]:
    if not isinstance(inventory_row, Mapping) or not isinstance(artifact_receipt, Mapping):
        return ["legacy admission replay artifact row is malformed"], None
    artifact_path = artifact_receipt.get("artifact_path")
    inventory_path = inventory_row.get("path")
    normalized_inventory_path = inventory_path
    if isinstance(inventory_path, str) and inventory_path != artifact_path:
        case_relative_path = f"cases/{case.get('case_id')}/{inventory_path}"
        normalized_inventory_path = (
            case_relative_path if case_relative_path == artifact_path else inventory_path
        )
    artifact_sha256 = artifact_receipt.get("artifact_sha256")
    inventory_sha256 = inventory_row.get("sha256", inventory_row.get("normalized_bundle_sha256"))
    errors = []
    if normalized_inventory_path != artifact_path or inventory_sha256 != artifact_sha256:
        errors.append("legacy replay inventory row does not match its artifact receipt")
    inventory_event_identity = inventory_row.get("selected_event_identity")
    if inventory_event_identity is not None and inventory_event_identity != artifact_receipt.get(
        "selected_event_identity"
    ):
        errors.append("legacy replay inventory event identity differs from its receipt")
    if not isinstance(artifact_path, str):
        errors.append("legacy replay artifact path is missing")
        return errors, None
    item = _admission_replay_observation(case, artifact_receipt)
    artifact_errors = _validate_replay_artifact(item, case, artifact_receipt, corpus_root)
    errors.extend(
        error for error in artifact_errors if error != "replay_receipt_input_binding_missing"
    )
    if _admission_projection_from_receipt(artifact_receipt) != projection:
        errors.append("selected projection differs from a verified legacy replay artifact")
    return errors, artifact_path


def _case_replay_input_binding_status(case: Mapping[str, Any]) -> str:
    receipt = case.get("replay_receipt")
    if (
        isinstance(receipt, Mapping)
        and receipt.get("schema_version") == CASE_ADMISSION_REPLAY_SCHEMA_VERSION
        and receipt.get("input_binding_status") in {"bound", "unknown_historical"}
    ):
        return str(receipt["input_binding_status"])
    return "unknown_legacy"


def _validate_admission_replay_pair(
    case: Mapping[str, Any],
    inventory_row: Any,
    artifact_receipt: Any,
    selected_projection: Any,
    corpus_root: Path,
) -> tuple[list[str], str | None, str | None, str | None]:
    errors = []
    if not isinstance(artifact_receipt, Mapping):
        return ["admission replay artifact receipt is malformed"], None, None, None
    if not isinstance(inventory_row, Mapping):
        return ["admission replay artifact inventory row is malformed"], None, None, None
    expected_inventory = {
        "path": artifact_receipt.get("artifact_path"),
        "sha256": artifact_receipt.get("artifact_sha256"),
        "run_id": artifact_receipt.get("run_id"),
        "selected_event_identity": artifact_receipt.get("selected_event_identity"),
    }
    if any(inventory_row.get(field) != value for field, value in expected_inventory.items()):
        errors.append("replay artifact inventory row does not match its artifact receipt")
    run_id = artifact_receipt.get("run_id")
    artifact_path = artifact_receipt.get("artifact_path")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("admission replay artifact receipt run_id is missing")
        run_id = None
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        errors.append("admission replay artifact receipt path is missing")
        artifact_path = None
    artifact_binding = artifact_receipt.get("input_binding")
    status = artifact_binding.get("status") if isinstance(artifact_binding, Mapping) else None
    if status not in {"bound", "unknown_historical"}:
        errors.append("admission replay artifact input-binding status is unsupported")
        status = None
    item = _admission_replay_observation(case, artifact_receipt)
    errors.extend(_validate_replay_artifact(item, case, artifact_receipt, corpus_root))
    if not isinstance(selected_projection, Mapping) or (
        _admission_projection_from_receipt(artifact_receipt) != selected_projection
    ):
        errors.append("selected projection differs from a verified replay artifact")
    return errors, run_id, artifact_path, status


def _validate_admission_replay_set(
    receipt: Mapping[str, Any],
    run_ids: list[str],
    artifact_paths: list[str],
    input_binding_statuses: list[str],
) -> list[str]:
    errors = []
    if len(run_ids) != len(set(run_ids)):
        errors.append("admission replay artifacts must have distinct run IDs")
    if len(artifact_paths) != len(set(artifact_paths)):
        errors.append("admission replay artifact paths must be unique")
    if len(set(input_binding_statuses)) != 1:
        errors.append("admission replay artifacts have mixed input-binding status")
    expected_binding_status = (
        input_binding_statuses[0] if len(set(input_binding_statuses)) == 1 else None
    )
    if receipt.get("input_binding_status") != expected_binding_status:
        errors.append("admission replay input-binding status differs from its receipts")
    return errors


def _admission_projection_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": receipt.get("scenario_id"),
        "seed": receipt.get("scenario_seed"),
        "planner_id": receipt.get("planner_id"),
        "planner_config_identity": receipt.get("planner_config_identity"),
        "source_revision": receipt.get("source_revision"),
        "outcome": receipt.get("outcome"),
        "termination_reason": receipt.get("termination_reason"),
        "metrics": receipt.get("metrics"),
        "selected_event_identity": receipt.get("selected_event_identity"),
    }


def _validate_legacy_admission_replay(receipt: Mapping[str, Any], corpus_root: Path) -> list[str]:
    if receipt.get("verification_status") != "repeated_current_revision_match":
        return ["unsupported admission replay receipt schema"]
    replays = receipt.get("replay_artifacts")
    if not isinstance(replays, list) or len(replays) != receipt.get("replay_count"):
        return ["admission replay artifact inventory is incomplete"]
    selected_projection = receipt.get("selected_projection")
    if not isinstance(selected_projection, Mapping):
        return ["legacy admission selected projection is missing"]
    errors = []
    for replay in replays:
        if not isinstance(replay, Mapping):
            errors.append("admission replay artifact row is malformed")
            continue
        errors.extend(_validate_legacy_replay_artifact(replay, selected_projection, corpus_root))
    return errors


def _validate_legacy_replay_artifact(
    replay: Mapping[str, Any],
    selected_projection: Mapping[str, Any],
    corpus_root: Path,
) -> list[str]:
    errors = []
    for path_key, digest_key in (
        ("path", "normalized_bundle_sha256"),
        ("provenance_path", "provenance_sha256_normalized"),
    ):
        try:
            artifact = _resolve_corpus_artifact(replay.get(path_key), corpus_root)
        except CorpusError as exc:
            errors.append(f"admission replay artifact is unavailable: {exc}")
            continue
        if _sha256_file(artifact) != replay.get(digest_key):
            errors.append(f"admission replay artifact digest differs: {path_key}")
            continue
        if path_key == "path":
            errors.extend(_validate_legacy_artifact_projection(artifact, selected_projection))
    return errors


def _validate_legacy_artifact_projection(
    artifact: Path, selected_projection: Mapping[str, Any]
) -> list[str]:
    try:
        episode = _read_single_jsonl_record(artifact)
        artifact_projection = _admission_projection_from_episode(episode, selected_projection)
    except CorpusError as exc:
        return [f"legacy admission replay artifact is invalid: {exc}"]
    consistency_errors = _replay_episode_status_projection_errors(episode)
    consistency_errors.extend(_replay_outcome_metric_consistency_errors(episode))
    consistency_errors.extend(_validate_legacy_replay_execution_evidence(episode))
    if artifact_projection != selected_projection:
        consistency_errors.append("selected projection differs from a verified replay artifact")
    return consistency_errors


def _validate_legacy_replay_execution_evidence(record: Mapping[str, Any]) -> list[str]:
    """Apply current execution eligibility checks when a v1 receipt lacks row metadata."""
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    kinematics = metadata.get("planner_kinematics")
    kinematics = kinematics if isinstance(kinematics, Mapping) else {}
    execution_mode = kinematics.get("execution_mode")
    readiness_status = {
        "native": "native",
        "adapter": "adapter",
        "mixed": "adapter",
    }.get(execution_mode)
    return _validate_replay_execution_evidence(
        {
            "execution_mode": execution_mode,
            "readiness_status": readiness_status,
            "availability_status": "available",
            "fallback_or_degraded": False,
        },
        record,
    )


def _admission_projection_from_episode(
    episode: Mapping[str, Any], selected_projection: Mapping[str, Any]
) -> dict[str, Any]:
    outcome = episode.get("outcome")
    metrics = episode.get("metrics")
    metadata = episode.get("algorithm_metadata")
    if not isinstance(outcome, Mapping) or not isinstance(metrics, Mapping):
        raise CorpusError("episode outcome or metrics are missing")
    if not isinstance(metadata, Mapping):
        raise CorpusError("episode planner metadata is missing")
    selected_outcome = selected_projection.get("outcome")
    selected_metrics = selected_projection.get("metrics")
    if not isinstance(selected_outcome, Mapping) or not isinstance(selected_metrics, Mapping):
        raise CorpusError("selected projection outcome or metrics are malformed")
    return {
        "scenario_id": episode.get("scenario_id"),
        "seed": episode.get("seed"),
        "planner_id": episode.get("algo"),
        "planner_config_identity": metadata.get("config_hash"),
        "source_revision": episode.get("git_hash"),
        "outcome": {key: outcome.get(key) for key in selected_outcome},
        "termination_reason": episode.get("termination_reason"),
        "metrics": {key: metrics.get(key) for key in selected_metrics},
        "selected_event_identity": _selected_event_identity(episode),
    }


def _admission_replay_observation(
    case: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "effective_scenario_sha256": case.get("effective_scenario_sha256"),
        "planner_id": receipt.get("planner_id"),
        "planner_config_identity": receipt.get("planner_config_identity"),
        "source_revision": receipt.get("source_revision"),
        "episode_sha256": receipt.get("episode_sha256"),
        "outcome": receipt.get("outcome"),
        "termination_reason": receipt.get("termination_reason"),
        "metrics": receipt.get("metrics"),
        "execution_mode": receipt.get("execution_mode"),
        "readiness_status": receipt.get("readiness_status"),
        "availability_status": receipt.get("availability_status"),
        "fallback_or_degraded": receipt.get("fallback_or_degraded"),
    }


def _validate_evaluation(item: Mapping[str, Any]) -> list[str]:
    errors = _validate_evaluation_identity(item)
    errors.extend(_validate_evaluation_execution_metadata(item))
    evidence_status = item.get("evidence_status")
    if evidence_status not in {"complete", "failed", "partial", "missing", "unknown"}:
        errors.append("evidence_status_invalid")
    if evidence_status == "complete":
        errors.extend(_validate_complete_evaluation(item))
    else:
        errors.extend(_validate_incomplete_evaluation(item))
    return errors


def _evaluation_digest(item: Mapping[str, Any]) -> str:
    identity = {
        key: item.get(key)
        for key in (
            "case_id",
            "effective_scenario_sha256",
            "planner_id",
            "planner_config_identity",
            "source_revision",
            "episode_sha256",
            "outcome",
            "termination_reason",
            "metrics",
            "execution_mode",
            "readiness_status",
            "availability_status",
            "fallback_or_degraded",
            "evidence_status",
            "error",
            "replay_receipt",
        )
    }
    return hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()


def _validate_evaluation_identity(item: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required_text = (
        "case_id",
        "effective_scenario_sha256",
        "planner_id",
        "planner_config_identity",
        "source_revision",
    )
    for field in required_text:
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field}_missing")
    if not _is_sha256(item.get("effective_scenario_sha256")):
        errors.append("effective_scenario_sha256_invalid")
    return errors


def _validate_replay_artifact(
    item: Mapping[str, Any],
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    corpus_root: Path,
) -> list[str]:
    """Verify a stored one-episode artifact against its receipt, case, and evaluation."""
    errors = _evaluation_receipt_binding_errors(item, case, receipt)
    try:
        _scenario_input_paths(case, corpus_root)
    except (CorpusError, OSError, ValueError, TypeError) as exc:
        errors.append(f"case_input_binding_invalid:{exc}")
    try:
        artifact = _resolve_corpus_artifact(receipt.get("artifact_path"), corpus_root)
    except CorpusError as exc:
        errors.append(f"replay_artifact_path_invalid:{exc}")
        return errors
    artifact_sha256 = _sha256_file(artifact) if artifact.is_file() else None
    if artifact_sha256 is None:
        errors.append("replay_artifact_missing")
        return errors
    if artifact_sha256 != receipt.get("artifact_sha256"):
        errors.append("replay_artifact_checksum_mismatch")
        return errors
    if artifact_sha256 != item.get("episode_sha256"):
        errors.append("episode_sha256_does_not_match_replay_artifact")
        return errors
    try:
        record = _read_single_jsonl_record(artifact)
        event_identity = _selected_event_identity(record)
    except CorpusError as exc:
        errors.append(f"replay_artifact_invalid:{exc}")
        return errors
    errors.extend(_validate_replay_record_projection(item, case, receipt, record, event_identity))
    errors.extend(_validate_replay_input_binding(case, receipt, record, corpus_root))
    errors.extend(_validate_replay_execution_evidence(item, record))
    return errors


def _validate_replay_input_binding(
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    record: Mapping[str, Any],
    corpus_root: Path,
) -> list[str]:
    """Bind exact receipts to captured runtime inputs; retain legacy uncertainty explicitly."""
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    recorded = provenance.get("case_input_identity")
    declared = receipt.get("input_binding")
    if not isinstance(declared, Mapping):
        return ["replay_receipt_input_binding_missing"]
    status = declared.get("status")
    if status == "unknown_historical":
        return _validate_unknown_historical_input_binding(
            case, receipt, declared, recorded, corpus_root
        )
    if status != "bound":
        return ["replay_receipt_input_binding_status_invalid"]
    return _validate_bound_replay_input_binding(case, receipt, declared, recorded, corpus_root)


def _validate_unknown_historical_input_binding(
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    declared: Mapping[str, Any],
    recorded: Any,
    corpus_root: Path,
) -> list[str]:
    if recorded is not None:
        return ["replay_receipt_marks_available_input_binding_unknown"]
    if not isinstance(declared.get("reason"), str) or not declared["reason"].strip():
        return ["replay_receipt_historical_input_uncertainty_reason_missing"]
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return ["replay_receipt_historical_run_id_missing"]
    if declared.get("run_id") != run_id:
        return ["replay_receipt_historical_run_id_mismatch"]
    return _validate_historical_replay_run_binding(case, receipt, corpus_root)


def _validate_bound_replay_input_binding(
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    declared: Mapping[str, Any],
    recorded: Any,
    corpus_root: Path,
) -> list[str]:
    if not isinstance(recorded, Mapping):
        return ["replay_artifact_direct_input_binding_missing"]
    try:
        expected = _case_runtime_input_binding(case, corpus_root)
    except (CorpusError, OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        return [f"replay_artifact_case_input_identity_invalid:{exc}"]
    binding_fields = (
        "schema_version",
        "status",
        "scenario_semantic_sha256",
        "route_overrides_sha256",
        "map_assets",
    )
    errors = []
    for field in binding_fields:
        if recorded.get(field) != expected.get(field):
            errors.append(f"replay_artifact_input_binding_{field}_mismatch")
        if declared.get(field) != recorded.get(field):
            errors.append(f"replay_receipt_input_binding_{field}_mismatch")
    run_id = recorded.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("replay_artifact_input_binding_run_id_missing")
    elif receipt.get("run_id") != run_id or declared.get("run_id") != run_id:
        errors.append("replay_receipt_run_id_mismatch")
    if recorded.get("reason_codes") != []:
        errors.append("replay_artifact_input_binding_has_unresolved_reasons")
    return errors


def _case_runtime_input_binding(case: Mapping[str, Any], corpus_root: Path) -> dict[str, Any]:
    """Recompute semantic scenario and exact route/map input identities from custody."""
    inputs = case.get("inputs")
    inputs = inputs if isinstance(inputs, Mapping) else {}
    paths = _scenario_input_paths(case, corpus_root)
    scenario_document = _load_yaml_object(paths["scenario"], "scenario")
    scenario_rows = scenario_document.get("scenarios")
    if (
        not isinstance(scenario_rows, list)
        or len(scenario_rows) != 1
        or not isinstance(scenario_rows[0], Mapping)
    ):
        raise CorpusError("case scenario input must contain exactly one scenario")
    return {
        "schema_version": EPISODE_INPUT_IDENTITY_SCHEMA,
        "status": "bound",
        "scenario_semantic_sha256": scenario_semantic_sha256(
            scenario_rows[0], seed=int(case["scenario_seed"])
        ),
        "route_overrides_sha256": inputs.get("route_overrides_sha256"),
        "map_assets": _map_asset_identity(inputs.get("map_assets")),
    }


def _validate_historical_replay_run_binding(
    case: Mapping[str, Any], receipt: Mapping[str, Any], corpus_root: Path
) -> list[str]:
    """Verify the historical run ID against its separately retained provenance sidecar."""
    case_receipt = case.get("replay_receipt")
    inventory = case_receipt.get("replay_artifacts") if isinstance(case_receipt, Mapping) else None
    if not isinstance(inventory, list):
        return ["historical_replay_run_provenance_inventory_missing"]
    matching = [
        row
        for row in inventory
        if isinstance(row, Mapping) and row.get("path") == receipt.get("artifact_path")
    ]
    if len(matching) != 1:
        return ["historical_replay_run_provenance_artifact_not_unique"]
    row = matching[0]
    if (
        row.get("sha256") != receipt.get("artifact_sha256")
        or row.get("run_id") != receipt.get("run_id")
        or row.get("selected_event_identity") != receipt.get("selected_event_identity")
    ):
        return ["historical_replay_run_provenance_inventory_mismatch"]
    provenance_path = row.get("provenance_path")
    provenance_sha256 = row.get("provenance_sha256_normalized")
    if not isinstance(provenance_path, str) or not _is_sha256(provenance_sha256):
        return ["historical_replay_run_provenance_receipt_missing"]
    try:
        provenance_file = _resolve_corpus_artifact(provenance_path, corpus_root)
        if _sha256_file(provenance_file) != provenance_sha256:
            return ["historical_replay_run_provenance_checksum_mismatch"]
        provenance = _read_json_object(provenance_file)
    except (CorpusError, OSError, ValueError, TypeError) as exc:
        return [f"historical_replay_run_provenance_invalid:{exc}"]
    run = provenance.get("run")
    if not isinstance(run, Mapping) or run.get("run_id") != receipt.get("run_id"):
        return ["historical_replay_run_id_does_not_match_provenance"]
    return []


def _validate_replay_record_projection(
    item: Mapping[str, Any],
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    record: Mapping[str, Any],
    event_identity: Mapping[str, Any],
) -> list[str]:
    errors = []
    if record.get("scenario_id") != case.get("scenario_id"):
        errors.append("replay_artifact_scenario_id_mismatch")
    if isinstance(record.get("seed"), bool) or record.get("seed") != case.get("scenario_seed"):
        errors.append("replay_artifact_scenario_seed_mismatch")
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if record.get("algo") != item.get("planner_id"):
        errors.append("replay_artifact_planner_id_mismatch")
    if metadata.get("config_hash") != item.get("planner_config_identity"):
        errors.append("replay_artifact_planner_config_mismatch")
    errors.extend(_target_config_snapshot_projection_errors(item, case, metadata))
    if record.get("git_hash") != item.get("source_revision"):
        errors.append("replay_artifact_source_revision_mismatch")
    if not _is_full_git_revision(item.get("source_revision")):
        errors.append("replay_artifact_source_revision_invalid")

    artifact_outcome = record.get("outcome")
    if not isinstance(artifact_outcome, dict) or any(
        artifact_outcome.get(key) != item.get("outcome", {}).get(key)
        for key in ("collision_event", "route_complete", "timeout_event")
    ):
        errors.append("replay_artifact_outcome_mismatch")
    if record.get("termination_reason") != item.get("termination_reason"):
        errors.append("replay_artifact_termination_reason_mismatch")
    errors.extend(_replay_episode_status_projection_errors(record))
    artifact_metrics = record.get("metrics")
    if not isinstance(artifact_metrics, dict) or any(
        artifact_metrics.get(key) != value for key, value in item.get("metrics", {}).items()
    ):
        errors.append("replay_artifact_selected_metrics_mismatch")
    errors.extend(_replay_outcome_metric_consistency_errors(record))
    errors.extend(_validate_replay_event_projection(item, case, receipt, event_identity))
    return errors


def _replay_outcome_metric_consistency_errors(record: Mapping[str, Any]) -> list[str]:
    """Check canonical outcome consistency against every raw replay metric."""
    outcome = record.get("outcome")
    metrics = record.get("metrics")
    if not isinstance(outcome, Mapping) or not isinstance(metrics, Mapping):
        return []
    contradictions = outcome_contradictions(
        termination_reason=str(record.get("termination_reason") or ""),
        outcome=outcome,
        metrics=metrics,
    )
    if not contradictions:
        return []
    return ["replay_artifact_outcome_metric_contradiction: " + "; ".join(contradictions)]


def _target_config_snapshot_projection_errors(
    item: Mapping[str, Any], case: Mapping[str, Any], metadata: Mapping[str, Any]
) -> list[str]:
    target = case.get("target_planner")
    if (
        isinstance(target, Mapping)
        and item.get("planner_id") == target.get("planner_id")
        and item.get("planner_config_identity") == target.get("config_identity")
        and metadata.get("config") != target.get("configuration_snapshot")
    ):
        return ["replay_artifact_target_configuration_snapshot_mismatch"]
    return []


def _replay_episode_status_projection_errors(record: Mapping[str, Any]) -> list[str]:
    termination_reason = record.get("termination_reason")
    if not isinstance(termination_reason, str) or termination_reason not in TERMINATION_REASONS:
        return ["replay_artifact_termination_reason_unsupported"]
    if record.get("status") != status_from_termination_reason(termination_reason):
        return ["replay_artifact_status_termination_mismatch"]
    return []


def _validate_replay_event_projection(
    item: Mapping[str, Any],
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    event_identity: Mapping[str, Any],
) -> list[str]:
    errors = []
    if event_identity != receipt.get("selected_event_identity"):
        errors.append("replay_artifact_event_identity_mismatch")
    expected_event_outcome = {
        "collision": item.get("outcome", {}).get("collision_event"),
        "goal_reached": item.get("outcome", {}).get("route_complete"),
        "timeout": item.get("outcome", {}).get("timeout_event"),
    }
    if event_identity.get("exact_events") != expected_event_outcome:
        errors.append("replay_artifact_event_outcome_mismatch")
    if event_identity.get("scenario_id") != case.get("scenario_id") or event_identity.get(
        "seed"
    ) != case.get("scenario_seed"):
        errors.append("replay_artifact_event_case_identity_mismatch")
    if event_identity.get("planner_id") != item.get("planner_id") or event_identity.get(
        "source_revision"
    ) != item.get("source_revision"):
        errors.append("replay_artifact_event_planner_identity_mismatch")
    return errors


def _validate_replay_execution_evidence(
    item: Mapping[str, Any], record: Mapping[str, Any]
) -> list[str]:
    errors = _replay_episode_validity_errors(record)
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    kinematics = metadata.get("planner_kinematics")
    kinematics = kinematics if isinstance(kinematics, dict) else {}
    actual_mode = kinematics.get("execution_mode")
    if actual_mode != item.get("execution_mode"):
        errors.append("replay_artifact_execution_mode_mismatch")
    expected_readiness = {
        "native": "native",
        "adapter": "adapter",
        "mixed": "adapter",
    }.get(actual_mode)
    if expected_readiness is None:
        errors.append("replay_artifact_execution_mode_unsupported")
    else:
        recorded_readiness = record.get("readiness_status") or metadata.get(
            "readiness_status", metadata.get("benchmark_readiness_status")
        )
        if recorded_readiness is not None and recorded_readiness != expected_readiness:
            errors.append("replay_artifact_readiness_mode_mismatch")
        if item.get("readiness_status") != expected_readiness:
            errors.append("replay_artifact_readiness_not_benchmark_capable")

    benchmark_availability = record.get("benchmark_availability")
    benchmark_availability = (
        benchmark_availability if isinstance(benchmark_availability, dict) else {}
    )
    recorded_availability = (
        record.get("availability_status")
        or benchmark_availability.get("availability_status")
        or metadata.get("availability_status")
    )
    if recorded_availability is not None and recorded_availability != "available":
        errors.append("replay_artifact_availability_not_available")
    elif metadata.get("status") not in {"ok", "available"}:
        errors.append("replay_artifact_planner_status_not_available")
    if item.get("availability_status") != "available":
        errors.append("replay_artifact_evaluation_availability_not_available")

    actual_fallback_or_degraded = _replay_fallback_marker(record, metadata)
    if actual_fallback_or_degraded is None:
        errors.append("replay_artifact_fallback_status_missing")
    elif item.get("fallback_or_degraded") is not actual_fallback_or_degraded:
        errors.append("replay_artifact_fallback_status_mismatch")
    return errors


def _replay_episode_validity_errors(record: Mapping[str, Any]) -> list[str]:
    if _replay_episode_is_invalid(record):
        return ["replay_artifact_invalid_run"]
    return []


def _replay_episode_is_invalid(record: Mapping[str, Any]) -> bool:
    status = record.get("status")
    if not isinstance(status, str):
        return True
    event_ledger = record.get("event_ledger")
    event_ledger = event_ledger if isinstance(event_ledger, dict) else {}
    exact_events = event_ledger.get("exact_events")
    exact_events = exact_events if isinstance(exact_events, dict) else {}
    return (
        status in {"invalid", "error"}
        or record.get("termination_reason") == "error"
        or exact_events.get("invalid_run") is not False
    )


def _replay_fallback_marker(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool | None:
    integrity = record.get("integrity")
    integrity = integrity if isinstance(integrity, dict) else {}
    effective_view = integrity.get("effective_view", {})
    effective_view = effective_view if isinstance(effective_view, dict) else {}
    # Preserve marker presence and raw types for the canonical detector. In particular,
    # dropping a present None boolean or fallback_reason would turn malformed evidence clean.
    runtime_metadata = {
        str(key): value
        for key, value in metadata.items()
        if str(key) not in {"config", "planner_contract", "safety_shield_contract"}
    }
    execution_fields = {
        "algorithm_metadata": runtime_metadata,
        "integrity": {"effective_view": effective_view},
    }
    for key in (
        "benchmark_availability",
        "readiness_status",
        "availability_status",
        "fallback_or_degraded",
    ):
        if key in record:
            execution_fields[key] = record[key]
    if runtime_fallback_or_degraded_marker(execution_fields) is not None:
        return True
    explicit_marker = record.get("fallback_or_degraded", metadata.get("fallback_or_degraded"))
    degraded = effective_view.get("degraded")
    if isinstance(degraded, bool):
        return degraded
    if isinstance(explicit_marker, bool):
        return explicit_marker
    return None


def _evaluation_receipt_binding_errors(
    item: Mapping[str, Any], case: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[str]:
    if not isinstance(receipt, Mapping):
        return ["replay_receipt_missing"]
    errors = []
    if receipt.get("schema_version") not in {
        EVALUATION_REPLAY_RECEIPT_SCHEMA_VERSION,
        LEGACY_EVALUATION_REPLAY_RECEIPT_SCHEMA_VERSION,
    }:
        errors.append("replay_receipt_schema_invalid")
    if receipt.get("verification_status") != "artifact_projection_match":
        errors.append("replay_receipt_verification_status_invalid")
    errors.extend(_validate_replay_receipt_bindings(item, case, receipt))
    errors.extend(_validate_replay_receipt_artifact_fields(item, receipt))
    errors.extend(_validate_replay_receipt_case_fields(receipt))
    return errors


def _validate_replay_receipt_bindings(
    item: Mapping[str, Any], case: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[str]:
    inputs = case.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    expected = {
        "case_id": item.get("case_id"),
        "effective_scenario_sha256": item.get("effective_scenario_sha256"),
        "scenario_id": case.get("scenario_id"),
        "scenario_seed": case.get("scenario_seed"),
        "scenario_input_sha256": inputs.get("scenario_sha256"),
        "route_overrides_sha256": inputs.get("route_overrides_sha256"),
        "map_assets": _map_asset_identity(inputs.get("map_assets")),
        "planner_id": item.get("planner_id"),
        "planner_config_identity": item.get("planner_config_identity"),
        "source_revision": item.get("source_revision"),
        "episode_sha256": item.get("episode_sha256"),
        "outcome": item.get("outcome"),
        "termination_reason": item.get("termination_reason"),
        "metrics": item.get("metrics"),
        "execution_mode": item.get("execution_mode"),
        "readiness_status": item.get("readiness_status"),
        "availability_status": item.get("availability_status"),
        "fallback_or_degraded": item.get("fallback_or_degraded"),
    }
    errors = []
    for field, value in expected.items():
        if receipt.get(field) != value:
            errors.append(f"replay_receipt_{field}_mismatch")
    return errors


def _validate_replay_receipt_artifact_fields(
    item: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[str]:
    errors = []
    if not _is_sha256(receipt.get("artifact_sha256")):
        errors.append("replay_receipt_artifact_sha256_invalid")
    if receipt.get("artifact_sha256") != item.get("episode_sha256"):
        errors.append("replay_receipt_episode_sha256_mismatch")
    for field in ("scenario_input_sha256", "route_overrides_sha256"):
        if not _is_sha256(receipt.get(field)):
            errors.append(f"replay_receipt_{field}_invalid")
    if not isinstance(receipt.get("map_assets"), list) or not receipt["map_assets"]:
        errors.append("replay_receipt_map_assets_missing")
    if not isinstance(receipt.get("artifact_path"), str) or not receipt["artifact_path"]:
        errors.append("replay_receipt_artifact_path_missing")
    return errors


def _validate_replay_receipt_case_fields(receipt: Mapping[str, Any]) -> list[str]:
    errors = []
    selected_event_identity = receipt.get("selected_event_identity")
    if not isinstance(selected_event_identity, dict):
        errors.append("replay_receipt_selected_event_identity_missing")
    else:
        episode_status = selected_event_identity.get("episode_status")
        if not isinstance(episode_status, str) or not episode_status.strip():
            errors.append("replay_receipt_episode_status_missing")
        if not isinstance(selected_event_identity.get("invalid_run"), bool):
            errors.append("replay_receipt_invalid_run_missing")
    if not isinstance(receipt.get("scenario_id"), str) or not receipt["scenario_id"].strip():
        errors.append("replay_receipt_scenario_id_missing")
    if not isinstance(receipt.get("scenario_seed"), int) or isinstance(
        receipt.get("scenario_seed"), bool
    ):
        errors.append("replay_receipt_scenario_seed_invalid")
    return errors


def _selected_event_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    ledger = record.get("event_ledger")
    if not isinstance(ledger, dict):
        raise CorpusError("replay episode event ledger is missing")
    exact_events = ledger.get("exact_events")
    event_fields = ("collision", "goal_reached", "timeout")
    if not isinstance(exact_events, dict) or any(
        not isinstance(exact_events.get(key), bool) for key in event_fields
    ):
        raise CorpusError("replay episode exact event identities are incomplete")
    if not isinstance(exact_events.get("invalid_run"), bool):
        raise CorpusError("replay episode invalid-run event identity is missing")
    episode_status = record.get("status")
    if not isinstance(episode_status, str) or not episode_status.strip():
        raise CorpusError("replay episode status is missing")
    return {
        "schema_version": ledger.get("schema_version"),
        "scenario_id": ledger.get("scenario_id"),
        "seed": ledger.get("seed"),
        "planner_id": ledger.get("planner"),
        "source_revision": ledger.get("software_commit"),
        "episode_status": episode_status,
        "invalid_run": exact_events["invalid_run"],
        "exact_events": {key: exact_events[key] for key in event_fields},
    }


def _resolve_corpus_artifact(path_value: Any, corpus_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise CorpusError("replay artifact path must be a non-empty relative path")
    relative = PurePosixPath(path_value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CorpusError(f"unsafe replay artifact path: {path_value}")
    path = (corpus_root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(corpus_root.resolve())
    except ValueError as exc:
        raise CorpusError(f"replay artifact path escapes corpus root: {path_value}") from exc
    if not path.is_file():
        raise CorpusError(f"replay artifact is missing: {path_value}")
    return path


def _validate_evaluation_execution_metadata(item: Mapping[str, Any]) -> list[str]:
    errors = []
    evidence_status = item.get("evidence_status")
    if not isinstance(item.get("metrics"), dict):
        errors.append("metrics_missing")
    for field in ("execution_mode", "readiness_status", "availability_status"):
        if not isinstance(item.get(field), str) or not item[field]:
            errors.append(f"{field}_missing")
    if not isinstance(item.get("fallback_or_degraded"), bool):
        errors.extend(
            []
            if evidence_status != "complete" and item.get("fallback_or_degraded") is None
            else ["fallback_or_degraded_missing"]
        )
    if "error" not in item:
        errors.append("error_status_missing")
    elif item["error"] is not None and not isinstance(item["error"], str):
        errors.append("error_status_invalid")
    return errors


def _validate_complete_evaluation(item: Mapping[str, Any]) -> list[str]:
    errors = []
    if not _is_full_git_revision(item.get("source_revision")):
        errors.append("source_revision_invalid")
    if not _is_sha256(item.get("episode_sha256")):
        errors.append("episode_sha256_invalid")
    if (
        not isinstance(item.get("termination_reason"), str)
        or not item["termination_reason"].strip()
    ):
        errors.append("termination_reason_missing")
    outcome = item.get("outcome")
    if not isinstance(outcome, dict) or any(
        not isinstance(outcome.get(field), bool)
        for field in ("collision_event", "route_complete", "timeout_event")
    ):
        errors.append("canonical_outcome_flags_missing")
    receipt = item.get("replay_receipt")
    if isinstance(receipt, dict):
        errors.extend(
            _evaluation_receipt_binding_errors(
                item,
                {
                    "scenario_id": receipt.get("scenario_id"),
                    "scenario_seed": receipt.get("scenario_seed"),
                    "inputs": {
                        "scenario_sha256": receipt.get("scenario_input_sha256"),
                        "route_overrides_sha256": receipt.get("route_overrides_sha256"),
                        "map_assets": receipt.get("map_assets"),
                    },
                },
                receipt,
            )
        )
    return errors


def _validate_incomplete_evaluation(item: Mapping[str, Any]) -> list[str]:
    errors = []
    episode_hash = item.get("episode_sha256")
    if episode_hash is not None and not _is_sha256(episode_hash):
        errors.append("episode_sha256_invalid")
    termination = item.get("termination_reason")
    if termination is not None and (not isinstance(termination, str) or not termination.strip()):
        errors.append("termination_reason_invalid")
    outcome = item.get("outcome")
    if outcome is not None and (
        not isinstance(outcome, dict)
        or any(
            outcome.get(field) is not None and not isinstance(outcome.get(field), bool)
            for field in ("collision_event", "route_complete", "timeout_event")
        )
    ):
        errors.append("canonical_outcome_flags_invalid")
    if not isinstance(item.get("error"), str) or not item["error"].strip():
        errors.append("incomplete_evaluation_requires_reason")
    return errors


def _evaluation_state(
    item: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    corpus_root: Path | None,
) -> dict[str, Any]:
    reasons = _validate_evaluation(item)
    if reasons:
        return {"status": "unknown", "reason_codes": reasons}
    if item["evidence_status"] != "complete":
        return {
            "status": "unknown",
            "reason_codes": [f"evaluation_evidence_{item['evidence_status']}"],
        }
    if not isinstance(item.get("replay_receipt"), Mapping):
        return {"status": "unknown", "reason_codes": ["replay_receipt_missing"]}
    if corpus_root is None:
        return {"status": "unknown", "reason_codes": ["replay_artifact_root_required"]}
    replay_errors = _validate_replay_artifact(item, case, item.get("replay_receipt"), corpus_root)
    if replay_errors:
        return {"status": "unknown", "reason_codes": sorted(set(replay_errors))}
    replay_receipt = item["replay_receipt"]
    input_binding = replay_receipt.get("input_binding")
    if (
        replay_receipt.get("schema_version") != EVALUATION_REPLAY_RECEIPT_SCHEMA_VERSION
        or not isinstance(input_binding, Mapping)
        or input_binding.get("status") != "bound"
    ):
        return {
            "status": "unknown",
            "reason_codes": ["replay_input_binding_unknown_historical"],
        }
    reasons = _complete_evaluation_eligibility_errors(item)
    if reasons:
        return {"status": "unknown", "reason_codes": sorted(set(reasons))}
    return _complete_outcome_state(item)


def _complete_evaluation_eligibility_errors(item: Mapping[str, Any]) -> list[str]:
    reasons = []
    expected_readiness = {
        "native": "native",
        "adapter": "adapter",
        "mixed": "adapter",
    }.get(item["execution_mode"])
    if expected_readiness is None:
        reasons.append("execution_mode_not_benchmark_capable")
    elif item["readiness_status"] != expected_readiness:
        reasons.append("readiness_not_benchmark_capable")
    if item["availability_status"] != "available":
        reasons.append("availability_not_available")
    if item["fallback_or_degraded"] is not False:
        reasons.append("fallback_or_degraded_execution")
    if item["error"] is not None:
        reasons.append("evaluation_error")
    metrics = item["metrics"]
    for key in ("success", "collisions", "total_collision_count"):
        if key == "success" and isinstance(metrics.get(key), bool):
            continue
        if _finite_number(metrics.get(key)) is None:
            reasons.append(f"metric_{key}_missing_or_invalid")
    reasons.extend(
        outcome_contradictions(
            termination_reason=item["termination_reason"],
            outcome=item["outcome"],
            metrics=metrics,
        )
    )
    return reasons


def _complete_outcome_state(item: Mapping[str, Any]) -> dict[str, Any]:
    outcome = item["outcome"]
    metrics = item["metrics"]
    if (
        outcome["route_complete"]
        and not outcome["collision_event"]
        and not outcome["timeout_event"]
    ):
        if item["termination_reason"] != "success" or float(metrics["success"]) <= 0.0:
            return {"status": "unknown", "reason_codes": ["success_termination_or_metric_mismatch"]}
        return {"status": "solved", "reason_codes": []}
    return {"status": "unsolved", "reason_codes": []}


def _scenario_input_paths(case: Mapping[str, Any], corpus_root: Path) -> dict[str, Path]:
    inputs = case.get("inputs")
    if not isinstance(inputs, dict):
        raise CorpusError("case has no materialized input paths")
    paths = {}
    for key in ("scenario_path", "route_overrides_path"):
        relative = inputs.get(key)
        if not isinstance(relative, str) or not relative:
            raise CorpusError(f"case input reference missing: {key}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise CorpusError(f"unsafe corpus input path: {relative}")
        path = (corpus_root / Path(*pure.parts)).resolve()
        try:
            path.relative_to(corpus_root.resolve())
        except ValueError as exc:
            raise CorpusError(f"corpus input path escapes root: {relative}") from exc
        if not path.is_file() or _sha256_file(path) != inputs.get(
            "scenario_sha256" if key == "scenario_path" else "route_overrides_sha256"
        ):
            raise CorpusError(f"corpus input missing or checksum mismatch: {relative}")
        paths["scenario" if key == "scenario_path" else "route"] = path
    scenario_document = _load_yaml_object(paths["scenario"], "scenario")
    scenario_rows = scenario_document.get("scenarios")
    if (
        not isinstance(scenario_rows, list)
        or len(scenario_rows) != 1
        or not isinstance(scenario_rows[0], dict)
    ):
        raise CorpusError("corpus case scenario input must contain exactly one scenario")
    scenario = scenario_rows[0]
    if _scenario_seed(scenario) != case.get("scenario_seed"):
        raise CorpusError("corpus case seed differs from its materialized scenario")
    route_payload = _load_yaml_object(paths["route"], "route overrides")
    paths.update(
        _validate_case_map_assets(
            scenario,
            paths["scenario"],
            inputs.get("map_assets"),
            corpus_root,
        )
    )
    if compute_case_effective_scenario_hash(
        scenario, route_payload, inputs.get("map_assets", [])
    ) != case.get("effective_scenario_sha256"):
        raise CorpusError("corpus effective-scenario/map identity differs from its stored inputs")
    return paths


def _validate_case_map_assets(
    scenario: Mapping[str, Any],
    scenario_path: Path,
    assets: Any,
    corpus_root: Path,
) -> dict[str, Path]:
    resolved, metadata = _resolve_case_map_asset_inventory(assets, corpus_root)
    map_id = scenario.get("map_id")
    if isinstance(map_id, str) and map_id.strip():
        _validate_registered_map_reference(map_id, resolved, metadata)
    else:
        _validate_direct_map_reference(scenario, scenario_path, resolved)
    return {f"{role}_asset": path for role, path in resolved.items()}


def _resolve_case_map_asset_inventory(
    assets: Any, corpus_root: Path
) -> tuple[dict[str, Path], dict[str, Mapping[str, Any]]]:
    if not isinstance(assets, list) or not assets:
        raise CorpusError("case map asset receipts are missing")
    resolved: dict[str, Path] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, Mapping):
            raise CorpusError("case map asset receipt is malformed")
        role = asset.get("role")
        relative = asset.get("path")
        if role not in {"map", "map_registry"} or role in resolved:
            raise CorpusError("case map asset roles are invalid or duplicated")
        if not isinstance(relative, str) or not relative:
            raise CorpusError(f"case map asset path is missing: {role}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise CorpusError(f"unsafe corpus map asset path: {relative}")
        path = (corpus_root / Path(*pure.parts)).resolve()
        try:
            path.relative_to(corpus_root.resolve())
        except ValueError as exc:
            raise CorpusError(f"corpus map asset path escapes root: {relative}") from exc
        if not path.is_file() or _sha256_file(path) != asset.get("sha256"):
            raise CorpusError(f"corpus map asset missing or checksum mismatch: {relative}")
        resolved[str(role)] = path
        metadata[str(role)] = asset
    return resolved, metadata


def _validate_registered_map_reference(
    map_id: str,
    resolved: Mapping[str, Path],
    metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(resolved) != {"map", "map_registry"}:
        raise CorpusError("map_id cases require both registry and map bytes")
    registry_path = resolved["map_registry"]
    try:
        registry_document = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CorpusError("stored map registry is unreadable") from exc
    row = _map_registry_entry(registry_document, map_id)
    if row is None:
        raise CorpusError("stored map registry does not contain the scenario map_id")
    declared_path = row.get("path") or row.get("map_file")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise CorpusError("stored map registry entry has no map path")
    pure_map = PurePosixPath(declared_path)
    if pure_map.is_absolute() or ".." in pure_map.parts:
        raise CorpusError("stored map registry path is unsafe")
    expected_map = (registry_path.parent / Path(*pure_map.parts)).resolve()
    if expected_map != resolved["map"].resolve():
        raise CorpusError("stored map file does not match the map registry entry")
    if row.get("source_sha256") != metadata["map"]["sha256"]:
        raise CorpusError("stored map file digest differs from its registry entry")


def _validate_direct_map_reference(
    scenario: Mapping[str, Any], scenario_path: Path, resolved: Mapping[str, Path]
) -> None:
    if set(resolved) != {"map"}:
        raise CorpusError("map_file cases require exactly one resolved map asset")
    map_file = scenario.get("map_file")
    if not isinstance(map_file, str) or not map_file.strip():
        raise CorpusError("scenario has no resolvable map reference")
    declared_map = Path(map_file)
    expected_map = (
        declared_map.resolve()
        if declared_map.is_absolute()
        else (scenario_path.parent / declared_map).resolve()
    )
    if expected_map != resolved["map"].resolve():
        raise CorpusError("stored map bytes do not match the scenario map_file reference")


def _case_input_paths(case: Mapping[str, Any], corpus_root: Path) -> dict[str, Path]:
    return _scenario_input_paths(case, corpus_root)


def _validate_scenario_structure(scenario_path: Path) -> list[str]:
    """Run the repository's canonical scenario loader/schema/asset validation."""
    report = validate_scenario_payload(str(scenario_path))
    if report.get("valid") is True and report.get("num_scenarios") == 1:
        return []
    errors = report.get("errors")
    return [
        str(error.get("code") or error.get("message") or "scenario_invalid")
        for error in errors
        if isinstance(error, dict)
    ] or [str(report.get("status") or "scenario_invalid")]


def _record_attempt(
    corpus: dict[str, Any],
    *,
    source_kind: str,
    source_id: str,
    decision: str,
    blockers: list[str],
    candidate_identity: str | None,
    duplicate_case_id: str | None = None,
    near_duplicate_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "source_kind": source_kind,
        "source_id": source_id,
        "decision": decision,
        "blockers": sorted(set(blockers)),
        "candidate_identity": candidate_identity,
        "duplicate_case_id": duplicate_case_id,
        "near_duplicate_report": dict(near_duplicate_report),
    }
    attempt["attempt_id"] = hashlib.sha256(_stable_json(attempt).encode("utf-8")).hexdigest()
    corpus["admission_attempts"] = _append_unique(
        corpus, "admission_attempts", attempt, key="attempt_id"
    )["admission_attempts"]
    return corpus, {**attempt, "case_id": None}


def _near_duplicate_report(
    case: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    mechanism = case.get("mechanism_group")
    matches = [
        item["case_id"]
        for item in cases
        if isinstance(mechanism, dict) and item.get("mechanism_group") == mechanism
    ]
    if matches:
        return {
            "status": "mechanism_group_matches_reported",
            "method": "same archived mechanism cluster key; report-only, never merged",
            "case_ids": sorted(matches),
        }
    return _unassessed_near_duplicates()


def _unassessed_near_duplicates() -> dict[str, Any]:
    return {
        "status": "not_assessed",
        "method": None,
        "reason": "no calibrated cross-case near-duplicate metric is available; no case is merged",
        "case_ids": [],
    }


def _merge_source_evidence(existing: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    evidence = existing.setdefault("supporting_source_evidence", [])
    key = _stable_json(incoming)
    if all(_stable_json(item) != key for item in evidence):
        evidence.append(dict(incoming))
        evidence.sort(key=_stable_json)


def _append_unique(
    corpus: dict[str, Any], collection: str, item: Mapping[str, Any], *, key: str
) -> dict[str, Any]:
    values = corpus.setdefault(collection, [])
    if not any(existing.get(key) == item.get(key) for existing in values):
        values.append(dict(item))
        values.sort(key=lambda value: str(value.get(key, "")))
    return corpus


def _materialize_case_artifacts(
    case: dict[str, Any], source_files: Mapping[str, Any], corpus_root: Path
) -> None:
    """Atomically copy a verified case's replay inputs and source evidence into corpus custody."""
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.startswith("case-"):
        raise CorpusError("case has no safe stable ID for artifact materialization")
    cases_dir = corpus_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    final_dir = cases_dir / case_id
    if final_dir.exists():
        raise CorpusError(f"case artifact path already exists without a corpus record: {case_id}")
    staging = Path(tempfile.mkdtemp(prefix=f".{case_id}.", dir=cases_dir))
    promoted = False
    try:
        copied_map_paths = _copy_case_inputs(case, source_files, staging, corpus_root)
        _copy_case_source_evidence(source_files, staging, corpus_root)
        _copy_historical_source_snapshots(source_files, staging, corpus_root)
        staging.replace(final_dir)
        promoted = True
        case["inputs"]["scenario_path"] = f"cases/{case_id}/inputs/scenario.yaml"
        case["inputs"]["route_overrides_path"] = f"cases/{case_id}/inputs/route_overrides.yaml"
        for asset in case["inputs"]["map_assets"]:
            asset["path"] = copied_map_paths[asset["role"]]
        for replay in case["replay_receipt"].get("replay_artifacts", []):
            for field in ("path", "provenance_path"):
                relative = replay.get(field)
                if isinstance(relative, str) and not relative.startswith(f"cases/{case_id}/"):
                    replay[field] = f"cases/{case_id}/{relative}"
        case["source_evidence"]["corpus_files"] = _case_file_inventory(final_dir, corpus_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if promoted:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def _copy_case_inputs(
    case: Mapping[str, Any],
    source_files: Mapping[str, Any],
    staging: Path,
    corpus_root: Path,
) -> dict[str, str]:
    for key, relative in {
        "scenario": "inputs/scenario.yaml",
        "route": "inputs/route_overrides.yaml",
    }.items():
        source = source_files.get(key)
        if not isinstance(source, Path):
            raise CorpusError(f"case source artifact is missing: {key}")
        _copy_artifact(source, staging / relative, corpus_root)
    source_assets = source_files.get("map_assets")
    if not isinstance(source_assets, Mapping):
        raise CorpusError("case source map assets are missing")
    copied: dict[str, str] = {}
    for asset in case.get("inputs", {}).get("map_assets", []):
        role = asset.get("role")
        source_path = asset.get("source_path")
        source = source_assets.get(role)
        if role == "map_registry":
            relative = PurePosixPath("inputs/maps/registry.yaml")
        elif role == "map" and isinstance(source_path, str):
            source_relative = PurePosixPath(source_path)
            if source_relative.is_absolute() or ".." in source_relative.parts:
                raise CorpusError("case map source path is unsafe")
            parts = (
                source_relative.parts[1:]
                if source_relative.parts[:1] == ("maps",)
                else source_relative.parts
            )
            relative = PurePosixPath("inputs/maps").joinpath(*parts)
        else:
            raise CorpusError("case map asset role or source path is invalid")
        if not isinstance(source, Path | bytes):
            raise CorpusError(f"case source map asset is missing: {role}")
        _copy_artifact(source, staging / Path(*relative.parts), corpus_root)
        copied[role] = (Path("cases") / str(case["case_id"]) / Path(*relative.parts)).as_posix()
    return copied


def _copy_case_source_evidence(
    source_files: Mapping[str, Any], staging: Path, corpus_root: Path
) -> None:
    destinations = {
        "historical_packet": "historical_issue_1501_failure_0002.json",
        "replay_validation": "replay_validation.json",
        "path_normalization": "path_normalization.json",
        "source_archive": "issue_1501_archive.json",
        "source_run_report": "issue_1501_adversarial_smoke_run.md",
        "replay_1": "replay_1.jsonl",
        "replay_1_provenance": "replay_1.provenance.json",
        "replay_2": "replay_2.jsonl",
        "replay_2_provenance": "replay_2.provenance.json",
    }
    for key, filename in destinations.items():
        source = source_files.get(key)
        if not isinstance(source, Path):
            raise CorpusError(f"case source evidence is missing: {key}")
        _copy_artifact(source, staging / "source_evidence" / filename, corpus_root)


def _copy_historical_source_snapshots(
    source_files: Mapping[str, Any], staging: Path, corpus_root: Path
) -> None:
    snapshots = source_files.get("historical_sources")
    if not isinstance(snapshots, Mapping) or not snapshots:
        raise CorpusError("historical source snapshots are missing")
    for name, content in sorted(snapshots.items()):
        safe_name = (
            isinstance(name, str)
            and PurePosixPath(name).name == name
            and name not in {"", ".", ".."}
        )
        if not safe_name or not isinstance(content, bytes):
            raise CorpusError("historical source snapshot has an unsafe name or value")
        _copy_artifact(
            content,
            staging / "source_evidence" / "historical_sources" / name,
            corpus_root,
        )


def _copy_artifact(source: Path | bytes, destination: Path, corpus_root: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, bytes):
        destination.write_bytes(source)
    else:
        shutil.copyfile(source, destination)
    return destination.resolve().relative_to(corpus_root.resolve()).as_posix()


def _case_file_inventory(case_dir: Path, corpus_root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": item.resolve().relative_to(corpus_root.resolve()).as_posix(),
            "sha256": _sha256_file(item),
        }
        for item in sorted(path for path in case_dir.rglob("*") if path.is_file())
    ]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusError(f"JSON root is not an object: {path}")
    return value


def _read_single_jsonl_record(path: Path) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError) as exc:
        raise CorpusError(f"could not read replay JSONL {path}: {exc}") from exc
    if len(lines) != 1:
        raise CorpusError(f"replay JSONL must contain exactly one record: {path}")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise CorpusError(f"replay JSONL contains malformed JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusError(f"replay JSONL record must be an object: {path}")
    return value


def _load_yaml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CorpusError(f"could not load {label} YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusError(f"{label} YAML root must be an object: {path}")
    return value


def _scenario_seed(scenario: Mapping[str, Any]) -> int:
    seeds = scenario.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1:
        raise CorpusError("scenario must bind exactly one seed")
    seed = seeds[0]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CorpusError("scenario seed must be an integer")
    return seed


def _require_finite_metric(metrics: Mapping[str, Any], key: str) -> float | bool:
    if key == "success" and isinstance(metrics.get(key), bool):
        return metrics[key]
    value = _finite_number(metrics.get(key))
    if value is None:
        raise CorpusError(f"selected replay metric is missing or non-finite: {key}")
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _stable_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_full_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _current_target_revision() -> str | None:
    """Resolve the exact HEAD when the module's source checkout is clean."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "--verify", "HEAD^{commit}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "-C", str(_ROOT), "status", "--porcelain", "--untracked-files=all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip()
    if (
        completed.returncode != 0
        or status.returncode != 0
        or status.stdout.strip()
        or not _is_full_git_revision(revision)
    ):
        return None
    return revision

#!/usr/bin/env python3
"""Build the issue #7980 speed-tier interpretation packet.

This projects the 24 canonical issue #5578 synthesis decisions into the existing
``result_interpretation_packet.v1`` contract. It does not rerun the campaign or
admit a benchmark claim. Authenticated-source wording additionally requires an
exact immutable-member receipt, preservation manifest, and row crosswalk;
without that proof the output is explicitly diagnostic/pending.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
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
EXPECTED_CLASSIFICATION_COUNTS = {
    "no_material_shift": 10,
    "inconclusive": 8,
    "intervention_not_activated": 6,
}
EXPECTED_NONACTIVATED_IDS = frozenset(
    f"prediction_planner__{tier}__{metric}"
    for tier in ("cap_3_0", "cap_4_0")
    for metric in ("collision_rate", "near_miss_rate", "success_rate")
)
EXPECTED_SYNTHESIS_SCHEMA = "robot_sf.issue_5578_speed_tier_synthesis_adapter.v1"
EXPECTED_EVIDENCE_STATUS = "native_grid_synthesis_complete_provenance_unverified"
SOURCE_RECEIPT_SCHEMA = "issue_7980_source_ingestion_receipt.v1"
SOURCE_PROOF_STATUSES = frozenset({"fixture_verified", "authenticated_immutable_source_hydrated"})
SOURCE_ARTIFACT_MEMBER = "synthesis.json.gz"
SOURCE_ARTIFACT_SOURCE_MEMBER = "synthesis.json"
SOURCE_ARTIFACT_MANIFEST_MEMBER = "campaign_preservation_manifest.json"
SOURCE_ARTIFACT_MANIFEST_SCHEMA = "campaign-preservation-manifest.v1"
PRODUCER_SCRIPT_PATH = "scripts/analysis/build_issue_7980_speed_tier_packet.py"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object or fail closed with path context."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _row_digest(row: Mapping[str, Any]) -> str:
    """Return the canonical digest used by the independent row crosswalk."""

    return hashlib.sha256(
        json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_nonempty_mapping(value: object, field: str) -> Mapping[str, Any]:
    """Return one non-empty mapping or reject a self-declared empty receipt block."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"source receipt {field} must be a non-empty mapping")
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    """Return one non-empty string from a receipt identity field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source receipt {field} must be a non-empty string")
    return value


def _require_sha256(value: object, field: str) -> str:
    """Return one lowercase SHA-256 digest from a receipt identity field."""

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"source receipt {field} must be a lowercase SHA-256 digest")
    return value


def _input_path(value: object, field: str) -> Path:
    """Resolve one explicit receipt input path without requiring it to be tracked."""

    raw = _require_nonempty_string(value, field)
    path = Path(raw)
    return (path if path.is_absolute() else (_REPO_ROOT / path)).resolve()


def _canonical_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Recompute the preservation tool's canonical manifest identity."""

    payload = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_row_crosswalk(crosswalk: object, rows: Sequence[Mapping[str, Any]]) -> None:
    """Require the independent receipt to cover each canonical source row exactly once."""

    if not isinstance(crosswalk, list) or len(crosswalk) != len(rows):
        raise ValueError("source receipt must contain one digest crosswalk entry per source row")
    expected = {str(row["test_id"]): _row_digest(row) for row in rows}
    observed: dict[str, str] = {}
    for entry in crosswalk:
        if not isinstance(entry, Mapping):
            raise ValueError("source receipt row crosswalk entries must be objects")
        test_id = entry.get("test_id")
        digest = entry.get("canonical_row_sha256")
        if not isinstance(test_id, str) or not isinstance(digest, str):
            raise ValueError(
                "source receipt crosswalk entries need test_id and canonical_row_sha256"
            )
        if test_id in observed:
            raise ValueError(f"source receipt contains duplicate row crosswalk entry: {test_id}")
        observed[test_id] = digest
    if observed != expected:
        raise ValueError("source receipt row crosswalk does not match independently ingested rows")


def _load_receipt_source_crosswalk(receipt: Mapping[str, Any], synthesis_path: Path) -> object:
    """Load and hash the independently supplied source crosswalk."""

    source_path_value = receipt.get("source_path")
    source_path = _input_path(source_path_value, "source_path")
    if source_path == synthesis_path.resolve():
        raise ValueError("source receipt source path must not reuse the supplied synthesis path")
    if receipt.get("source_sha256") != _sha256(source_path):
        raise ValueError("source receipt source path digest does not match supplied source bytes")
    source_crosswalk = _load_json(source_path)
    crosswalk = source_crosswalk.get("row_crosswalk")
    if "row_crosswalk" in receipt and crosswalk != receipt["row_crosswalk"]:
        raise ValueError(
            "source receipt does not match the independently supplied source crosswalk"
        )
    return crosswalk


def _expected_authenticated_artifact(
    recovery_manifest: Mapping[str, Any], synthesis_sha256: str
) -> dict[str, str]:
    """Resolve the pinned W&B artifact identity from reviewed recovery metadata."""

    durable = _require_nonempty_mapping(
        recovery_manifest.get("durable_artifact"), "recovery durable_artifact"
    )
    qualified_name = _require_nonempty_string(
        durable.get("artifact_name"), "recovery durable_artifact.artifact_name"
    )
    version = _require_nonempty_string(durable.get("version"), "recovery durable_artifact.version")
    if not qualified_name.endswith(f":{version}"):
        raise ValueError("reviewed recovery artifact name and version disagree")
    manifest_sha256 = _require_sha256(
        durable.get("manifest_sha256"), "recovery durable_artifact.manifest_sha256"
    )
    return {
        "qualified_name": qualified_name,
        "version": version,
        "member": SOURCE_ARTIFACT_MEMBER,
        "source_member": SOURCE_ARTIFACT_SOURCE_MEMBER,
        "source_sha256": synthesis_sha256,
        "manifest_digest": f"sha256:{manifest_sha256}",
    }


def _validated_authenticated_identity(
    receipt: Mapping[str, Any],
    *,
    recovery_manifest: Mapping[str, Any],
    synthesis_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, str], str]:
    """Cross-check receipt identity blocks against reviewed artifact metadata."""

    source_artifact = _require_nonempty_mapping(receipt.get("source_artifact"), "source_artifact")
    hydration = _require_nonempty_mapping(
        receipt.get("immutable_hydration_receipt"), "immutable_hydration_receipt"
    )
    expected = _expected_authenticated_artifact(recovery_manifest, synthesis_sha256)
    for field, expected_value in expected.items():
        if source_artifact.get(field) != expected_value:
            raise ValueError(
                f"source receipt source_artifact {field} does not match reviewed artifact identity"
            )
        if hydration.get(field) != expected_value:
            raise ValueError(
                f"source receipt immutable_hydration_receipt {field} does not match "
                "reviewed artifact identity"
            )
    if hydration.get("status") != "verified":
        raise ValueError("authenticated source receipt hydration status must be verified")

    member_sha256 = _require_sha256(
        source_artifact.get("member_sha256"), "source_artifact.member_sha256"
    )
    if hydration.get("member_sha256") != member_sha256:
        raise ValueError("source receipt artifact and hydration member digests disagree")
    return source_artifact, hydration, expected, member_sha256


def _validated_authenticated_member(
    hydration: Mapping[str, Any],
    *,
    member_sha256: str,
    synthesis_sha256: str,
    synthesis_path: Path,
) -> tuple[bytes, Path]:
    """Cross-check compressed member bytes against the supplied source bytes."""

    member_path = _input_path(hydration.get("hydrated_member_path"), "hydrated_member_path")
    if _sha256(member_path) != member_sha256:
        raise ValueError("authenticated source member digest drifted")

    source_bytes = synthesis_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != synthesis_sha256:
        raise ValueError("authenticated source bytes do not match the supplied synthesis digest")
    try:
        decompressed = gzip.decompress(member_path.read_bytes())
    except (OSError, EOFError) as exc:
        raise ValueError("authenticated source member is not a valid gzip payload") from exc
    if decompressed != source_bytes:
        raise ValueError("authenticated source member does not decompress to supplied source bytes")
    return source_bytes, member_path


def _validate_authenticated_manifest(
    hydration: Mapping[str, Any],
    *,
    expected_manifest_digest: str,
    member_path: Path,
    member_sha256: str,
    source_bytes: bytes,
    synthesis_sha256: str,
) -> None:
    """Cross-check the pinned preservation manifest and its synthesis member row."""

    manifest_path = _input_path(
        hydration.get("preservation_manifest_path"), "preservation_manifest_path"
    )
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != SOURCE_ARTIFACT_MANIFEST_SCHEMA:
        raise ValueError("authenticated preservation manifest schema is unsupported")
    manifest_digest = _canonical_manifest_digest(manifest)
    if manifest.get("manifest_digest") != manifest_digest:
        raise ValueError("authenticated preservation manifest digest is internally inconsistent")
    if manifest_digest != expected_manifest_digest:
        raise ValueError(
            "authenticated preservation manifest digest does not match reviewed custody"
        )
    if hydration.get("manifest_member") != SOURCE_ARTIFACT_MANIFEST_MEMBER:
        raise ValueError(
            "authenticated source receipt names the wrong preservation manifest member"
        )

    manifest_rows = manifest.get("files")
    if not isinstance(manifest_rows, list):
        raise ValueError("authenticated preservation manifest files must be a list")
    source_entries = [
        entry
        for entry in manifest_rows
        if isinstance(entry, Mapping) and entry.get("path") == SOURCE_ARTIFACT_SOURCE_MEMBER
    ]
    if len(source_entries) != 1:
        raise ValueError(
            "authenticated preservation manifest must contain one synthesis source row"
        )
    source_entry = source_entries[0]
    expected_entry = {
        "path": SOURCE_ARTIFACT_SOURCE_MEMBER,
        "stored_path": SOURCE_ARTIFACT_MEMBER,
        "sha256": synthesis_sha256,
        "stored_sha256": member_sha256,
        "bytes": len(source_bytes),
        "stored_bytes": member_path.stat().st_size,
    }
    for field, expected_value in expected_entry.items():
        if source_entry.get(field) != expected_value:
            raise ValueError(
                f"authenticated preservation manifest source row {field} does not match receipt"
            )


def _validate_authenticated_source_rows(
    synthesis_path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Cross-check direct synthesis rows against the independent digest crosswalk."""

    source_payload = _load_json(synthesis_path)
    source_rows = source_payload.get("decision_table")
    if not isinstance(source_rows, list) or not all(
        isinstance(row, Mapping) for row in source_rows
    ):
        raise ValueError("authenticated synthesis decision_table must contain source row objects")
    source_crosswalk = [
        {
            "test_id": row.get("test_id"),
            "canonical_row_sha256": _row_digest(row),
        }
        for row in source_rows
    ]
    try:
        _validate_row_crosswalk(source_crosswalk, rows)
    except ValueError as exc:
        raise ValueError("authenticated synthesis source rows do not match validated rows") from exc


def _validate_authenticated_hydration(
    receipt: Mapping[str, Any],
    *,
    recovery_manifest: Mapping[str, Any],
    synthesis_sha256: str,
    synthesis_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Bind authenticated artifact identity, member bytes, manifest, and source rows."""

    _, hydration, expected, member_sha256 = _validated_authenticated_identity(
        receipt,
        recovery_manifest=recovery_manifest,
        synthesis_sha256=synthesis_sha256,
    )
    source_bytes, member_path = _validated_authenticated_member(
        hydration,
        member_sha256=member_sha256,
        synthesis_sha256=synthesis_sha256,
        synthesis_path=synthesis_path,
    )
    _validate_authenticated_manifest(
        hydration,
        expected_manifest_digest=expected["manifest_digest"],
        member_path=member_path,
        member_sha256=member_sha256,
        source_bytes=source_bytes,
        synthesis_sha256=synthesis_sha256,
    )
    _validate_authenticated_source_rows(synthesis_path, rows)


def _validate_source_receipt(
    receipt: Mapping[str, Any],
    *,
    synthesis_sha256: str,
    synthesis_path: Path,
    rows: Sequence[Mapping[str, Any]],
    recovery_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate fixture or authenticated source custody without trusting declarations alone."""

    if receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA:
        raise ValueError("source receipt schema_version is unsupported")
    status = receipt.get("source_ingestion_status")
    if status not in SOURCE_PROOF_STATUSES:
        raise ValueError(
            "authenticated immutable source hydration is unavailable; "
            "authenticated-source packet generation is refused"
        )
    if receipt.get("independent_of_packet") is not True:
        raise ValueError("source receipt must declare independence from the packet")
    crosswalk = _load_receipt_source_crosswalk(receipt, synthesis_path)
    if receipt.get("synthesis_sha256") != synthesis_sha256:
        raise ValueError("source receipt synthesis digest does not match supplied synthesis")
    _validate_row_crosswalk(crosswalk, rows)
    source_artifact = _require_nonempty_mapping(receipt.get("source_artifact"), "source_artifact")
    if status == "fixture_verified":
        if source_artifact.get("kind") != "deterministic_fixture_crosswalk":
            raise ValueError(
                "fixture source receipt must identify a deterministic fixture crosswalk"
            )
        if source_artifact.get("path") != receipt.get("source_path"):
            raise ValueError("fixture source receipt artifact path must match its crosswalk path")
        if receipt.get("immutable_hydration_receipt") is not None:
            raise ValueError("fixture source receipt must not declare immutable hydration")
        return dict(receipt)
    _validate_authenticated_hydration(
        receipt,
        recovery_manifest=recovery_manifest,
        synthesis_sha256=synthesis_sha256,
        synthesis_path=synthesis_path,
        rows=rows,
    )
    return dict(receipt)


def _source_receipt_source_ref(
    receipt: Mapping[str, Any],
    *,
    producer_commit: str,
) -> dict[str, Any]:
    """Project the receipt's durable source crosswalk into a packet ``SourceRef``.

    The receipt itself may be an ignored, task-local hydration record.  The generic packet
    contract only permits repository-relative durable source files, so the metric binding points
    at the independently supplied crosswalk that the receipt validates.  The receipt's complete
    custody identity remains encoded in each metric sensitivity binding.
    """

    source_path = _input_path(receipt.get("source_path"), "source_path")
    try:
        relative_path = source_path.relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            "source receipt crosswalk must be a durable repository file for packet declaration"
        ) from exc
    tracked_commit = _tracked_source_commit(relative_path, source_path)
    source_sha256 = _require_sha256(receipt.get("source_sha256"), "source_sha256")
    return {
        "source_id": "issue_7980_source_receipt",
        "path": relative_path,
        "sha256": source_sha256,
        "kind": "independent_source_crosswalk",
        "commit": producer_commit,
        "tracked_commit": tracked_commit,
        "command": (
            f"scripts/analysis/build_issue_7980_speed_tier_packet.py "
            "--synthesis <verified-synthesis.json> "
            "--source-receipt <verified-source-receipt.json> "
            f"--producer-commit {producer_commit}"
        ),
        "description": (
            "Independent source-row crosswalk retained and validated by the supplied "
            "issue #7980 source-ingestion receipt."
        ),
        "direction": "not_applicable",
    }


def _tracked_source_commit(relative_path: str, source_path: Path) -> str:
    """Require the current source bytes to equal the latest tracked blob."""
    tracked_commit = _git("log", "-1", "--format=%H", "--", relative_path)
    if not tracked_commit:
        raise ValueError(
            "source receipt crosswalk must be tracked before it can be declared in the packet"
        )
    try:
        committed_blob = _git("rev-parse", f"{tracked_commit}:{relative_path}")
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "source receipt crosswalk tracked commit does not contain the declared path"
        ) from exc
    current_blob = _git("hash-object", str(source_path))
    if committed_blob != current_blob:
        raise ValueError(
            "source receipt crosswalk bytes do not match the latest tracked commit blob"
        )
    return tracked_commit


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


def _validate_producer_commit(commit: str) -> str:
    """Require a full commit whose builder blob exactly matches the executing builder."""

    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("producer commit must be one full 40-character commit SHA")
    resolved = _git("rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved != commit:
        raise ValueError("producer commit did not resolve to the exact requested commit")
    try:
        committed_blob = _git("rev-parse", f"{commit}:{PRODUCER_SCRIPT_PATH}")
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"producer commit does not contain {PRODUCER_SCRIPT_PATH}") from exc
    current_blob = _git("hash-object", str(_REPO_ROOT / PRODUCER_SCRIPT_PATH))
    if committed_blob != current_blob:
        raise ValueError("producer commit does not contain the executing builder implementation")
    return resolved


def _producer_command(commit: str) -> str:
    """Return a checkout-pinned, non-self-referential reproduction command."""

    return (
        f"git worktree add --detach <fresh-worktree> {commit} && "
        "cd <fresh-worktree> && "
        "scripts/dev/run_worktree_shared_venv.sh -- python "
        f"{PRODUCER_SCRIPT_PATH} --synthesis <verified-synthesis.json> "
        f"--producer-commit {commit}"
    )


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


def _validate_synthesis_row_identity(row: Mapping[str, Any]) -> str:
    """Require a source row ID to agree with its planner, tier, and metric fields."""

    test_id = row.get("test_id")
    if not isinstance(test_id, str) or not test_id:
        raise ValueError("every synthesis decision row needs a non-empty string test_id")
    row_identity: dict[str, str] = {}
    for field in ("planner_id", "speed_tier_id", "metric"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{test_id}: {field} must be a non-empty string")
        row_identity[field] = value
    expected_test_id = (
        f"{row_identity['planner_id']}__{row_identity['speed_tier_id']}__{row_identity['metric']}"
    )
    if test_id != expected_test_id:
        raise ValueError(
            f"{test_id}: test_id must match planner_id, speed_tier_id, and metric "
            f"({expected_test_id})"
        )
    return test_id


def _validate_synthesis_row(
    raw_row: object,
    *,
    numeric_fields: Sequence[str],
) -> dict[str, Any]:
    """Validate one canonical decision row and its activation-classification contract."""

    if not isinstance(raw_row, dict):
        raise ValueError("every synthesis decision row must be an object")
    row = dict(raw_row)
    test_id = _validate_synthesis_row_identity(row)
    _require_finite(row, numeric_fields, test_id)
    n_scenarios = row.get("n_scenarios")
    if isinstance(n_scenarios, bool) or not isinstance(n_scenarios, int) or n_scenarios != 6:
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


def _validate_classification_accounting(
    rows: Sequence[Mapping[str, Any]], recovery_manifest: Mapping[str, Any]
) -> None:
    """Require the frozen classification counts and inactive-row roster."""

    expected_counts = recovery_manifest["descriptive_synthesis"]["classification_counts"]
    if expected_counts != EXPECTED_CLASSIFICATION_COUNTS:
        raise ValueError(
            "recovery manifest classification accounting does not match the frozen 10/8/6 "
            "issue #7980 contract"
        )
    observed_counts = Counter(str(row["classification"]) for row in rows)
    if dict(observed_counts) != expected_counts:
        raise ValueError(
            f"classification accounting mismatch: {dict(observed_counts)} != {expected_counts}"
        )
    observed_nonactivated_ids = {
        row["test_id"] for row in rows if not row["intervention_activated"]
    }
    if observed_nonactivated_ids != EXPECTED_NONACTIVATED_IDS:
        raise ValueError(
            "non-activated source rows must match the six prediction_planner cap-3/cap-4 contrasts"
        )


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
        validated_rows.append(row)
    _validate_classification_accounting(validated_rows, recovery_manifest)
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
    source_receipt: Mapping[str, Any] | None,
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
        "source_artifact": dict(source_receipt["source_artifact"])
        if source_receipt is not None
        else {
            "status": "pending",
            "reason": "independent source-ingestion receipt is unavailable",
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
    source_receipt: Mapping[str, Any] | None,
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
            source_receipt=source_receipt,
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
        "source_ids": ["recovery_manifest", "issue_7980_source_receipt"]
        if source_receipt is not None
        else ["recovery_manifest"],
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
            "this diagnostic/pending packet grants no admission."
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
    source_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Build a diagnostic packet, optionally backed by an independent source receipt."""

    synthesis_path = synthesis_path.resolve()
    recovery_manifest_path = _repo_path(recovery_manifest_path)
    previous_packet_path = _repo_path(previous_packet_path)
    preregistration_path = _repo_path(preregistration_path)
    synthesis = _load_json(synthesis_path)
    recovery_manifest = _load_json(recovery_manifest_path)
    previous_packet = _load_json(previous_packet_path)
    preregistration = _read_preregistration(preregistration_path)
    source_receipt = None
    if source_receipt_path is not None:
        source_receipt = _load_json(_repo_path(source_receipt_path))
    synthesis_sha256 = _sha256(synthesis_path)
    rows, paired_denominator, thresholds = _validate_synthesis(
        synthesis,
        synthesis_sha256=synthesis_sha256,
        recovery_manifest=recovery_manifest,
        preregistration=preregistration,
    )
    if source_receipt is not None:
        source_receipt = _validate_source_receipt(
            source_receipt,
            synthesis_sha256=synthesis_sha256,
            synthesis_path=synthesis_path,
            rows=rows,
            recovery_manifest=recovery_manifest,
        )
    authenticated_source_bound = (
        source_receipt is not None
        and source_receipt.get("source_ingestion_status")
        == "authenticated_immutable_source_hydrated"
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
            source_receipt=source_receipt,
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
    sources = [dict(recovery_source)]
    if source_receipt is not None:
        sources.append(
            _source_receipt_source_ref(
                source_receipt,
                producer_commit=producer_commit,
            )
        )

    packet = {
        "schema_version": "result_interpretation_packet.v1",
        "packet_id": "issue_7980_robot_speed_tier_contrast_binding_diagnostic",
        "question": {
            "question_id": "q_7980_robot_speed_tier_contrast_binding",
            "text": (
                "What do the exact 24 speed-tier contrasts establish before any "
                "separate benchmark-admission decision?"
            ),
            "issue_refs": [5578, 6102, 7980],
        },
        "evidence": {
            "evidence_id": "issue_7980_robot_speed_tier_contrast_binding",
            "tier": "smoke_diagnostic",
            "admission_state": "diagnostic_only",
            "rationale": (
                "All registered statistics are bound to an authenticated immutable-member "
                "receipt, but this packet preserves the non-admitted diagnostic boundary."
                if authenticated_source_bound
                else "Authenticated immutable-member custody is pending; this source-bound "
                "successor remains diagnostic only."
            ),
        },
        "sources": sources,
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
                    (
                        "The authenticated-source classification accounting is "
                        if authenticated_source_bound
                        else "The source-bound diagnostic classification accounting is "
                    )
                    + f"{counts['no_material_shift']} no_material_shift, "
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
            "command": _producer_command(producer_commit),
            "status": "draft",
        },
        "findings": [
            "All 24 registered planner-by-tier-by-metric contrasts are present exactly once.",
            "Every contrast binds the pooled effect, paired denominator, both directional tests and bounds, multiplicity, activation state, and the supplied synthesis digest.",
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
            "Authenticated-source wording requires an exact artifact/version/member receipt, preservation-manifest digest, compressed-member digest, source digest, and row crosswalk; otherwise status remains diagnostic/pending.",
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
    parser.add_argument(
        "--source-receipt",
        type=Path,
        default=None,
        help="Independent source-ingestion receipt; omit to emit diagnostic/pending wording.",
    )
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
    if producer_commit is None:
        raise ValueError(
            "--producer-commit is required when writing; commit the exact builder first so "
            "the generated packet does not self-reference its own output commit"
        )
    producer_commit = _validate_producer_commit(producer_commit)
    packet = build_packet(
        synthesis_path=args.synthesis,
        recovery_manifest_path=args.recovery_manifest,
        previous_packet_path=args.previous_packet,
        preregistration_path=args.preregistration,
        producer_commit=producer_commit,
        source_receipt_path=args.source_receipt,
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

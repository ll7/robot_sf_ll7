"""Fail-closed admission checks for a preserved post-execution release.

The ordinary release doctor answers ``can this packet be submitted?``.  A
preserved campaign needs a different question after the scheduler is terminal:
``can these exact rows, corrected by the reviewed validator, be published?``.
This module joins the derived receipt, publication bundle, historical packet,
and private queue/job ledgers without changing any of them.

The post-execution path deliberately does not require a failed queue row to be
dispatchable.  It does require that the row consumed exactly one admitted
attempt and that the job's failure was a validator/publication-gate failure (or
that a trusted derived evaluation receipt records that fact).  This keeps a
terminal scheduler failure from being silently converted into a successful
submission while allowing a separately reviewed, checksum-preserving
revalidation to close the release.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.artifact_publication import (
    PublicationPreflightError,
    verify_publication_bundle_preflight,
)
from robot_sf.benchmark.release_doctor import (
    _ci_check,
    _dissertation_check,
    _git_check,
    _tag_check,
    _zenodo_check,
)

FROZEN_SOURCE_SHA = "b1d5ab6de708385c0828c99501a9d1c29727ec11"
EXPECTED_BASE_SHA = "cd831d7582c117ac9529065e7d1c60386933c92d"
EXPECTED_RELEASE_TAG = "paper-matrix-v2-h600-s30-2026-08-cd831d7582c1"
EXPECTED_CAMPAIGN_ID = "issue7742_release_full-s30-h600-b1d5ab6de708-v1_20260825"
EXPECTED_JOB_ID = "14890"
EXPECTED_VALIDATOR_SHA = "bd4bc4b4018b24c887c8e91ad834bc6898d7aad2"
EXPECTED_ARMS = 14
EXPECTED_EPISODE_CELLS = 20_160
EXPECTED_STRESS_CELLS = 70
EXPECTED_RUNTIME_SMOKE_CELLS = 14

_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$", re.IGNORECASE)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_FORBIDDEN_MARKER_KEYS = {
    "fallback_count",
    "fallback_rows",
    "degraded_count",
    "degraded_rows",
    "unavailable_count",
    "unavailable_rows",
    "failed_count",
    "failed_rows",
}
_MANIFEST_SENTINEL = "ROBOT_SF_RELEASE_MANIFEST_VALIDATION="
_CREDENTIAL_PATTERNS = (
    b"authorization: bearer ",
    b"access_token=",
    b'"access_token":',
    b'"api_key":',
    b'"password":',
    b'"secret":',
)


@dataclass(frozen=True)
class PostExecutionReleaseDoctorCheck:
    """One sanitized post-execution release check."""

    name: str
    status: str
    summary: str


def _read_mapping(path: Path, *, yaml_ok: bool = False) -> dict[str, Any]:
    """Read one JSON/YAML mapping without leaking its contents in errors.

    Returns:
        Parsed object mapping.
    """
    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if yaml_ok else json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid structured artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"structured artifact is not an object: {path.name}")
    return payload


def _sha256(path: Path) -> str:
    """Hash one regular file.

    Returns:
        Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str | None:
    """Normalize a SHA-256 value, accepting an optional ``sha256:`` prefix.

    Returns:
        Digest without a prefix, or ``None`` for malformed input.
    """
    text = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(text):
        return None
    return text.removeprefix("sha256:")


def _int(value: Any) -> int | None:
    """Parse only an integral, non-boolean value.

    Returns:
        Parsed integer, or ``None`` for non-integral input.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    return None


def _check(name: str, problems: list[str], success: str) -> PostExecutionReleaseDoctorCheck:
    """Build a check while preserving stable, credential-free summaries.

    Returns:
        Sanitized check result.
    """
    summary = success if not problems else "; ".join(dict.fromkeys(problems))
    return PostExecutionReleaseDoctorCheck(name, "pass" if not problems else "fail", summary)


def _adapt_check(check: Any) -> PostExecutionReleaseDoctorCheck:
    """Convert a shared release-doctor check without changing its safe summary.

    Returns:
        Equivalent post-execution check.
    """
    return PostExecutionReleaseDoctorCheck(check.name, check.status, check.summary)


def _contained_file(path: Path, root: Path) -> bool:
    """Return whether an existing regular file is contained by ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return path.is_file()


def _source_manifest_check(  # noqa: C901, PLR0912
    manifest_path: Path | None,
    *,
    repo: Path,
    source_sha: str,
    base_sha: str,
    tag: str,
) -> PostExecutionReleaseDoctorCheck:
    """Validate the frozen v0.2 manifest without importing from a moving checkout.

    Returns:
        Frozen manifest validation check.
    """
    problems: list[str] = []
    if manifest_path is None or not _contained_file(manifest_path, repo):
        return _check("manifest", ["frozen release manifest is missing or outside source repo"], "")
    try:
        payload = _read_mapping(manifest_path, yaml_ok=True)
    except ValueError as exc:
        return _check("manifest", [str(exc)], "")
    if payload.get("schema_version") != "benchmark-release-manifest.v0.2":
        problems.append("frozen release manifest is not v0.2")
    if payload.get("latest_main_base_commit") != base_sha:
        problems.append("frozen release manifest base commit does not match")
    if payload.get("release_tag") != tag:
        problems.append("frozen release manifest tag does not match")
    matrix = payload.get("matrix")
    expected_matrix = {
        "planner_arms": EXPECTED_ARMS,
        "scenarios": 48,
        "seeds": 30,
        "expected_episode_cells": EXPECTED_EPISODE_CELLS,
        "horizon_steps": 600,
    }
    if not isinstance(matrix, dict):
        problems.append("frozen release matrix is missing")
    else:
        for key, expected in expected_matrix.items():
            if _int(matrix.get(key)) != expected:
                problems.append(f"frozen release matrix {key} does not match")
    if source_sha != FROZEN_SOURCE_SHA:
        problems.append("manifest validation source is not the frozen release candidate")

    validation_script = """
import json
import sys
from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest

manifest = load_release_manifest(sys.argv[1])
report = validate_release_manifest(manifest)
identity = {
    "release_tag": manifest.release_tag,
    "latest_main_base_commit": manifest.latest_main_base_commit,
    "expected_episode_cells": manifest.expected_episode_cells,
    "expected_horizon_steps": manifest.expected_horizon_steps,
    "publication_channel": manifest.publication_channel,
    "concept_doi": manifest.concept_doi,
    "version_doi": manifest.version_doi,
    "snqi_claim_policy": manifest.snqi_claim_policy,
    "planner_count": len(manifest.planner_keys),
    "seed_count": len(manifest.resolved_seeds),
}
print("ROBOT_SF_RELEASE_MANIFEST_VALIDATION=" + json.dumps({"report": report, "identity": identity}, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo.resolve())
    try:
        completed = subprocess.run(
            [sys.executable, "-c", validation_script, str(manifest_path)],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        problems.append("complete frozen manifest validation could not run")
    else:
        lines = [
            line.removeprefix(_MANIFEST_SENTINEL)
            for line in completed.stdout.splitlines()
            if line.startswith(_MANIFEST_SENTINEL)
        ]
        try:
            complete = (
                json.loads(lines[-1]) if completed.returncode == 0 and len(lines) == 1 else {}
            )
        except json.JSONDecodeError:
            complete = {}
        report = complete.get("report") if isinstance(complete, dict) else None
        identity = complete.get("identity") if isinstance(complete, dict) else None
        if not isinstance(report, dict) or report.get("status") != "valid":
            problems.append("complete frozen v0.2 manifest validation did not pass")
        expected_identity = {
            "release_tag": tag,
            "latest_main_base_commit": base_sha,
            "expected_episode_cells": EXPECTED_EPISODE_CELLS,
            "expected_horizon_steps": 600,
            "publication_channel": "direct_zenodo_benchmark_dataset",
            "concept_doi": "10.5281/zenodo.22077447",
            "version_doi": "10.5281/zenodo.22077448",
            "snqi_claim_policy": "advisory_no_ranking",
            "planner_count": EXPECTED_ARMS,
            "seed_count": 30,
        }
        if identity != expected_identity:
            problems.append("complete frozen v0.2 manifest identity does not match")

    pinned_files = (
        ("canonical_campaign_config", "campaign_config_sha256"),
        ("scenario.matrix_path", "scenario.matrix_sha256"),
        ("publication.metadata_path", "publication.metadata_sha256"),
    )
    for path_key, digest_key in pinned_files:
        path_parent: Any = payload
        for part in path_key.split("."):
            path_parent = path_parent.get(part) if isinstance(path_parent, dict) else None
        digest_parent: Any = payload
        for part in digest_key.split("."):
            digest_parent = digest_parent.get(part) if isinstance(digest_parent, dict) else None
        if not isinstance(path_parent, str) or _digest(digest_parent) is None:
            problems.append(f"manifest pin {path_key} is incomplete")
            continue
        candidate = (manifest_path.parent / path_parent).resolve()
        if not _contained_file(candidate, repo):
            problems.append(f"manifest pin {path_key} is missing or outside source repo")
        elif _sha256(candidate) != _digest(digest_parent):
            problems.append(f"manifest pin {path_key} digest does not match")
    return _check(
        "manifest",
        problems,
        "frozen v0.2 manifest, 20,160-cell matrix, and input hashes are exact",
    )


def _disk_check(repo: Path, minimum_free_gib: float) -> PostExecutionReleaseDoctorCheck:
    """Require sufficient local space for publication and cold verification.

    Returns:
        Disk-capacity check.
    """
    free_gib = shutil.disk_usage(repo).free / (1024**3)
    problems = [] if free_gib >= minimum_free_gib else ["release filesystem has insufficient space"]
    return _check(
        "disk_capacity",
        problems,
        f"{free_gib:.1f} GiB free; requires {minimum_free_gib:.1f} GiB",
    )


def _acceptance_problems(  # noqa: C901
    payload: Any,
    *,
    source_sha: str,
    label: str,
    expected_cells: int = EXPECTED_EPISODE_CELLS,
) -> list[str]:
    """Validate one full-release acceptance mapping.

    Returns:
        Credential-free contract violations.
    """
    if not isinstance(payload, dict):
        return [f"{label} is missing"]
    problems: list[str] = []
    if payload.get("status") != "valid":
        problems.append(f"{label} status is not valid")
    if payload.get("benchmark_success") is not True:
        problems.append(f"{label} benchmark_success is not true")
    if payload.get("blockers") != []:
        problems.append(f"{label} has blockers")
    for key, expected in (
        ("expected_planner_arms", EXPECTED_ARMS),
        ("expected_scenario_count", 48),
        ("expected_seed_count", 30),
        ("expected_episode_cells", expected_cells),
        ("observed_episode_rows", expected_cells),
        ("unique_episode_identities", expected_cells),
        ("successful_planner_arms", EXPECTED_ARMS),
    ):
        if _int(payload.get(key)) != expected:
            problems.append(f"{label} {key} does not match the fixed release contract")
    for key in ("missing_episode_identities", "unexpected_episode_identities"):
        if _int(payload.get(key)) != 0:
            problems.append(f"{label} has {key}")
    forbidden = payload.get("forbidden_status_counts")
    if not isinstance(forbidden, dict) or any(_int(value) != 0 for value in forbidden.values()):
        problems.append(f"{label} contains forbidden status counts")
    commits = payload.get("source_commits")
    if commits != [source_sha]:
        problems.append(f"{label} source commit is not the frozen source")
    return problems


def _positive_forbidden_markers(payload: Any) -> bool:
    """Find explicit positive runtime failure markers in a structured result.

    Returns:
        ``True`` when a forbidden marker has a positive value.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in _FORBIDDEN_MARKER_KEYS:
                if _int(value) not in {None, 0} or (isinstance(value, bool) and value):
                    return True
            if _positive_forbidden_markers(value):
                return True
    elif isinstance(payload, list):
        return any(_positive_forbidden_markers(item) for item in payload)
    return False


def _receipt_check(  # noqa: C901, PLR0912
    receipt_path: Path | None,
    *,
    source_sha: str,
    expected_tag: str,
    expected_base_sha: str,
    expected_campaign_id: str,
    expected_validator_sha: str | None,
) -> tuple[PostExecutionReleaseDoctorCheck, dict[str, Any] | None]:
    """Validate the derived revalidation receipt and its claim boundary.

    Returns:
        Sanitized check and parsed receipt when readable.
    """
    if receipt_path is None or not receipt_path.is_file():
        return PostExecutionReleaseDoctorCheck(
            "derived_revalidation", "fail", "derived revalidation receipt is missing"
        ), None
    try:
        receipt = _read_mapping(receipt_path)
    except ValueError as exc:
        return PostExecutionReleaseDoctorCheck("derived_revalidation", "fail", str(exc)), None

    problems: list[str] = []
    if receipt.get("schema_version") != "benchmark-derived-revalidation.v1":
        problems.append("derived receipt schema is not benchmark-derived-revalidation.v1")
    if receipt.get("mode") != "preserved_rows_corrected_validator":
        problems.append("derived receipt does not identify preserved-row validator revalidation")
    source = receipt.get("source")
    if not isinstance(source, dict) or source.get("execution_commit") != source_sha:
        problems.append("derived receipt source commit does not match frozen source")

    for key in ("acceptance", "projection_acceptance", "source_acceptance"):
        problems.extend(_acceptance_problems(receipt.get(key), source_sha=source_sha, label=key))

    validator = receipt.get("validator")
    if not isinstance(validator, dict):
        problems.append("derived receipt validator provenance is missing")
    else:
        validator_commit = str(validator.get("commit") or "").lower()
        expected_reviewed = str(validator.get("expected_reviewed_commit") or "").lower()
        if not _COMMIT_RE.fullmatch(validator_commit) or validator_commit != expected_reviewed:
            problems.append("derived receipt validator commit is not self-consistent")
        if expected_validator_sha is not None:
            if not _COMMIT_RE.fullmatch(expected_validator_sha):
                problems.append("expected validator commit is malformed")
            elif validator_commit != expected_validator_sha.lower():
                problems.append(
                    "derived receipt validator commit does not match expected validator"
                )
        if _digest(validator.get("file_sha256")) is None:
            problems.append("derived receipt validator file digest is missing")

    snqi = receipt.get("snqi")
    if not isinstance(snqi, dict) or snqi.get("ranking_authority") is not False:
        problems.append("SNQI ranking authority was not disabled")
    elif str(snqi.get("status") or "").lower() not in {"advisory", "reconciled_advisory_only"}:
        problems.append("SNQI is not marked advisory")

    reconciliation = receipt.get("publication_reconciliation")
    if not isinstance(reconciliation, dict):
        problems.append("publication reconciliation provenance is missing")
    else:
        if reconciliation.get("scientific_execution_changed") is not False:
            problems.append("derived receipt claims a scientific execution change")
        if reconciliation.get("simulation_rerun") is not False:
            problems.append("derived receipt does not prove that no simulation was rerun")
        sidecars = reconciliation.get("sidecar_path_binding")
        if (
            not isinstance(sidecars, dict)
            or _int(sidecars.get("row_count")) != EXPECTED_EPISODE_CELLS
        ):
            problems.append("derived sidecar provenance does not cover all episode rows")
        timeout = reconciliation.get("goal_timeout_boundary")
        if isinstance(timeout, dict) and timeout.get("timing_evidence_fabricated") is True:
            problems.append("goal/timeout timing evidence was fabricated")

    publication_inputs = receipt.get("publication_inputs")
    if not isinstance(publication_inputs, dict):
        problems.append("derived publication inputs are missing")

    if _positive_forbidden_markers(receipt):
        problems.append("derived receipt contains a positive forbidden runtime marker")

    # The receipt itself intentionally carries the source identity; the tag,
    # base, and campaign are checked against the bundle's resolved manifest and
    # release result below.  Keeping this check separate avoids trusting a
    # single duplicated JSON field.
    del expected_tag, expected_base_sha, expected_campaign_id
    return _check(
        "derived_revalidation",
        problems,
        "derived receipt is valid, source-bound, and advisory-safe",
    ), receipt


def _bundle_check(  # noqa: C901, PLR0912, PLR0915
    bundle: Path | None,
    *,
    receipt_path: Path | None,
    receipt: dict[str, Any] | None,
    source_sha: str,
    expected_tag: str,
    expected_base_sha: str,
    expected_campaign_id: str,
) -> tuple[PostExecutionReleaseDoctorCheck, dict[str, Any] | None]:
    """Verify the publication bundle, preflight contract, and release identity.

    Returns:
        Sanitized check and parsed release result when readable.
    """
    if bundle is None or not bundle.is_dir():
        return PostExecutionReleaseDoctorCheck(
            "publication_bundle", "fail", "publication bundle directory is missing"
        ), None
    problems: list[str] = []
    manifest_path = bundle / "publication_manifest.json"
    checksums_path = bundle / "checksums.sha256"
    payload_root = bundle / "payload"
    receipt_in_bundle = payload_root / "provenance" / "derived_revalidation_receipt.json"
    result_path = payload_root / "release" / "release_result.json"
    resolved_manifest_path = payload_root / "release" / "release_manifest.resolved.json"
    for path, label in (
        (manifest_path, "publication manifest"),
        (checksums_path, "bundle checksums"),
        (payload_root, "bundle payload"),
        (receipt_in_bundle, "bundled derived receipt"),
        (result_path, "bundled release result"),
        (resolved_manifest_path, "bundled resolved release manifest"),
    ):
        exists = path.is_dir() if path == payload_root else path.is_file()
        if not exists:
            problems.append(f"{label} is missing")

    if not problems:
        try:
            preflight = verify_publication_bundle_preflight(bundle)
        except (OSError, ValueError, PublicationPreflightError) as exc:
            problems.append(f"publication preflight could not run: {type(exc).__name__}")
        else:
            if preflight.get("status") != "pass":
                problems.append("publication bundle preflight is not pass")
        try:
            publication_manifest = _read_mapping(manifest_path)
            result = _read_mapping(result_path)
            resolved_manifest = _read_mapping(resolved_manifest_path)
        except ValueError as exc:
            problems.append(str(exc))
            publication_manifest = result = resolved_manifest = None
    else:
        publication_manifest = result = resolved_manifest = None

    if not problems:
        if receipt is None or receipt_path is None:
            problems.append("supplied derived receipt is unavailable for bundle binding")
        else:
            try:
                bundled_receipt = _read_mapping(receipt_in_bundle)
                if _sha256(receipt_in_bundle) != _sha256(receipt_path):
                    problems.append("bundled derived receipt differs from supplied receipt")
                if bundled_receipt != receipt:
                    problems.append(
                        "bundled derived receipt content does not match supplied receipt"
                    )
            except (OSError, ValueError):
                problems.append("bundled derived receipt is invalid")

        benchmark_release = result.get("benchmark_release") if isinstance(result, dict) else None
        if not isinstance(benchmark_release, dict):
            problems.append("release result benchmark identity is missing")
        else:
            if benchmark_release.get("release_tag") != expected_tag:
                problems.append("release result tag does not match expected tag")
            if benchmark_release.get("latest_main_base_commit") != expected_base_sha:
                problems.append("release result base commit does not match expected base")
        if result.get("campaign_id") != expected_campaign_id:
            problems.append("release result campaign ID does not match expected campaign")
        if result.get("release_status") != "ok" or result.get("release_exit_code") != 0:
            problems.append("derived release result is not successful")
        if result.get("publication_preflight_status") != "pass":
            problems.append("release result does not record a passing publication preflight")
        for key in ("full_release_acceptance", "release_acceptance"):
            problems.extend(
                _acceptance_problems(
                    result.get(key), source_sha=source_sha, label=f"release_result.{key}"
                )
            )
        if (
            result.get("benchmark_success") is not True
            or result.get("release_benchmark_success") is not True
        ):
            problems.append("release result benchmark success is not true")
        if isinstance(resolved_manifest, dict):
            provenance = resolved_manifest.get("provenance")
            if (
                not isinstance(provenance, dict)
                or provenance.get("latest_main_base_commit") != expected_base_sha
            ):
                problems.append("resolved manifest base commit does not match expected base")
        if isinstance(publication_manifest, dict):
            channels = publication_manifest.get("publication_channels")
            if isinstance(channels, dict) and channels.get("release_tag") not in {
                None,
                expected_tag,
            }:
                problems.append("publication manifest tag does not match expected tag")

    return _check(
        "publication_bundle", problems, "publication bundle and independent preflight are valid"
    ), result if isinstance(result, dict) else None


def _publication_preflight_check(
    path: Path | None,
    bundle: Path | None,
) -> PostExecutionReleaseDoctorCheck:
    """Validate the recorded preflight and bind it to the supplied bundle.

    Returns:
        Sanitized preflight check.
    """
    if path is None or not path.is_file():
        return PostExecutionReleaseDoctorCheck(
            "publication_preflight", "fail", "publication preflight receipt is missing"
        )
    try:
        preflight = _read_mapping(path)
    except ValueError as exc:
        return PostExecutionReleaseDoctorCheck("publication_preflight", "fail", str(exc))
    problems: list[str] = []
    if preflight.get("schema_version") != "publication-preflight.v1":
        problems.append("publication preflight schema is not publication-preflight.v1")
    if preflight.get("status") != "pass":
        problems.append("publication preflight receipt is not pass")
    if _int(preflight.get("violation_count")) != 0 or preflight.get("violations") != []:
        problems.append("publication preflight records violations")
    declared_bundle = preflight.get("bundle_dir")
    if bundle is None or not isinstance(declared_bundle, str):
        problems.append("publication preflight is not bound to the supplied bundle")
    else:
        try:
            declared_path = Path(declared_bundle)
            if not declared_path.is_absolute():
                declared_path = path.parent / declared_path
            if declared_path.resolve() != bundle.resolve():
                problems.append("publication preflight bundle differs from supplied bundle")
        except OSError:
            problems.append("publication preflight bundle path is invalid")
    return _check(
        "publication_preflight", problems, "recorded publication preflight is pass and bundle-bound"
    )


def _load_rows(path: Path | None) -> list[dict[str, Any]]:
    """Load a private queue/jobs list from YAML or JSON.

    Returns:
        Ledger rows.
    """
    if path is None or not path.is_file():
        raise ValueError("private queue/job ledger is missing")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("private queue/job ledger is invalid") from exc
    if isinstance(payload, dict):
        for key in ("jobs", "queues", "rows"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("private queue/job ledger must be a list of objects")
    return payload


def _producer_file_digest(derived_receipt: dict[str, Any] | None, relative_path: str) -> str | None:
    """Read one producer digest from the signed cross-root map.

    Returns:
        Bare SHA-256 digest when the exact map entry is valid, otherwise ``None``.
    """
    if not isinstance(derived_receipt, dict):
        return None
    cross_root = derived_receipt.get("cross_root_binding")
    retrieved = cross_root.get("retrieved_file_map") if isinstance(cross_root, dict) else None
    entry = retrieved.get(relative_path) if isinstance(retrieved, dict) else None
    return _digest(entry.get("sha256")) if isinstance(entry, dict) else None


def _two_distinct_strings(value: Any) -> bool:
    """Return whether a receipt field is exactly two distinct non-empty strings."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, str) and item.strip() for item in value)
        and len(set(value)) == 2
    )


def _stream_has_credential_shape(handle: Any) -> bool:
    """Scan a binary stream for high-confidence credential shapes.

    Returns:
        Whether a credential-shaped byte sequence was found.
    """
    tail = b""
    while chunk := handle.read(1024 * 1024):
        sample = (tail + chunk).lower()
        if any(pattern in sample for pattern in _CREDENTIAL_PATTERNS):
            return True
        tail = sample[-64:]
    return False


def _stream_digest_and_credential_shape(handle: Any) -> tuple[str, bool]:
    """Hash a stream while checking its decompressed content for credentials.

    Returns:
        Lowercase SHA-256 digest and whether a credential shape was found.
    """
    digest = hashlib.sha256()
    tail = b""
    found = False
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
        sample = (tail + chunk).lower()
        found = found or any(pattern in sample for pattern in _CREDENTIAL_PATTERNS)
        tail = sample[-64:]
    return digest.hexdigest(), found


def _credential_scan_problems(  # noqa: C901, PLR0912
    bundle: Path | None,
    receipt: dict[str, Any] | None,
    archive: Path | None = None,
) -> list[str]:
    """Reject recorded credentials and high-confidence secret shapes without echoing values.

    Returns:
        Credential-free problem descriptions.
    """
    problems: list[str] = []
    if not isinstance(receipt, dict) or receipt.get("credentials") != "not_recorded":
        problems.append("derived receipt does not declare credentials not_recorded")
    if bundle is None or not bundle.is_dir():
        return problems
    try:
        files = [path for path in bundle.rglob("*") if path.is_file()]
    except OSError:
        return [*problems, "publication payload could not be scanned for credentials"]
    for path in files:
        try:
            with path.open("rb") as handle:
                if _stream_has_credential_shape(handle):
                    problems.append("publication payload contains credential-shaped content")
                    return problems
        except OSError:
            problems.append("publication payload could not be scanned for credentials")
            return problems
    if archive is None or not archive.is_file():
        problems.append("publication archive is unavailable for credential scanning")
        return problems
    try:
        with archive.open("rb") as raw_archive:
            if _stream_has_credential_shape(raw_archive):
                problems.append("publication archive contains credential-shaped raw bytes")
                return problems
        bundle_files = {path.relative_to(bundle).as_posix(): path for path in files}
        archive_files: dict[str, str] = {}
        with tarfile.open(archive, "r:*") as handle:
            for member in handle:
                if _stream_has_credential_shape(io.BytesIO(member.name.encode("utf-8"))):
                    problems.append(
                        "publication archive contains credential-shaped member metadata"
                    )
                    return problems
                if member.isdir():
                    continue
                if not member.isfile():
                    problems.append("publication archive contains a non-file/non-directory member")
                    return problems
                member_path = Path(member.name)
                parts = member_path.parts
                if (
                    member_path.is_absolute()
                    or not parts
                    or parts[0] != bundle.name
                    or any(part in {"", ".", ".."} for part in parts)
                ):
                    problems.append("publication archive member path differs from bundle layout")
                    return problems
                relative = Path(*parts[1:]).as_posix()
                if not relative or relative in archive_files:
                    problems.append("publication archive has an empty or duplicate file member")
                    return problems
                extracted = handle.extractfile(member)
                if extracted is None:
                    problems.append("publication archive file member could not be read")
                    return problems
                member_digest, has_credential = _stream_digest_and_credential_shape(extracted)
                if has_credential:
                    problems.append("publication archive contains credential-shaped content")
                    return problems
                archive_files[relative] = member_digest
        if set(archive_files) != set(bundle_files):
            problems.append("publication archive file set differs from supplied bundle")
            return problems
        for relative, member_digest in archive_files.items():
            if member_digest != _sha256(bundle_files[relative]):
                problems.append("publication archive file bytes differ from supplied bundle")
                return problems
    except (OSError, tarfile.TarError):
        problems.append("publication archive could not be scanned for credentials")
    return problems


def _derived_evaluation_problems(  # noqa: C901, PLR0912, PLR0915
    path: Path | None,
    *,
    source_sha: str,
    campaign_id: str,
    job_id: str,
    public_revalidation_receipt: Path | None,
    publication_bundle: Path | None,
    publication_archive: Path | None,
) -> tuple[list[str], bool]:
    """Validate and cross-bind the required private derived evaluation receipt.

    Returns:
        Credential-free problems and whether the receipt is accepted.
    """
    if path is None:
        return ["private derived evaluation receipt is required"], False
    try:
        payload = _read_mapping(path)
    except ValueError as exc:
        return [str(exc)], False
    problems: list[str] = []
    if payload.get("schema") != "robot-sf-derived-evaluation-receipt.v1":
        problems.append("derived evaluation receipt schema does not match")
    if payload.get("evaluation_status") != "complete" or payload.get("evidence_valid") is not True:
        problems.append("derived evaluation receipt is not complete and valid")
    if payload.get("scientific_outcome") != "not_applicable":
        problems.append("derived evaluation receipt rewrites scientific outcome")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("public_commit") != source_sha:
        problems.append("derived evaluation receipt source SHA does not match")
    if payload.get("producer_campaign_id") != campaign_id:
        problems.append("derived evaluation receipt campaign does not match")
    if str(payload.get("job_id")) != job_id:
        problems.append("derived evaluation receipt job does not match")

    execution = payload.get("execution")
    if not isinstance(execution, dict):
        problems.append("derived evaluation execution boundary is missing")
    else:
        expected_execution = {
            "simulation_rerun": False,
            "scientific_execution_status": "completed",
            "scheduler_state": "FAILED",
            "exit_code": "2:0",
            "execution_status": "failed",
            "completion_status": "failed",
            "no_rerun_authorized": True,
        }
        for key, expected in expected_execution.items():
            if execution.get(key) != expected:
                problems.append(f"derived evaluation execution {key} does not match")

    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        problems.append("derived evaluation acceptance is missing")
    else:
        expected_acceptance = {
            "status": "valid",
            "expected_planner_arms": EXPECTED_ARMS,
            "successful_planner_arms": EXPECTED_ARMS,
            "expected_episode_cells": EXPECTED_EPISODE_CELLS,
            "observed_episode_rows": EXPECTED_EPISODE_CELLS,
            "unique_episode_identities": EXPECTED_EPISODE_CELLS,
            "missing_episode_identities": 0,
            "unexpected_episode_identities": 0,
            "duplicate_episode_identities": 0,
        }
        for key, expected in expected_acceptance.items():
            actual = acceptance.get(key)
            if actual != expected and _int(actual) != expected:
                problems.append(f"derived evaluation acceptance {key} does not match")
        if acceptance.get("blockers") != [] or acceptance.get("forbidden_status_counts") != {}:
            problems.append("derived evaluation acceptance has blockers or forbidden markers")
        if acceptance.get("source_commits") != [source_sha]:
            problems.append("derived evaluation acceptance source does not match")
    snqi = payload.get("snqi")
    if (
        not isinstance(snqi, dict)
        or snqi.get("calibration_status") != "fail"
        or snqi.get("policy") != "warn"
        or snqi.get("status") != "advisory"
        or snqi.get("ranking_authority") is not False
        or snqi.get("ranking_claims_admitted") is not False
    ):
        problems.append("derived evaluation SNQI boundary is not advisory/no-ranking")

    derived = payload.get("derived_bundle")
    if not isinstance(derived, dict):
        problems.append("derived publication bundle binding is missing")
    else:
        release_result = (
            publication_bundle / "payload" / "release" / "release_result.json"
            if publication_bundle is not None
            else None
        )
        expected_paths = {
            "archive": publication_archive,
            "revalidation_receipt": public_revalidation_receipt,
            "release_result": release_result,
        }
        for key in ("archive", "revalidation_receipt", "release_result"):
            item = derived.get(key)
            if not isinstance(item, dict):
                problems.append(f"derived publication {key} binding is missing")
                continue
            bound_path = Path(str(item.get("path") or ""))
            expected_digest = _digest(item.get("sha256"))
            valid_file = bound_path.is_absolute() and bound_path.is_file()
            if not valid_file or expected_digest is None:
                problems.append(f"derived publication {key} binding is invalid")
            elif _sha256(bound_path) != expected_digest:
                problems.append(f"derived publication {key} digest does not match")
            expected_path = expected_paths[key]
            if expected_path is None or not expected_path.is_file():
                problems.append(f"doctor input for derived publication {key} is missing")
            elif expected_digest != _sha256(expected_path):
                problems.append(f"private/public derived publication {key} digests differ")
            if (
                key == "archive"
                and valid_file
                and _int(item.get("bytes")) != bound_path.stat().st_size
            ):
                problems.append("derived publication archive size does not match")

    preservation = payload.get("preservation")
    if not isinstance(preservation, dict):
        problems.append("derived preservation binding is missing")
    else:
        expected_preservation = {
            "status": "verified",
            "two_copy_satisfied": True,
            "remote_state": "COMMITTED",
            "readback_match": True,
            "digest_mismatches": 0,
        }
        for key, expected in expected_preservation.items():
            if preservation.get(key) != expected:
                problems.append(f"derived preservation {key} does not match")
        receipt_path = Path(str(preservation.get("receipt_path") or ""))
        if not receipt_path.is_absolute() or not receipt_path.is_file():
            problems.append("derived preservation receipt path is invalid")
        else:
            try:
                preservation_receipt = _read_mapping(receipt_path)
            except ValueError:
                problems.append("derived preservation receipt is invalid")
            else:
                if preservation_receipt.get("status") != "verified":
                    problems.append("derived preservation receipt is not verified")
                if preservation_receipt.get("campaign_id") != payload.get("derived_campaign_id"):
                    problems.append("derived preservation campaign does not match")
                if preservation_receipt.get("receipt_digest") != preservation.get("receipt_digest"):
                    problems.append("derived preservation receipt digest does not match")
                if preservation_receipt.get("manifest_digest") != preservation.get(
                    "manifest_digest"
                ):
                    problems.append("derived preservation manifest digest does not match")
                policy = preservation_receipt.get("two_copy_policy")
                if (
                    not isinstance(policy, dict)
                    or policy.get("satisfied") is not True
                    or _int(policy.get("verified_copies")) != 2
                    or not _two_distinct_strings(policy.get("distinct_failure_domains"))
                    or not _two_distinct_strings(policy.get("distinct_backend_classes"))
                ):
                    problems.append("derived preservation receipt does not prove two copies")
    if _positive_forbidden_markers(payload):
        problems.append("derived evaluation receipt contains a positive forbidden marker")
    return problems, not problems


def _private_execution_check(  # noqa: C901, PLR0912, PLR0913
    queue_path: Path | None,
    jobs_path: Path | None,
    *,
    source_sha: str,
    campaign_id: str,
    job_id: str,
    derived_evaluation_receipt: Path | None,
    public_revalidation_receipt: Path | None,
    publication_bundle: Path | None,
    publication_archive: Path | None,
) -> PostExecutionReleaseDoctorCheck:
    """Join terminal queue/job identity without requiring dispatchability.

    Returns:
        Sanitized private execution check.
    """
    problems: list[str] = []
    try:
        queues = _load_rows(queue_path)
        jobs = _load_rows(jobs_path)
    except ValueError as exc:
        return PostExecutionReleaseDoctorCheck("private_execution", "fail", str(exc))
    queue_matches = [row for row in queues if str(row.get("campaign")) == campaign_id]
    job_matches = [row for row in jobs if str(row.get("job_id")) == job_id]
    if len(queue_matches) != 1:
        problems.append("private queue does not contain exactly one matching campaign row")
    if len(job_matches) != 1:
        problems.append("private jobs ledger does not contain exactly one matching job")
    if not queue_matches or not job_matches:
        return _check(
            "private_execution", problems, "terminal queue/job identity is consumed exactly once"
        )
    queue, job = queue_matches[0], job_matches[0]
    if queue.get("queue_id") is None or queue.get("queue_id") != job.get(
        "queue_id", queue.get("queue_id")
    ):
        # jobs.yaml historically omits queue_id; in that shape the campaign is
        # the stable join and the queue row still binds the attempt identity.
        if job.get("queue_id") is not None:
            problems.append("queue/job queue identity differs")
    if queue.get("expected_public_commit") != source_sha or job.get("public_commit") != source_sha:
        problems.append("queue/job source SHA does not match frozen source")
    if queue.get("campaign") != job.get("campaign"):
        problems.append("queue/job campaign identity differs")
    if _int(queue.get("attempts")) != 1 or _int(queue.get("max_attempts")) != 1:
        problems.append("queue row does not prove one consumed attempt")
    if str(queue.get("state") or "").lower() not in {"failed", "complete", "done", "analyzed"}:
        problems.append("queue row is not terminal")
    if str(queue.get("go") or "").lower() not in {"false", "0"}:
        problems.append("terminal queue row is still dispatchable")
    if str(job.get("state") or "").lower() not in {"analyzed", "finished", "failed", "complete"}:
        problems.append("job row is not terminal")

    eval_problems, eval_accepted = _derived_evaluation_problems(
        derived_evaluation_receipt,
        source_sha=source_sha,
        campaign_id=campaign_id,
        job_id=job_id,
        public_revalidation_receipt=public_revalidation_receipt,
        publication_bundle=publication_bundle,
        publication_archive=publication_archive,
    )
    if eval_problems:
        problems.extend(eval_problems)
    failure_text = " ".join(
        str(queue.get(key) or "") + " " + str(job.get(key) or "")
        for key in (
            "go_reason",
            "block_reason",
            "terminal_triage_reason",
            "summary",
            "submission_evidence",
        )
    ).lower()
    if str(job_id).lower() not in failure_text:
        # The terminal job ledger supplies the structured ID; the historical
        # queue narrative must also name the consumed scheduler identity.
        problems.append("queue/job records do not name the expected consumed job")
    validator_only = (
        str(job.get("slurm_state") or "").upper() == "FAILED"
        and str(job.get("exit_code") or "") == "2:0"
        and str(job.get("artifact_status") or "").lower() in {"verified", "preserved"}
        and any(
            term in failure_text for term in ("validator", "publication gate", "publication-grade")
        )
    )
    if _positive_forbidden_markers(queue) or _positive_forbidden_markers(job):
        problems.append("private queue/job records contain a positive forbidden runtime marker")
    if not validator_only:
        problems.append("terminal job is not bound to the validator-only scheduler failure")
    if not eval_accepted:
        problems.append("terminal job is not bound to the accepted derived evaluation receipt")
    return _check(
        "private_execution",
        problems,
        "terminal queue/job identity is consumed exactly once; dispatchability not required",
    )


def _historical_provenance_check(  # noqa: C901, PLR0912, PLR0915
    bundle: Path | None,
    packet_path: Path | None,
    queue_path: Path | None,
    derived_receipt: dict[str, Any] | None,
    *,
    source_sha: str,
    expected_tag: str,
    expected_base_sha: str,
    campaign_id: str,
) -> PostExecutionReleaseDoctorCheck:
    """Verify admitted packet, checkpoint, runtime-smoke, and stress lineage.

    Returns:
        Sanitized historical provenance check.
    """
    packet: dict[str, Any] | None = None
    bundled_packet: dict[str, Any] | None = None
    bundled_packet_path = bundle / "payload" / "launch_packet.yaml" if bundle is not None else None
    if bundled_packet_path is not None and bundled_packet_path.is_file():
        try:
            bundled_packet = _read_mapping(bundled_packet_path, yaml_ok=True)
        except ValueError as exc:
            return PostExecutionReleaseDoctorCheck("historical_provenance", "fail", str(exc))
    if packet_path is not None:
        if not packet_path.is_file():
            return PostExecutionReleaseDoctorCheck(
                "historical_provenance", "fail", "historical admitted launch packet is missing"
            )
        try:
            packet = _read_mapping(packet_path, yaml_ok=True)
        except ValueError as exc:
            return PostExecutionReleaseDoctorCheck("historical_provenance", "fail", str(exc))
    elif bundled_packet is not None:
        packet = bundled_packet
    else:
        return PostExecutionReleaseDoctorCheck(
            "historical_provenance", "fail", "historical admitted launch packet is missing"
        )

    problems: list[str] = []
    producer_packet_sha = _producer_file_digest(derived_receipt, "launch_packet.yaml")
    if packet_path is None or producer_packet_sha != _sha256(packet_path):
        problems.append("producer launch packet differs from the signed cross-root map")
    if packet_path is not None and bundled_packet is not None:
        for key in ("schema", "queue_id", "campaign_id", "campaign", "status"):
            if packet.get(key) != bundled_packet.get(key):
                problems.append(f"historical launch packet {key} differs from bundled packet")
        packet_identity = packet.get("identity")
        bundled_identity = bundled_packet.get("identity")
        if not isinstance(packet_identity, dict) or packet_identity != bundled_identity:
            problems.append("historical launch packet identity differs from bundled packet")
    if (
        packet.get("schema") != "robot-sf-launch-packet.v1"
        or packet.get("status") != "admitted_frozen"
    ):
        problems.append("historical launch packet is not admitted_frozen")
    if packet.get("campaign_id") != campaign_id or packet.get("campaign") != campaign_id:
        problems.append("historical launch packet campaign does not match")
    identity = packet.get("identity")
    if not isinstance(identity, dict):
        problems.append("historical launch packet identity is missing")
        identity = {}
    if identity.get("public_source_commit") != source_sha:
        problems.append("historical launch packet source SHA does not match")
    if (
        identity.get("release_tag") != expected_tag
        or identity.get("latest_main_base_commit") != expected_base_sha
    ):
        problems.append("historical launch packet tag/base identity does not match")
    if (
        not isinstance(identity.get("checkpoint_receipt_path"), str)
        or _digest(identity.get("checkpoint_receipt_sha256")) is None
    ):
        problems.append("historical checkpoint receipt is not checksum-bound")
    try:
        queue_rows = _load_rows(queue_path)
    except ValueError as exc:
        problems.append(str(exc))
    else:
        queue_matches = [row for row in queue_rows if row.get("campaign") == campaign_id]
        if len(queue_matches) != 1:
            problems.append("historical packet has no unique private queue identity")
        else:
            queue = queue_matches[0]
            if queue.get("queue_id") != packet.get("queue_id"):
                problems.append("historical packet queue ID differs from tracked queue")
            if packet_path is None:
                problems.append("tracked queue cannot bind a missing producer launch packet")
            else:
                artifact_manifest = str(queue.get("artifact_manifest") or "")
                artifact_path, separator, artifact_digest = artifact_manifest.partition(" sha256:")
                if not artifact_path.endswith(packet_path.name):
                    problems.append("tracked queue artifact manifest does not match packet")
                if not separator:
                    problems.append("tracked queue artifact manifest is not packet-digest-bound")
                elif (
                    producer_packet_sha is None
                    or not _SHA256_RE.fullmatch(artifact_digest)
                    or artifact_digest.lower() != producer_packet_sha
                ):
                    problems.append("tracked queue does not bind the producer launch packet digest")
    smoke = packet.get("accepted_runtime_smoke")
    if not isinstance(smoke, dict) or smoke.get("status") != "accepted_preserved_verified":
        problems.append("historical runtime smoke is not accepted and preserved")
    else:
        if (
            smoke.get("public_source_commit") != source_sha
            or _int(smoke.get("expected_episode_cells")) != EXPECTED_RUNTIME_SMOKE_CELLS
        ):
            problems.append("historical runtime smoke identity does not match")
        if (
            _int(smoke.get("fallback_or_degraded_rows")) != 0
            or smoke.get("exit_code") != "0:0"
            or smoke.get("derived_exit_code") != "0:0"
        ):
            problems.append("historical runtime smoke is not clean")
    stress = packet.get("accepted_hybrid_stress")
    if not isinstance(stress, dict) or stress.get("status") != "accepted_preserved_verified":
        problems.append("historical hybrid stress is not accepted and preserved")
    else:
        if (
            stress.get("public_source_commit") != source_sha
            or _int(stress.get("expected_episode_cells")) != EXPECTED_STRESS_CELLS
        ):
            problems.append("historical hybrid stress identity does not match")
        for key in ("gate_receipt_sha256", "release_result_sha256", "preservation_manifest_sha256"):
            if _digest(stress.get(key)) is None:
                problems.append(f"historical hybrid stress {key} is not checksum-bound")

    if bundle is not None:
        payload = bundle / "payload"
        checkpoint_path = payload / "checkpoint_staging_receipt.json"
        smoke_path = payload / "runtime_smoke_release_result.json"
        stress_path = payload / "accepted_hybrid_stress_gate.json"
        for path, label in (
            (checkpoint_path, "checkpoint"),
            (smoke_path, "runtime smoke"),
            (stress_path, "hybrid stress"),
        ):
            if not path.is_file():
                problems.append(f"publication payload omits historical {label} evidence")
        if checkpoint_path.is_file():
            try:
                checkpoint_payload = _read_mapping(checkpoint_path)
            except ValueError:
                problems.append("publication checkpoint evidence is invalid")
            else:
                if (
                    checkpoint_payload.get("schema_version")
                    != "campaign-checkpoint-staging-receipt.v1"
                    or checkpoint_payload.get("status") != "ok"
                    or checkpoint_payload.get("mode") != "enforced_staged"
                    or checkpoint_payload.get("submit_safe") is not True
                    or _int(checkpoint_payload.get("checked")) != 5
                    or _int(checkpoint_payload.get("resolved")) != 5
                ):
                    problems.append("publication checkpoint evidence is not accepted staged input")
                # Publication intentionally replaces workstation-absolute checkpoint paths
                # with repository-relative paths.  Bind the original producer receipt to
                # the admitted packet through the independently checksummed cross-root map,
                # while validating the portable receipt semantically above.
                producer_checkpoint_sha = _producer_file_digest(
                    derived_receipt, "checkpoint_staging_receipt.json"
                )
                if producer_checkpoint_sha != _digest(identity.get("checkpoint_receipt_sha256")):
                    problems.append("producer checkpoint receipt differs from the admitted packet")
        if smoke_path.is_file():
            try:
                smoke_result = _read_mapping(smoke_path)
                if smoke_result.get("benchmark_success") is not True:
                    problems.append("publication runtime-smoke result is not exact-source success")
                for source_key in ("git_hash", "source_commit", "public_source_commit"):
                    if source_key in smoke_result and smoke_result[source_key] != source_sha:
                        problems.append("publication runtime-smoke result source is not exact")
            except ValueError:
                problems.append("publication runtime-smoke result is invalid")
        if stress_path.is_file():
            try:
                stress_result = _read_mapping(stress_path)
                if (
                    stress_result.get("status") != "accepted_preserved_verified"
                    or stress_result.get("public_source_commit") != source_sha
                ):
                    problems.append(
                        "publication hybrid-stress evidence is not exact-source accepted"
                    )
            except ValueError:
                problems.append("publication hybrid-stress evidence is invalid")

    return _check(
        "historical_provenance",
        problems,
        "admitted packet, checkpoint, runtime-smoke, and stress lineage are bound",
    )


def collect_post_execution_release_doctor_report(  # noqa: PLR0913
    *,
    repo: Path,
    derived_revalidation_receipt: Path | None,
    publication_bundle: Path | None,
    publication_archive: Path | None,
    publication_preflight: Path | None,
    private_queue: Path | None,
    private_jobs: Path | None,
    manifest_path: Path | None = None,
    private_launch_packet: Path | None = None,
    private_evaluation_receipt: Path | None = None,
    dissertation: Path | None = None,
    token_file: Path | None = None,
    minimum_free_gib: float = 100.0,
    require_zenodo_webhook_disabled: bool = True,
    expected_source_sha: str = FROZEN_SOURCE_SHA,
    expected_base_sha: str = EXPECTED_BASE_SHA,
    tag: str = EXPECTED_RELEASE_TAG,
    expected_campaign_id: str = EXPECTED_CAMPAIGN_ID,
    expected_job_id: str = EXPECTED_JOB_ID,
    expected_validator_sha: str = EXPECTED_VALIDATOR_SHA,
) -> dict[str, Any]:
    """Collect the read-only post-execution publication checks.

    Returns:
        Credential-free doctor report.
    """
    checks: list[PostExecutionReleaseDoctorCheck] = []
    source_problems: list[str] = []
    if not isinstance(expected_source_sha, str) or not _COMMIT_RE.fullmatch(expected_source_sha):
        source_problems.append("expected source SHA is malformed")
    elif expected_source_sha != FROZEN_SOURCE_SHA:
        source_problems.append("post-execution doctor is fixed to the frozen b1d5 source")
    checks.append(_check("source_identity", source_problems, "frozen source SHA is exact"))
    release_identity_problems: list[str] = []
    if expected_base_sha != EXPECTED_BASE_SHA:
        release_identity_problems.append(
            "expected base SHA does not match the frozen release contract"
        )
    if tag != EXPECTED_RELEASE_TAG:
        release_identity_problems.append("release tag does not match the frozen release contract")
    if expected_campaign_id != EXPECTED_CAMPAIGN_ID:
        release_identity_problems.append("campaign ID does not match the frozen release contract")
    if expected_job_id != EXPECTED_JOB_ID:
        release_identity_problems.append("job ID does not match the frozen release contract")
    if expected_validator_sha != EXPECTED_VALIDATOR_SHA:
        release_identity_problems.append(
            "validator SHA does not match the reviewed frozen release contract"
        )
    checks.append(
        _check(
            "release_identity",
            release_identity_problems,
            "release tag, base, campaign, and consumed job are exact",
        )
    )
    if source_problems or release_identity_problems:
        failed = [check.name for check in checks if check.status != "pass"]
        return {
            "schema_version": "robot-sf-post-execution-release-doctor.v1",
            "status": "blocked",
            "expected_source_sha": expected_source_sha,
            "expected_base_sha": expected_base_sha,
            "release_tag": tag,
            "expected_campaign_id": expected_campaign_id,
            "expected_job_id": expected_job_id,
            "checks": [asdict(check) for check in checks],
            "failed_checks": failed,
        }
    checks.extend(
        [
            _adapt_check(_git_check(repo, expected_source_sha)),
            _adapt_check(_ci_check(repo, expected_source_sha)),
            _adapt_check(_tag_check(repo, tag)),
            _source_manifest_check(
                manifest_path,
                repo=repo,
                source_sha=expected_source_sha,
                base_sha=expected_base_sha,
                tag=tag,
            ),
            _disk_check(repo, minimum_free_gib),
        ]
    )
    checks.extend(
        _adapt_check(check)
        for check in _zenodo_check(
            repo,
            token_file,
            require_hook_disabled=require_zenodo_webhook_disabled,
        )
    )
    checks.append(_adapt_check(_dissertation_check(dissertation)))
    receipt_check, receipt = _receipt_check(
        derived_revalidation_receipt,
        source_sha=expected_source_sha,
        expected_tag=tag,
        expected_base_sha=expected_base_sha,
        expected_campaign_id=expected_campaign_id,
        expected_validator_sha=expected_validator_sha,
    )
    checks.append(receipt_check)
    bundle_check, _ = _bundle_check(
        publication_bundle,
        receipt_path=derived_revalidation_receipt,
        receipt=receipt,
        source_sha=expected_source_sha,
        expected_tag=tag,
        expected_base_sha=expected_base_sha,
        expected_campaign_id=expected_campaign_id,
    )
    checks.append(bundle_check)
    checks.append(
        _check(
            "credential_safety",
            _credential_scan_problems(publication_bundle, receipt, publication_archive),
            "receipts and publication payload contain no recorded credentials",
        )
    )

    checks.append(_publication_preflight_check(publication_preflight, publication_bundle))

    checks.append(
        _private_execution_check(
            private_queue,
            private_jobs,
            source_sha=expected_source_sha,
            campaign_id=expected_campaign_id,
            job_id=expected_job_id,
            derived_evaluation_receipt=private_evaluation_receipt,
            public_revalidation_receipt=derived_revalidation_receipt,
            publication_bundle=publication_bundle,
            publication_archive=publication_archive,
        )
    )
    checks.append(
        _historical_provenance_check(
            publication_bundle,
            private_launch_packet,
            private_queue,
            receipt,
            source_sha=expected_source_sha,
            expected_tag=tag,
            expected_base_sha=expected_base_sha,
            campaign_id=expected_campaign_id,
        )
    )
    failed = [check.name for check in checks if check.status != "pass"]
    return {
        "schema_version": "robot-sf-post-execution-release-doctor.v1",
        "status": "pass" if not failed else "blocked",
        "expected_source_sha": expected_source_sha,
        "expected_base_sha": expected_base_sha,
        "release_tag": tag,
        "expected_campaign_id": expected_campaign_id,
        "expected_job_id": expected_job_id,
        "checks": [asdict(check) for check in checks],
        "failed_checks": failed,
    }


__all__ = [
    "EXPECTED_BASE_SHA",
    "EXPECTED_CAMPAIGN_ID",
    "EXPECTED_EPISODE_CELLS",
    "EXPECTED_RELEASE_TAG",
    "EXPECTED_VALIDATOR_SHA",
    "FROZEN_SOURCE_SHA",
    "PostExecutionReleaseDoctorCheck",
    "collect_post_execution_release_doctor_report",
]

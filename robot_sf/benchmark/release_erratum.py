"""Fail-closed contracts and scientific-equality proof for release errata.

An erratum is a new publication identity derived from an immutable predecessor.
It may repair publication metadata, but it must not silently change benchmark
episode rows or component metrics.  This module keeps those two identities
separate and produces the compact equality evidence embedded in the successor.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

ERRATUM_CONTRACT_SCHEMA = "benchmark-release-erratum.v1"
ERRATUM_RECEIPT_SCHEMA = "benchmark-release-erratum-receipt.v1"
ERRATUM_SCOPE = "derived_publication_metadata_only"
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOI_RE = re.compile(r"^10\.5281/zenodo\.(\d+)$")
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]+$")
_EPISODE_MEMBER_RE = re.compile(r"^[^/]+/payload/runs/([^/]+)/episodes\.jsonl$")
_MAX_ARCHIVE_MEMBERS = 20_000
_MAX_EXPANDED_BYTES = 4 * 1024**3
_MAX_EPISODE_FILE_BYTES = 256 * 1024**2
_MAX_EPISODE_ROW_BYTES = 16 * 1024**2
_SCIENTIFIC_ROW_STATUSES = frozenset({"success", "collision", "failure"})


class ReleaseErratumError(RuntimeError):
    """Raised when an erratum identity or scientific equality check fails."""


@dataclass(frozen=True)
class ErratumContract:
    """Immutable predecessor, successor, and scientific identities."""

    correction_id: str
    predecessor_version_doi: str
    predecessor_archive_sha256: str
    predecessor_archive_size_bytes: int
    predecessor_github_release_tag: str
    source_sha: str
    planner_arms: int
    scenario_count: int
    seed_count: int
    episode_rows: int
    builder_sha: str
    concept_doi: str
    successor_version_doi: str
    successor_github_release_tag: str
    metadata_path: Path
    metadata_sha256: str


@dataclass(frozen=True)
class _RowDigests:
    """Canonical leaf digests retained only while two snapshots are compared."""

    row_sha256: str
    component_sha256: str


@dataclass(frozen=True)
class ScientificSnapshot:
    """Compact deterministic identity for one complete benchmark matrix."""

    source_sha: str
    planner_arms: int
    scenario_count: int
    seed_count: int
    episode_rows: int
    episode_identity_manifest_sha256: str
    component_leaf_manifest_sha256: str
    canonical_row_manifest_sha256: str
    per_arm: Mapping[str, Mapping[str, Any]]
    _rows: Mapping[tuple[str, str, int, str], _RowDigests] = field(repr=False, compare=False)

    def public_dict(self) -> dict[str, Any]:
        """Return the credential-free compact form suitable for a receipt."""
        return {
            "source_sha": self.source_sha,
            "planner_arms": self.planner_arms,
            "scenario_count": self.scenario_count,
            "seed_count": self.seed_count,
            "episode_rows": self.episode_rows,
            "episode_identity_manifest_sha256": self.episode_identity_manifest_sha256,
            "component_leaf_manifest_sha256": self.component_leaf_manifest_sha256,
            "canonical_row_manifest_sha256": self.canonical_row_manifest_sha256,
            "per_arm": dict(self.per_arm),
        }


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseErratumError(f"{label} must be an object")
    return value


def _required_text(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseErratumError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _required_positive_int(payload: Mapping[str, Any], key: str, *, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReleaseErratumError(f"{label}.{key} must be a positive integer")
    return value


def _doi_record_id(value: str, *, label: str) -> str:
    match = _DOI_RE.fullmatch(value)
    if match is None:
        raise ReleaseErratumError(f"{label} must be a Zenodo DOI")
    return match.group(1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseErratumError("scientific row is not canonical JSON") from exc


def _manifest_digest(rows: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_regular_file(path: Path, *, label: str) -> Path:
    candidate = Path(path.absolute())
    if any(parent.is_symlink() for parent in candidate.parents) or candidate.is_symlink():
        raise ReleaseErratumError(f"{label} contains a symlink")
    if not candidate.is_file():
        raise ReleaseErratumError(f"{label} is missing")
    return candidate


def _validate_metadata_file(
    path: Path,
    *,
    digest: str,
    successor_tag: str,
    predecessor_doi: str,
) -> None:
    metadata_path = _safe_regular_file(path, label="erratum metadata")
    if _sha256_file(metadata_path) != digest:
        raise ReleaseErratumError("erratum metadata SHA-256 does not match the contract")
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReleaseErratumError("erratum metadata is not readable JSON") from exc
    metadata = _require_mapping(document, label="erratum metadata").get("metadata")
    metadata = _require_mapping(metadata, label="erratum metadata.metadata")
    related = metadata.get("related_identifiers")
    if not isinstance(related, list):
        raise ReleaseErratumError("erratum metadata related_identifiers must be a list")
    source_url = f"https://github.com/ll7/robot_sf_ll7/releases/tag/{successor_tag}"
    source_matches = [
        item
        for item in related
        if isinstance(item, Mapping)
        and item.get("identifier") == source_url
        and item.get("relation") == "isSupplementTo"
    ]
    predecessor_matches = [
        item
        for item in related
        if isinstance(item, Mapping)
        and item.get("identifier") == predecessor_doi
        and item.get("relation") == "isNewVersionOf"
    ]
    if len(source_matches) != 1:
        raise ReleaseErratumError("erratum metadata must bind exactly one successor GitHub tag")
    if len(predecessor_matches) != 1:
        raise ReleaseErratumError("erratum metadata must bind exactly one predecessor version DOI")
    description = str(metadata.get("description", "")).casefold()
    required_terms = ("erratum", "no simulation", "unchanged", "snqi", "advisory", "ranking")
    if not all(term in description for term in required_terms):
        raise ReleaseErratumError(
            "erratum metadata must state the correction, unchanged rows, no simulation rerun, "
            "and advisory/no-ranking SNQI boundary"
        )


def load_erratum_contract(  # noqa: C901, PLR0912, PLR0915
    path: Path, *, repository_root: Path
) -> ErratumContract:
    """Load one exact, repository-bound successor publication contract.

    Returns:
        The validated immutable erratum contract.
    """
    contract_path = _safe_regular_file(path, label="erratum contract")
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReleaseErratumError("erratum contract is not readable JSON") from exc
    payload = _require_mapping(payload, label="erratum contract")
    if payload.get("schema_version") != ERRATUM_CONTRACT_SCHEMA:
        raise ReleaseErratumError("erratum contract schema_version is unsupported")
    if payload.get("correction_scope") != ERRATUM_SCOPE:
        raise ReleaseErratumError("erratum correction_scope is not derived-metadata-only")

    supersedes = _require_mapping(payload.get("supersedes"), label="supersedes")
    scientific = _require_mapping(payload.get("scientific_identity"), label="scientific_identity")
    derivation = _require_mapping(payload.get("derivation"), label="derivation")
    successor = _require_mapping(payload.get("successor"), label="successor")
    verdict = _require_mapping(payload.get("corrected_verdict"), label="corrected_verdict")

    predecessor_doi = _required_text(supersedes, "version_doi", label="supersedes")
    predecessor_tag = _required_text(supersedes, "github_release_tag", label="supersedes")
    predecessor_digest = _required_text(supersedes, "archive_sha256", label="supersedes").lower()
    source_sha = _required_text(scientific, "source_sha", label="scientific_identity").lower()
    builder_sha = _required_text(derivation, "builder_sha", label="derivation").lower()
    concept_doi = _required_text(successor, "concept_doi", label="successor")
    successor_doi = _required_text(successor, "version_doi", label="successor")
    successor_tag = _required_text(successor, "github_release_tag", label="successor")
    metadata_digest = _required_text(successor, "metadata_sha256", label="successor").lower()

    _doi_record_id(predecessor_doi, label="supersedes.version_doi")
    concept_record_id = _doi_record_id(concept_doi, label="successor.concept_doi")
    successor_record_id = _doi_record_id(successor_doi, label="successor.version_doi")
    if len({predecessor_doi, concept_doi, successor_doi}) != 3:
        raise ReleaseErratumError("predecessor, concept, and successor DOIs must be distinct")
    if concept_record_id == successor_record_id:
        raise ReleaseErratumError("successor concept and version records must be distinct")
    if _SHA256_RE.fullmatch(predecessor_digest) is None:
        raise ReleaseErratumError("supersedes.archive_sha256 must be a lowercase SHA-256")
    if _SHA256_RE.fullmatch(metadata_digest) is None:
        raise ReleaseErratumError("successor.metadata_sha256 must be a lowercase SHA-256")
    if _SHA1_RE.fullmatch(source_sha) is None or _SHA1_RE.fullmatch(builder_sha) is None:
        raise ReleaseErratumError("scientific source and builder SHAs must be full lowercase SHAs")
    if not _TAG_RE.fullmatch(predecessor_tag) or not _TAG_RE.fullmatch(successor_tag):
        raise ReleaseErratumError("erratum GitHub release tags are invalid")
    if successor_tag != f"{predecessor_tag}-erratum.1":
        raise ReleaseErratumError("successor tag must be the predecessor tag plus -erratum.1")
    if derivation.get("simulation_rerun") is not False:
        raise ReleaseErratumError("erratum derivation must record simulation_rerun=false")
    if supersedes.get("old_publication_retained") is not True:
        raise ReleaseErratumError("erratum must retain the immutable predecessor publication")
    if verdict.get("publication_preflight_status") != "pass":
        raise ReleaseErratumError("corrected publication preflight status must be pass")
    if verdict.get("publication_preflight_violations") != []:
        raise ReleaseErratumError("corrected publication preflight violations must be empty")
    if verdict.get("release_status") != "ok":
        raise ReleaseErratumError("corrected release status must be ok")
    if verdict.get("ranking_claims_admitted") is not False:
        raise ReleaseErratumError("erratum must not admit SNQI ranking claims")

    planner_arms = _required_positive_int(scientific, "planner_arms", label="scientific_identity")
    scenario_count = _required_positive_int(
        scientific, "scenario_count", label="scientific_identity"
    )
    seed_count = _required_positive_int(scientific, "seed_count", label="scientific_identity")
    episode_rows = _required_positive_int(scientific, "episode_rows", label="scientific_identity")
    if planner_arms * scenario_count * seed_count != episode_rows:
        raise ReleaseErratumError("scientific matrix cardinality does not equal episode_rows")

    raw_metadata_path = _required_text(successor, "metadata_path", label="successor")
    metadata_relative = Path(raw_metadata_path)
    if metadata_relative.is_absolute() or ".." in metadata_relative.parts:
        raise ReleaseErratumError("successor.metadata_path must be repository-relative")
    root = Path(repository_root).resolve()
    metadata_path = (root / metadata_relative).resolve()
    if not metadata_path.is_relative_to(root):
        raise ReleaseErratumError("successor.metadata_path escapes the repository")
    _validate_metadata_file(
        metadata_path,
        digest=metadata_digest,
        successor_tag=successor_tag,
        predecessor_doi=predecessor_doi,
    )

    return ErratumContract(
        correction_id=_required_text(payload, "correction_id", label="erratum contract"),
        predecessor_version_doi=predecessor_doi,
        predecessor_archive_sha256=predecessor_digest,
        predecessor_archive_size_bytes=_required_positive_int(
            supersedes, "archive_size_bytes", label="supersedes"
        ),
        predecessor_github_release_tag=predecessor_tag,
        source_sha=source_sha,
        planner_arms=planner_arms,
        scenario_count=scenario_count,
        seed_count=seed_count,
        episode_rows=episode_rows,
        builder_sha=builder_sha,
        concept_doi=concept_doi,
        successor_version_doi=successor_doi,
        successor_github_release_tag=successor_tag,
        metadata_path=metadata_path,
        metadata_sha256=metadata_digest,
    )


def _validate_archive_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseErratumError("predecessor archive contains an unsafe member path")
    if not (member.isdir() or member.isreg()):
        raise ReleaseErratumError("predecessor archive contains a non-regular member")
    if member.size < 0:
        raise ReleaseErratumError("predecessor archive contains a negative member size")


def _read_episode_rows(stream: BinaryIO, *, arm: str) -> Iterable[Mapping[str, Any]]:
    for line_number, raw_line in enumerate(stream, start=1):
        if len(raw_line) > _MAX_EPISODE_ROW_BYTES:
            raise ReleaseErratumError(f"{arm} episodes.jsonl contains an oversized row")
        if not raw_line.strip():
            raise ReleaseErratumError(f"{arm} episodes.jsonl contains an empty row")
        try:
            row = json.loads(raw_line)
        except (UnicodeError, ValueError) as exc:
            raise ReleaseErratumError(
                f"{arm} episodes.jsonl row {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, Mapping):
            raise ReleaseErratumError(f"{arm} episodes.jsonl row {line_number} is not an object")
        yield row


def _snapshot_from_arm_rows(  # noqa: C901
    arm_rows: Mapping[str, Iterable[Mapping[str, Any]]], *, contract: ErratumContract
) -> ScientificSnapshot:
    rows: dict[tuple[str, str, int, str], _RowDigests] = {}
    scenarios: set[str] = set()
    seeds: set[int] = set()
    per_arm_rows: dict[str, list[tuple[tuple[str, str, int, str], _RowDigests]]] = {}
    for arm in sorted(arm_rows):
        if not arm or "/" in arm or arm in {".", ".."}:
            raise ReleaseErratumError("scientific snapshot contains an invalid planner arm")
        arm_entries: list[tuple[tuple[str, str, int, str], _RowDigests]] = []
        for row in arm_rows[arm]:
            scenario = row.get("scenario_id")
            seed = row.get("seed")
            episode_id = row.get("episode_id")
            if not isinstance(scenario, str) or not scenario:
                raise ReleaseErratumError(f"{arm} row has an invalid scenario_id")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ReleaseErratumError(f"{arm} row has an invalid seed")
            if not isinstance(episode_id, str) or not episode_id:
                raise ReleaseErratumError(f"{arm} row has an invalid episode_id")
            if row.get("status") not in _SCIENTIFIC_ROW_STATUSES:
                raise ReleaseErratumError(f"{arm}/{episode_id} has a forbidden row status")
            for source_path, source_value in (
                ("git_hash", row.get("git_hash")),
                (
                    "provenance.git_hash",
                    _require_mapping(row.get("provenance"), label="provenance").get("git_hash"),
                ),
                (
                    "result_provenance.repo_commit",
                    _require_mapping(row.get("result_provenance"), label="result_provenance").get(
                        "repo_commit"
                    ),
                ),
            ):
                if source_value != contract.source_sha:
                    raise ReleaseErratumError(
                        f"{arm}/{episode_id} {source_path} differs from the frozen source"
                    )
            metrics = _require_mapping(row.get("metrics"), label=f"{arm}/{episode_id}.metrics")
            identity = (arm, scenario, seed, episode_id)
            if identity in rows:
                raise ReleaseErratumError(f"duplicate scientific identity: {identity!r}")
            digests = _RowDigests(
                row_sha256=hashlib.sha256(_canonical_bytes(row)).hexdigest(),
                component_sha256=hashlib.sha256(_canonical_bytes(metrics)).hexdigest(),
            )
            rows[identity] = digests
            arm_entries.append((identity, digests))
            scenarios.add(scenario)
            seeds.add(seed)
        per_arm_rows[arm] = arm_entries

    if len(per_arm_rows) != contract.planner_arms:
        raise ReleaseErratumError("scientific snapshot planner-arm count is incorrect")
    if len(rows) != contract.episode_rows:
        raise ReleaseErratumError("scientific snapshot episode-row count is incorrect")
    if len(scenarios) != contract.scenario_count or len(seeds) != contract.seed_count:
        raise ReleaseErratumError("scientific snapshot scenario/seed cardinality is incorrect")

    expected_per_arm = contract.scenario_count * contract.seed_count
    per_arm: dict[str, Mapping[str, Any]] = {}
    for arm, entries in sorted(per_arm_rows.items()):
        if len(entries) != expected_per_arm:
            raise ReleaseErratumError(f"scientific snapshot arm {arm} has the wrong row count")
        ordered = sorted(entries, key=lambda item: item[0])
        per_arm[arm] = {
            "episode_rows": len(ordered),
            "canonical_row_manifest_sha256": _manifest_digest(
                [list(identity) + [digests.row_sha256] for identity, digests in ordered]
            ),
            "component_leaf_manifest_sha256": _manifest_digest(
                [list(identity) + [digests.component_sha256] for identity, digests in ordered]
            ),
        }

    ordered_rows = sorted(rows.items(), key=lambda item: item[0])
    return ScientificSnapshot(
        source_sha=contract.source_sha,
        planner_arms=len(per_arm),
        scenario_count=len(scenarios),
        seed_count=len(seeds),
        episode_rows=len(rows),
        episode_identity_manifest_sha256=_manifest_digest(
            [list(identity) for identity, _ in ordered_rows]
        ),
        component_leaf_manifest_sha256=_manifest_digest(
            [list(identity) + [digests.component_sha256] for identity, digests in ordered_rows]
        ),
        canonical_row_manifest_sha256=_manifest_digest(
            [list(identity) + [digests.row_sha256] for identity, digests in ordered_rows]
        ),
        per_arm=per_arm,
        _rows=rows,
    )


def snapshot_campaign(campaign_root: Path, *, contract: ErratumContract) -> ScientificSnapshot:
    """Compute the scientific leaf identity of an extracted campaign root.

    Returns:
        A compact snapshot with private comparison leaves retained in memory.
    """
    root = Path(campaign_root).resolve()
    runs = root / "runs"
    if not runs.is_dir() or runs.is_symlink():
        raise ReleaseErratumError("successor campaign runs directory is missing or unsafe")
    episode_files = sorted(runs.glob("*/episodes.jsonl"))
    arm_rows: dict[str, Iterable[Mapping[str, Any]]] = {}
    open_streams: list[BinaryIO] = []
    try:
        for path in episode_files:
            safe_path = _safe_regular_file(path, label="successor episodes.jsonl")
            if not safe_path.is_relative_to(runs):
                raise ReleaseErratumError("successor episode file escapes the runs directory")
            arm = safe_path.parent.name
            stream = safe_path.open("rb")
            open_streams.append(stream)
            arm_rows[arm] = _read_episode_rows(stream, arm=arm)
        return _snapshot_from_arm_rows(arm_rows, contract=contract)
    finally:
        for stream in open_streams:
            stream.close()


def snapshot_predecessor_archive(  # noqa: C901
    archive_path: Path, *, contract: ErratumContract
) -> ScientificSnapshot:
    """Verify the immutable predecessor archive and compute its scientific leaves.

    Returns:
        The predecessor's compact scientific snapshot.
    """
    archive = _safe_regular_file(archive_path, label="predecessor archive")
    if archive.stat().st_size != contract.predecessor_archive_size_bytes:
        raise ReleaseErratumError("predecessor archive byte count does not match the contract")
    if _sha256_file(archive) != contract.predecessor_archive_sha256:
        raise ReleaseErratumError("predecessor archive SHA-256 does not match the contract")
    try:
        bundle = tarfile.open(archive, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseErratumError("predecessor archive cannot be opened") from exc
    with bundle:
        members = bundle.getmembers()
        if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
            raise ReleaseErratumError("predecessor archive member count is invalid")
        expanded_bytes = 0
        episode_members: dict[str, tarfile.TarInfo] = {}
        roots: set[str] = set()
        for member in members:
            _validate_archive_member(member)
            expanded_bytes += member.size
            if expanded_bytes > _MAX_EXPANDED_BYTES:
                raise ReleaseErratumError("predecessor archive expands beyond the safety limit")
            parts = PurePosixPath(member.name).parts
            roots.add(parts[0])
            match = _EPISODE_MEMBER_RE.fullmatch(member.name)
            if match is not None:
                arm = match.group(1)
                if member.size > _MAX_EPISODE_FILE_BYTES:
                    raise ReleaseErratumError("predecessor episode member exceeds the safety limit")
                if arm in episode_members:
                    raise ReleaseErratumError("predecessor archive repeats a planner episode file")
                episode_members[arm] = member
        if len(roots) != 1:
            raise ReleaseErratumError("predecessor archive must have exactly one bundle root")
        arm_rows: dict[str, Iterable[Mapping[str, Any]]] = {}
        streams: list[BinaryIO] = []
        try:
            for arm, member in sorted(episode_members.items()):
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ReleaseErratumError("predecessor episode member is unreadable")
                streams.append(stream)
                arm_rows[arm] = _read_episode_rows(stream, arm=arm)
            return _snapshot_from_arm_rows(arm_rows, contract=contract)
        finally:
            for stream in streams:
                stream.close()


def compare_scientific_snapshots(
    predecessor: ScientificSnapshot, successor: ScientificSnapshot
) -> dict[str, Any]:
    """Require exact canonical row and component-leaf equality.

    Returns:
        A compact equality proof when every leaf matches.
    """
    predecessor_keys = set(predecessor._rows)
    successor_keys = set(successor._rows)
    missing = sorted(predecessor_keys - successor_keys)
    unexpected = sorted(successor_keys - predecessor_keys)
    changed_rows = sorted(
        key
        for key in predecessor_keys & successor_keys
        if predecessor._rows[key].row_sha256 != successor._rows[key].row_sha256
    )
    changed_components = sorted(
        key
        for key in predecessor_keys & successor_keys
        if predecessor._rows[key].component_sha256 != successor._rows[key].component_sha256
    )
    if missing or unexpected or changed_rows or changed_components:
        summary = {
            "missing": [list(item) for item in missing[:10]],
            "unexpected": [list(item) for item in unexpected[:10]],
            "changed_rows": [list(item) for item in changed_rows[:10]],
            "changed_components": [list(item) for item in changed_components[:10]],
        }
        raise ReleaseErratumError(
            "predecessor/successor scientific leaves differ: " + json.dumps(summary, sort_keys=True)
        )
    if predecessor.public_dict() != successor.public_dict():
        raise ReleaseErratumError("predecessor/successor compact scientific manifests differ")
    return {
        "status": "identical",
        "identity_set_equal": True,
        "canonical_rows_equal": True,
        "component_leaves_equal": True,
        "simulation_rerun": False,
        "episode_rows": predecessor.episode_rows,
        "planner_arms": predecessor.planner_arms,
        "episode_identity_manifest_sha256": predecessor.episode_identity_manifest_sha256,
        "component_leaf_manifest_sha256": predecessor.component_leaf_manifest_sha256,
        "canonical_row_manifest_sha256": predecessor.canonical_row_manifest_sha256,
    }


def build_erratum_receipt(
    *,
    contract: ErratumContract,
    predecessor: ScientificSnapshot,
    successor: ScientificSnapshot,
) -> dict[str, Any]:
    """Build the self-contained machine-readable successor correction receipt.

    Returns:
        A JSON-ready correction receipt.
    """
    equality = compare_scientific_snapshots(predecessor, successor)
    return {
        "schema_version": ERRATUM_RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "correction_id": contract.correction_id,
        "correction_scope": ERRATUM_SCOPE,
        "supersedes": {
            "version_doi": contract.predecessor_version_doi,
            "archive_sha256": contract.predecessor_archive_sha256,
            "archive_size_bytes": contract.predecessor_archive_size_bytes,
            "github_release_tag": contract.predecessor_github_release_tag,
            "old_publication_retained": True,
        },
        "successor": {
            "concept_doi": contract.concept_doi,
            "version_doi": contract.successor_version_doi,
            "github_release_tag": contract.successor_github_release_tag,
            "metadata_sha256": contract.metadata_sha256,
            "relation": "isNewVersionOf",
        },
        "scientific_identity": successor.public_dict(),
        "scientific_equality": equality,
        "derivation": {
            "builder_sha": contract.builder_sha,
            "scientific_source_sha": contract.source_sha,
            "simulation_rerun": False,
        },
        "corrected_verdict": {
            "publication_preflight_status": "pass",
            "publication_preflight_violations": [],
            "release_status": "ok",
            "ranking_authority": False,
            "ranking_claims_admitted": False,
        },
        "changed_derived_path_classes": [
            "release/release_result.json",
            "release/release_manifest.resolved.json",
            "release/zenodo_metadata.erratum.json",
            "provenance/benchmark_release_erratum.json",
            "release_metadata/*",
            "publication_manifest.json",
            "checksums.sha256",
            "archive_container",
        ],
        "credentials": "not_recorded",
    }

"""Fail-closed contracts and scientific-equality proof for release errata.

An erratum is a new publication identity derived from an immutable predecessor.
It may repair publication metadata, but it must not silently change benchmark
episode rows or component metrics.  This module keeps those two identities
separate and produces the compact equality evidence embedded in the successor.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tarfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from robot_sf.benchmark.release_tag_identity import check_canonical_source_tag

ERRATUM_CONTRACT_SCHEMA = "benchmark-release-erratum.v1"
ERRATUM_RECEIPT_SCHEMA = "benchmark-release-erratum-receipt.v1"
ERRATUM_SCOPE = "derived_publication_metadata_only"
SCIENTIFIC_CANONICALIZATION = "robot-sf-scientific-json.v1"
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
    validator_sha: str
    orchestration_sha: str
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
    episode_file_sha256: Mapping[str, str]
    episode_file_manifest_sha256: str
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
            "episode_file_sha256": dict(self.episode_file_sha256),
            "episode_file_manifest_sha256": self.episode_file_manifest_sha256,
            "per_arm": dict(self.per_arm),
        }


def _validate_contract_sha_and_matrix(contract: ErratumContract) -> None:
    """Validate immutable SHAs, sizes, digests, and cardinality."""
    sha_values = (
        contract.source_sha,
        contract.builder_sha,
        contract.validator_sha,
        contract.orchestration_sha,
    )
    if any(_SHA1_RE.fullmatch(value) is None for value in sha_values):
        raise ReleaseErratumError(
            "scientific source, builder, validator, and orchestration SHAs must be "
            "full lowercase SHAs"
        )
    if contract.source_sha in {
        contract.builder_sha,
        contract.validator_sha,
        contract.orchestration_sha,
    }:
        raise ReleaseErratumError("erratum implementation SHAs must differ from scientific source")
    if contract.builder_sha != contract.validator_sha:
        raise ReleaseErratumError(
            "erratum correction builder and validator must name the same accepted commit"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            contract.predecessor_archive_size_bytes,
            contract.planner_arms,
            contract.scenario_count,
            contract.seed_count,
            contract.episode_rows,
        )
    ):
        raise ReleaseErratumError(
            "erratum sizes and matrix cardinalities must be positive integers"
        )
    if (
        contract.planner_arms * contract.scenario_count * contract.seed_count
        != contract.episode_rows
    ):
        raise ReleaseErratumError("scientific matrix cardinality does not equal episode_rows")
    if _SHA256_RE.fullmatch(contract.predecessor_archive_sha256) is None:
        raise ReleaseErratumError("predecessor archive digest must be a lowercase SHA-256")
    if _SHA256_RE.fullmatch(contract.metadata_sha256) is None:
        raise ReleaseErratumError("successor metadata digest must be a lowercase SHA-256")


def _validate_contract_dois_and_tags(contract: ErratumContract) -> None:
    """Validate distinct DOI coordinates and canonical source-tag lineage."""
    doi_values = (
        contract.predecessor_version_doi,
        contract.concept_doi,
        contract.successor_version_doi,
    )
    if any(_DOI_RE.fullmatch(value) is None for value in doi_values) or len(set(doi_values)) != 3:
        raise ReleaseErratumError(
            "predecessor, concept, and successor DOIs must be valid and distinct"
        )
    predecessor_tag = contract.predecessor_github_release_tag
    successor_tag = contract.successor_github_release_tag
    if not contract.correction_id.strip():
        raise ReleaseErratumError("erratum correction_id must be non-empty")
    if not _TAG_RE.fullmatch(predecessor_tag) or not _TAG_RE.fullmatch(successor_tag):
        raise ReleaseErratumError("erratum GitHub release tags are invalid")
    problems = check_canonical_source_tag(predecessor_tag, contract.source_sha)
    if problems or re.search(r"-erratum\.[1-9][0-9]*$", predecessor_tag):
        raise ReleaseErratumError(
            "predecessor tag must end in the exact scientific source SHA without an erratum suffix"
        )
    if successor_tag != f"{predecessor_tag}-erratum.1":
        raise ReleaseErratumError("successor tag must be the predecessor tag plus -erratum.1")
    if check_canonical_source_tag(successor_tag, contract.source_sha):
        raise ReleaseErratumError("successor tag does not carry the exact scientific source SHA")


def validate_erratum_contract_identity(contract: ErratumContract) -> None:
    """Validate identities needed by every direct or file-backed caller.

    Loading a checked-in contract additionally validates its metadata file.
    This boundary covers the immutable identity rules that must also apply to
    direct dataclass callers used by the derivation and cold-audit paths.
    """
    _validate_contract_sha_and_matrix(contract)
    _validate_contract_dois_and_tags(contract)


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


def _canonical_scientific_value(value: Any) -> list[Any]:  # noqa: C901
    """Return a type-tagged JSON value with deterministic non-finite floats.

    Historical episode JSONL uses Python's explicit ``NaN`` and
    ``Infinity`` tokens for unavailable diagnostics. They are scientific leaf
    values and must compare exactly, but they cannot be emitted by strict
    RFC-8259 JSON. A tagged representation also avoids collisions with literal
    strings or user-provided objects that happen to resemble sentinel values.

    Returns:
        A strict-JSON-compatible, type-tagged canonical value.
    """
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if math.isnan(value):
            return ["float", "nan"]
        if math.isinf(value):
            return ["float", "+inf" if value > 0 else "-inf"]
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, list):
        return ["list", [_canonical_scientific_value(item) for item in value]]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ReleaseErratumError("scientific row object keys must be strings")
        return [
            "object",
            [[key, _canonical_scientific_value(value[key])] for key in sorted(value)],
        ]
    raise ReleaseErratumError("scientific row contains a non-JSON value")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _canonical_scientific_value(value),
            ensure_ascii=False,
            allow_nan=False,
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
        if isinstance(item, Mapping) and item.get("relation") == "isNewVersionOf"
    ]
    if len(source_matches) != 1 or source_matches[0].get("scheme") != "url":
        raise ReleaseErratumError("erratum metadata must bind exactly one successor GitHub tag")
    if (
        len(predecessor_matches) != 1
        or predecessor_matches[0].get("identifier") != predecessor_doi
        or predecessor_matches[0].get("scheme") != "doi"
    ):
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
    root = Path(repository_root.absolute())
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ReleaseErratumError("erratum repository root contains a symlink")
    if not root.is_dir():
        raise ReleaseErratumError("erratum repository root is missing")
    contract_path = _safe_regular_file(path, label="erratum contract")
    if not contract_path.resolve().is_relative_to(root.resolve()):
        raise ReleaseErratumError("erratum contract is outside the repository root")
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
    validator_sha = _required_text(derivation, "validator_sha", label="derivation").lower()
    orchestration_sha = _required_text(derivation, "orchestration_sha", label="derivation").lower()
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
    if any(
        _SHA1_RE.fullmatch(value) is None
        for value in (source_sha, builder_sha, validator_sha, orchestration_sha)
    ):
        raise ReleaseErratumError(
            "scientific source, builder, validator, and orchestration SHAs must be full lowercase SHAs"
        )
    if source_sha in {builder_sha, validator_sha, orchestration_sha}:
        raise ReleaseErratumError("erratum implementation SHAs must differ from scientific source")
    if not _TAG_RE.fullmatch(predecessor_tag) or not _TAG_RE.fullmatch(successor_tag):
        raise ReleaseErratumError("erratum GitHub release tags are invalid")
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
    metadata_candidate = Path((root / metadata_relative).absolute())
    if metadata_candidate.is_symlink() or any(
        parent.is_symlink() for parent in metadata_candidate.parents
    ):
        raise ReleaseErratumError("successor.metadata_path contains a symlink")
    metadata_path = metadata_candidate.resolve()
    if not metadata_path.is_relative_to(root.resolve()):
        raise ReleaseErratumError("successor.metadata_path escapes the repository")
    _validate_metadata_file(
        metadata_path,
        digest=metadata_digest,
        successor_tag=successor_tag,
        predecessor_doi=predecessor_doi,
    )

    contract = ErratumContract(
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
        validator_sha=validator_sha,
        orchestration_sha=orchestration_sha,
        concept_doi=concept_doi,
        successor_version_doi=successor_doi,
        successor_github_release_tag=successor_tag,
        metadata_path=metadata_path,
        metadata_sha256=metadata_digest,
    )
    validate_erratum_contract_identity(contract)
    return contract


def _validate_archive_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseErratumError("predecessor archive contains an unsafe member path")
    if not (member.isdir() or member.isreg()):
        raise ReleaseErratumError("predecessor archive contains a non-regular member")
    if member.size < 0:
        raise ReleaseErratumError("predecessor archive contains a negative member size")


def _read_episode_rows(
    stream: BinaryIO, *, arm: str, digest: Any | None = None
) -> Iterable[Mapping[str, Any]]:
    for line_number, raw_line in enumerate(stream, start=1):
        if digest is not None:
            digest.update(raw_line)
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


def _snapshot_from_arm_rows(  # noqa: C901, PLR0912
    arm_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    contract: ErratumContract,
    episode_file_hashers: Mapping[str, Any],
) -> ScientificSnapshot:
    validate_erratum_contract_identity(contract)
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
                (
                    "event_ledger.software_commit",
                    _require_mapping(row.get("event_ledger"), label="event_ledger").get(
                        "software_commit"
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
    if set(episode_file_hashers) != set(per_arm_rows):
        raise ReleaseErratumError("scientific snapshot episode-file digest set is incorrect")
    episode_file_sha256 = {arm: hasher.hexdigest() for arm, hasher in episode_file_hashers.items()}
    if any(_SHA256_RE.fullmatch(digest) is None for digest in episode_file_sha256.values()):
        raise ReleaseErratumError("scientific snapshot episode-file digest is invalid")
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
        episode_file_sha256={arm: episode_file_sha256[arm] for arm in sorted(episode_file_sha256)},
        episode_file_manifest_sha256=_manifest_digest(
            [[arm, episode_file_sha256[arm]] for arm in sorted(episode_file_sha256)]
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
    discovered_episode_files = sorted(runs.rglob("episodes.jsonl"))
    if discovered_episode_files != episode_files:
        raise ReleaseErratumError(
            "successor campaign contains episode files outside runs/<arm>/episodes.jsonl"
        )
    arm_rows: dict[str, Iterable[Mapping[str, Any]]] = {}
    episode_file_hashers: dict[str, Any] = {}
    open_streams: list[BinaryIO] = []
    try:
        for path in episode_files:
            safe_path = _safe_regular_file(path, label="successor episodes.jsonl")
            if not safe_path.is_relative_to(runs):
                raise ReleaseErratumError("successor episode file escapes the runs directory")
            arm = safe_path.parent.name
            stream = safe_path.open("rb")
            open_streams.append(stream)
            hasher = hashlib.sha256()
            episode_file_hashers[arm] = hasher
            arm_rows[arm] = _read_episode_rows(stream, arm=arm, digest=hasher)
        snapshot = _snapshot_from_arm_rows(
            arm_rows,
            contract=contract,
            episode_file_hashers=episode_file_hashers,
        )
        return snapshot
    finally:
        for stream in open_streams:
            stream.close()


def snapshot_predecessor_archive(  # noqa: C901, PLR0912
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
        member_names: set[str] = set()
        roots: set[str] = set()
        for member in members:
            _validate_archive_member(member)
            if member.name in member_names:
                raise ReleaseErratumError("predecessor archive contains duplicate member names")
            member_names.add(member.name)
            expanded_bytes += member.size
            if expanded_bytes > _MAX_EXPANDED_BYTES:
                raise ReleaseErratumError("predecessor archive expands beyond the safety limit")
            parts = PurePosixPath(member.name).parts
            roots.add(parts[0])
            match = _EPISODE_MEMBER_RE.fullmatch(member.name)
            if PurePosixPath(member.name).name == "episodes.jsonl" and match is None:
                raise ReleaseErratumError(
                    "predecessor archive contains episode files outside payload/runs/<arm>"
                )
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
        episode_file_hashers: dict[str, Any] = {}
        streams: list[BinaryIO] = []
        try:
            for arm, member in sorted(episode_members.items()):
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ReleaseErratumError("predecessor episode member is unreadable")
                streams.append(stream)
                hasher = hashlib.sha256()
                episode_file_hashers[arm] = hasher
                arm_rows[arm] = _read_episode_rows(stream, arm=arm, digest=hasher)
            return _snapshot_from_arm_rows(
                arm_rows,
                contract=contract,
                episode_file_hashers=episode_file_hashers,
            )
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
    changed_episode_files = sorted(
        arm
        for arm in set(predecessor.episode_file_sha256) & set(successor.episode_file_sha256)
        if predecessor.episode_file_sha256[arm] != successor.episode_file_sha256[arm]
    )
    if (
        missing
        or unexpected
        or changed_rows
        or changed_components
        or predecessor.episode_file_sha256 != successor.episode_file_sha256
    ):
        summary = {
            "missing": [list(item) for item in missing[:10]],
            "unexpected": [list(item) for item in unexpected[:10]],
            "changed_rows": [list(item) for item in changed_rows[:10]],
            "changed_components": [list(item) for item in changed_components[:10]],
            "changed_episode_files": changed_episode_files[:10],
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
        "episode_file_bytes_equal": True,
        "simulation_rerun": False,
        "episode_rows": predecessor.episode_rows,
        "planner_arms": predecessor.planner_arms,
        "episode_identity_manifest_sha256": predecessor.episode_identity_manifest_sha256,
        "component_leaf_manifest_sha256": predecessor.component_leaf_manifest_sha256,
        "canonical_row_manifest_sha256": predecessor.canonical_row_manifest_sha256,
        "episode_file_sha256": dict(predecessor.episode_file_sha256),
        "episode_file_manifest_sha256": predecessor.episode_file_manifest_sha256,
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
    validate_erratum_contract_identity(contract)
    _validate_metadata_file(
        contract.metadata_path,
        digest=contract.metadata_sha256,
        successor_tag=contract.successor_github_release_tag,
        predecessor_doi=contract.predecessor_version_doi,
    )
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
        "scientific_canonicalization": {
            "schema": SCIENTIFIC_CANONICALIZATION,
            "nonfinite_float_policy": "preserve_nan_positive_infinity_and_negative_infinity",
            "finite_float_policy": "python_float_hex",
        },
        "derivation": {
            "builder_sha": contract.builder_sha,
            "validator_sha": contract.validator_sha,
            "orchestration_sha": contract.orchestration_sha,
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


def _load_published_erratum_receipt(receipt_path: Path) -> tuple[Path, Mapping[str, Any]]:
    """Load the outer published receipt contract.

    Returns:
        The safe receipt path and parsed mapping.
    """
    path = _safe_regular_file(receipt_path, label="published erratum receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReleaseErratumError("published erratum receipt is not readable JSON") from exc
    receipt = _require_mapping(receipt, label="published erratum receipt")
    if receipt.get("schema_version") != ERRATUM_RECEIPT_SCHEMA:
        raise ReleaseErratumError("published erratum receipt schema is unsupported")
    if receipt.get("correction_scope") != ERRATUM_SCOPE:
        raise ReleaseErratumError("published erratum receipt scope is not metadata-only")
    return path, receipt


def _validate_published_erratum_verdict(receipt: Mapping[str, Any]) -> None:
    """Require exact equality, advisory claims, and the canonicalization contract."""
    verdict = _require_mapping(receipt.get("corrected_verdict"), label="receipt.corrected_verdict")
    equality = _require_mapping(
        receipt.get("scientific_equality"), label="receipt.scientific_equality"
    )
    canonicalization = _require_mapping(
        receipt.get("scientific_canonicalization"),
        label="receipt.scientific_canonicalization",
    )
    expected_verdict = {
        "publication_preflight_status": "pass",
        "publication_preflight_violations": [],
        "release_status": "ok",
        "ranking_authority": False,
        "ranking_claims_admitted": False,
    }
    if any(verdict.get(key) != value for key, value in expected_verdict.items()):
        raise ReleaseErratumError("published erratum receipt has a contradictory verdict")
    expected_equality = {
        "status": "identical",
        "identity_set_equal": True,
        "canonical_rows_equal": True,
        "component_leaves_equal": True,
        "episode_file_bytes_equal": True,
        "simulation_rerun": False,
    }
    if any(equality.get(key) != value for key, value in expected_equality.items()):
        raise ReleaseErratumError("published erratum receipt does not claim exact equality")
    if canonicalization.get("schema") != SCIENTIFIC_CANONICALIZATION:
        raise ReleaseErratumError("published erratum receipt canonicalization is unsupported")


def _published_erratum_contract(  # noqa: C901
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    metadata_path: Path,
    expected_tag: str,
    expected_doi: str,
    expected_source_sha: str | None,
) -> tuple[ErratumContract, Mapping[str, Any], Mapping[str, Any]]:
    """Validate receipt identities.

    Returns:
        The snapshot contract, scientific identity, and equality mapping.
    """

    supersedes = _require_mapping(receipt.get("supersedes"), label="receipt.supersedes")
    successor = _require_mapping(receipt.get("successor"), label="receipt.successor")
    scientific = _require_mapping(
        receipt.get("scientific_identity"), label="receipt.scientific_identity"
    )
    equality = _require_mapping(
        receipt.get("scientific_equality"), label="receipt.scientific_equality"
    )
    derivation = _require_mapping(receipt.get("derivation"), label="receipt.derivation")
    if successor.get("github_release_tag") != expected_tag:
        raise ReleaseErratumError("published erratum receipt successor tag is incorrect")
    if successor.get("version_doi") != expected_doi:
        raise ReleaseErratumError("published erratum receipt successor DOI is incorrect")
    if successor.get("relation") != "isNewVersionOf":
        raise ReleaseErratumError("published erratum receipt relation is incorrect")
    concept_doi = _required_text(successor, "concept_doi", label="receipt.successor")
    if _DOI_RE.fullmatch(concept_doi) is None or concept_doi == expected_doi:
        raise ReleaseErratumError("published erratum receipt concept DOI is invalid")
    metadata_digest = _required_text(
        successor, "metadata_sha256", label="receipt.successor"
    ).lower()
    if _SHA256_RE.fullmatch(metadata_digest) is None:
        raise ReleaseErratumError("published erratum receipt metadata digest is invalid")
    predecessor_doi = _required_text(supersedes, "version_doi", label="receipt.supersedes")
    predecessor_digest = _required_text(
        supersedes, "archive_sha256", label="receipt.supersedes"
    ).lower()
    predecessor_tag = _required_text(supersedes, "github_release_tag", label="receipt.supersedes")
    if _DOI_RE.fullmatch(predecessor_doi) is None or predecessor_doi == expected_doi:
        raise ReleaseErratumError("published erratum receipt predecessor DOI is invalid")
    if _SHA256_RE.fullmatch(predecessor_digest) is None:
        raise ReleaseErratumError("published erratum receipt predecessor digest is invalid")
    predecessor_size = _required_positive_int(
        supersedes, "archive_size_bytes", label="receipt.supersedes"
    )
    if supersedes.get("old_publication_retained") is not True:
        raise ReleaseErratumError("published erratum receipt does not retain its predecessor")
    if expected_tag != f"{predecessor_tag}-erratum.1":
        raise ReleaseErratumError("published erratum receipt tag lineage is incorrect")

    source_sha = _required_text(scientific, "source_sha", label="receipt.scientific_identity")
    builder_sha = _required_text(derivation, "builder_sha", label="receipt.derivation")
    validator_sha = _required_text(derivation, "validator_sha", label="receipt.derivation")
    orchestration_sha = _required_text(derivation, "orchestration_sha", label="receipt.derivation")
    if any(
        _SHA1_RE.fullmatch(value) is None
        for value in (source_sha, builder_sha, validator_sha, orchestration_sha)
    ):
        raise ReleaseErratumError("published erratum receipt contains an invalid Git SHA")
    if source_sha in {builder_sha, validator_sha, orchestration_sha}:
        raise ReleaseErratumError("published erratum receipt conflates source and implementation")
    if derivation.get("scientific_source_sha") != source_sha:
        raise ReleaseErratumError(
            "published erratum receipt derivation source differs from scientific identity"
        )
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ReleaseErratumError(
            "published erratum receipt source differs from the GitHub tag target"
        )
    if derivation.get("simulation_rerun") is not False:
        raise ReleaseErratumError("published erratum receipt claims a simulation rerun")
    contract = ErratumContract(
        correction_id=_required_text(receipt, "correction_id", label="published erratum receipt"),
        predecessor_version_doi=predecessor_doi,
        predecessor_archive_sha256=predecessor_digest,
        predecessor_archive_size_bytes=predecessor_size,
        predecessor_github_release_tag=predecessor_tag,
        source_sha=source_sha,
        planner_arms=_required_positive_int(
            scientific, "planner_arms", label="receipt.scientific_identity"
        ),
        scenario_count=_required_positive_int(
            scientific, "scenario_count", label="receipt.scientific_identity"
        ),
        seed_count=_required_positive_int(
            scientific, "seed_count", label="receipt.scientific_identity"
        ),
        episode_rows=_required_positive_int(
            scientific, "episode_rows", label="receipt.scientific_identity"
        ),
        builder_sha=builder_sha,
        validator_sha=validator_sha,
        orchestration_sha=orchestration_sha,
        concept_doi=concept_doi,
        successor_version_doi=expected_doi,
        successor_github_release_tag=expected_tag,
        metadata_path=metadata_path,
        metadata_sha256=metadata_digest,
    )
    validate_erratum_contract_identity(contract)
    _validate_metadata_file(
        metadata_path,
        digest=metadata_digest,
        successor_tag=expected_tag,
        predecessor_doi=predecessor_doi,
    )
    return (
        contract,
        scientific,
        equality,
    )


def _assert_published_equality_digests(
    scientific: Mapping[str, Any], equality: Mapping[str, Any]
) -> None:
    """Require equality evidence to repeat all scientific leaf digests."""
    keys = (
        "episode_identity_manifest_sha256",
        "component_leaf_manifest_sha256",
        "canonical_row_manifest_sha256",
        "episode_file_manifest_sha256",
        "episode_file_sha256",
        "episode_rows",
        "planner_arms",
    )
    if any(equality.get(key) != scientific.get(key) for key in keys):
        raise ReleaseErratumError("published erratum equality digest is inconsistent")


def _load_publication_document(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load one required bundle-local publication document.

    Returns:
        The parsed mapping.
    """
    safe_path = _safe_regular_file(path, label=label)
    try:
        document = json.loads(safe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReleaseErratumError(f"{label} is not readable JSON") from exc
    return _require_mapping(document, label=label)


def _assert_current_alias_values(
    payload: Mapping[str, Any],
    *,
    contract: ErratumContract,
    label: str,
    required: bool,
) -> None:
    """Validate one level of current-publication aliases."""
    tag_keys = ("release_tag", "release_id", "benchmark_release_tag", "benchmark_release_id")
    doi_keys = ("doi", "version_doi")
    tag_values = [payload[key] for key in tag_keys if key in payload]
    doi_values = [payload[key] for key in doi_keys if key in payload]
    concept_values = [payload["concept_doi"]] if "concept_doi" in payload else []
    if (required and not tag_values) or any(
        value != contract.successor_github_release_tag for value in tag_values
    ):
        raise ReleaseErratumError(f"{label} contains a stale release-tag alias")
    if (required and not doi_values) or any(
        value != contract.successor_version_doi for value in doi_values
    ):
        raise ReleaseErratumError(f"{label} contains a stale version-DOI alias")
    if (required and not concept_values) or any(
        value != contract.concept_doi for value in concept_values
    ):
        raise ReleaseErratumError(f"{label} contains a stale concept-DOI alias")


def _assert_publication_aliases(
    payload: Mapping[str, Any], *, contract: ErratumContract, label: str
) -> None:
    """Reject any present current-publication alias that remains stale."""
    _assert_current_alias_values(
        payload,
        contract=contract,
        label=label,
        required=True,
    )
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        _assert_current_alias_values(
            provenance,
            contract=contract,
            label=f"{label}.provenance",
            required=False,
        )
    publication = payload.get("publication")
    if isinstance(publication, Mapping):
        expected = {
            "concept_doi": contract.concept_doi,
            "version_doi": contract.successor_version_doi,
            "predecessor_version_doi": contract.predecessor_version_doi,
        }
        if any(publication.get(key) != value for key, value in expected.items()):
            raise ReleaseErratumError(f"{label}.publication contains a stale DOI alias")


def _assert_predecessor_alias_values(
    payload: Mapping[str, Any],
    *,
    contract: ErratumContract,
    label: str,
    required: bool,
) -> None:
    """Validate one level of predecessor execution aliases."""
    tag_values = [
        payload[key]
        for key in ("release_tag", "release_id", "benchmark_release_tag", "benchmark_release_id")
        if key in payload
    ]
    doi_values = [payload[key] for key in ("doi", "version_doi") if key in payload]
    if (required and not tag_values) or any(
        value != contract.predecessor_github_release_tag for value in tag_values
    ):
        raise ReleaseErratumError(f"{label} contains a stale predecessor tag alias")
    if (required and not doi_values) or any(
        value != contract.predecessor_version_doi for value in doi_values
    ):
        raise ReleaseErratumError(f"{label} contains a stale predecessor DOI alias")
    if "concept_doi" in payload and payload["concept_doi"] != contract.concept_doi:
        raise ReleaseErratumError(f"{label} contains a stale predecessor concept DOI")
    source_values = [
        payload[key]
        for key in ("source_sha", "source_commit", "scientific_source_sha")
        if key in payload
    ]
    if any(value != contract.source_sha for value in source_values):
        raise ReleaseErratumError(f"{label} contains a stale scientific source SHA")


def _assert_predecessor_execution_aliases(
    payload: Mapping[str, Any], *, contract: ErratumContract, label: str
) -> None:
    """Require preserved execution coordinates to name only the predecessor."""
    _assert_predecessor_alias_values(
        payload,
        contract=contract,
        label=label,
        required=True,
    )
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        _assert_predecessor_alias_values(
            provenance,
            contract=contract,
            label=f"{label}.provenance",
            required=False,
        )


def _validate_published_release_documents(
    campaign_root: Path, *, contract: ErratumContract
) -> None:
    """Validate current-publication identities and verdicts in the downloaded payload."""
    manifest = _load_publication_document(
        campaign_root / "release/release_manifest.resolved.json",
        label="published resolved manifest",
    )
    result = _load_publication_document(
        campaign_root / "release/release_result.json",
        label="published release result",
    )
    summary = _load_publication_document(
        campaign_root / "reports/campaign_summary.json",
        label="published campaign summary",
    )
    _assert_publication_aliases(manifest, contract=contract, label="published resolved manifest")
    _assert_publication_aliases(result, contract=contract, label="published release result")
    publication = _require_mapping(
        manifest.get("publication"), label="published resolved manifest.publication"
    )
    if (
        publication.get("concept_doi") != contract.concept_doi
        or publication.get("version_doi") != contract.successor_version_doi
        or publication.get("predecessor_version_doi") != contract.predecessor_version_doi
        or publication.get("bundle_metadata_path") != "release/zenodo_metadata.erratum.json"
        or publication.get("metadata_sha256") != contract.metadata_sha256
        or publication.get("correction_scope") != ERRATUM_SCOPE
    ):
        raise ReleaseErratumError("published resolved manifest publication identity is stale")
    for key in ("benchmark_release", "resolved_manifest"):
        nested = _require_mapping(result.get(key), label=f"published release result.{key}")
        _assert_publication_aliases(
            nested,
            contract=contract,
            label=f"published release result.{key}",
        )
    for key in ("scientific_execution_benchmark_release", "scientific_execution_resolved_manifest"):
        execution = _require_mapping(result.get(key), label=f"published release result.{key}")
        _assert_predecessor_execution_aliases(
            execution,
            contract=contract,
            label=f"published release result.{key}",
        )
    result_derivation = _require_mapping(
        result.get("derivation"), label="published release result.derivation"
    )
    expected_derivation = {
        "builder_sha": contract.builder_sha,
        "validator_sha": contract.validator_sha,
        "orchestration_sha": contract.orchestration_sha,
        "scientific_source_sha": contract.source_sha,
        "simulation_rerun": False,
        "correction_id": contract.correction_id,
        "predecessor_version_doi": contract.predecessor_version_doi,
    }
    if any(result_derivation.get(key) != value for key, value in expected_derivation.items()):
        raise ReleaseErratumError("published release result derivation identity is stale")
    summary_release = _require_mapping(
        summary.get("benchmark_release"), label="published campaign summary.benchmark_release"
    )
    summary_campaign = _require_mapping(
        summary.get("campaign"), label="published campaign summary.campaign"
    )
    _assert_publication_aliases(
        summary_release,
        contract=contract,
        label="published campaign summary.benchmark_release",
    )
    _assert_publication_aliases(
        summary_campaign,
        contract=contract,
        label="published campaign summary.campaign",
    )
    summary_execution = _require_mapping(
        summary_campaign.get("scientific_execution_release_identity"),
        label="published campaign summary.campaign.scientific_execution_release_identity",
    )
    _assert_predecessor_execution_aliases(
        summary_execution,
        contract=contract,
        label="published campaign summary.campaign.scientific_execution_release_identity",
    )
    if (
        result.get("publication_preflight_status") != "pass"
        or result.get("publication_preflight_violations") != []
        or result.get("release_status") != "ok"
        or result.get("ranking_claims_admitted") is not False
    ):
        raise ReleaseErratumError("published release result has a contradictory verdict")
    manifest_provenance = _require_mapping(
        manifest.get("provenance"), label="published resolved manifest.provenance"
    )
    if manifest_provenance.get("scientific_source_sha") != contract.source_sha:
        raise ReleaseErratumError("published resolved manifest lost the scientific source SHA")
    if (
        manifest_provenance.get("erratum_builder_sha") != contract.builder_sha
        or manifest_provenance.get("erratum_validator_sha") != contract.validator_sha
        or manifest_provenance.get("erratum_orchestration_sha") != contract.orchestration_sha
    ):
        raise ReleaseErratumError("published resolved manifest implementation identity is stale")

    derived_receipt = _load_publication_document(
        campaign_root / "provenance/derived_revalidation_receipt.json",
        label="published derived revalidation receipt",
    )
    if derived_receipt.get("schema_version") != "benchmark-derived-revalidation.v1":
        raise ReleaseErratumError("published derived revalidation receipt schema is unsupported")
    derived_source = _require_mapping(
        derived_receipt.get("source"), label="published derived revalidation receipt.source"
    )
    derived_validator = _require_mapping(
        derived_receipt.get("validator"),
        label="published derived revalidation receipt.validator",
    )
    if (
        derived_source.get("execution_commit") != contract.source_sha
        or derived_validator.get("commit") != contract.validator_sha
    ):
        raise ReleaseErratumError("published derived revalidation receipt identity is stale")


def validate_erratum_receipt_against_campaign(
    receipt_path: Path,
    *,
    campaign_root: Path,
    metadata_path: Path,
    expected_tag: str,
    expected_doi: str,
    expected_source_sha: str | None = None,
) -> dict[str, Any]:
    """Validate an embedded receipt and recompute its successor scientific leaves.

    This cold-audit helper does not trust the receipt's claimed successor
    digests. It rebuilds them from the downloaded bundle. The predecessor
    archive remains independently verified during derivation and is bound by
    DOI, size, and SHA-256 in the receipt.

    Returns:
        Compact public observations for the cold audit receipt.
    """
    root = Path(campaign_root).resolve()
    expected_receipt_path = root / "provenance/benchmark_release_erratum.json"
    expected_metadata_path = root / "release/zenodo_metadata.erratum.json"
    if Path(receipt_path).resolve() != expected_receipt_path:
        raise ReleaseErratumError("published erratum receipt is outside its canonical payload path")
    if Path(metadata_path).resolve() != expected_metadata_path:
        raise ReleaseErratumError(
            "published erratum metadata is outside its canonical payload path"
        )
    path, receipt = _load_published_erratum_receipt(receipt_path)
    _validate_published_erratum_verdict(receipt)
    contract, scientific, equality = _published_erratum_contract(
        receipt,
        receipt_path=path,
        metadata_path=metadata_path,
        expected_tag=expected_tag,
        expected_doi=expected_doi,
        expected_source_sha=expected_source_sha,
    )
    observed = snapshot_campaign(campaign_root, contract=contract)
    if observed.public_dict() != dict(scientific):
        raise ReleaseErratumError("published successor scientific leaves differ from its receipt")
    _assert_published_equality_digests(scientific, equality)
    _validate_published_release_documents(root, contract=contract)

    return {
        "status": "pass",
        "source_sha": contract.source_sha,
        "builder_sha": contract.builder_sha,
        "validator_sha": contract.validator_sha,
        "orchestration_sha": contract.orchestration_sha,
        "predecessor_version_doi": contract.predecessor_version_doi,
        "predecessor_archive_sha256": contract.predecessor_archive_sha256,
        "concept_doi": contract.concept_doi,
        "successor_version_doi": expected_doi,
        "episode_rows": observed.episode_rows,
        "planner_arms": observed.planner_arms,
        "episode_identity_manifest_sha256": observed.episode_identity_manifest_sha256,
        "component_leaf_manifest_sha256": observed.component_leaf_manifest_sha256,
        "canonical_row_manifest_sha256": observed.canonical_row_manifest_sha256,
        "episode_file_sha256": dict(observed.episode_file_sha256),
        "episode_file_manifest_sha256": observed.episode_file_manifest_sha256,
        "ranking_claims_admitted": False,
    }

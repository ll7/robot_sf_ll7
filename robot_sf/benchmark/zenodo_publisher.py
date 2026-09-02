"""Direct, credential-safe Zenodo publisher for benchmark-data releases."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from robot_sf.common.optional_import import try_import

ZENODO_API_BASE = "https://zenodo.org/api"
ZENODO_STATE_SCHEMA = "robot-sf-zenodo-deposition.v1"
ZENODO_VERIFICATION_SCHEMA = "robot-sf-zenodo-verification.v2"
_REMOTE_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_TAG_RE = re.compile(r"^https://github\.com/ll7/robot_sf_ll7/releases/tag/[^/?#]+$")
_ZENODO_DOI_RE = re.compile(r"^10\.5281/zenodo\.\d+$")
_CLAIM_BOUNDARY_TERMS = ("snqi", "advisory", "ranking")
_CREDENTIAL_KEYS = ("token", "authorization", "password", "secret")


class ZenodoPublisherError(RuntimeError):
    """Raised for a rejected local contract or Zenodo API response."""


class _Response(Protocol):
    """Small response protocol used by the publisher and mocked tests."""

    status_code: int

    def json(self) -> Any:
        """Return the decoded response body."""

    def iter_content(self, *, chunk_size: int) -> Any:
        """Yield bounded chunks from a streamed response body."""

    def raise_for_status(self) -> None:
        """Raise for an unsuccessful HTTP response."""


class _Session(Protocol):
    """Subset of ``requests.Session`` consumed by this module."""

    headers: dict[str, str]

    def post(self, url: str, **kwargs: Any) -> _Response:
        """Issue POST."""

    def put(self, url: str, **kwargs: Any) -> _Response:
        """Issue PUT."""

    def get(self, url: str, **kwargs: Any) -> _Response:
        """Issue GET."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a public receipt/state payload deterministically.

    Returns:
        Canonical UTF-8 JSON bytes.
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any], excluded_key: str) -> str:
    """Hash a payload after removing its self-integrity block.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    body = dict(payload)
    body.pop(excluded_key, None)
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _assert_credential_free(payload: Any) -> None:
    """Reject state-like payloads containing credential-shaped fields."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key).casefold()
            if any(secret_key in key_text for secret_key in _CREDENTIAL_KEYS):
                raise ZenodoPublisherError("Zenodo state must not contain credentials")
            _assert_credential_free(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_credential_free(value)


def _source_tag(metadata: Mapping[str, Any]) -> str:
    """Return the one exact GitHub release URL bound to dataset metadata."""
    related = metadata.get("related_identifiers")
    if not isinstance(related, list):
        raise ZenodoPublisherError(
            "Zenodo metadata must relate the dataset to the exact source tag"
        )
    matches = [
        item.get("identifier")
        for item in related
        if isinstance(item, Mapping) and item.get("relation") == "isSupplementTo"
    ]
    if (
        len(matches) != 1
        or not isinstance(matches[0], str)
        or not _SOURCE_TAG_RE.fullmatch(matches[0])
    ):
        raise ZenodoPublisherError(
            "Zenodo metadata must contain exactly one exact source tag release identity"
        )
    return matches[0]


def _metadata_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return caller-owned metadata fields, excluding Zenodo's reservation hint."""
    return {str(key): value for key, value in metadata.items() if key not in {"prereserve_doi"}}


_ZENODO_LICENSE_ALIASES = {
    "GPL-3.0-only": "gpl-3.0",
    "gpl-3.0": "gpl-3.0",
}


def _canonical_creator_for_comparison(creator: Any) -> Any:
    """Remove only Zenodo's null optional affiliation from a creator entry.

    Returns:
        The creator with a null affiliation omitted, or the original value.
    """
    if not isinstance(creator, Mapping):
        return creator
    canonical = dict(creator)
    if canonical.get("affiliation") is None:
        canonical.pop("affiliation", None)
    return canonical


def _canonical_metadata_value_for_comparison(key: str, value: Any) -> Any:
    """Canonicalize the known Zenodo read-back aliases for one metadata field.

    The API normalizes the configured GPL identifier and adds a null optional
    creator affiliation. All other fields and values remain unchanged so that
    title, type, source identity, and non-null creator metadata stay strict.

    Returns:
        The comparison value with only known Zenodo aliases canonicalized.
    """
    if key == "license" and isinstance(value, str):
        return _ZENODO_LICENSE_ALIASES.get(value, value)
    if key == "creators" and isinstance(value, list):
        return [_canonical_creator_for_comparison(creator) for creator in value]
    return value


def _metadata_sha256(metadata: Mapping[str, Any]) -> str:
    """Hash the exact user-controlled metadata contract.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(_canonical_bytes(_metadata_contract(metadata))).hexdigest()


def _binding_value(binding: Any, key: str) -> Any:
    """Read one release-binding field from a mapping or manifest-like object.

    Returns:
        The requested field value, or ``None`` when it is absent.
    """
    if isinstance(binding, Mapping):
        return binding.get(key)
    return getattr(binding, key, None)


def _normalize_release_binding(binding: Any) -> dict[str, Any]:
    """Normalize and validate the public fields needed to bind a Zenodo release.

    Returns:
        Normalized release identity and metadata path fields.
    """
    if binding is None:
        raise ZenodoPublisherError("Zenodo release binding is missing")
    metadata_path_value = _binding_value(binding, "metadata_path")
    if metadata_path_value is None:
        raise ZenodoPublisherError("Zenodo release binding metadata_path is missing")
    metadata_path = Path(str(metadata_path_value)).resolve()
    metadata_sha256 = str(_binding_value(binding, "metadata_sha256") or "").strip().lower()
    if _SHA256_RE.fullmatch(metadata_sha256) is None:
        raise ZenodoPublisherError("Zenodo release binding metadata_sha256 is invalid")
    release_tag = str(_binding_value(binding, "release_tag") or "").strip()
    if not release_tag:
        raise ZenodoPublisherError("Zenodo release binding release_tag is missing")
    expected_source_tag = (
        release_tag
        if release_tag.startswith("https://")
        else f"https://github.com/ll7/robot_sf_ll7/releases/tag/{release_tag}"
    )
    if _SOURCE_TAG_RE.fullmatch(expected_source_tag) is None:
        raise ZenodoPublisherError("Zenodo release binding release_tag is invalid")
    concept_doi = str(_binding_value(binding, "concept_doi") or "").strip()
    version_doi = str(_binding_value(binding, "version_doi") or "").strip()
    if _ZENODO_DOI_RE.fullmatch(concept_doi) is None:
        raise ZenodoPublisherError("Zenodo release binding concept_doi is invalid")
    if _ZENODO_DOI_RE.fullmatch(version_doi) is None:
        raise ZenodoPublisherError("Zenodo release binding version_doi is invalid")
    if concept_doi == version_doi:
        raise ZenodoPublisherError("Zenodo release binding concept and version DOI must differ")
    return {
        "metadata_path": metadata_path,
        "metadata_sha256": metadata_sha256,
        "release_tag": release_tag,
        "source_tag": expected_source_tag,
        "concept_doi": concept_doi,
        "version_doi": version_doi,
    }


def build_release_binding(manifest: Any) -> dict[str, Any]:
    """Build a publisher binding from a validated benchmark release manifest.

    The function intentionally accepts a manifest-like object instead of
    importing the release protocol module, keeping the generic publisher
    independent from the benchmark manifest implementation.

    Returns:
        Normalized release identity and metadata path fields.
    """
    return _normalize_release_binding(manifest)


def _state_release_binding(
    binding: Mapping[str, Any], *, metadata_contract_sha256: str
) -> dict[str, str]:
    """Return the credential-free binding persisted in deposition state."""
    return {
        "metadata_sha256": str(binding["metadata_sha256"]),
        "metadata_contract_sha256": metadata_contract_sha256,
        "release_tag": str(binding["release_tag"]),
        "concept_doi": str(binding["concept_doi"]),
        "version_doi": str(binding["version_doi"]),
    }


def _validate_release_binding_file(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Read and validate the exact metadata file named by a release binding.

    Returns:
        Validated metadata loaded from the bound file.
    """
    return load_dataset_metadata(
        binding["metadata_path"],
        expected_source_tag=binding["source_tag"],
        expected_metadata_sha256=binding["metadata_sha256"],
    )


def _validate_release_binding_metadata(
    metadata: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Require operation metadata to be byte-bound to the release manifest.

    Returns:
        Validated metadata loaded from the bound file.
    """
    file_metadata = _validate_release_binding_file(binding)
    if _metadata_contract(file_metadata) != _metadata_contract(metadata):
        raise ZenodoPublisherError(
            "Zenodo operation metadata does not match the release manifest metadata file"
        )
    return file_metadata


def _assert_deposition_identity(state: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    """Require reserved Zenodo identity to equal the manifest's concept/version DOIs."""
    concept_record_id = str(state.get("concept_record_id") or "")
    expected_concept_record_id = str(binding["concept_doi"].rsplit(".", 1)[-1])
    if concept_record_id != expected_concept_record_id:
        raise ZenodoPublisherError(
            "Zenodo deposition concept DOI does not match the release manifest"
        )
    if state.get("doi") != binding["version_doi"]:
        raise ZenodoPublisherError(
            "Zenodo deposition version DOI does not match the release manifest"
        )


def _validate_state_binding(
    state: dict[str, Any],
    binding: Mapping[str, Any] | None,
    *,
    metadata_contract_sha256: str | None = None,
) -> None:
    """Validate or adopt the manifest binding carried by deposition state."""
    stored = state.get("release_binding")
    if stored is not None and not isinstance(stored, Mapping):
        raise ZenodoPublisherError("Zenodo state release binding is malformed")
    if binding is None:
        if stored is None:
            return
        if not isinstance(stored, Mapping):  # pragma: no cover - narrowed above
            raise ZenodoPublisherError("Zenodo state release binding is malformed")
        if (
            metadata_contract_sha256 is not None
            and stored.get("metadata_contract_sha256") != metadata_contract_sha256
        ):
            raise ZenodoPublisherError("Zenodo state metadata does not match its release binding")
        return

    _assert_deposition_identity(state, binding)
    expected_contract_sha256 = metadata_contract_sha256 or str(
        stored.get("metadata_contract_sha256") if isinstance(stored, Mapping) else ""
    )
    if not expected_contract_sha256:
        raise ZenodoPublisherError("Zenodo release binding metadata contract is missing")
    expected_state_binding = _state_release_binding(
        binding, metadata_contract_sha256=expected_contract_sha256
    )
    if stored is not None and dict(stored) != expected_state_binding:
        raise ZenodoPublisherError("Zenodo state release binding does not match the manifest")
    state["release_binding"] = expected_state_binding


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the public benchmark-data metadata contract.

    Returns:
        A shallow copy of the validated metadata.
    """
    if not isinstance(metadata, Mapping):
        raise ZenodoPublisherError("Zenodo metadata must be a JSON object")
    if metadata.get("upload_type") != "dataset":
        raise ZenodoPublisherError("Zenodo benchmark publication must use upload_type=dataset")
    if metadata.get("license") != "GPL-3.0-only":
        raise ZenodoPublisherError("Zenodo benchmark publication license must be GPL-3.0-only")
    creators = metadata.get("creators")
    if not isinstance(creators, list) or not creators:
        raise ZenodoPublisherError("Zenodo metadata must name at least one creator")
    if any(
        not isinstance(creator, Mapping)
        or not isinstance(creator.get("name"), str)
        or not creator["name"].strip()
        for creator in creators
    ):
        raise ZenodoPublisherError("Zenodo metadata creators must contain non-empty names")
    _source_tag(metadata)
    if metadata.get("access_right") != "open":
        raise ZenodoPublisherError("Zenodo benchmark publication must use access_right=open")
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ZenodoPublisherError("Zenodo metadata description is required")
    description_lower = description.casefold()
    if not all(term in description_lower for term in _CLAIM_BOUNDARY_TERMS):
        raise ZenodoPublisherError(
            "Zenodo metadata description must state the SNQI advisory/no-ranking claim boundary"
        )
    return dict(metadata)


def _seal_payload(payload: Mapping[str, Any], integrity_key: str, schema: str) -> dict[str, Any]:
    """Attach a deterministic self-hash to a public state/receipt payload.

    Returns:
        A credential-free payload containing its integrity block.
    """
    sealed = dict(payload)
    sealed[integrity_key] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-utf8-v1",
        "excluded_json_key": integrity_key,
        "receipt_sha256": _payload_sha256(sealed, integrity_key),
        "schema_version": schema,
    }
    _assert_credential_free(sealed)
    return sealed


def _seal_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Seal deposition state for persistence and in-memory handoff.

    Returns:
        A state object with a self-integrity block.
    """
    return _seal_payload(state, "integrity", ZENODO_STATE_SCHEMA)


def _verify_integrity(payload: Mapping[str, Any], *, key: str, schema: str) -> None:
    """Require a matching self-hash for a state or verification receipt."""
    integrity = payload.get(key)
    if not isinstance(integrity, Mapping) or integrity.get("schema_version") != schema:
        raise ZenodoPublisherError("Zenodo state/receipt integrity block is missing")
    expected = _payload_sha256(payload, key)
    if integrity.get("receipt_sha256") != expected:
        raise ZenodoPublisherError("Zenodo state/receipt integrity self-hash mismatch")
    _assert_credential_free(payload)


def _validate_state_for_operation(state: Mapping[str, Any]) -> None:
    """Validate identity and integrity before a mutating deposition operation."""
    if not isinstance(state, Mapping) or not state.get("deposition_id"):
        raise ZenodoPublisherError("deposition state has no deposition_id")
    if not isinstance(state, Mapping) or state.get("schema_version") != ZENODO_STATE_SCHEMA:
        raise ZenodoPublisherError("invalid Zenodo deposition state")
    _verify_integrity(state, key="integrity", schema=ZENODO_STATE_SCHEMA)


def _file_inventory(state: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate and index the nonempty expected file inventory.

    Returns:
        An index by filename and a list of contract violations.
    """
    raw_files = state.get("files")
    problems: list[str] = []
    if not isinstance(raw_files, list) or not raw_files:
        return {}, ["expected file inventory is empty"]
    indexed: dict[str, Any] = {}
    for item in raw_files:
        if not isinstance(item, Mapping):
            problems.append("expected file inventory contains a malformed entry")
            continue
        name = item.get("name")
        size = item.get("size")
        digest = item.get("sha256")
        if not isinstance(name, str) or not name or name in indexed:
            problems.append("expected file inventory contains a missing or duplicate filename")
            continue
        if not isinstance(size, int) or size <= 0:
            problems.append(f"expected file {name} is empty or has an invalid size")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            problems.append(f"expected file {name} has an invalid SHA-256")
        indexed[name] = dict(item)
    return indexed, problems


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local file.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_remote_file(response: _Response) -> tuple[int, str]:
    """Hash and count a remote file without materializing its response body.

    Returns:
        A ``(byte_count, sha256)`` tuple for the streamed response.

    Raises:
        ZenodoPublisherError: If the response cannot provide valid byte chunks.
    """
    iterator = getattr(response, "iter_content", None)
    if not callable(iterator):
        raise ZenodoPublisherError("Zenodo response does not support streamed downloads")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        for chunk in iterator(chunk_size=_REMOTE_DOWNLOAD_CHUNK_SIZE):
            if not chunk:
                continue
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise ZenodoPublisherError("Zenodo streamed download yielded a non-byte chunk")
            chunk_bytes = bytes(chunk)
            byte_count += len(chunk_bytes)
            digest.update(chunk_bytes)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    return byte_count, digest.hexdigest()


def read_token_file(path: str | Path) -> str:
    """Read a Zenodo token from a mode-0600 file without logging it.

    Returns:
        The non-empty token text.
    """
    token_path = Path(path).resolve()
    if not token_path.is_file():
        raise ZenodoPublisherError(f"Zenodo token file not found: {token_path}")
    if token_path.stat().st_mode & 0o077:
        raise ZenodoPublisherError("Zenodo token file permissions must be 0600 or stricter")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ZenodoPublisherError("Zenodo token file is empty")
    return token


def build_session(token_file: str | Path) -> _Session:
    """Build an authenticated requests session using header-only credentials.

    Returns:
        An authenticated session. The token is never placed in URLs.
    """
    requests = try_import("requests")
    if requests is None:  # pragma: no cover - all-extras environment owns requests
        raise ZenodoPublisherError("requests is required; run `uv sync --all-extras`")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {read_token_file(token_file)}"})
    return session


def load_dataset_metadata(
    path: str | Path,
    *,
    expected_source_tag: str | None = None,
    expected_metadata_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and validate benchmark-dataset deposition metadata.

    Returns:
        Metadata suitable for the Zenodo deposition API.
    """
    metadata_path = Path(path).resolve()
    if expected_metadata_sha256 is not None:
        expected_digest = str(expected_metadata_sha256).strip().lower()
        if _SHA256_RE.fullmatch(expected_digest) is None:
            raise ZenodoPublisherError("expected Zenodo metadata SHA-256 is invalid")
        if not metadata_path.is_file() or _sha256_file(metadata_path) != expected_digest:
            raise ZenodoPublisherError(
                "Zenodo metadata file SHA-256 does not match the release manifest"
            )
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ZenodoPublisherError("Zenodo metadata file could not be read") from exc
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, Mapping):
        raise ZenodoPublisherError("metadata file must contain a top-level metadata object")
    normalized = _validate_metadata(metadata)
    source_tag = _source_tag(normalized)
    if expected_source_tag is not None:
        expected_url = (
            expected_source_tag
            if expected_source_tag.startswith("https://")
            else f"https://github.com/ll7/robot_sf_ll7/releases/tag/{expected_source_tag}"
        )
        if source_tag != expected_url:
            raise ZenodoPublisherError("Zenodo metadata source tag does not match expected tag")
    normalized["prereserve_doi"] = True
    return normalized


def _json_object(response: _Response, operation: str) -> dict[str, Any]:
    """Require a successful JSON-object API response.

    Returns:
        Parsed response object.
    """
    try:
        response.raise_for_status()
        payload = response.json()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ZenodoPublisherError(f"Zenodo {operation} request failed") from exc
    if not isinstance(payload, dict):
        raise ZenodoPublisherError(f"Zenodo {operation} response was not a JSON object")
    return payload


def _public_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the non-secret deposition identity needed by later modes.

    Returns:
        A credential-free state object.
    """
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    preregistered = (
        metadata.get("prereserve_doi") if isinstance(metadata.get("prereserve_doi"), dict) else {}
    )
    return {
        "schema_version": ZENODO_STATE_SCHEMA,
        "deposition_id": payload.get("id"),
        "record_id": payload.get("record_id"),
        "concept_record_id": payload.get("conceptrecid"),
        "doi": payload.get("doi") or preregistered.get("doi"),
        "state": payload.get("state"),
        "submitted": bool(payload.get("submitted")),
        "files": [],
    }


def reserve(
    session: _Session,
    metadata: dict[str, Any],
    *,
    api_base: str = ZENODO_API_BASE,
    release_binding: Any | None = None,
) -> dict[str, Any]:
    """Create a fresh deposition and reserve its version DOI.

    Returns:
        Credential-free deposition state.
    """
    normalized_metadata = _validate_metadata(metadata)
    binding = _normalize_release_binding(release_binding) if release_binding is not None else None
    file_metadata = (
        _validate_release_binding_metadata(normalized_metadata, binding)
        if binding is not None
        else None
    )
    normalized_metadata["prereserve_doi"] = True
    response = session.post(
        f"{api_base.rstrip('/')}/deposit/depositions",
        json={"metadata": normalized_metadata},
        timeout=60,
    )
    payload = _json_object(response, "reserve")
    state = _public_state(payload)
    if not state["deposition_id"] or not state["doi"] or not state["concept_record_id"]:
        raise ZenodoPublisherError(
            "Zenodo reserve response omitted deposition/concept/DOI identity"
        )
    if binding is not None:
        _assert_deposition_identity(state, binding)
        state["release_binding"] = _state_release_binding(
            binding,
            metadata_contract_sha256=_metadata_sha256(file_metadata or normalized_metadata),
        )
    return _seal_state(state)


def recover(
    session: _Session,
    deposition_id: int,
    metadata: Mapping[str, Any],
    *,
    api_base: str = ZENODO_API_BASE,
    release_binding: Any,
) -> dict[str, Any]:
    """Recover sealed state for one manifest-bound unpublished deposition.

    Recovery is deliberately read-only: it retrieves the exact deposition ID,
    validates the remote identity and metadata against the frozen release
    binding, and reconstructs only the credential-free state emitted by
    :func:`reserve`.

    Returns:
        Credential-free sealed deposition state.
    """
    if isinstance(deposition_id, bool) or not isinstance(deposition_id, int) or deposition_id <= 0:
        raise ZenodoPublisherError("Zenodo recovery deposition ID must be a positive integer")
    normalized_metadata = _validate_metadata(metadata)
    binding = _normalize_release_binding(release_binding)
    file_metadata = _validate_release_binding_metadata(normalized_metadata, binding)
    payload = _json_object(
        session.get(
            f"{api_base.rstrip('/')}/deposit/depositions/{deposition_id}",
            timeout=60,
        ),
        "recover draft",
    )
    submitted = payload.get("submitted")
    if not isinstance(submitted, bool):
        raise ZenodoPublisherError("Zenodo recovery response omitted or invalid submitted state")
    state = _public_state(payload)
    if state.get("deposition_id") != deposition_id:
        raise ZenodoPublisherError("Zenodo recovery response changed the requested deposition ID")
    if not state.get("record_id") or not state.get("concept_record_id") or not state.get("doi"):
        raise ZenodoPublisherError(
            "Zenodo recovery response omitted deposition/concept/DOI identity"
        )
    if state.get("submitted") is not False or state.get("state") != "unsubmitted":
        raise ZenodoPublisherError("Zenodo recovery requires an unpublished draft deposition")
    _assert_deposition_identity(state, binding)

    remote_metadata = payload.get("metadata")
    if not isinstance(remote_metadata, Mapping):
        raise ZenodoPublisherError("Zenodo recovery response omitted deposition metadata")
    for key, value in _metadata_contract(normalized_metadata).items():
        if _canonical_metadata_value_for_comparison(
            key, remote_metadata.get(key)
        ) != _canonical_metadata_value_for_comparison(key, value):
            raise ZenodoPublisherError(
                f"Zenodo recovered draft metadata.{key} does not match release metadata"
            )

    state["release_binding"] = _state_release_binding(
        binding,
        metadata_contract_sha256=_metadata_sha256(file_metadata),
    )
    return _seal_state(state)


def upload(
    session: _Session,
    state: dict[str, Any],
    files: list[Path],
    *,
    api_base: str = ZENODO_API_BASE,
    release_binding: Any | None = None,
) -> dict[str, Any]:
    """Upload files to an unpublished deposition and record local SHA-256 values.

    Returns:
        Updated credential-free deposition state.
    """
    _validate_state_for_operation(state)
    binding = _normalize_release_binding(release_binding) if release_binding is not None else None
    binding_metadata = _validate_release_binding_file(binding) if binding is not None else None
    _validate_state_binding(
        state,
        binding,
        metadata_contract_sha256=(
            _metadata_sha256(binding_metadata) if binding_metadata is not None else None
        ),
    )
    deposition_id = state.get("deposition_id")
    deposition = _json_object(
        session.get(f"{api_base.rstrip('/')}/deposit/depositions/{deposition_id}", timeout=60),
        "retrieve draft",
    )
    if bool(deposition.get("submitted")):
        raise ZenodoPublisherError("cannot upload files to a published Zenodo deposition")
    links = deposition.get("links")
    bucket = links.get("bucket") if isinstance(links, dict) else None
    if not isinstance(bucket, str) or not bucket.startswith("https://"):
        raise ZenodoPublisherError("Zenodo draft response omitted a secure upload bucket")
    if not files:
        raise ZenodoPublisherError("upload requires at least one nonempty file")
    uploaded: list[dict[str, Any]] = []
    for file_path in files:
        resolved = file_path.resolve()
        if not resolved.is_file():
            raise ZenodoPublisherError(f"upload file not found: {resolved}")
        size = resolved.stat().st_size
        if size <= 0:
            raise ZenodoPublisherError(f"upload file is empty: {resolved.name}")
        with resolved.open("rb") as stream:
            response = session.put(
                f"{bucket.rstrip('/')}/{quote(resolved.name)}",
                data=stream,
                timeout=3600,
            )
        remote = _json_object(response, f"upload {resolved.name}")
        uploaded.append(
            {
                "name": resolved.name,
                "size": size,
                "sha256": _sha256_file(resolved),
                "zenodo_checksum": remote.get("checksum"),
            }
        )
    updated = dict(state)
    updated["files"] = uploaded
    updated.pop("verification_receipt", None)
    return _seal_state(updated)


def publish(  # noqa: C901
    session: _Session,
    state: dict[str, Any],
    metadata: Mapping[str, Any] | None = None,
    *,
    api_base: str = ZENODO_API_BASE,
    release_binding: Any | None = None,
) -> dict[str, Any]:
    """Irreversibly publish a deposition admitted by a draft verification receipt.

    Returns:
        Updated published deposition state.
    """
    _validate_state_for_operation(state)
    deposition_id = state.get("deposition_id")
    if metadata is None:
        raise ZenodoPublisherError("publish requires the exact expected metadata")
    normalized_metadata = _validate_metadata(metadata)
    binding = _normalize_release_binding(release_binding) if release_binding is not None else None
    file_metadata = (
        _validate_release_binding_metadata(normalized_metadata, binding)
        if binding is not None
        else None
    )
    _validate_state_binding(
        state,
        binding,
        metadata_contract_sha256=(
            _metadata_sha256(file_metadata or normalized_metadata)
            if binding is not None
            else _metadata_sha256(normalized_metadata)
        ),
    )
    if bool(state.get("submitted")):
        raise ZenodoPublisherError("Zenodo deposition is already published")
    expected_files, file_problems = _file_inventory(state)
    if file_problems:
        raise ZenodoPublisherError("cannot publish: " + "; ".join(file_problems))
    receipt = state.get("verification_receipt")
    if not isinstance(receipt, Mapping):
        raise ZenodoPublisherError("publish requires a prior verification receipt")
    _verify_integrity(receipt, key="integrity", schema=ZENODO_VERIFICATION_SCHEMA)
    if receipt.get("status") != "pass" or receipt.get("publication_state") != "draft":
        raise ZenodoPublisherError("publish requires a passing draft verification receipt")
    if receipt.get("deposition_id") != state.get("deposition_id"):
        raise ZenodoPublisherError("verification receipt deposition identity does not match state")
    if receipt.get("metadata_sha256") != _metadata_sha256(normalized_metadata):
        raise ZenodoPublisherError("verification receipt metadata does not match expected metadata")
    if receipt.get("source_tag") != _source_tag(normalized_metadata):
        raise ZenodoPublisherError("verification receipt source tag does not match metadata")
    if state.get("release_binding") is not None and receipt.get("release_binding") != state.get(
        "release_binding"
    ):
        raise ZenodoPublisherError("verification receipt release binding does not match state")
    if state.get("release_binding") is not None and receipt.get(
        "manifest_metadata_sha256"
    ) != state["release_binding"].get("metadata_sha256"):
        raise ZenodoPublisherError("verification receipt metadata checksum does not match state")
    receipt_files = receipt.get("files")
    expected_receipt_files = [
        {"name": name, "size": item["size"], "sha256": item["sha256"]}
        for name, item in sorted(expected_files.items())
    ]
    if receipt_files != expected_receipt_files:
        raise ZenodoPublisherError("verification receipt file inventory does not match state")
    response = session.post(
        f"{api_base.rstrip('/')}/deposit/depositions/{deposition_id}/actions/publish",
        timeout=120,
    )
    published = _public_state(_json_object(response, "publish"))
    if not published["submitted"]:
        raise ZenodoPublisherError("Zenodo publish response did not mark the deposition submitted")
    for key in ("deposition_id", "record_id", "concept_record_id", "doi"):
        if published.get(key) != state.get(key):
            raise ZenodoPublisherError(f"Zenodo publish response changed {key}")
    published["files"] = list(state["files"])
    if state.get("release_binding") is not None:
        published["release_binding"] = dict(state["release_binding"])
    published["verification_receipt"] = dict(receipt)
    published["published_from_receipt_sha256"] = receipt["integrity"]["receipt_sha256"]
    return _seal_state(published)


def verify(  # noqa: C901, PLR0912, PLR0915
    session: _Session,
    state: dict[str, Any],
    metadata: Mapping[str, Any],
    *,
    api_base: str = ZENODO_API_BASE,
    release_binding: Any | None = None,
) -> dict[str, Any]:
    """Verify a draft or published deposition and, on pass, seal a receipt.

    Returns:
        A machine-readable verification report. Passing draft verification is
        also sealed into ``state`` for publication admission.
    """
    _validate_state_for_operation(state)
    normalized_metadata = _validate_metadata(metadata)
    binding = _normalize_release_binding(release_binding) if release_binding is not None else None
    file_metadata = (
        _validate_release_binding_metadata(normalized_metadata, binding)
        if binding is not None
        else None
    )
    _validate_state_binding(
        state,
        binding,
        metadata_contract_sha256=(
            _metadata_sha256(file_metadata or normalized_metadata)
            if binding is not None
            else _metadata_sha256(normalized_metadata)
        ),
    )
    deposition_id = state.get("deposition_id")
    remote = _json_object(
        session.get(f"{api_base.rstrip('/')}/deposit/depositions/{deposition_id}", timeout=60),
        "verify",
    )
    remote_state = _public_state(remote)
    remote_metadata = remote.get("metadata") if isinstance(remote.get("metadata"), Mapping) else {}
    problems: list[str] = []

    for key in ("deposition_id", "record_id", "concept_record_id", "doi"):
        if remote_state.get(key) != state.get(key):
            problems.append(f"{key} does not match reserved state")
    expected_submitted = state.get("submitted")
    if not isinstance(expected_submitted, bool):
        problems.append("state submitted flag is missing or invalid")
    remote_submitted = remote.get("submitted")
    if not isinstance(remote_submitted, bool):
        problems.append("remote deposition submitted flag is missing or invalid")
    publication_state = (
        "published"
        if remote_submitted is True
        else "draft"
        if remote_submitted is False
        else "unknown"
    )
    if isinstance(expected_submitted, bool) and remote_submitted != expected_submitted:
        problems.append("remote deposition draft/published state does not match state")

    expected_metadata = _metadata_contract(normalized_metadata)
    for key, value in expected_metadata.items():
        remote_value = remote_metadata.get(key)
        if _canonical_metadata_value_for_comparison(
            key, remote_value
        ) != _canonical_metadata_value_for_comparison(key, value):
            problems.append(f"metadata.{key} does not match requested metadata")
    try:
        source_tag = _source_tag(normalized_metadata)
    except (
        ZenodoPublisherError
    ) as exc:  # guarded by _validate_metadata, kept explicit for type narrowing
        problems.append(str(exc))
        source_tag = ""

    expected_files, file_problems = _file_inventory(state)
    problems.extend(file_problems)
    file_inventory_source = remote
    if remote_submitted is True:
        record_id = state.get("record_id")
        public_record = _json_object(
            session.get(f"{api_base.rstrip('/')}/records/{record_id}", timeout=60),
            "verify published record",
        )
        if public_record.get("id") != record_id:
            problems.append("published record id does not match reserved state")
        if str(public_record.get("conceptrecid") or "") != str(
            state.get("concept_record_id") or ""
        ):
            problems.append("published record concept id does not match reserved state")
        if public_record.get("doi") != state.get("doi"):
            problems.append("published record DOI does not match reserved state")
        if public_record.get("status") != "published":
            problems.append("published record status is not published")
        file_inventory_source = public_record

    remote_files_value = file_inventory_source.get("files")
    remote_files = remote_files_value if isinstance(remote_files_value, list) else []
    if not remote_files:
        problems.append("remote file inventory is empty")
    remote_by_name: dict[str, Mapping[str, Any]] = {}
    for item in remote_files:
        if not isinstance(item, Mapping):
            problems.append("remote file inventory contains a malformed entry")
            continue
        name = item.get("filename") or item.get("key")
        if not isinstance(name, str) or not name:
            problems.append("remote file inventory contains an unnamed entry")
            continue
        if name in remote_by_name:
            problems.append(f"remote file inventory contains duplicate filename {name}")
            continue
        remote_by_name[name] = item
    if set(expected_files) != set(remote_by_name):
        problems.append("remote file inventory does not match uploaded file inventory")

    for name, expected in expected_files.items():
        remote_file = remote_by_name.get(name)
        if remote_file is None:
            continue
        remote_size = remote_file.get("size")
        # Zenodo's legacy deposit/depositions draft response may omit size even
        # after a completed bucket upload. The streamed cold download below is
        # still authoritative for exact bytes and SHA-256. Published records,
        # and any draft that does advertise size, remain fail-closed.
        if remote_size is None and remote_submitted is False:
            pass
        elif not isinstance(remote_size, int) or isinstance(remote_size, bool):
            problems.append(f"remote file {name} has an invalid size")
        elif remote_size <= 0:
            problems.append(f"remote file {name} is empty")
        elif remote_size != expected.get("size"):
            problems.append(f"remote file {name} size does not match uploaded bytes")
        links = remote_file.get("links")
        download_url = (
            (links.get("self") if remote_submitted is True else links.get("download"))
            if isinstance(links, Mapping)
            else None
        )
        if not isinstance(download_url, str) or not download_url.startswith("https://"):
            problems.append(f"remote file {name} has no secure download URL")
            continue
        response = session.get(download_url, stream=True, timeout=3600)
        if response.status_code >= 400:
            problems.append(f"remote file {name} download failed")
            continue
        try:
            downloaded_size, observed_sha = _stream_remote_file(response)
        except (OSError, RuntimeError, TypeError, ValueError, ZenodoPublisherError):
            problems.append(f"remote file {name} download failed")
            continue
        if downloaded_size <= 0:
            problems.append(f"remote file {name} is empty")
            continue
        if downloaded_size != expected.get("size"):
            problems.append(f"remote file {name} downloaded size does not match uploaded bytes")
        if observed_sha != expected.get("sha256"):
            problems.append(f"remote file {name} SHA-256 does not match uploaded bytes")

    report: dict[str, Any] = {
        "schema_version": ZENODO_VERIFICATION_SCHEMA,
        "status": "pass" if not problems else "fail",
        "problem_count": len(problems),
        "problems": problems,
        "doi": remote_state.get("doi"),
        "concept_record_id": remote_state.get("concept_record_id"),
        "submitted": remote_state.get("submitted"),
        "publication_state": publication_state,
        "file_count": len(remote_files),
    }
    if not problems:
        receipt_files = [
            {"name": name, "size": item["size"], "sha256": item["sha256"]}
            for name, item in sorted(expected_files.items())
        ]
        receipt = _seal_payload(
            {
                "schema_version": ZENODO_VERIFICATION_SCHEMA,
                "status": "pass",
                "publication_state": publication_state,
                "deposition_id": state["deposition_id"],
                "record_id": state.get("record_id"),
                "concept_record_id": state.get("concept_record_id"),
                "doi": state.get("doi"),
                "metadata_sha256": _metadata_sha256(normalized_metadata),
                "source_tag": source_tag,
                "files": receipt_files,
                **(
                    {
                        "release_binding": dict(state["release_binding"]),
                        "manifest_metadata_sha256": state["release_binding"]["metadata_sha256"],
                    }
                    if state.get("release_binding") is not None
                    else {}
                ),
            },
            "integrity",
            ZENODO_VERIFICATION_SCHEMA,
        )
        report["receipt"] = receipt
        updated_state = dict(state)
        updated_state["verification_receipt"] = receipt
        sealed_state = _seal_state(updated_state)
        state.clear()
        state.update(sealed_state)
    return report


def load_state(path: str | Path) -> dict[str, Any]:
    """Load a credential-free publisher state file.

    Returns:
        Parsed state object.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ZenodoPublisherError("invalid Zenodo deposition state file") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != ZENODO_STATE_SCHEMA:
        raise ZenodoPublisherError("invalid Zenodo deposition state file")
    _verify_integrity(payload, key="integrity", schema=ZENODO_STATE_SCHEMA)
    return payload


def write_state(path: str | Path, state: dict[str, Any]) -> None:
    """Persist credential-free publisher state without overwriting an existing identity."""
    if not isinstance(state, Mapping) or state.get("schema_version") != ZENODO_STATE_SCHEMA:
        raise ZenodoPublisherError("invalid Zenodo deposition state")
    _assert_credential_free(state)
    _verify_integrity(state, key="integrity", schema=ZENODO_STATE_SCHEMA)
    output = Path(path)
    if output.exists():
        existing = load_state(output)
        if existing.get("deposition_id") != state.get("deposition_id"):
            raise ZenodoPublisherError("refusing to overwrite a different Zenodo deposition state")
    output.parent.mkdir(parents=True, exist_ok=True)
    sealed = _seal_state(state)
    output.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output, 0o600)


__all__ = [
    "ZENODO_API_BASE",
    "ZENODO_STATE_SCHEMA",
    "ZENODO_VERIFICATION_SCHEMA",
    "ZenodoPublisherError",
    "build_release_binding",
    "build_session",
    "load_dataset_metadata",
    "load_state",
    "publish",
    "read_token_file",
    "recover",
    "reserve",
    "upload",
    "verify",
    "write_state",
]

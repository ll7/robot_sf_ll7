"""Direct, credential-safe Zenodo publisher for benchmark-data releases."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlsplit

from robot_sf.benchmark.release_tag_identity import check_canonical_source_tag
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
_APPROVED_ZENODO_API_HOSTS = frozenset({"zenodo.org"})
_KNOWN_ZENODO_STATES = frozenset({"unsubmitted", "inprogress", "done", "error"})
_STABLE_ZENODO_STATES = frozenset({"unsubmitted", "done"})
_REMOTE_VERSION_FIELDS = ("modified", "version", "revision")


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

    def delete(self, url: str, **kwargs: Any) -> _Response:
        """Issue DELETE."""


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


def _source_tag(metadata: Mapping[str, Any], *, require_url_scheme: bool = False) -> str:
    """Return the one exact GitHub release URL bound to dataset metadata.

    ``new_version`` uses the stricter form because it is about to mutate a
    published Zenodo concept. The source relation must therefore identify the
    GitHub URL with Zenodo's explicit ``url`` scheme, rather than relying on
    the identifier text alone. The default remains permissive for legacy
    metadata reads and non-mutating recovery of older records.
    """
    related = metadata.get("related_identifiers")
    if not isinstance(related, list):
        raise ZenodoPublisherError(
            "Zenodo metadata must relate the dataset to the exact source tag"
        )
    matches = [
        item
        for item in related
        if isinstance(item, Mapping) and item.get("relation") == "isSupplementTo"
    ]
    if (
        len(matches) != 1
        or not isinstance(matches[0].get("identifier"), str)
        or not _SOURCE_TAG_RE.fullmatch(matches[0]["identifier"])
        or (require_url_scheme and matches[0].get("scheme") != "url")
    ):
        raise ZenodoPublisherError(
            "Zenodo metadata must contain exactly one exact source tag release identity"
        )
    return matches[0]["identifier"]


def _expected_source_tag_url(value: str, *, label: str) -> str:
    """Normalize one exact GitHub release tag or URL.

    Returns:
        The validated canonical GitHub release URL.
    """
    if not isinstance(value, str) or not value.strip():
        raise ZenodoPublisherError(f"{label} is invalid")
    candidate = value.strip()
    expected_url = (
        candidate
        if candidate.startswith("https://")
        else f"https://github.com/ll7/robot_sf_ll7/releases/tag/{candidate}"
    )
    if _SOURCE_TAG_RE.fullmatch(expected_url) is None:
        raise ZenodoPublisherError(f"{label} is invalid")
    return expected_url


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


def _validated_api_base(api_base: str) -> str:
    """Require the production Zenodo HTTPS API base without URL credentials/queries.

    The sandbox is intentionally unsupported: its official test DOI prefix is
    ``10.5072`` while this publisher's persisted identity schema is pinned to
    production ``10.5281/zenodo.<record-id>`` values.

    Returns:
        The trimmed API base URL.
    """
    if not isinstance(api_base, str) or not api_base.strip():
        raise ZenodoPublisherError("Zenodo API base must be a non-empty HTTPS URL")
    candidate = api_base.strip().rstrip("/")
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ZenodoPublisherError("Zenodo API base must be a valid HTTPS URL") from exc
    normalized_hostname = hostname.casefold() if hostname is not None else None
    normalized_path = parsed.path.rstrip("/")
    if normalized_hostname not in _APPROVED_ZENODO_API_HOSTS:
        raise ZenodoPublisherError("Zenodo API base must use an approved Zenodo HTTPS origin")
    if normalized_hostname == "zenodo.org" and port not in {None, 443}:
        raise ZenodoPublisherError("Zenodo API base must use an approved Zenodo HTTPS origin")
    if (
        parsed.scheme.casefold() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or normalized_path != "/api"
    ):
        raise ZenodoPublisherError("Zenodo API base must be a valid HTTPS URL")
    return candidate.rstrip("/")


def _validated_remote_url(value: Any, api_base: str, label: str) -> str:
    """Validate one server-supplied URL against the configured API origin.

    Server-supplied bucket and download links are untrusted data. They must be
    HTTPS URLs on the exact configured API origin, with no userinfo, query, or
    fragment that could smuggle credentials or redirect the authenticated
    session to an unintended resource.

    Returns:
        The validated URL.
    """
    if not isinstance(value, str) or not value.strip():
        raise ZenodoPublisherError(f"Zenodo {label} is not a valid same-origin HTTPS URL")
    try:
        base = urlsplit(_validated_api_base(api_base))
        candidate = urlsplit(value.strip())
        base_hostname = base.hostname
        candidate_hostname = candidate.hostname
        base_port = base.port or 443
        candidate_port = candidate.port or 443
    except ValueError as exc:
        raise ZenodoPublisherError(f"Zenodo {label} is not a valid same-origin HTTPS URL") from exc
    if (
        candidate.scheme.casefold() != "https"
        or not candidate_hostname
        or candidate_hostname.casefold() != (base_hostname or "").casefold()
        or candidate_port != base_port
        or candidate.username is not None
        or candidate.password is not None
        or candidate.query
        or candidate.fragment
        or not candidate.path
    ):
        raise ZenodoPublisherError(f"Zenodo {label} is not a valid same-origin HTTPS URL")
    api_path = base.path.rstrip("/")
    normalized_path = posixpath.normpath(unquote(candidate.path))
    if normalized_path != api_path and not normalized_path.startswith(f"{api_path}/"):
        raise ZenodoPublisherError(f"Zenodo {label} is not a valid same-API HTTPS URL")
    return value.strip()


def _validated_latest_draft_link(latest_draft: Any, api_base: str) -> tuple[str, int]:
    """Validate and extract the deposition ID from a same-API draft link.

    Returns:
        The validated link and its positive deposition identifier.
    """
    validated_base = _validated_api_base(api_base)
    base = urlsplit(validated_base)
    if not isinstance(latest_draft, str) or not latest_draft.strip():
        raise ZenodoPublisherError("Zenodo new-version response omitted links.latest_draft")
    try:
        candidate = _validated_remote_url(latest_draft, validated_base, "links.latest_draft")
    except ZenodoPublisherError as exc:
        raise ZenodoPublisherError("Zenodo links.latest_draft is not a valid same-API URL") from exc
    try:
        link = urlsplit(candidate)
        base_hostname = base.hostname
        link_hostname = link.hostname
        base_port = base.port or 443
        link_port = link.port or 443
    except ValueError as exc:
        raise ZenodoPublisherError("Zenodo links.latest_draft is not a valid same-API URL") from exc
    base_path = base.path.rstrip("/")
    link_path = link.path.rstrip("/")
    relative_path = link_path[len(base_path) :].lstrip("/") if base_path else link_path.lstrip("/")
    parts = relative_path.split("/")
    deposition_text = (
        parts[-1] if len(parts) == 3 and parts[:2] == ["deposit", "depositions"] else ""
    )
    if (
        link_hostname is None
        or link_hostname.casefold() != (base_hostname or "").casefold()
        or link_port != base_port
        or (base_path and not (link_path == base_path or link_path.startswith(base_path + "/")))
        or not re.fullmatch(r"[1-9][0-9]*", deposition_text)
    ):
        raise ZenodoPublisherError("Zenodo links.latest_draft is not a valid same-API URL")
    return candidate, int(deposition_text)


def _positive_deposition_id(value: Any, label: str) -> int:
    """Require an API deposition/record identifier to be a positive integer.

    Returns:
        The validated identifier.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ZenodoPublisherError(f"Zenodo {label} must be a positive integer")
    return value


def _positive_decimal_id(value: Any, label: str) -> str:
    """Require and canonicalize a positive decimal identifier string.

    Returns:
        The identifier without leading zeroes.
    """
    if isinstance(value, bool):
        raise ZenodoPublisherError(f"Zenodo {label} must be a positive decimal identifier")
    if isinstance(value, int):
        if value <= 0:
            raise ZenodoPublisherError(f"Zenodo {label} must be a positive decimal identifier")
        return str(value)
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise ZenodoPublisherError(f"Zenodo {label} must be a positive decimal identifier")
    normalized = value.lstrip("0")
    if not normalized:
        raise ZenodoPublisherError(f"Zenodo {label} must be a positive decimal identifier")
    return normalized


def _validated_version_doi(value: Any, record_id: int, label: str) -> str:
    """Require a production version DOI whose record suffix matches its ID.

    Returns:
        The validated DOI.
    """
    if not isinstance(value, str) or _ZENODO_DOI_RE.fullmatch(value) is None:
        raise ZenodoPublisherError(f"Zenodo {label} is invalid")
    if value.rsplit(".", 1)[-1] != str(record_id):
        raise ZenodoPublisherError(f"Zenodo {label} does not match record ID")
    return value


def _successor_version_doi(payload: Mapping[str, Any]) -> Any:
    """Extract a version DOI from the direct or pre-reserved legacy API field.

    Returns:
        The direct or pre-reserved DOI, if present.
    """
    direct_doi = payload.get("doi")
    if direct_doi is not None and direct_doi != "":
        return direct_doi
    metadata = payload.get("metadata")
    preregistered = metadata.get("prereserve_doi") if isinstance(metadata, Mapping) else None
    return preregistered.get("doi") if isinstance(preregistered, Mapping) else None


def _validate_successor_concept(
    payload: Mapping[str, Any], *, expected_concept_doi: str, operation: str
) -> str:
    """Validate the legacy concept record ID and any advertised concept DOI.

    Returns:
        The normalized concept record identifier.
    """
    concept_record_id = payload.get("conceptrecid")
    if isinstance(concept_record_id, bool) or not str(concept_record_id or "").isdigit():
        raise ZenodoPublisherError(f"Zenodo {operation} omitted a positive concept record ID")
    if int(str(concept_record_id)) <= 0:
        raise ZenodoPublisherError(f"Zenodo {operation} omitted a positive concept record ID")
    expected_concept_record_id = expected_concept_doi.rsplit(".", 1)[-1]
    if str(concept_record_id) != expected_concept_record_id:
        raise ZenodoPublisherError(
            f"Zenodo {operation} concept ID does not match expected concept DOI"
        )
    metadata = payload.get("metadata")
    observed_values = [payload.get("conceptdoi"), payload.get("concept_doi")]
    if isinstance(metadata, Mapping):
        observed_values.extend([metadata.get("conceptdoi"), metadata.get("concept_doi")])
    if any(value is not None and value != expected_concept_doi for value in observed_values):
        raise ZenodoPublisherError(
            f"Zenodo {operation} concept DOI does not match expected concept DOI"
        )
    return str(concept_record_id)


def _validate_successor_version_doi(
    payload: Mapping[str, Any],
    *,
    record_id: int,
    expected_predecessor_doi: str,
    expected_concept_doi: str,
    operation: str,
) -> str:
    """Validate a distinct version DOI and its record-ID binding.

    Returns:
        The validated version DOI.
    """
    version_doi = _successor_version_doi(payload)
    if not isinstance(version_doi, str) or _ZENODO_DOI_RE.fullmatch(version_doi) is None:
        raise ZenodoPublisherError(f"Zenodo {operation} omitted a valid version DOI")
    if version_doi in {expected_predecessor_doi, expected_concept_doi}:
        raise ZenodoPublisherError(f"Zenodo {operation} reused a predecessor or concept DOI")
    if version_doi.rsplit(".", 1)[-1] != str(record_id):
        raise ZenodoPublisherError(f"Zenodo {operation} version DOI does not match record ID")
    return version_doi


def _validate_unpublished_draft(payload: Mapping[str, Any], operation: str) -> None:
    """Require an API payload to describe a stable unpublished draft."""
    if payload.get("submitted") is not False or payload.get("state") != "unsubmitted":
        raise ZenodoPublisherError(f"Zenodo {operation} must remain an unpublished draft")


def _validate_successor_payload(
    payload: Mapping[str, Any],
    *,
    predecessor_deposition_id: int,
    expected_predecessor_doi: str,
    expected_concept_doi: str,
    latest_draft_id: int | None = None,
    operation: str,
) -> dict[str, Any]:
    """Validate one new-version draft/read-back identity and return public state.

    Returns:
        A normalized credential-free state object.
    """
    deposition_id = _positive_deposition_id(payload.get("id"), f"{operation} deposition ID")
    record_id = _positive_deposition_id(payload.get("record_id"), f"{operation} record ID")
    if latest_draft_id is not None and deposition_id != latest_draft_id:
        raise ZenodoPublisherError(f"Zenodo {operation} changed the latest_draft deposition ID")
    predecessor_record_id = int(expected_predecessor_doi.rsplit(".", 1)[-1])
    if predecessor_deposition_id in {deposition_id, record_id} or predecessor_record_id in {
        deposition_id,
        record_id,
    }:
        raise ZenodoPublisherError(
            f"Zenodo {operation} reused the predecessor deposition/record ID"
        )

    concept_record_id = _validate_successor_concept(
        payload, expected_concept_doi=expected_concept_doi, operation=operation
    )
    version_doi = _validate_successor_version_doi(
        payload,
        record_id=record_id,
        expected_predecessor_doi=expected_predecessor_doi,
        expected_concept_doi=expected_concept_doi,
        operation=operation,
    )
    _validate_unpublished_draft(payload, operation)

    state = _public_state(dict(payload))
    state.update(
        {
            "deposition_id": deposition_id,
            "record_id": record_id,
            "concept_record_id": str(concept_record_id),
            "doi": version_doi,
            "state": "unsubmitted",
            "submitted": False,
        }
    )
    return state


def _validate_successor_metadata_readback(
    expected_metadata: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    """Require every user-controlled field to match the requested metadata contract."""
    observed = payload.get("metadata")
    if not isinstance(observed, Mapping):
        raise ZenodoPublisherError("Zenodo new-version PUT response omitted metadata")
    for key, value in _metadata_contract(expected_metadata).items():
        expected = _canonical_metadata_value_for_comparison(key, value)
        actual = _canonical_metadata_value_for_comparison(key, observed.get(key))
        if expected != actual:
            raise ZenodoPublisherError(
                f"Zenodo new-version metadata readback mismatch at metadata.{key}"
            )


def _successor_predecessor_doi(metadata: Mapping[str, Any]) -> str | None:
    """Return a validated predecessor DOI when metadata declares a successor relation."""
    related = metadata.get("related_identifiers")
    predecessor_relations = (
        [
            item
            for item in related
            if isinstance(item, Mapping) and item.get("relation") == "isNewVersionOf"
        ]
        if isinstance(related, list)
        else []
    )
    if not predecessor_relations:
        return None
    if (
        len(predecessor_relations) != 1
        or not isinstance(predecessor_relations[0].get("identifier"), str)
        or _ZENODO_DOI_RE.fullmatch(predecessor_relations[0]["identifier"]) is None
        or predecessor_relations[0].get("scheme") != "doi"
    ):
        raise ZenodoPublisherError(
            "Zenodo successor metadata must contain exactly one isNewVersionOf predecessor DOI"
        )
    return predecessor_relations[0]["identifier"]


def _validate_new_version_relation(
    metadata: Mapping[str, Any], *, expected_predecessor_doi: str
) -> None:
    """Require exactly one DOI relation to the immutable predecessor version."""
    if _successor_predecessor_doi(metadata) != expected_predecessor_doi:
        raise ZenodoPublisherError(
            "Zenodo successor metadata must contain exactly one isNewVersionOf predecessor DOI"
        )


def _validate_predecessor_payload(
    payload: Mapping[str, Any],
    *,
    predecessor_deposition_id: int,
    expected_predecessor_doi: str,
    expected_concept_doi: str,
    expected_predecessor_source_url: str,
) -> None:
    """Bind the mutating new-version action to one exact published predecessor."""
    deposition_id = _positive_deposition_id(payload.get("id"), "predecessor deposition response ID")
    record_id = _positive_deposition_id(payload.get("record_id"), "predecessor record ID")
    if deposition_id != predecessor_deposition_id or record_id != predecessor_deposition_id:
        raise ZenodoPublisherError("Zenodo predecessor response changed the requested identity")
    if payload.get("doi") != expected_predecessor_doi:
        raise ZenodoPublisherError("Zenodo predecessor DOI does not match the requested version")
    _validate_successor_concept(
        payload,
        expected_concept_doi=expected_concept_doi,
        operation="predecessor",
    )
    if payload.get("submitted") is not True or payload.get("state") != "done":
        raise ZenodoPublisherError("Zenodo predecessor must be a published deposition")
    predecessor_metadata = payload.get("metadata")
    if not isinstance(predecessor_metadata, Mapping):
        raise ZenodoPublisherError("Zenodo predecessor metadata is malformed")
    if _source_tag(predecessor_metadata) != expected_predecessor_source_url:
        raise ZenodoPublisherError("Zenodo predecessor source tag does not match the expected tag")


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
    """Validate sealed state identity and lifecycle before any API URL use."""
    if not isinstance(state, Mapping):
        raise ZenodoPublisherError("invalid Zenodo deposition state")
    if not state.get("deposition_id"):
        raise ZenodoPublisherError("deposition state has no deposition_id")
    if state.get("schema_version") != ZENODO_STATE_SCHEMA:
        raise ZenodoPublisherError("invalid Zenodo deposition state")
    _verify_integrity(state, key="integrity", schema=ZENODO_STATE_SCHEMA)

    _positive_deposition_id(state.get("deposition_id"), "state deposition ID")
    record_id = _positive_deposition_id(state.get("record_id"), "state record ID")
    concept_record_id = _positive_decimal_id(
        state.get("concept_record_id"), "state concept record ID"
    )
    if state.get("concept_record_id") != concept_record_id:
        raise ZenodoPublisherError("Zenodo state concept record ID is not canonical")
    _validated_version_doi(state.get("doi"), record_id, "state version DOI")
    submitted = state.get("submitted")
    lifecycle = state.get("state")
    if not isinstance(submitted, bool):
        raise ZenodoPublisherError("Zenodo state submitted flag is invalid")
    if not isinstance(lifecycle, str) or lifecycle not in _KNOWN_ZENODO_STATES:
        raise ZenodoPublisherError("Zenodo state lifecycle is invalid")
    if (submitted and lifecycle != "done") or (not submitted and lifecycle == "done"):
        raise ZenodoPublisherError("Zenodo state lifecycle is inconsistent")
    if lifecycle not in _STABLE_ZENODO_STATES:
        raise ZenodoPublisherError("Zenodo state lifecycle is not an admissible operation state")


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


def _validated_upload_files(files: list[Path]) -> list[tuple[Path, int, str]]:
    """Resolve and hash a complete, unique local upload inventory before mutation.

    Returns:
        Tuples containing the resolved path, byte size, and SHA-256 digest.
    """
    if not files:
        raise ZenodoPublisherError("upload requires at least one nonempty file")
    validated: list[tuple[Path, int, str]] = []
    names: set[str] = set()
    for file_path in files:
        resolved = file_path.resolve()
        if not resolved.is_file():
            raise ZenodoPublisherError(f"upload file not found: {resolved}")
        name = resolved.name
        if name in names:
            raise ZenodoPublisherError(f"upload contains duplicate filename: {name}")
        names.add(name)
        size = resolved.stat().st_size
        if size <= 0:
            raise ZenodoPublisherError(f"upload file is empty: {name}")
        validated.append((resolved, size, _sha256_file(resolved)))
    return validated


def _draft_file_inventory(
    raw_files: Any,
    *,
    deposition_id: int,
    api_base: str,
) -> dict[str, tuple[str, str]]:
    """Validate and index a complete remote draft-file inventory.

    Returns:
        File IDs and exact deletion URLs keyed by safe, unique filename.
    """
    if not isinstance(raw_files, list):
        raise ZenodoPublisherError("Zenodo draft response omitted its file inventory")
    inventory: dict[str, tuple[str, str]] = {}
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_files):
        if not isinstance(item, Mapping):
            raise ZenodoPublisherError(
                f"Zenodo draft file inventory contains a malformed entry at index {index}"
            )
        name = item.get("filename")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            raise ZenodoPublisherError(
                f"Zenodo draft file inventory contains an unsafe filename at index {index}"
            )
        if name in inventory:
            raise ZenodoPublisherError(f"Zenodo draft contains duplicate file {name}")
        file_id = item.get("id")
        if (
            not isinstance(file_id, str)
            or re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]{0,127}", file_id) is None
        ):
            raise ZenodoPublisherError(f"Zenodo draft file {name} has an invalid file ID")
        if file_id in seen_ids:
            raise ZenodoPublisherError(f"Zenodo draft contains duplicate file ID {file_id}")
        seen_ids.add(file_id)
        links = item.get("links")
        self_url = links.get("self") if isinstance(links, Mapping) else None
        validated_url = _validated_remote_url(
            self_url,
            api_base,
            f"draft file {name} identity",
        )
        expected_url = (
            f"{api_base}/deposit/depositions/{deposition_id}/files/{quote(file_id, safe='')}"
        )
        if validated_url != expected_url:
            raise ZenodoPublisherError(
                f"Zenodo draft file {name} URL does not match its deposition"
            )
        inventory[name] = (file_id, validated_url)
    return inventory


def _validate_successor_cleanup_state(state: Mapping[str, Any]) -> None:
    """Require new-version provenance before removing inherited draft files."""
    predecessor_id = _positive_deposition_id(
        state.get("predecessor_deposition_id"),
        "state predecessor deposition ID",
    )
    predecessor_doi = _validated_version_doi(
        state.get("predecessor_doi"),
        predecessor_id,
        "state predecessor DOI",
    )
    if predecessor_id in {state.get("deposition_id"), state.get("record_id")}:
        raise ZenodoPublisherError("Zenodo successor state reuses its predecessor identity")
    predecessor = state.get("predecessor")
    if not isinstance(predecessor, Mapping) or dict(predecessor) != {
        "deposition_id": predecessor_id,
        "doi": predecessor_doi,
    }:
        raise ZenodoPublisherError("Zenodo successor state predecessor binding is invalid")
    concept_doi = state.get("concept_doi")
    expected_concept_doi = f"10.5281/zenodo.{state.get('concept_record_id')}"
    if concept_doi != expected_concept_doi or concept_doi == predecessor_doi:
        raise ZenodoPublisherError("Zenodo successor state concept DOI binding is invalid")


def _list_draft_files(
    session: _Session,
    *,
    deposition_id: int,
    api_base: str,
    operation: str,
) -> dict[str, tuple[str, str]]:
    """Fetch and validate the current file inventory of one exact draft.

    Returns:
        File IDs and exact deletion URLs keyed by safe, unique filename.
    """
    response = session.get(
        f"{api_base}/deposit/depositions/{deposition_id}/files",
        timeout=60,
        allow_redirects=False,
    )
    try:
        if response.status_code >= 300:
            raise RuntimeError("HTTP redirect or failure")
        response.raise_for_status()
        payload = response.json()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ZenodoPublisherError(f"Zenodo {operation} request failed") from exc
    return _draft_file_inventory(
        payload,
        deposition_id=deposition_id,
        api_base=api_base,
    )


def _delete_draft_extra(session: _Session, *, name: str, url: str) -> None:
    """Delete one prevalidated extra from an unpublished draft."""
    response = session.delete(url, timeout=60, allow_redirects=False)
    try:
        if response.status_code != 204:
            raise RuntimeError("unexpected delete response")
        response.raise_for_status()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ZenodoPublisherError(f"Zenodo delete extra draft file {name} request failed") from exc


def _admit_successor_cleanup(
    state: Mapping[str, Any],
    *,
    initial_inventory: Mapping[str, tuple[str, str]],
    expected_names: set[str],
) -> set[str]:
    """Validate successor provenance when the draft contains extra files.

    Returns:
        The names of stable pre-existing files eligible for later cleanup.
    """
    extra_names = set(initial_inventory) - expected_names
    if extra_names:
        _validate_successor_cleanup_state(state)
    return extra_names


def _reconcile_inherited_draft_files(
    session: _Session,
    *,
    initial_inventory: Mapping[str, tuple[str, str]],
    expected_names: set[str],
    extra_names: set[str],
    deposition_id: int,
    api_base: str,
) -> None:
    """Remove only stable, pre-existing extras from a proven successor draft."""
    post_upload_inventory = _list_draft_files(
        session,
        deposition_id=deposition_id,
        api_base=api_base,
        operation="post-upload draft file inventory",
    )
    if set(post_upload_inventory) != expected_names | extra_names:
        raise ZenodoPublisherError(
            "Zenodo draft file inventory changed unexpectedly after upload; refusing deletion"
        )
    if not extra_names:
        return
    if any(post_upload_inventory[name] != initial_inventory[name] for name in sorted(extra_names)):
        raise ZenodoPublisherError(
            "Zenodo inherited draft file identity changed after upload; refusing deletion"
        )
    for name in sorted(extra_names):
        _delete_draft_extra(
            session,
            name=name,
            url=post_upload_inventory[name][1],
        )
    final_inventory = _list_draft_files(
        session,
        deposition_id=deposition_id,
        api_base=api_base,
        operation="post-delete draft file inventory",
    )
    if set(final_inventory) != expected_names:
        raise ZenodoPublisherError(
            "Zenodo draft file inventory does not match intended upload after cleanup"
        )


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
        if response.status_code >= 300:
            raise RuntimeError("HTTP redirect or failure")
        response.raise_for_status()
        payload = response.json()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ZenodoPublisherError(f"Zenodo {operation} request failed") from exc
    if not isinstance(payload, dict):
        raise ZenodoPublisherError(f"Zenodo {operation} response was not a JSON object")
    return payload


def _remote_optimistic_binding(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """Return a non-secret binding for a remote optimistic version field.

    Zenodo depositions expose ``modified`` as a last-change timestamp. Some
    controlled mirrors may expose a similarly useful ``version`` or
    ``revision`` field instead. Persist only a digest of the value so a
    malicious server cannot cause a credential-shaped string to be echoed in a
    receipt.

    Returns:
        A field name and SHA-256 digest, or ``None`` when no known field exists.
    """
    for field in _REMOTE_VERSION_FIELDS:
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value):
            raise ZenodoPublisherError(f"Zenodo remote {field} optimistic version is invalid")
        digest = hashlib.sha256(_canonical_bytes({"field": field, "value": value})).hexdigest()
        return {"field": field, "sha256": digest}
    return None


def _receipt_contract(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return receipt fields that bind one verified remote state."""
    return {key: value for key, value in receipt.items() if key != "integrity"}


def _public_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only a validated, credential-free deposition identity.

    Zenodo response fields are untrusted input. Identity and lifecycle values
    are normalized here before they can be interpolated into later requests or
    sealed into a state/verification receipt.

    Returns:
        A validated credential-free state object.
    """
    if not isinstance(payload, Mapping):
        raise ZenodoPublisherError("Zenodo deposition response is not a JSON object")
    deposition_id = _positive_deposition_id(payload.get("id"), "deposition response ID")
    record_id = _positive_deposition_id(payload.get("record_id"), "record response ID")
    concept_record_id = _positive_decimal_id(
        payload.get("conceptrecid"), "concept record response ID"
    )
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    preregistered = (
        metadata.get("prereserve_doi")
        if isinstance(metadata.get("prereserve_doi"), Mapping)
        else {}
    )
    direct_doi = payload.get("doi")
    version_doi = (
        direct_doi if direct_doi is not None and direct_doi != "" else preregistered.get("doi")
    )
    version_doi = _validated_version_doi(version_doi, record_id, "deposition version DOI")
    submitted = payload.get("submitted")
    state = payload.get("state")
    if not isinstance(submitted, bool):
        raise ZenodoPublisherError("Zenodo deposition response submitted state is invalid")
    if not isinstance(state, str) or state not in _KNOWN_ZENODO_STATES:
        raise ZenodoPublisherError("Zenodo deposition response state is invalid")
    if (submitted and state != "done") or (not submitted and state == "done"):
        raise ZenodoPublisherError("Zenodo deposition response state is inconsistent")
    return {
        "schema_version": ZENODO_STATE_SCHEMA,
        "deposition_id": deposition_id,
        "record_id": record_id,
        "concept_record_id": concept_record_id,
        "doi": version_doi,
        "state": state,
        "submitted": submitted,
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
    validated_base = _validated_api_base(api_base)
    normalized_metadata = _validate_metadata(metadata)
    binding = _normalize_release_binding(release_binding) if release_binding is not None else None
    file_metadata = (
        _validate_release_binding_metadata(normalized_metadata, binding)
        if binding is not None
        else None
    )
    normalized_metadata["prereserve_doi"] = True
    response = session.post(
        f"{validated_base}/deposit/depositions",
        json={"metadata": normalized_metadata},
        timeout=60,
        allow_redirects=False,
    )
    payload = _json_object(response, "reserve")
    state = _public_state(payload)
    _validate_unpublished_draft(state, "reserve response")
    if binding is not None:
        _assert_deposition_identity(state, binding)
        state["release_binding"] = _state_release_binding(
            binding,
            metadata_contract_sha256=_metadata_sha256(file_metadata or normalized_metadata),
        )
    return _seal_state(state)


def _validate_new_version_tag_lineage(
    *, predecessor_tag: str, source_sha: str, successor_tag: str
) -> tuple[str, str]:
    """Validate the complete local tag lineage.

    Returns:
        The canonical predecessor and successor GitHub release URLs.
    """
    predecessor_tag_problems = check_canonical_source_tag(predecessor_tag, source_sha)
    if predecessor_tag_problems or re.search(r"-erratum\.[1-9][0-9]*$", predecessor_tag):
        raise ZenodoPublisherError(
            "expected predecessor tag must end in the exact lowercase scientific source SHA"
        )
    if successor_tag != f"{predecessor_tag}-erratum.1":
        raise ZenodoPublisherError(
            "expected successor tag must be the exact predecessor tag plus -erratum.1"
        )
    if check_canonical_source_tag(successor_tag, source_sha):
        raise ZenodoPublisherError("expected successor tag has invalid source-SHA lineage")
    return (
        _expected_source_tag_url(predecessor_tag, label="expected predecessor source tag"),
        _expected_source_tag_url(successor_tag, label="expected successor source tag"),
    )


def new_version(  # noqa: C901, PLR0913
    session: _Session,
    metadata: Mapping[str, Any],
    *,
    predecessor_deposition_id: int,
    expected_predecessor_doi: str,
    expected_concept_doi: str,
    expected_predecessor_tag: str,
    expected_source_sha: str,
    expected_successor_tag: str,
    api_base: str = ZENODO_API_BASE,
    release_binding: Any | None = None,
) -> dict[str, Any]:
    """Create a fail-closed successor draft for one published Zenodo version.

    ``release_binding`` is optional only for the initial successor reservation:
    Zenodo assigns the successor version DOI as part of the ``newversion``
    action, so that DOI cannot be included in a binding beforehand.  In that
    explicit pre-reservation mode the returned state is unbound and must be
    passed through a manifest-bound ``upload``, ``verify``, or ``publish``
    operation after the returned DOI has been frozen into the release
    manifest.  When a binding is supplied, the returned draft identity is
    checked before any metadata mutation.

    The legacy deposit API creates the successor through the predecessor's
    ``actions/newversion`` endpoint. The returned ``links.latest_draft`` link
    is treated as untrusted until it is proven to be an HTTPS URL on the same
    API, and the draft metadata is replaced and checked through a PUT before
    any state is sealed.

    Returns:
        Credential-free sealed state for the unpublished successor draft.
    """
    validated_base = _validated_api_base(api_base)
    predecessor_id = _positive_deposition_id(predecessor_deposition_id, "predecessor deposition ID")
    expected_predecessor_source_url, expected_source_url = _validate_new_version_tag_lineage(
        predecessor_tag=expected_predecessor_tag,
        source_sha=expected_source_sha,
        successor_tag=expected_successor_tag,
    )
    if (
        not isinstance(expected_predecessor_doi, str)
        or _ZENODO_DOI_RE.fullmatch(expected_predecessor_doi) is None
    ):
        raise ZenodoPublisherError("expected predecessor DOI is invalid")
    if (
        not isinstance(expected_concept_doi, str)
        or _ZENODO_DOI_RE.fullmatch(expected_concept_doi) is None
    ):
        raise ZenodoPublisherError("expected concept DOI is invalid")
    if expected_predecessor_doi == expected_concept_doi:
        raise ZenodoPublisherError("expected predecessor and concept DOIs must differ")
    if predecessor_id != int(expected_predecessor_doi.rsplit(".", 1)[-1]):
        raise ZenodoPublisherError(
            "predecessor deposition ID does not match the expected predecessor DOI"
        )

    normalized_metadata = _validate_metadata(metadata)
    if _source_tag(normalized_metadata, require_url_scheme=True) != expected_source_url:
        raise ZenodoPublisherError(
            "new-version metadata source tag does not match the expected successor tag"
        )
    _validate_new_version_relation(
        normalized_metadata,
        expected_predecessor_doi=expected_predecessor_doi,
    )
    preregistered = normalized_metadata.get("prereserve_doi")
    if preregistered is not None and preregistered is not True:
        raise ZenodoPublisherError(
            "new-version metadata must not carry an inherited prereserved DOI"
        )
    binding = _normalize_release_binding(release_binding) if release_binding is not None else None
    file_metadata = (
        _validate_release_binding_metadata(normalized_metadata, binding)
        if binding is not None
        else None
    )
    base = validated_base.rstrip("/")

    predecessor = _json_object(
        session.get(
            f"{base}/deposit/depositions/{predecessor_id}",
            timeout=60,
            allow_redirects=False,
        ),
        "retrieve predecessor",
    )
    _validate_predecessor_payload(
        predecessor,
        predecessor_deposition_id=predecessor_id,
        expected_predecessor_doi=expected_predecessor_doi,
        expected_concept_doi=expected_concept_doi,
        expected_predecessor_source_url=expected_predecessor_source_url,
    )

    created = _json_object(
        session.post(
            f"{base}/deposit/depositions/{predecessor_id}/actions/newversion",
            timeout=60,
            allow_redirects=False,
        ),
        "new-version",
    )
    links = created.get("links")
    latest_draft = links.get("latest_draft") if isinstance(links, Mapping) else None
    latest_draft_url, latest_draft_id = _validated_latest_draft_link(latest_draft, validated_base)
    draft = _json_object(
        session.get(latest_draft_url, timeout=60, allow_redirects=False),
        "new-version draft",
    )
    state = _validate_successor_payload(
        draft,
        predecessor_deposition_id=predecessor_id,
        expected_predecessor_doi=expected_predecessor_doi,
        expected_concept_doi=expected_concept_doi,
        latest_draft_id=latest_draft_id,
        operation="new-version draft",
    )
    if binding is not None:
        # This must remain immediately after the generic successor validation:
        # a server-returned DOI that is valid in isolation but differs from the
        # reviewed manifest must not reach the metadata PUT below.
        _assert_deposition_identity(state, binding)

    successor_metadata = _metadata_contract(normalized_metadata)
    updated = _json_object(
        session.put(
            f"{base}/deposit/depositions/{state['deposition_id']}",
            json={"metadata": successor_metadata},
            timeout=60,
            allow_redirects=False,
        ),
        "new-version metadata update",
    )
    updated_state = _validate_successor_payload(
        updated,
        predecessor_deposition_id=predecessor_id,
        expected_predecessor_doi=expected_predecessor_doi,
        expected_concept_doi=expected_concept_doi,
        latest_draft_id=state["deposition_id"],
        operation="new-version metadata readback",
    )
    for key in ("deposition_id", "record_id", "concept_record_id", "doi"):
        if updated_state.get(key) != state.get(key):
            raise ZenodoPublisherError(f"Zenodo new-version metadata readback changed {key}")
    _validate_successor_metadata_readback(successor_metadata, updated)

    updated_state.update(
        {
            "concept_doi": expected_concept_doi,
            "predecessor_deposition_id": predecessor_id,
            "predecessor_doi": expected_predecessor_doi,
            "source_tag": expected_source_url,
            "predecessor": {
                "deposition_id": predecessor_id,
                "doi": expected_predecessor_doi,
            },
        }
    )
    if binding is not None:
        _assert_deposition_identity(updated_state, binding)
        updated_state["release_binding"] = _state_release_binding(
            binding,
            metadata_contract_sha256=_metadata_sha256(file_metadata or normalized_metadata),
        )
    return _seal_state(updated_state)


def _restore_recovered_successor_lineage(
    state: dict[str, Any],
    metadata: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    """Restore cleanup provenance when recovered metadata declares a successor."""
    predecessor_doi = _successor_predecessor_doi(metadata)
    if predecessor_doi is None:
        return
    predecessor_id = int(predecessor_doi.rsplit(".", 1)[-1])
    predecessor_doi = _validated_version_doi(
        predecessor_doi,
        predecessor_id,
        "recovered predecessor DOI",
    )
    source_tag = _source_tag(metadata, require_url_scheme=True)
    if source_tag != binding["source_tag"]:
        raise ZenodoPublisherError(
            "Zenodo recovered successor source tag does not match release binding"
        )
    state.update(
        {
            "concept_doi": binding["concept_doi"],
            "predecessor_deposition_id": predecessor_id,
            "predecessor_doi": predecessor_doi,
            "predecessor": {
                "deposition_id": predecessor_id,
                "doi": predecessor_doi,
            },
            "source_tag": source_tag,
        }
    )
    _validate_successor_cleanup_state(state)


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
    binding, and reconstructs credential-free state. When the bound metadata
    identifies a successor, the predecessor lineage required for safe inherited
    file reconciliation is restored as well.

    Returns:
        Credential-free sealed deposition state.
    """
    validated_base = _validated_api_base(api_base)
    if isinstance(deposition_id, bool) or not isinstance(deposition_id, int) or deposition_id <= 0:
        raise ZenodoPublisherError("Zenodo recovery deposition ID must be a positive integer")
    normalized_metadata = _validate_metadata(metadata)
    binding = _normalize_release_binding(release_binding)
    file_metadata = _validate_release_binding_metadata(normalized_metadata, binding)
    payload = _json_object(
        session.get(
            f"{validated_base}/deposit/depositions/{deposition_id}",
            timeout=60,
            allow_redirects=False,
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

    _restore_recovered_successor_lineage(state, normalized_metadata, binding)

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
    working_state = deepcopy(state)
    validated_base = _validated_api_base(api_base)
    _validate_state_for_operation(working_state)
    binding = _normalize_release_binding(release_binding) if release_binding is not None else None
    binding_metadata = _validate_release_binding_file(binding) if binding is not None else None
    _validate_state_binding(
        working_state,
        binding,
        metadata_contract_sha256=(
            _metadata_sha256(binding_metadata) if binding_metadata is not None else None
        ),
    )
    local_files = _validated_upload_files(files)
    deposition_id = _positive_deposition_id(
        working_state.get("deposition_id"),
        "state deposition ID",
    )
    deposition = _json_object(
        session.get(
            f"{validated_base}/deposit/depositions/{deposition_id}",
            timeout=60,
            allow_redirects=False,
        ),
        "retrieve draft",
    )
    remote_state = _public_state(deposition)
    for key in ("deposition_id", "record_id", "concept_record_id", "doi"):
        if remote_state.get(key) != working_state.get(key):
            raise ZenodoPublisherError("Zenodo draft response changed reserved identity")
    _validate_unpublished_draft(remote_state, "upload response")
    links = deposition.get("links")
    bucket = links.get("bucket") if isinstance(links, dict) else None
    try:
        bucket = _validated_remote_url(bucket, validated_base, "draft upload bucket")
    except ZenodoPublisherError as exc:
        raise ZenodoPublisherError(
            "Zenodo draft response omitted a secure upload bucket (invalid Zenodo URL)"
        ) from exc
    if not isinstance(bucket, str):  # pragma: no cover - narrowed by _validated_remote_url
        raise ZenodoPublisherError(
            "Zenodo draft response omitted a secure upload bucket (invalid Zenodo URL)"
        )
    expected_names = {resolved.name for resolved, _, _ in local_files}
    initial_inventory = _draft_file_inventory(
        deposition.get("files"),
        deposition_id=deposition_id,
        api_base=validated_base,
    )
    extra_names = _admit_successor_cleanup(
        working_state,
        initial_inventory=initial_inventory,
        expected_names=expected_names,
    )
    uploaded: list[dict[str, Any]] = []
    for resolved, size, sha256 in local_files:
        with resolved.open("rb") as stream:
            response = session.put(
                f"{bucket.rstrip('/')}/{quote(resolved.name)}",
                data=stream,
                timeout=3600,
                allow_redirects=False,
            )
        _json_object(response, f"upload {resolved.name}")
        if resolved.stat().st_size != size or _sha256_file(resolved) != sha256:
            raise ZenodoPublisherError(f"upload file changed during transfer: {resolved.name}")
        uploaded.append(
            {
                "name": resolved.name,
                "size": size,
                "sha256": sha256,
            }
        )
    _reconcile_inherited_draft_files(
        session,
        initial_inventory=initial_inventory,
        expected_names=expected_names,
        extra_names=extra_names,
        deposition_id=deposition_id,
        api_base=validated_base,
    )
    updated = dict(working_state)
    updated["files"] = uploaded
    updated.pop("verification_receipt", None)
    return _seal_state(updated)


def publish(  # noqa: C901, PLR0912
    session: _Session,
    state: dict[str, Any],
    metadata: Mapping[str, Any] | None = None,
    *,
    api_base: str = ZENODO_API_BASE,
    release_binding: Any | None = None,
) -> dict[str, Any]:
    """Irreversibly publish a deposition admitted by a draft verification receipt.

    The legacy Zenodo publish endpoint has no documented conditional
    compare-and-publish precondition. Fresh verification is therefore required
    immediately before the publish request, but cannot close the final
    time-of-check/time-of-use window; callers must run :func:`verify` again
    after publication before treating the public record as accepted.

    Returns:
        Updated published deposition state.
    """
    working_state = deepcopy(state)
    validated_base = _validated_api_base(api_base)
    _validate_state_for_operation(working_state)
    deposition_id = working_state.get("deposition_id")
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
        working_state,
        binding,
        metadata_contract_sha256=(
            _metadata_sha256(file_metadata or normalized_metadata)
            if binding is not None
            else _metadata_sha256(normalized_metadata)
        ),
    )
    if bool(working_state.get("submitted")):
        raise ZenodoPublisherError("Zenodo deposition is already published")
    expected_files, file_problems = _file_inventory(working_state)
    if file_problems:
        raise ZenodoPublisherError("cannot publish: " + "; ".join(file_problems))
    receipt = working_state.get("verification_receipt")
    if not isinstance(receipt, Mapping):
        raise ZenodoPublisherError("publish requires a prior verification receipt")
    _verify_integrity(receipt, key="integrity", schema=ZENODO_VERIFICATION_SCHEMA)
    if receipt.get("status") != "pass" or receipt.get("publication_state") != "draft":
        raise ZenodoPublisherError("publish requires a passing draft verification receipt")
    if receipt.get("deposition_id") != working_state.get("deposition_id"):
        raise ZenodoPublisherError("verification receipt deposition identity does not match state")
    if receipt.get("metadata_sha256") != _metadata_sha256(normalized_metadata):
        raise ZenodoPublisherError("verification receipt metadata does not match expected metadata")
    if receipt.get("source_tag") != _source_tag(normalized_metadata):
        raise ZenodoPublisherError("verification receipt source tag does not match metadata")
    if working_state.get("release_binding") is not None and receipt.get(
        "release_binding"
    ) != working_state.get("release_binding"):
        raise ZenodoPublisherError("verification receipt release binding does not match state")
    if working_state.get("release_binding") is not None and receipt.get(
        "manifest_metadata_sha256"
    ) != working_state["release_binding"].get("metadata_sha256"):
        raise ZenodoPublisherError("verification receipt metadata checksum does not match state")
    receipt_files = receipt.get("files")
    expected_receipt_files = [
        {"name": name, "size": item["size"], "sha256": item["sha256"]}
        for name, item in sorted(expected_files.items())
    ]
    if receipt_files != expected_receipt_files:
        raise ZenodoPublisherError("verification receipt file inventory does not match state")
    # Keep every validation and receipt mutation on private copies. The caller's
    # state remains byte-identical, including when publication admission fails.
    verification_state = deepcopy(working_state)
    fresh_report = verify(
        session,
        verification_state,
        normalized_metadata,
        api_base=validated_base,
        release_binding=binding,
    )
    if fresh_report.get("status") != "pass":
        problems = fresh_report.get("problems")
        detail = (
            "; ".join(str(problem) for problem in problems)
            if isinstance(problems, list)
            else "unknown drift"
        )
        raise ZenodoPublisherError(f"publish fresh draft verification failed: {detail}")
    fresh_receipt = fresh_report.get("receipt")
    if not isinstance(fresh_receipt, Mapping):
        raise ZenodoPublisherError("publish fresh draft verification omitted a receipt")
    _verify_integrity(fresh_receipt, key="integrity", schema=ZENODO_VERIFICATION_SCHEMA)
    if _receipt_contract(fresh_receipt) != _receipt_contract(receipt):
        raise ZenodoPublisherError("publish remote draft changed since verification receipt")
    receipt = dict(fresh_receipt)
    response = session.post(
        f"{validated_base}/deposit/depositions/{deposition_id}/actions/publish",
        timeout=120,
        allow_redirects=False,
    )
    published = _public_state(_json_object(response, "publish"))
    if not published["submitted"]:
        raise ZenodoPublisherError("Zenodo publish response did not mark the deposition submitted")
    for key in ("deposition_id", "record_id", "concept_record_id", "doi"):
        if published.get(key) != working_state.get(key):
            raise ZenodoPublisherError(f"Zenodo publish response changed {key}")
    published["files"] = list(working_state["files"])
    if working_state.get("release_binding") is not None:
        published["release_binding"] = dict(working_state["release_binding"])
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
        A machine-readable verification report. Passing verification seals a
        receipt into ``state`` for publication admission; failures leave the
        caller's state unchanged.
    """
    working_state = deepcopy(state)
    validated_base = _validated_api_base(api_base)
    _validate_state_for_operation(working_state)
    normalized_metadata = _validate_metadata(metadata)
    binding = _normalize_release_binding(release_binding) if release_binding is not None else None
    file_metadata = (
        _validate_release_binding_metadata(normalized_metadata, binding)
        if binding is not None
        else None
    )
    _validate_state_binding(
        working_state,
        binding,
        metadata_contract_sha256=(
            _metadata_sha256(file_metadata or normalized_metadata)
            if binding is not None
            else _metadata_sha256(normalized_metadata)
        ),
    )
    deposition_id = working_state.get("deposition_id")
    remote = _json_object(
        session.get(
            f"{validated_base}/deposit/depositions/{deposition_id}",
            timeout=60,
            allow_redirects=False,
        ),
        "verify",
    )
    remote_state = _public_state(remote)
    remote_metadata = remote.get("metadata") if isinstance(remote.get("metadata"), Mapping) else {}
    problems: list[str] = []
    try:
        remote_optimistic = _remote_optimistic_binding(remote)
    except ZenodoPublisherError as exc:
        problems.append(str(exc))
        remote_optimistic = None

    if remote_state["state"] not in _STABLE_ZENODO_STATES:
        raise ZenodoPublisherError("Zenodo verify response has an unsupported lifecycle state")
    for key in ("deposition_id", "record_id", "concept_record_id", "doi"):
        if remote_state.get(key) != working_state.get(key):
            problems.append(f"{key} does not match reserved state")
    expected_submitted = working_state.get("submitted")
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

    expected_files, file_problems = _file_inventory(working_state)
    problems.extend(file_problems)
    file_inventory_source = remote
    if remote_submitted is True:
        record_id = working_state.get("record_id")
        public_record = _json_object(
            session.get(
                f"{validated_base}/records/{record_id}",
                timeout=60,
                allow_redirects=False,
            ),
            "verify published record",
        )
        if public_record.get("id") != record_id:
            problems.append("published record id does not match reserved state")
        if str(public_record.get("conceptrecid") or "") != str(
            working_state.get("concept_record_id") or ""
        ):
            problems.append("published record concept id does not match reserved state")
        if public_record.get("doi") != working_state.get("doi"):
            problems.append("published record DOI does not match reserved state")
        if public_record.get("status") != "published":
            problems.append("published record status is not published")
        file_inventory_source = public_record

    remote_files_value = file_inventory_source.get("files")
    remote_files = remote_files_value if isinstance(remote_files_value, list) else []
    if not remote_files:
        problems.append("remote file inventory is empty")
    remote_by_name: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(remote_files):
        if not isinstance(item, Mapping):
            problems.append("remote file inventory contains a malformed entry")
            continue
        name = item.get("filename") or item.get("key")
        if not isinstance(name, str) or not name:
            problems.append(f"remote file inventory contains an unnamed entry at index {index}")
            continue
        if name in remote_by_name:
            problems.append(f"remote file inventory contains a duplicate entry at index {index}")
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
        try:
            download_url = _validated_remote_url(
                download_url,
                validated_base,
                f"remote file {name} download",
            )
        except ZenodoPublisherError:
            problems.append(f"remote file {name} has no secure download URL (download failed)")
            continue
        response = session.get(
            download_url,
            stream=True,
            timeout=3600,
            allow_redirects=False,
        )
        if response.status_code >= 300:
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
                "deposition_id": working_state["deposition_id"],
                "record_id": working_state.get("record_id"),
                "concept_record_id": working_state.get("concept_record_id"),
                "doi": working_state.get("doi"),
                "metadata_sha256": _metadata_sha256(normalized_metadata),
                "source_tag": source_tag,
                "files": receipt_files,
                **(
                    {"remote_optimistic": remote_optimistic}
                    if remote_optimistic is not None
                    else {}
                ),
                **(
                    {
                        "release_binding": dict(working_state["release_binding"]),
                        "manifest_metadata_sha256": working_state["release_binding"][
                            "metadata_sha256"
                        ],
                    }
                    if working_state.get("release_binding") is not None
                    else {}
                ),
            },
            "integrity",
            ZENODO_VERIFICATION_SCHEMA,
        )
        report["receipt"] = receipt
        updated_state = dict(working_state)
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
    "new_version",
    "publish",
    "read_token_file",
    "recover",
    "reserve",
    "upload",
    "verify",
    "write_state",
]

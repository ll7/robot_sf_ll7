"""Direct, credential-safe Zenodo publisher for benchmark-data releases."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from robot_sf.common.optional_import import try_import

ZENODO_API_BASE = "https://zenodo.org/api"
ZENODO_STATE_SCHEMA = "robot-sf-zenodo-deposition.v1"


class ZenodoPublisherError(RuntimeError):
    """Raised for a rejected local contract or Zenodo API response."""


class _Response(Protocol):
    """Small response protocol used by the publisher and mocked tests."""

    status_code: int
    content: bytes

    def json(self) -> Any:
        """Return the decoded response body."""

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


def load_dataset_metadata(path: str | Path) -> dict[str, Any]:
    """Load and validate benchmark-dataset deposition metadata.

    Returns:
        Metadata suitable for the Zenodo deposition API.
    """
    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise ZenodoPublisherError("metadata file must contain a top-level metadata object")
    if metadata.get("upload_type") != "dataset":
        raise ZenodoPublisherError("Zenodo benchmark publication must use upload_type=dataset")
    if metadata.get("license") != "GPL-3.0-only":
        raise ZenodoPublisherError("Zenodo benchmark publication license must be GPL-3.0-only")
    creators = metadata.get("creators")
    if not isinstance(creators, list) or not creators:
        raise ZenodoPublisherError("Zenodo metadata must name at least one creator")
    related = metadata.get("related_identifiers")
    if not isinstance(related, list) or not any(
        isinstance(item, dict)
        and item.get("relation") == "isSupplementTo"
        and "/releases/tag/" in str(item.get("identifier", ""))
        for item in related
    ):
        raise ZenodoPublisherError(
            "Zenodo metadata must relate the dataset to the exact source tag"
        )
    metadata["prereserve_doi"] = True
    return metadata


def _json_object(response: _Response, operation: str) -> dict[str, Any]:
    """Require a successful JSON-object API response.

    Returns:
        Parsed response object.
    """
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # requests raises provider-specific exceptions
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
) -> dict[str, Any]:
    """Create a fresh deposition and reserve its version DOI.

    Returns:
        Credential-free deposition state.
    """
    response = session.post(
        f"{api_base.rstrip('/')}/deposit/depositions",
        json={"metadata": metadata},
        timeout=60,
    )
    payload = _json_object(response, "reserve")
    state = _public_state(payload)
    if not state["deposition_id"] or not state["doi"] or not state["concept_record_id"]:
        raise ZenodoPublisherError(
            "Zenodo reserve response omitted deposition/concept/DOI identity"
        )
    return state


def upload(
    session: _Session,
    state: dict[str, Any],
    files: list[Path],
    *,
    api_base: str = ZENODO_API_BASE,
) -> dict[str, Any]:
    """Upload files to an unpublished deposition and record local SHA-256 values.

    Returns:
        Updated credential-free deposition state.
    """
    deposition_id = state.get("deposition_id")
    if not deposition_id:
        raise ZenodoPublisherError("deposition state has no deposition_id")
    deposition = _json_object(
        session.get(f"{api_base.rstrip('/')}/deposit/depositions/{deposition_id}", timeout=60),
        "retrieve draft",
    )
    links = deposition.get("links")
    bucket = links.get("bucket") if isinstance(links, dict) else None
    if not isinstance(bucket, str) or not bucket.startswith("https://"):
        raise ZenodoPublisherError("Zenodo draft response omitted a secure upload bucket")
    uploaded: list[dict[str, Any]] = []
    for file_path in files:
        resolved = file_path.resolve()
        if not resolved.is_file():
            raise ZenodoPublisherError(f"upload file not found: {resolved}")
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
                "size": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
                "zenodo_checksum": remote.get("checksum"),
            }
        )
    updated = dict(state)
    updated["files"] = uploaded
    return updated


def publish(
    session: _Session,
    state: dict[str, Any],
    *,
    api_base: str = ZENODO_API_BASE,
) -> dict[str, Any]:
    """Irreversibly publish a prepared Zenodo deposition.

    Returns:
        Updated published deposition state.
    """
    deposition_id = state.get("deposition_id")
    if not deposition_id:
        raise ZenodoPublisherError("deposition state has no deposition_id")
    response = session.post(
        f"{api_base.rstrip('/')}/deposit/depositions/{deposition_id}/actions/publish",
        timeout=120,
    )
    published = _public_state(_json_object(response, "publish"))
    published["files"] = list(state.get("files", []))
    if not published["submitted"]:
        raise ZenodoPublisherError("Zenodo publish response did not mark the deposition submitted")
    return published


def verify(  # noqa: C901
    session: _Session,
    state: dict[str, Any],
    metadata: dict[str, Any],
    *,
    api_base: str = ZENODO_API_BASE,
) -> dict[str, Any]:
    """Verify DOI, concept, metadata type/license/source tag, and uploaded file inventory.

    Returns:
        A machine-readable verification report.
    """
    deposition_id = state.get("deposition_id")
    if not deposition_id:
        raise ZenodoPublisherError("deposition state has no deposition_id")
    remote = _json_object(
        session.get(f"{api_base.rstrip('/')}/deposit/depositions/{deposition_id}", timeout=60),
        "verify",
    )
    remote_state = _public_state(remote)
    remote_metadata = remote.get("metadata") if isinstance(remote.get("metadata"), dict) else {}
    problems: list[str] = []
    for key in ("doi", "concept_record_id"):
        if remote_state.get(key) != state.get(key):
            problems.append(f"{key} does not match reserved state")
    for key in ("title", "upload_type", "license", "creators", "related_identifiers"):
        if remote_metadata.get(key) != metadata.get(key):
            problems.append(f"metadata.{key} does not match requested metadata")
    remote_files = remote.get("files") if isinstance(remote.get("files"), list) else []
    expected_files = {str(item.get("name")): item for item in state.get("files", [])}
    remote_by_name = {
        str(item.get("filename") or item.get("key")): item
        for item in remote_files
        if isinstance(item, dict)
    }
    observed_names = set(remote_by_name)
    if set(expected_files) != observed_names:
        problems.append("remote file inventory does not match uploaded file inventory")
    for name, expected in expected_files.items():
        remote_file = remote_by_name.get(name)
        if not isinstance(remote_file, dict):
            continue
        links = remote_file.get("links")
        download_url = links.get("download") if isinstance(links, dict) else None
        if not isinstance(download_url, str) or not download_url.startswith("https://"):
            problems.append(f"remote file {name} has no secure download URL")
            continue
        response = session.get(download_url, timeout=3600)
        if response.status_code >= 400:
            problems.append(f"remote file {name} download failed")
            continue
        observed_sha = hashlib.sha256(response.content).hexdigest()
        if observed_sha != expected.get("sha256"):
            problems.append(f"remote file {name} SHA-256 does not match uploaded bytes")
    return {
        "schema_version": "robot-sf-zenodo-verification.v1",
        "status": "pass" if not problems else "fail",
        "problem_count": len(problems),
        "problems": problems,
        "doi": remote_state.get("doi"),
        "concept_record_id": remote_state.get("concept_record_id"),
        "submitted": remote_state.get("submitted"),
        "file_count": len(remote_files),
    }


def load_state(path: str | Path) -> dict[str, Any]:
    """Load a credential-free publisher state file.

    Returns:
        Parsed state object.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != ZENODO_STATE_SCHEMA:
        raise ZenodoPublisherError("invalid Zenodo deposition state file")
    return payload


def write_state(path: str | Path, state: dict[str, Any]) -> None:
    """Persist credential-free publisher state without overwriting an existing identity."""
    output = Path(path)
    if output.exists():
        existing = load_state(output)
        if existing.get("deposition_id") != state.get("deposition_id"):
            raise ZenodoPublisherError("refusing to overwrite a different Zenodo deposition state")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output, 0o600)


__all__ = [
    "ZENODO_API_BASE",
    "ZENODO_STATE_SCHEMA",
    "ZenodoPublisherError",
    "build_session",
    "load_dataset_metadata",
    "load_state",
    "publish",
    "read_token_file",
    "reserve",
    "upload",
    "verify",
    "write_state",
]

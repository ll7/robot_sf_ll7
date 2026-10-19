"""Versioned receipt stage identities and crash-safe, same-source episode custody.

JSONL is a rebuildable view. The immutable validated completion record is the commit point.
Only receipt acquisition opts in by writing a descriptor next to an arm's JSONL.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import ValidationError

DESCRIPTOR = "receipt_execution.json"


def canonical_bytes(value: Any) -> bytes:
    """Encode an identity without platform whitespace or non-finite JSON numbers.

    Returns:
        The validated stage result."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def stage_key(stage: str, inputs: Any) -> str:
    """Hash a domain-separated, explicitly versioned canonical stage representation.

    Returns:
        The validated stage result."""
    return hashlib.sha256(
        canonical_bytes({"schema_version": f"receipt-{stage}.v1", "inputs": inputs})
    ).hexdigest()


def file_digest(path: Path) -> str:
    """Stream an artifact digest without loading checkpoints into memory.

    Returns:
        The validated stage result."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    """Validate temporary JSON then atomically publish, durably and without clobbering."""
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".receipt-tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if json.loads(Path(temporary).read_bytes()) != value:
            raise ValueError("temporary receipt JSON failed validation")
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError(
                        "immutable receipt record differs; refusing overwrite"
                    ) from None
        else:
            os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_row(row: dict, schema: dict) -> None:
    """Use the production writer's schema and semantic checks before committing a row."""
    from robot_sf.benchmark.map_runner.map_runner_jsonl import write_validated_to_handle  # noqa: PLC0415 - defer heavy runtime imports until the stage executes.

    write_validated_to_handle(io.StringIO(), schema, row)
    canonical_bytes(row)


def publish_episode(out_path: Path, schema: dict, row: dict) -> bool:
    """Commit completed observations, including collisions/timeouts, before JSONL append.

    Returns:
        The validated stage result."""
    descriptor_path = out_path.parent / DESCRIPTOR
    if not descriptor_path.is_file():
        return False
    descriptor = json.loads(descriptor_path.read_bytes())
    if descriptor.get("schema_version") != "receipt-execution.v1":
        raise ValueError("unsupported receipt execution descriptor")
    validate_row(row, schema)
    episode_id = row.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("completion requires an episode identity")
    record = {
        "schema_version": "receipt-episode.v1",
        **{key: descriptor[key] for key in ("execution_key", "source_sha", "run_id")},
        "episode_id": episode_id,
        "row_sha256": hashlib.sha256(canonical_bytes(row)).hexdigest(),
        "row": row,
    }
    name = hashlib.sha256(episode_id.encode()).hexdigest() + ".json"
    atomic_json(out_path.parent / "completed" / name, record, immutable=True)
    return True


def recover_episodes(out_path: Path, schema: dict, descriptor: dict) -> list[dict]:  # noqa: C901
    """Rebuild the view from valid commits; quarantine corrupt commits for missing-only retry.

    Called with the sweep's exclusive lock. Unpublished temporaries are reaped. Existing JSONL
    without completion provenance is refused: historical rows never acquire a new source identity.

    Returns:
        The validated stage result."""
    directory = out_path.parent
    marker = directory / DESCRIPTOR
    if marker.exists() and json.loads(marker.read_bytes()) != descriptor:
        raise ValueError("receipt execution/source differs; same-source continuation only")
    if not marker.exists() and out_path.exists() and out_path.stat().st_size:
        raise ValueError("legacy JSONL has no atomic completion provenance; use fresh output")
    atomic_json(marker, descriptor, immutable=True)
    rows, seen = [], set()
    for temporary in directory.glob("completed/.receipt-tmp-*"):
        temporary.unlink()
    for path in sorted(directory.glob("completed/*.json")):
        try:
            record = json.loads(path.read_bytes())
            if record.get("schema_version") != "receipt-episode.v1":
                raise ValueError("unknown completion schema")
            if any(
                record.get(key) != descriptor[key]
                for key in ("execution_key", "source_sha", "run_id")
            ):
                raise ValueError("completion provenance mismatch")
            row = record["row"]
            validate_row(row, schema)
            episode_id = row["episode_id"]
            if (
                record["episode_id"] != episode_id
                or record["row_sha256"] != hashlib.sha256(canonical_bytes(row)).hexdigest()
            ):
                raise ValueError("completion checksum mismatch")
            if path.stem != hashlib.sha256(episode_id.encode()).hexdigest() or episode_id in seen:
                raise ValueError("completion identity mismatch")
        except (ValueError, KeyError, TypeError, ValidationError):
            quarantine = directory / "invalid"
            quarantine.mkdir(exist_ok=True)
            os.replace(path, quarantine / f"{path.stem}-{file_digest(path)}.json")
            continue
        seen.add(episode_id)
        rows.append(row)
    descriptor_fd, temporary = tempfile.mkstemp(prefix=".receipt-tmp-", dir=directory)
    try:
        with os.fdopen(descriptor_fd, "wb") as handle:
            for row in rows:
                handle.write(canonical_bytes(row) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out_path)
        _sync_directory(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return rows

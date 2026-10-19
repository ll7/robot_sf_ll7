"""Versioned receipt stage identities and crash-safe, same-source episode custody.

JSONL is a rebuildable view. The immutable validated completion record is the commit point.
Only receipt acquisition opts in by writing a descriptor next to an arm's JSONL.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import ValidationError

DESCRIPTOR = "receipt_execution.json"


def encode_row(value: Any) -> Any:
    """Losslessly encode non-finite observation sentinels as explicit versioned JSON tags.

    Returns:
        Standard JSON data with a reserved tag for unbounded/undefined numeric observations.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "$receipt_float.v1": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
        }
    if isinstance(value, dict):
        if "$receipt_float.v1" in value:
            raise ValueError("reserved receipt numeric tag in raw observation")
        return {key: encode_row(child) for key, child in value.items()}
    if isinstance(value, list):
        return [encode_row(child) for child in value]
    return value


def decode_row(value: Any) -> Any:
    """Restore the observation's original numeric semantics for the production reader.

    Returns:
        Original row values, including infinity/NaN sentinels.
    """
    if isinstance(value, dict):
        if set(value) == {"$receipt_float.v1"}:
            return {"nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf")}[
                value["$receipt_float.v1"]
            ]
        return {key: decode_row(child) for key, child in value.items()}
    if isinstance(value, list):
        return [decode_row(child) for child in value]
    return value


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
        if canonical_bytes(json.loads(Path(temporary).read_bytes())) != canonical_bytes(value):
            raise ValueError("temporary receipt JSON failed validation")
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if canonical_bytes(json.loads(path.read_bytes())) != canonical_bytes(value):
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
    canonical_bytes(encode_row(row))


def find_receipt_file(out_path: Path, name: str = DESCRIPTOR) -> Path | None:
    """Resolve an arm's acquisition marker, including the pre-arm campaign marker.

    Returns:
        Closest receipt file within the arm, runs directory or campaign root.
    """
    for directory in [out_path.parent, out_path.parent.parent, out_path.parent.parent.parent]:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def publish_episode(out_path: Path, schema: dict, row: dict) -> bool:
    """Commit completed observations, including collisions/timeouts, before JSONL append.

    Returns:
        The validated stage result."""
    descriptor_path = find_receipt_file(out_path)
    if descriptor_path is None:
        return False
    descriptor = json.loads(descriptor_path.read_bytes())
    if descriptor.get("schema_version") != "receipt-execution.v1":
        raise ValueError("unsupported receipt execution descriptor")
    validate_row(row, schema)
    atomic_json(out_path.parent / DESCRIPTOR, descriptor, immutable=True)
    episode_id = row.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("completion requires an episode identity")
    record = {
        "schema_version": "receipt-episode.v1",
        **{key: descriptor[key] for key in ("execution_key", "source_sha", "run_id")},
        "episode_id": episode_id,
        "row_sha256": hashlib.sha256(canonical_bytes(encode_row(row))).hexdigest(),
        "row": encode_row(row),
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
            row = decode_row(record["row"])
            validate_row(row, schema)
            episode_id = row["episode_id"]
            if (
                record["episode_id"] != episode_id
                or record["row_sha256"]
                != hashlib.sha256(canonical_bytes(encode_row(row))).hexdigest()
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
                handle.write((json.dumps(row, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out_path)
        _sync_directory(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return rows


def rebuild_receipt_summary_inputs(ctx: Any, execution: Any) -> Any:
    """Fold every banked observation through the production metadata aggregator.

    Returns:
        Batch execution inputs for reporting, without scheduling an episode.
    """
    from robot_sf.benchmark.map_runner import map_runner as runner  # noqa: PLC0415
    from robot_sf.benchmark.map_runner import map_runner_batch_runner as batches  # noqa: PLC0415

    if find_receipt_file(ctx.out_path) is None:
        return execution
    rows = [json.loads(line) for line in ctx.out_path.read_text().splitlines() if line.strip()]
    totals = batches._initial_feasibility_totals()
    samples, native, adapted, runtime = False, 0, 0, None
    for row in rows:
        validate_row(row, ctx.schema)
        samples, native, adapted, runtime = batches._fold_worker_metadata(
            row,
            apply_worker_metadata_bridge=runner._apply_worker_metadata_bridge,
            feasibility_totals=totals,
            runtime_algorithm_contract=runtime,
            adapter_samples_seen=samples,
            adapter_native_steps=native,
            adapter_adapted_steps=adapted,
        )
    ctx.jobs = runner._build_seed_jobs(
        ctx.filtered, suite_seeds=ctx.suite_seeds, suite_key=ctx.suite_key
    )
    return execution._replace(
        wrote=len(rows),
        episode_records=rows,
        adapter_native_steps=native,
        adapter_adapted_steps=adapted,
        adapter_samples_seen=samples,
        runtime_algorithm_contract=runtime,
        feasibility_totals=totals,
    )


def compare_episode_observation(head: dict, baseline: dict) -> tuple[bool, int, bool, int]:
    """Compare observations through the pinned producer's outcome semantics.

    Returns:
        Head/baseline success and collision observations after metric compatibility checks.
    """
    from robot_sf.benchmark.metric_definitions import require_uniform_metric_schema  # noqa: PLC0415
    from scripts.ci import make_behaviour_receipt as producer  # noqa: PLC0415

    require_uniform_metric_schema([head, baseline])
    return (
        producer._row_success(head),
        producer._row_collisions(head),
        producer._row_success(baseline),
        producer._row_collisions(baseline),
    )

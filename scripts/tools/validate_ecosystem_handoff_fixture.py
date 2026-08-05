#!/usr/bin/env python3
"""Validate a Robot SF ecosystem handoff packet without importing Robot SF.

The validator uses only the standard library and ``jsonschema``. It validates
the positive packet and confirms that every bundled negative variant fails for
its declared reason code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

FIXTURE_ID = "robot_sf.ecosystem_handoff.v1"
FIXTURE_SCHEMA_VERSION = "robot_sf_ecosystem_handoff_fixture.v1"
ARTIFACT_SCHEMA_VERSION = "robot_sf_ecosystem_artifact_manifest.v1"
NEGATIVE_SCHEMA_VERSION = "robot_sf_ecosystem_negative_fixture.v1"


class PacketValidationError(ValueError):
    """A deterministic packet validation failure with a machine reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        """Create a validation failure with a machine-readable reason."""
        super().__init__(message)
        self.reason_code = reason_code


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PacketValidationError("duplicate_json_key", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except PacketValidationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise PacketValidationError("malformed_json", f"cannot load {path.name}: {exc}") from exc


def _load_object(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise PacketValidationError("malformed_json", f"{path.name} must contain an object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_relative(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.name == "":
        raise PacketValidationError("unsafe_path", f"unsafe packet path: {raw!r}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PacketValidationError("unsafe_path", f"packet path escapes root: {raw!r}")
    return resolved


def _verify_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise PacketValidationError("missing_checksums", "SHA256SUMS is missing")
    seen: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            digest, raw_path = line.split("  ", 1)
        except ValueError as exc:
            raise PacketValidationError(
                "malformed_checksums", f"malformed checksum line: {line!r}"
            ) from exc
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise PacketValidationError("malformed_checksums", f"invalid SHA-256: {digest!r}")
        path = _safe_relative(root, raw_path)
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            raise PacketValidationError(
                "duplicate_checksum_path", f"duplicate checksum path: {relative}"
            )
        seen.add(relative)
        if not path.is_file() or path.is_symlink():
            raise PacketValidationError(
                "missing_checksum_target", f"checksum target missing: {relative}"
            )
        if _sha256(path) != digest:
            raise PacketValidationError("checksum_mismatch", f"checksum mismatch: {relative}")
    expected = {
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise PacketValidationError(
            "checksum_coverage",
            f"checksum coverage mismatch; missing={missing}, extra={extra}",
        )


def _verify_artifact_manifest(root: Path) -> None:
    manifest = _load_object(root / "artifact_manifest.json")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise PacketValidationError(
            "unsupported_major_version", "unsupported artifact manifest version"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise PacketValidationError(
            "malformed_artifact_manifest", "artifact manifest files must be a list"
        )
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PacketValidationError(
                "malformed_artifact_manifest", "artifact entry must be an object"
            )
        path_value = entry.get("path")
        if not isinstance(path_value, str) or path_value in seen:
            raise PacketValidationError(
                "duplicate_artifact_identity", f"duplicate artifact path: {path_value!r}"
            )
        seen.add(path_value)
        path = _safe_relative(root, path_value)
        if not path.is_file():
            raise PacketValidationError("missing_artifact", f"artifact is missing: {path_value}")
        expected_digest = entry.get("sha256")
        if expected_digest != _sha256(path):
            raise PacketValidationError(
                "checksum_mismatch", f"artifact digest mismatch: {path_value}"
            )
        if entry.get("size_bytes") != path.stat().st_size:
            raise PacketValidationError(
                "artifact_size_mismatch", f"artifact size mismatch: {path_value}"
            )
    if manifest.get("self_excluded_from_manifest") is not True:
        raise PacketValidationError(
            "malformed_artifact_manifest", "artifact manifest must self-exclude"
        )


def _verify_fixture_manifest(root: Path) -> dict[str, Any]:
    manifest = _load_object(root / "fixture_manifest.json")
    if manifest.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise PacketValidationError(
            "unsupported_major_version", "unsupported fixture schema version"
        )
    if manifest.get("fixture_id") != FIXTURE_ID:
        raise PacketValidationError("fixture_identity", "fixture ID does not match the v1 contract")
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or contract.get("version") != "1.0.0":
        raise PacketValidationError(
            "contract_binding", "fixture does not bind contract version 1.0.0"
        )
    digest = contract.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise PacketValidationError("contract_binding", "fixture contract digest is invalid")
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("admissible") is not False:
        raise PacketValidationError(
            "evidence_admissibility", "fixture must be explicitly inadmissible"
        )
    config = manifest.get("config")
    if not isinstance(config, dict) or config.get("seed") != 271828:
        raise PacketValidationError("config_identity", "fixture seed/config identity is invalid")
    return manifest


def _validate_episode_row(line: str, index: int, schema: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate one episode row."""
    try:
        row = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
    except (PacketValidationError, json.JSONDecodeError) as exc:
        if isinstance(exc, PacketValidationError):
            raise
        raise PacketValidationError("malformed_episode", f"invalid JSON at line {index}") from exc
    if not isinstance(row, dict):
        raise PacketValidationError("malformed_episode", f"episode line {index} is not an object")
    try:
        jsonschema.validate(row, schema)
    except jsonschema.ValidationError as exc:
        raise PacketValidationError(
            "episode_schema", f"episode line {index} failed schema: {exc.message}"
        ) from exc
    if row.get("row_status") in {"fallback", "degraded"}:
        raise PacketValidationError(
            "fallback_or_degraded_success", f"episode line {index} is fallback/degraded"
        )
    if row.get("evidence_admissible") is not False:
        raise PacketValidationError("evidence_admissibility", f"episode line {index} is admissible")
    provenance = row.get("result_provenance")
    if not isinstance(provenance, dict) or not provenance.get("repo_commit"):
        raise PacketValidationError("missing_provenance", f"episode line {index} lacks provenance")
    return row


def _load_episode_rows(root: Path) -> list[dict[str, Any]]:
    """Load and validate all episode rows in one packet."""
    schema = _load_object(root / "episode.schema.v1.json")
    try:
        lines = (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PacketValidationError("missing_episodes", str(exc)) from exc
    if not lines:
        raise PacketValidationError("missing_episodes", "episodes.jsonl is empty")
    return [_validate_episode_row(line, index, schema) for index, line in enumerate(lines, start=1)]


def _verify_provenance(root: Path, rows: list[dict[str, Any]]) -> None:
    provenance_path = root / "episodes.jsonl.provenance.json"
    if not provenance_path.is_file():
        raise PacketValidationError("missing_provenance", "episodes provenance manifest is missing")
    provenance = _load_object(provenance_path)
    if provenance.get("schema_version") != "benchmark_result_provenance.v1":
        raise PacketValidationError("unsupported_major_version", "unsupported provenance version")
    if provenance.get("completeness", {}).get("status") != "complete":
        raise PacketValidationError("missing_provenance", "provenance is not complete")
    if len(provenance.get("rows", [])) != len(rows):
        raise PacketValidationError(
            "provenance_row_mismatch", "provenance row count differs from episodes"
        )


def _verify_derived_inputs(root: Path, rows: list[dict[str, Any]]) -> None:
    table = root / "table_input.csv"
    with table.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    if [row.get("episode_id") for row in parsed] != [row.get("episode_id") for row in rows]:
        raise PacketValidationError(
            "derived_input_mismatch", "table input does not preserve episode IDs"
        )
    figure = _load_object(root / "figure_input.json")
    points = figure.get("series", [{}])[0].get("points", [])
    if [point.get("episode_id") for point in points] != [row.get("episode_id") for row in rows]:
        raise PacketValidationError(
            "derived_input_mismatch", "figure input does not preserve episode IDs"
        )


def _validate_positive(root: Path) -> None:
    manifest = _verify_fixture_manifest(root)
    _verify_checksums(root)
    _verify_artifact_manifest(root)
    rows = _load_episode_rows(root)
    _verify_provenance(root, rows)
    _verify_derived_inputs(root, rows)
    expected_variants = manifest.get("negative_variants")
    if expected_variants != [
        "checksum_mismatch",
        "missing_provenance",
        "unsupported_major_version",
        "fallback_or_degraded_success",
        "duplicate_artifact_identity",
    ]:
        raise PacketValidationError(
            "fixture_identity", "negative variant roster is not the v1 roster"
        )


def _validate_variant(variant_root: Path) -> str:
    expected = _load_object(variant_root / "expected_failure.json")
    if expected.get("schema_version") != NEGATIVE_SCHEMA_VERSION:
        raise PacketValidationError(
            "unsupported_major_version", "negative fixture declaration is invalid"
        )
    expected_reason = expected.get("expected_reason_code")
    try:
        _verify_checksums(variant_root)
        _verify_fixture_manifest(variant_root)
        _verify_artifact_manifest(variant_root)
        rows = _load_episode_rows(variant_root)
        _verify_provenance(variant_root, rows)
        _verify_derived_inputs(variant_root, rows)
    except PacketValidationError as exc:
        return (
            exc.reason_code
            if exc.reason_code == expected_reason
            else f"unexpected:{exc.reason_code}"
        )
    return "unexpected:accepted"


def validate_packet(root: Path) -> int:
    """Validate one positive packet and every declared negative variant."""
    root = root.resolve()
    try:
        _validate_positive(root)
        variants = sorted((root / "negative").iterdir())
        if not variants:
            raise PacketValidationError(
                "missing_negative_variants", "negative variant directory is empty"
            )
        for variant in variants:
            observed = _validate_variant(variant)
            expected = _load_object(variant / "expected_failure.json").get("expected_reason_code")
            if observed != expected:
                raise PacketValidationError(
                    "negative_expectation",
                    f"{variant.name}: expected {expected!r}, observed {observed!r}",
                )
    except (PacketValidationError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: ecosystem handoff packet and {len(variants)} negative variants")
    return 0


def main() -> int:
    """Run the standalone packet validator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    args = parser.parse_args()
    return validate_packet(args.packet_dir)


if __name__ == "__main__":
    raise SystemExit(main())

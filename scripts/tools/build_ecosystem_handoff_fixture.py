#!/usr/bin/env python3
"""Build the small Robot SF ecosystem handoff conformance packet.

The packet is generated from the checked-in producer contract and existing
episode, provenance, CSV, schema, and checksum writers. It is diagnostic
conformance material only; it is not benchmark or scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from robot_sf.benchmark.aggregate import write_episode_csv
from robot_sf.benchmark.result_provenance import (
    build_row_result_provenance,
    validate_result_provenance_manifest,
)
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.benchmark.utils import _config_hash
from robot_sf.evidence.writers import sha256_file, write_sha256sums

REPOSITORY_URL = "https://github.com/ll7/robot_sf_ll7"
FIXTURE_ID = "robot_sf.ecosystem_handoff.v1"
FIXTURE_VERSION = "1.0.0"
FIXTURE_SCHEMA_VERSION = "robot_sf_ecosystem_handoff_fixture.v1"
ARTIFACT_SCHEMA_VERSION = "robot_sf_ecosystem_artifact_manifest.v1"
PACKET_CLOCK = "2026-01-01T00:00:00Z"
SEED = 271828
SCENARIO_ID = "conformance.minimal_success"
DEFAULT_OUTPUT = Path("tests/fixtures/ecosystem_handoff/v1")
CONTRACT_PATH = Path("robot_sf/benchmark/contracts/robot_sf_ecosystem_contract.v1.json")
CONTRACT_DIGEST_PATH = Path("robot_sf/benchmark/contracts/robot_sf_ecosystem_contract.v1.sha256")
EPISODE_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
PACKET_EPISODE_SCHEMA_ID = "https://robot-sf.dev/ecosystem-handoff/episode.schema.v1.json"


class FixtureBuildError(ValueError):
    """Raised when the producer contract or packet cannot be built."""


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FixtureBuildError(f"expected JSON object in {path}")
    return payload


def _contract_identity(root: Path) -> dict[str, Any]:
    contract_path = root / CONTRACT_PATH
    contract = _load_json(contract_path)
    fixtures = contract.get("canonical_public_fixtures")
    expected = {
        "fixture_id": FIXTURE_ID,
        "fixture_version": FIXTURE_VERSION,
        "path": (DEFAULT_OUTPUT / "fixture_manifest.json").as_posix(),
    }
    if not isinstance(fixtures, list) or expected not in fixtures:
        raise FixtureBuildError(
            "producer contract does not declare the expected canonical fixture "
            f"{expected!r}; regenerate the contract after updating the registry"
        )
    digest_block = contract.get("contract_digest")
    if not isinstance(digest_block, dict) or not isinstance(digest_block.get("value"), str):
        raise FixtureBuildError("producer contract has no SHA-256 contract digest")
    digest = digest_block["value"]
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise FixtureBuildError("producer contract digest is not lowercase SHA-256")
    sidecar = CONTRACT_DIGEST_PATH
    if not sidecar.is_file():
        raise FixtureBuildError(f"missing contract digest sidecar: {sidecar}")
    sidecar_fields = sidecar.read_text(encoding="ascii").split()
    if sidecar_fields != [sha256_file(contract_path), contract_path.name]:
        raise FixtureBuildError("contract digest sidecar disagrees with contract payload")
    return {
        "schema_version": contract.get("schema_version"),
        "version": contract.get("contract_version"),
        "digest": digest,
        "path": CONTRACT_PATH.as_posix(),
    }


def _config_identity() -> tuple[dict[str, Any], str]:
    config = {
        "fixture_id": FIXTURE_ID,
        "fixture_version": FIXTURE_VERSION,
        "clock": PACKET_CLOCK,
        "seed": SEED,
        "scenario_id": SCENARIO_ID,
        "execution_mode": "conformance",
    }
    return config, _config_hash(config)


def _episode_record(*, contract_digest: str, config_hash: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "version": "v1",
        "episode_id": "ecosystem-handoff-v1-episode-0001",
        "scenario_id": SCENARIO_ID,
        "scenario_params": {
            "fixture_id": FIXTURE_ID,
            "fixture_version": FIXTURE_VERSION,
            "execution_mode": "conformance",
        },
        "seed": SEED,
        "horizon": 4,
        "algo": "fixture_conformance",
        "observation_mode": "conformance",
        "observation_level": "oracle_full_state",
        "benchmark_track": "conformance",
        "track_schema_version": "conformance-track.v1",
        "algorithm_metadata": {
            "algorithm": "fixture_conformance",
            "canonical_algorithm": "fixture_conformance",
            "status": "ok",
            "baseline_category": "diagnostic",
        },
        "metric_parameters": {
            "threshold_signature": "ecosystem-handoff-v1",
            "threshold_profile": {
                "profile_id": "ecosystem-handoff-conformance-v1",
                "collision_distance_m": 0.0,
                "near_miss_distance_m": 0.0,
                "comfort_force_threshold": 0.0,
            },
        },
        "metrics": {
            "success": 1.0,
            "steps": 4,
            "collisions": 0,
            "path_length": 1.0,
            "goal_distance": 0.0,
            "time_to_yield_s": 4.0,
        },
        "termination_reason": "success",
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
        "integrity": {"contradictions": []},
        "config_hash": config_hash,
        # This is deliberately a digest-bound producer identity. A Git commit
        # is carried by a later revision envelope, not invented in a stable
        # fixture whose own commit would make regeneration self-referential.
        "git_hash": f"contract:{contract_digest}",
        "source_identity": {
            "repository_url": REPOSITORY_URL,
            "contract_digest": contract_digest,
        },
        "evidence_status": "diagnostic_only",
        "evidence_admissible": False,
        "row_status": "native",
        "timestamps": {"start": PACKET_CLOCK, "end": "2026-01-01T00:00:04Z"},
        "timing": {"steps_per_second": 1.0},
        "notes": "Conformance-only fixture; not scientific or benchmark evidence.",
        "tags": ["fixture", "conformance", "diagnostic-only"],
        "identity": {"fixture_id": FIXTURE_ID, "episode_index": 0},
    }
    record["result_provenance"] = build_row_result_provenance(
        episode_id=record["episode_id"],
        scenario_id=record["scenario_id"],
        seed=SEED,
        config_hash=config_hash,
        repo_commit=f"contract:{contract_digest}",
        raw_artifact_path="episodes.jsonl",
        jsonl_line=0,
        dt=1.0,
        horizon=4,
        record_forces=False,
        active_observation_mode="conformance",
        active_observation_level="oracle_full_state",
        postprocessing_steps=[
            {"step": "fixture_serialization", "status": "completed"},
            {"step": "conformance_validation", "status": "completed"},
        ],
    )
    return record


def _packet_episode_schema() -> dict[str, Any]:
    """Return the compact packet-boundary schema used by external validators."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": PACKET_EPISODE_SCHEMA_ID,
        "title": "Robot SF ecosystem handoff episode (v1)",
        "type": "object",
        "additionalProperties": True,
        "required": [
            "version",
            "episode_id",
            "scenario_id",
            "seed",
            "metrics",
            "termination_reason",
            "outcome",
            "integrity",
            "evidence_status",
            "evidence_admissible",
            "row_status",
            "result_provenance",
        ],
        "properties": {
            "version": {"const": "v1"},
            "episode_id": {"type": "string", "minLength": 1},
            "scenario_id": {"type": "string", "minLength": 1},
            "seed": {"type": "integer"},
            "metrics": {
                "type": "object",
                "required": ["success", "steps", "collisions", "path_length", "goal_distance"],
                "properties": {
                    "success": {"type": "number"},
                    "steps": {"type": "number"},
                    "collisions": {"type": "number"},
                    "path_length": {"type": "number"},
                    "goal_distance": {"type": "number"},
                },
            },
            "termination_reason": {"type": "string", "minLength": 1},
            "outcome": {"type": "object"},
            "integrity": {"type": "object"},
            "evidence_status": {"const": "diagnostic_only"},
            "evidence_admissible": {"const": False},
            "row_status": {"enum": ["native", "fallback", "degraded"]},
            "result_provenance": {
                "type": "object",
                "required": ["repo_commit", "episode_id", "scenario_id", "seed"],
                "properties": {
                    "repo_commit": {"type": "string", "minLength": 1},
                    "episode_id": {"type": "string", "minLength": 1},
                    "scenario_id": {"type": "string", "minLength": 1},
                    "seed": {"type": "integer"},
                },
            },
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _provenance_manifest(
    *, root: Path, record: dict[str, Any], contract: dict[str, Any], config_hash: str
) -> dict[str, Any]:
    episodes_path = root / "episodes.jsonl"
    manifest = {
        "schema_version": "benchmark_result_provenance.v1",
        "run": {
            "run_id": "ecosystem-handoff-v1",
            "repo_commit": f"contract:{contract['digest']}",
            "python_version": "3.11+",
            "invocation": "uv run python scripts/tools/build_ecosystem_handoff_fixture.py",
            "benchmark_profile": "conformance",
            "runner": "ecosystem_handoff_fixture",
            "protocol_version": "0.1.0",
            "execution_context": {
                "hostname": "portable",
                "cpu_model": "not-captured",
                "python_version": "3.11+",
                "platform": "portable",
                "thread_env": {},
            },
        },
        "inputs": {
            "schema_path": {
                "path": "episode.schema.v1.json",
                "sha256": sha256_file(root / "episode.schema.v1.json"),
                "artifact_status": "available",
            },
            "scenario_matrix": {
                "path": "fixture_manifest.json",
                "sha256": sha256_file(root / "fixture_manifest.json"),
                "artifact_status": "available",
            },
            "algo_config": {
                "path": "fixture_manifest.json",
                "sha256": sha256_file(root / "fixture_manifest.json"),
                "artifact_status": "available",
            },
        },
        "campaign_identity": {
            "scenario_matrix_hash": config_hash,
            "config_hash": config_hash,
            "suite_key": FIXTURE_ID,
            "total_jobs": 1,
            "written": 1,
        },
        "raw_artifacts": [
            {
                "kind": "episodes_jsonl",
                "path": "episodes.jsonl",
                "sha256": sha256_file(episodes_path),
                "artifact_status": "available",
            }
        ],
        "rows": [record["result_provenance"]],
        "derived_artifacts": [
            {
                "kind": "table_input",
                "path": "table_input.csv",
                "sha256": sha256_file(root / "table_input.csv"),
            },
            {
                "kind": "figure_input",
                "path": "figure_input.json",
                "sha256": sha256_file(root / "figure_input.json"),
            },
        ],
        "completeness": {
            "status": "complete",
            "required_fields_checked": [
                "schema_version",
                "run",
                "inputs",
                "campaign_identity",
                "completeness",
            ],
        },
    }
    validate_result_provenance_manifest(manifest)
    return manifest


def _write_artifact_manifest(root: Path) -> None:
    entries: list[dict[str, Any]] = []
    kinds = {
        "fixture_manifest.json": "fixture_identity",
        "contract_reference.json": "contract_identity",
        "episode.schema.v1.json": "schema",
        "episodes.jsonl": "episode_rows",
        "episodes.jsonl.provenance.json": "provenance",
        "table_input.csv": "table_ready_input",
        "figure_input.json": "figure_ready_input",
    }
    for name, kind in kinds.items():
        path = root / name
        if not path.is_file():
            raise FixtureBuildError(f"missing generated artifact before manifest: {path}")
        entries.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "kind": kind,
            }
        )
    _write_json(
        root / "artifact_manifest.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "fixture_id": FIXTURE_ID,
            "claim_boundary": "conformance-only; not scientific or benchmark evidence",
            "self_excluded_from_manifest": True,
            "files": entries,
        },
    )


def _write_portable_checksums(root: Path) -> None:
    # Use the production writer first, then normalize its labels to packet-local
    # paths so an examiner can run `sha256sum -c SHA256SUMS` from this directory
    # without knowing the producer checkout path.
    write_sha256sums(root)
    files = sorted(path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    lines = ["# AI-GENERATED NEEDS-REVIEW"]
    lines.extend(f"{sha256_file(path)}  {path.name}" for path in files)
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _positive_packet(root: Path, contract: dict[str, Any]) -> None:
    config, config_hash = _config_identity()
    schema_source = _repo_root() / EPISODE_SCHEMA_PATH
    record = _episode_record(contract_digest=contract["digest"], config_hash=config_hash)
    # Validate with the authoritative production schema before writing a
    # compact standalone schema for external consumers. The packet schema is
    # intentionally distinct from the canonical schema to avoid distributing
    # duplicate schema content in every negative variant.
    validate_episode(record, load_schema(schema_source))
    packet_schema_path = root / "episode.schema.v1.json"
    _write_json(packet_schema_path, _packet_episode_schema())
    validate_episode(record, load_schema(packet_schema_path))
    _write_jsonl(root / "episodes.jsonl", [record])
    table_path = root / "table_input.csv"
    write_episode_csv([record], table_path)
    # The production writer uses platform-neutral CSV semantics but emits CRLF
    # by default. Normalize the packet to LF so Git and POSIX consumers see one
    # stable byte representation.
    table_path.write_bytes(table_path.read_bytes().replace(b"\r\n", b"\n"))
    _write_json(
        root / "figure_input.json",
        {
            "schema_version": "robot_sf_ecosystem_figure_input.v1",
            "fixture_id": FIXTURE_ID,
            "claim_boundary": "conformance-only; not a scientific figure",
            "series": [
                {
                    "series_id": "fixture_conformance",
                    "points": [
                        {
                            "episode_id": record["episode_id"],
                            "steps": record["metrics"]["steps"],
                            "collisions": record["metrics"]["collisions"],
                            "success": record["metrics"]["success"],
                        }
                    ],
                }
            ],
        },
    )
    fixture_manifest = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_id": FIXTURE_ID,
        "fixture_version": FIXTURE_VERSION,
        "contract": contract,
        "producer": {
            "repository_url": REPOSITORY_URL,
            "source_identity": {
                "kind": "digest_bound_contract",
                "contract_digest": contract["digest"],
            },
        },
        "config": {**config, "sha256": config_hash},
        "evidence": {
            "class": "diagnostic",
            "admissible": False,
            "claim_boundary": "Conformance material only; it supports no planner, safety, dissertation, or benchmark claim.",
        },
        "payload": {
            "contract_reference": "contract_reference.json",
            "schema": "episode.schema.v1.json",
            "episodes": "episodes.jsonl",
            "provenance": "episodes.jsonl.provenance.json",
            "artifact_manifest": "artifact_manifest.json",
            "table": "table_input.csv",
            "figure": "figure_input.json",
            "checksums": "SHA256SUMS",
        },
        "expected_consumer_assertions": [
            "contract_version_and_digest_match",
            "positive_episode_schema_and_provenance_validate",
            "fallback_and_degraded_rows_are_not_admissible_success",
            "artifact_manifest_and_sha256sums_verify",
            "table_and_figure_inputs_are_derived_from_episode_rows",
        ],
        "negative_variants": [
            "checksum_mismatch",
            "missing_provenance",
            "unsupported_major_version",
            "fallback_or_degraded_success",
            "duplicate_artifact_identity",
        ],
    }
    _write_json(root / "fixture_manifest.json", fixture_manifest)
    _write_json(
        root / "contract_reference.json",
        {
            "schema_version": "robot_sf_ecosystem_contract_reference.v1",
            "fixture_id": FIXTURE_ID,
            "contract_version": contract["version"],
            "contract_digest": contract["digest"],
            "contract_path": contract["path"],
        },
    )
    _write_json(
        root / "episodes.jsonl.provenance.json",
        _provenance_manifest(root=root, record=record, contract=contract, config_hash=config_hash),
    )
    _write_artifact_manifest(root)
    _write_portable_checksums(root)


def _copy_variant_base(root: Path, variant_root: Path) -> None:
    variant_root.mkdir(parents=True, exist_ok=True)
    for path in root.iterdir():
        if path.name in {"SHA256SUMS", "negative"} or not path.is_file():
            continue
        shutil.copy2(path, variant_root / path.name)


def _load_artifact_manifest(variant_root: Path) -> tuple[Path, dict[str, Any]]:
    """Load one mutable artifact manifest and return its path."""
    manifest_path = variant_root / "artifact_manifest.json"
    return manifest_path, _load_json(manifest_path)


def _mutate_checksum_mismatch(manifest: dict[str, Any]) -> None:
    """Make one inner artifact digest invalid."""
    for entry in manifest["files"]:
        if entry["path"] == "episodes.jsonl":
            entry["sha256"] = "0" * 64
            return
    raise FixtureBuildError("positive artifact manifest lacks episodes.jsonl")


def _mutate_missing_provenance(variant_root: Path, manifest: dict[str, Any]) -> None:
    """Remove the row provenance artifact and its manifest entry."""
    (variant_root / "episodes.jsonl.provenance.json").unlink()
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != "episodes.jsonl.provenance.json"
    ]


def _refresh_manifest_entry(manifest: dict[str, Any], variant_root: Path, name: str) -> None:
    """Refresh one manifest entry after changing its file."""
    path = variant_root / name
    for entry in manifest["files"]:
        if entry["path"] == name:
            entry["sha256"] = sha256_file(path)
            entry["size_bytes"] = path.stat().st_size
            return
    raise FixtureBuildError(f"artifact manifest lacks {name}")


def _mutate_unsupported_major(variant_root: Path, manifest: dict[str, Any]) -> None:
    """Change the fixture schema major while keeping its outer checksums valid."""
    fixture_path = variant_root / "fixture_manifest.json"
    fixture = _load_json(fixture_path)
    fixture["schema_version"] = "robot_sf_ecosystem_handoff_fixture.v2"
    fixture["fixture_version"] = "2.0.0"
    _write_json(fixture_path, fixture)
    _refresh_manifest_entry(manifest, variant_root, "fixture_manifest.json")


def _mutate_fallback_success(variant_root: Path, manifest: dict[str, Any]) -> None:
    """Mark the successful row as a fallback while preserving inadmissibility."""
    episode_path = variant_root / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text().splitlines()]
    rows[0]["row_status"] = "fallback"
    _write_jsonl(episode_path, rows)
    _refresh_manifest_entry(manifest, variant_root, "episodes.jsonl")


def _mutate_duplicate_artifact(manifest: dict[str, Any]) -> None:
    """Duplicate one artifact identity in the manifest."""
    manifest["files"].append(dict(manifest["files"][0]))


def _mutate_variant(variant_root: Path, variant_id: str) -> None:
    """Apply one documented adversarial mutation to a packet copy."""
    manifest_path, manifest = _load_artifact_manifest(variant_root)
    handlers = {
        "checksum_mismatch": lambda: _mutate_checksum_mismatch(manifest),
        "missing_provenance": lambda: _mutate_missing_provenance(variant_root, manifest),
        "unsupported_major_version": lambda: _mutate_unsupported_major(variant_root, manifest),
        "fallback_or_degraded_success": lambda: _mutate_fallback_success(variant_root, manifest),
        "duplicate_artifact_identity": lambda: _mutate_duplicate_artifact(manifest),
    }
    try:
        handler = handlers[variant_id]
    except KeyError as exc:
        raise FixtureBuildError(f"unknown negative variant {variant_id}") from exc
    handler()
    _write_json(manifest_path, manifest)


def _write_negative_variants(root: Path) -> None:
    variants = {
        "checksum_mismatch": "checksum_mismatch",
        "missing_provenance": "missing_provenance",
        "unsupported_major_version": "unsupported_major_version",
        "fallback_or_degraded_success": "fallback_or_degraded_success",
        "duplicate_artifact_identity": "duplicate_artifact_identity",
    }
    negative_root = root / "negative"
    if negative_root.exists():
        shutil.rmtree(negative_root)
    for variant_id, reason_code in variants.items():
        variant_root = negative_root / variant_id
        _copy_variant_base(root, variant_root)
        _mutate_variant(variant_root, variant_id)
        _write_json(
            variant_root / "expected_failure.json",
            {
                "schema_version": "robot_sf_ecosystem_negative_fixture.v1",
                "variant_id": variant_id,
                "expected_reason_code": reason_code,
            },
        )
        _write_portable_checksums(variant_root)


def generate(output_dir: Path, *, overwrite: bool = False) -> None:
    """Generate a deterministic positive packet and its negative variants."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FixtureBuildError(
                f"output directory is not empty: {output_dir}; pass --overwrite to regenerate"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _contract_identity(_repo_root())
    _positive_packet(output_dir, contract)
    _write_negative_variants(output_dir)


def _compare_trees(expected: Path, actual: Path) -> list[str]:
    expected_files = {path.relative_to(expected) for path in expected.rglob("*") if path.is_file()}
    actual_files = {path.relative_to(actual) for path in actual.rglob("*") if path.is_file()}
    errors: list[str] = []
    for relative in sorted(expected_files | actual_files):
        expected_path = expected / relative
        actual_path = actual / relative
        if not expected_path.is_file():
            errors.append(f"unexpected generated file: {relative}")
        elif not actual_path.is_file():
            errors.append(f"missing generated file: {relative}")
        elif expected_path.read_bytes() != actual_path.read_bytes():
            errors.append(f"byte mismatch: {relative}")
    return errors


def check(output_dir: Path) -> int:
    """Compare the checked-in packet with a fresh deterministic generation."""
    if not output_dir.is_dir():
        print(f"fixture output is missing: {output_dir}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="robot-sf-ecosystem-handoff-") as temp_dir:
        generated = Path(temp_dir) / "v1"
        generate(generated, overwrite=False)
        errors = _compare_trees(generated, output_dir.resolve())
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"ecosystem handoff fixture is deterministic: {output_dir}")
    return 0


def validate(output_dir: Path) -> int:
    """Run the standalone validator against one packet directory."""
    validator = _repo_root() / "scripts/tools/validate_ecosystem_handoff_fixture.py"
    result = subprocess.run(
        [sys.executable, str(validator), "--packet-dir", str(output_dir)], check=False
    )
    return result.returncode


def main() -> int:
    """Run the fixture generator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.check and args.validate:
        parser.error("--check and --validate are mutually exclusive")
    try:
        if args.check:
            return check(args.output_dir)
        if args.validate:
            return validate(args.output_dir)
        generate(args.output_dir, overwrite=args.overwrite)
    except (FixtureBuildError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote ecosystem handoff fixture: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

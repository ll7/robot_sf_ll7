"""Copy completed acquisition evidence and bind portable custody metadata.

Run from the delivery checkout with a relative --source checkout path. Scientific
values and binary trajectories are preserved; only artifact references and their
byte checksums are rebound after the acquisition driver's path projection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "docs/context/robsweep"


def digest(path: Path) -> str:
    """Hash an actual artifact, not a logical record identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_inputs(value: dict) -> None:
    """Resolve producer input hashes against the archived byte-identical copy."""
    if not isinstance(value.get("inputs"), dict):
        return
    for item in value["inputs"].values():
        if isinstance(item, dict) and str(item.get("path", "")).startswith(
            ("configs/benchmarks/robsweep/", "docs/context/robsweep/inputs/")
        ):
            item["path"] = "docs/context/robsweep/inputs/" + Path(item["path"]).name
            if item.get("sha256") != digest(ROOT / item["path"]):
                raise ValueError("Archived input bytes disagree with producer provenance")


def bind_raw_artifacts(value: dict) -> None:
    """Rebind byte checksums only after the projected row artifact exists."""
    artifacts = value.get("raw_artifacts")
    if not isinstance(artifacts, list):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Producer raw-artifact entry must be an object")
        path = artifact.get("path", "")
        if path.endswith(".jsonl"):
            artifact["path"] = path + ".gz"
            artifact["sha256"] = digest(ROOT / artifact["path"])
            artifact["kind"] = "portable_episodes_jsonl_gzip"
            artifact["artifact_status"] = "available"


def bind(value: object) -> object:
    """Bind available projected rows and archived inputs recursively."""
    if isinstance(value, list):
        return [bind(item) for item in value]
    if isinstance(value, str) and value.startswith("configs/benchmarks/robsweep/"):
        return "docs/context/robsweep/inputs/" + Path(value).name
    if not isinstance(value, dict):
        return value
    value = {key: bind(item) for key, item in value.items()}
    bind_raw_artifacts(value)
    if "raw_artifact" in value and str(value["raw_artifact"]).endswith(".jsonl"):
        value["raw_artifact"] = str(value["raw_artifact"]) + ".gz"
    bind_inputs(value)
    return value


def check_acquisition(source_bundle: Path) -> int:
    """Refuse partial or failed batches before public projection."""
    receipts = list((source_bundle / "evidence/episodes").glob("*_execution.json"))
    if len(receipts) != 73 or list((source_bundle / "evidence/episodes").glob("*.jsonl")):
        raise ValueError("Acquisition must complete all 73 batches before custody projection")
    for path in receipts:
        payload = json.loads(path.read_text())
        result = payload["result"]
        prefix = payload.get("completed_prefix_rows", 0)
        if prefix and (
            payload.get("planner") != "predictive_mppi"
            or not payload.get("prefix_bytes_preserved")
            or payload.get("total_retained_records") != 300
        ):
            raise ValueError(f"Invalid continuation custody: {path.name}")
        if (
            result.get("failed_jobs", 0)
            or result.get("error")
            or result.get("written", 0) + prefix != 300
        ):
            raise ValueError(f"Incomplete/failed acquisition receipt: {path.name}")
    return len(receipts)


def check_compressed_paths() -> None:
    """Check compressed traces and archived source as well as visible text."""
    for path in (BUNDLE / "evidence").rglob("*.gz"):
        with gzip.open(path, "rt") as handle:
            if "/home/" in handle.read():
                raise ValueError(f"Compressed home path remains: {path.relative_to(ROOT)}")


def preserve_input_identity() -> None:
    """Separate exact source bytes from relocated retrieval pointers."""
    study_path = BUNDLE / "inputs/study.json"
    (BUNDLE / "inputs/study.source.txt").write_bytes(study_path.read_bytes())
    study_path.write_text(json.dumps(bind(json.loads(study_path.read_text())), indent=2) + "\n")
    for name, collection, key in (
        ("acquisition_source.json", "source_files", "path"),
        ("input_locations.json", "files", "repository_config_path"),
    ):
        path = BUNDLE / "evidence" / name
        payload = json.loads(path.read_text())
        for item in payload[collection]:
            original = item.get(key, "")
            if original.startswith("configs/benchmarks/robsweep/"):
                item["acquisition_path_parts"] = list(Path(item.pop(key)).parts)
            if item.get("artifact_path", "").endswith("/study.json"):
                item["artifact_path"] = "docs/context/robsweep/inputs/study.source.txt"
        path.write_text(json.dumps(payload, indent=2) + "\n")


def project(source: Path) -> None:
    """Preserve complete inputs and outputs with resolvable portable references."""
    source = source.resolve()
    source_bundle = source / "docs/context/robsweep"
    receipt_count = check_acquisition(source_bundle)
    for name in ("inputs", "evidence"):
        shutil.copytree(source_bundle / name, BUNDLE / name, dirs_exist_ok=True)
    (BUNDLE / "evidence/SHA256SUMS").unlink(missing_ok=True)
    preserve_input_identity()
    transformations = []

    for path in sorted((BUNDLE / "evidence").rglob("*")):
        if not path.is_file() or path.suffix in (".npz", ".gz"):
            continue
        original = path.read_text()
        projected = original.replace(str(source) + "/", "").replace(str(source), ".")
        if path.suffix == ".json":
            projected = json.dumps(bind(json.loads(projected)), indent=2) + "\n"
        if "/home/" in projected:
            raise ValueError(f"Unprojected home path remains: {path.relative_to(ROOT)}")
        if projected != original:
            path.write_text(projected)
            transformations.append(
                {"artifact_path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            )
    check_compressed_paths()
    custody = {
        "schema": "examiner_portable_custody.v1",
        "operations": [
            "Strip acquisition checkout prefix from text references.",
            "Bind producer raw-artifact path and sha256 to portable compressed rows.",
            "Bind input paths to byte-identical archived configs.",
            "Retain exact study metadata in study.source.txt; relocate retrieval pointers in study.json. Original config layout is retained as path components.",
        ],
        "scientific_values_changed": False,
        "source_receipts_checked": receipt_count,
        "transformed_artifacts": transformations,
        "original_byte_checksums": "Superseded raw JSONL checksums are replaced by actual portable gzip checksums; no original-byte parity certificate is claimed.",
        "logical_hashes": "Embedded campaign/config/context hashes identify structured producer inputs, not whole-file byte checksums. hash_locations.json identifies the containing artifact and JSON pointer.",
    }
    (BUNDLE / "evidence/portable_custody.json").write_text(json.dumps(custody, indent=2) + "\n")


def locate_hashes() -> None:
    """Give embedded digests a durable containing artifact and exact pointer."""
    entries = {}

    def visit(value: object, pointer: str, artifact: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                visit(item, pointer + "/" + escaped, artifact)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, pointer + f"/{i}", artifact)
        elif isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value):
            key = (value, artifact)
            if key not in entries:
                entries[key] = {
                    "sha256": value,
                    "artifact_path": artifact,
                    "json_pointer": pointer,
                    "occurrences": 0,
                }
            entries[key]["occurrences"] += 1

    for path in sorted((BUNDLE / "evidence").rglob("*.json")):
        if path.name == "hash_locations.json":
            continue
        visit(json.loads(path.read_text()), "", str(path.relative_to(ROOT)))
    (BUNDLE / "evidence/hash_locations.json").write_text(
        json.dumps(list(entries.values()), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    project(args.source)
    locate_hashes()
